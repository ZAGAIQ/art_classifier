import os
import csv
import numpy as np
import tensorflow as tf
import keras
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='templates')

# Global variables to store models and embeddings database
classifier_model = None
clip_model = None
clip_processor = None
db_embeddings = None
db_labels = None
db_filenames = None
imp_embeddings = None
imp_filenames = None
metadata_map = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_metadata_from_csv(csv_path):
    metadata = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Metadata file {csv_path} not found.")
        return metadata
    try:
        with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "")
                if filename:
                    basename = os.path.basename(filename)
                    metadata[basename] = {
                        "artist": row.get("artist", "Unknown Artist").title(),
                        "title": row.get("description", "Unknown Title").replace("-", " ").title()
                    }
    except Exception as e:
        print(f"Warning: Error parsing {csv_path}: {e}")
    return metadata

def parse_filename_fallback(filename):
    basename = os.path.basename(filename)
    name_w_ext, _ = os.path.splitext(basename)
    if "_" in name_w_ext:
        parts = name_w_ext.split("_", 1)
        artist = parts[0].replace("-", " ").title()
        title = parts[1].replace("-", " ").title()
    else:
        artist = "Unknown Artist"
        title = name_w_ext.replace("-", " ").title()
    return {"artist": artist, "title": title}

def init_app():
    global classifier_model, clip_model, clip_processor
    global db_embeddings, db_labels, db_filenames
    global imp_embeddings, imp_filenames, metadata_map

    # 1. Load Keras Classifier
    model_path = "impressionism_classifier.keras"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model '{model_path}' not found. Please train model first.")
    print("Loading trained classifier...")
    classifier_model = keras.models.load_model(model_path)

    # 2. Load Embeddings Database
    embeddings_path = "embeddings.npz"
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings database '{embeddings_path}' not found.")
    print("Loading embeddings database...")
    db = np.load(embeddings_path)
    db_embeddings = db["embeddings"]
    db_labels = db["labels"]
    db_filenames = db["filenames"]

    # Filter for Impressionism images only for recommendation
    imp_mask = (db_labels == "Impressionism")
    imp_embeddings = db_embeddings[imp_mask]
    imp_filenames = db_filenames[imp_mask]
    print(f"Loaded {len(imp_embeddings)} Impressionist embeddings for recommendation.")

    # 3. Load Metadata
    metadata_csv = os.path.join("balanced_art_dataset", "classes.csv")
    metadata_map = load_metadata_from_csv(metadata_csv)

    # 4. Load CLIP Model
    print(f"Loading CLIP model on {device}...")
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    print("Initialization complete!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dataset/<path:path>')
def send_dataset_file(path):
    # Safe serving of images from dataset
    return send_from_directory('balanced_art_dataset', path)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image file selected"}), 400
    
    threshold = float(request.form.get('threshold', 0.5))
    top_n = int(request.form.get('top_n', 6))

    try:
        # Load and convert image
        img = Image.open(file.stream).convert("RGB")
        
        # Extract features using CLIP
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            query_features = clip_model.get_image_features(**inputs)
            query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
            query_emb = query_features.cpu().numpy().flatten()

        # Predict probability
        pred_prob = float(classifier_model.predict(np.expand_dims(query_emb, axis=0), verbose=0)[0][0])
        is_impressionism = pred_prob >= threshold

        similar_images = []
        if is_impressionism and len(imp_embeddings) > 0:
            # Calculate Cosine Similarities
            similarities = np.dot(imp_embeddings, query_emb)
            # Get top N indices
            top_indices = np.argsort(similarities)[::-1][:top_n]

            for idx in top_indices:
                sim_score = float(similarities[idx])
                sim_percentage = max(0.0, sim_score) * 100
                rel_path = imp_filenames[idx]
                basename = os.path.basename(rel_path)

                # Retrieve metadata
                meta = metadata_map.get(basename)
                if not meta:
                    meta = parse_filename_fallback(rel_path)

                similar_images.append({
                    "filename": rel_path.replace("\\", "/"),
                    "similarity": sim_percentage,
                    "artist": meta["artist"],
                    "title": meta["title"]
                })

        return jsonify({
            "is_impressionism": is_impressionism,
            "probability": pred_prob,
            "similar_images": similar_images
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

if __name__ == '__main__':
    init_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
