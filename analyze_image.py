import os
import argparse
import csv
import torch
import numpy as np
import keras
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def load_metadata_from_csv(csv_path):
    """
    Parses classes.csv and returns a mapping from basename to metadata dictionary.
    """
    metadata = {}
    if not os.path.exists(csv_path):
        print(f"Warning: Metadata file {csv_path} not found. Will fallback to parsing filenames.")
        return metadata
        
    print(f"Loading metadata from {csv_path}...")
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
    """
    Generates artist and title fallback from filename (e.g. 'claude-monet_water-lilies.jpg').
    """
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

def main():
    parser = argparse.ArgumentParser(description="Analyze an image for Impressionist characteristics and suggest similar works.")
    parser.add_argument("image_path", type=str, help="Path to the image file to analyze.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for Impressionism detection (0.0 to 1.0).")
    parser.add_argument("--top_n", type=int, default=5, help="Number of similar artworks to recommend.")
    args = parser.parse_args()
    
    # 1. Load Trained Keras Model
    model_path = "impressionism_classifier.keras"
    if not os.path.exists(model_path):
        print(f"Error: Trained model file '{model_path}' not found. Please run train_classifier.py first.")
        return
        
    print("Loading trained classifier...")
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return
        
    # 2. Load Embeddings Database
    embeddings_path = "embeddings.npz"
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings database '{embeddings_path}' not found. Please run extract_embeddings.py first.")
        return
        
    print("Loading embeddings database...")
    db = np.load(embeddings_path)
    db_embeddings = db["embeddings"]
    db_labels = db["labels"]
    db_filenames = db["filenames"]
    
    # Filter for Impressionism images only for recommendation
    imp_mask = (db_labels == "Impressionism")
    imp_embeddings = db_embeddings[imp_mask]
    imp_filenames = db_filenames[imp_mask]
    
    if len(imp_embeddings) == 0:
        print("Error: No Impressionist images found in the embeddings database.")
        return
        
    # 3. Load Metadata
    metadata_csv = os.path.join("balanced_art_dataset", "classes.csv")
    metadata_map = load_metadata_from_csv(metadata_csv)
    
    # 4. Load CLIP Model for Query Feature Extraction
    print("Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        clip_model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading CLIP: {e}")
        return
        
    # 5. Process query image and extract embedding
    print(f"Processing query image: {args.image_path}...")
    if not os.path.exists(args.image_path):
        print(f"Error: Query image file '{args.image_path}' not found.")
        return
        
    try:
        query_image = Image.open(args.image_path).convert("RGB")
        inputs = processor(images=query_image, return_tensors="pt").to(device)
        with torch.no_grad():
            query_features = clip_model.get_image_features(**inputs)
            # L2 normalize the embedding
            query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
            query_emb = query_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting features from query image: {e}")
        return
        
    # 6. Predict Impressionism Probability
    # Reshape for Keras (1, 512)
    pred_prob = float(model.predict(np.expand_dims(query_emb, axis=0), verbose=0)[0][0])
    
    print("\n" + "="*50)
    print(f"Результаты анализа изображения: {os.path.basename(args.image_path)}")
    print(f"Вероятность импрессионизма: {pred_prob * 100:.2f}% (Порог: {args.threshold * 100:.2f}%)")
    print("="*50)
    
    if pred_prob >= args.threshold:
        print("Статус: Изображение содержит характерные признаки импрессионистической живописи.")
        print(f"\nПоиск наиболее схожих произведений в базе данных (Топ-{args.top_n}):")
        
        # 7. Compute Cosine Similarities (Dot product of normalized vectors)
        similarities = np.dot(imp_embeddings, query_emb)
        
        # Get top N indices
        top_indices = np.argsort(similarities)[::-1][:args.top_n]
        
        for idx in top_indices:
            sim_score = similarities[idx]
            sim_percentage = max(0.0, float(sim_score)) * 100
            
            rel_path = imp_filenames[idx]
            basename = os.path.basename(rel_path)
            
            # Retrieve metadata
            meta = metadata_map.get(basename)
            if not meta:
                meta = parse_filename_fallback(rel_path)
                
            print(f"- Сходство: {sim_percentage:.2f}% | Автор: {meta['artist']} | Название: '{meta['title']}' (Файл: {rel_path})")
    else:
        print("Статус: Изображение НЕ содержит выраженных признаков импрессионистической живописи.")
        
    print("="*50)

if __name__ == "__main__":
    main()
