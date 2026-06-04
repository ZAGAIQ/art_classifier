import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def main():
    data_dir = "balanced_art_dataset"
    subdirs = ["Baroque", "Cubism", "Impressionism", "Minimalism", "Post_Impressionism", "Ukiyo_e"]
    output_file = "embeddings.npz"
    
    # 1. Load CLIP Model
    print("Loading CLIP (openai/clip-vit-base-patch32)...")
    model_name = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # 2. Gather all image paths
    image_paths = []
    labels = []
    relative_paths = []
    
    for sd in subdirs:
        sd_path = os.path.join(data_dir, sd)
        if not os.path.isdir(sd_path):
            print(f"Warning: Directory {sd} not found, skipping.")
            continue
        
        for fname in os.listdir(sd_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(sd_path, fname))
                labels.append(sd)
                relative_paths.append(f"{sd}/{fname}")
                
    total_images = len(image_paths)
    print(f"Found {total_images} images in total across all classes.")
    
    # 3. Extract embeddings in batches
    batch_size = 64
    all_embeddings = []
    valid_labels = []
    valid_relative_paths = []
    
    print("Extracting embeddings (this might take a few minutes)...")
    
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_rel_paths = relative_paths[i:i+batch_size]
        
        imgs = []
        indices_to_keep = []
        
        for idx, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(img)
                indices_to_keep.append(idx)
            except Exception as e:
                print(f"\nWarning: Could not open {path}. Error: {e}. Skipping.")
                
        if not imgs:
            continue
            
        try:
            # Process batch
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                # L2 normalize the embeddings
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                features_np = features.cpu().numpy()
                
            all_embeddings.append(features_np)
            for idx in indices_to_keep:
                valid_labels.append(batch_labels[idx])
                valid_relative_paths.append(batch_rel_paths[idx])
        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {e}. Skipping batch.")
            
        if (i // batch_size) % 5 == 0 or i + batch_size >= total_images:
            print(f"Processed {min(i + batch_size, total_images)} / {total_images} images.")
            
    if not all_embeddings:
        print("No embeddings were successfully extracted!")
        return
        
    embeddings_concat = np.concatenate(all_embeddings, axis=0)
    print(f"\nSuccessfully extracted embeddings shape: {embeddings_concat.shape}")
    
    # 4. Save to disk
    print(f"Saving to {output_file}...")
    np.savez_compressed(
        output_file,
        embeddings=embeddings_concat,
        labels=np.array(valid_labels),
        filenames=np.array(valid_relative_paths)
    )
    print("Embeddings saved successfully!")

if __name__ == "__main__":
    main()
