import argparse
import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import login
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from timm.data import create_transform, resolve_data_config
from timm.layers import SwiGLUPacked
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

VISUALIZATION_MAX_WIDTH = 2048

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster foreground patch features and sample representative patches."
    )
    parser.add_argument("--sample_dir", required=True, type=str, help="Path to the sample directory.")
    parser.add_argument("--login_token", required=True, type=str, help="Hugging Face token.")
    parser.add_argument("--model_name", required=True, type=str, help="Patch clustering model.")
    parser.add_argument("--method", default="mymethod", type=str, help="Name of the embedding method for output directories.")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters for the foreground.")
    parser.add_argument("--n_patches", type=int, default=25, help="Number of sample patches to save per cluster.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Softmax temperature for clustering.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def load_model(hf_token: str, model_name: str):
    login(token=hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model '{model_name}' on device: {device}...")
    try:
        model = timm.create_model(f"hf-hub:{model_name}", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model.eval().to(device)
        config = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**config)
        print("Model and transform loaded successfully.")
        return model, transform, device
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)


def softmax(x, temperature=0.1):
    e_x = np.exp((x - np.max(x, axis=1, keepdims=True)) / temperature)
    return e_x / e_x.sum(axis=1, keepdims=True)

def get_group_color(group_index: int, k_total: int, alpha=180):
    cmap_name = "tab10" if k_total <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    color_index = group_index % len(cmap.colors)
    color_float = cmap(color_index)
    return tuple(int(c * 255) for c in color_float[:3]) + (alpha,)

def get_font(font_size: int) -> ImageFont.ImageFont:
    for font_path in ["DejaVuSans.ttf", "arial.ttf", "Verdana.ttf"]:
        try: return ImageFont.truetype(font_path, size=font_size)
        except IOError: continue
    return ImageFont.load_default()

def main():
    args = parse_args()
    set_seed(args.seed)

    sample_dir = Path(args.sample_dir)
    original_image_path = sample_dir / "data" / "histology.tif"
    foreground_json_path = sample_dir / "foreground_patches.json"
    results_dir = sample_dir / f"{args.method}_results"

    print(f"\nProcessing Sample: {sample_dir.name}")
    for path in [original_image_path, foreground_json_path]:
        if not path.exists():
            print(f"Error: Required input file not found: '{path}'. Aborting."); exit(1)
    if results_dir.exists(): shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    full_res_image = None
    try:
        print("Opening full-resolution TIF image...")
        full_res_image = Image.open(original_image_path)
        original_image_size = full_res_image.size
        
        with open(foreground_json_path, 'r') as f:
            foreground_metadata = json.load(f)
        
        if not foreground_metadata:
            print("No foreground patches found in JSON file. Exiting."); return

        model, transform, device = load_model(args.login_token, args.model_name)

        # --- Step 1: Extract features for foreground patches ---
        print(f"\n--- Extracting {args.method} features for {len(foreground_metadata)} patches ---")
        all_embeddings = []
        pbar = tqdm(foreground_metadata, desc="Extracting embeddings")
        for meta in pbar:
            coords = meta["coordinates"]
            patch_pil = full_res_image.crop((coords['left'], coords['top'], coords['right'], coords['bottom']))
            tensor = transform(patch_pil).unsqueeze(0).to(device)
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(tensor)
                else:
                    output = model(tensor)
            
            cls_token = output[:, 0]
            patch_tokens = output[:, 5:].mean(dim=1)
            embedding = torch.cat([cls_token, patch_tokens], dim=-1).cpu().numpy().squeeze()
            
            output_fname = f"id{meta['id']:05d}_top{coords['top']}_left{coords['left']}_bottom{coords['bottom']}_right{coords['right']}.npy"
            
            all_embeddings.append(embedding)
            meta['patch_filename'] = output_fname
        
        all_embeddings = np.array(all_embeddings)
        
        # --- Step 2: Cluster embeddings using Cosine Similarity equivalent ---
        print(f"\n--- Soft-Clustering Foreground into k={args.k} Groups ---")
        normalized_embeddings = normalize(all_embeddings, norm='l2', axis=1)
        
        kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init='auto').fit(normalized_embeddings)
        
        normalized_centroids = normalize(kmeans.cluster_centers_, norm='l2', axis=1)
        
        similarities = np.dot(normalized_embeddings, normalized_centroids.T)
        probabilities = softmax(similarities, temperature=args.temperature)
        hard_labels = np.argmax(probabilities, axis=1)
        
        for i, meta in enumerate(foreground_metadata):
            meta["group_id"] = int(hard_labels[i] + 1)
            meta["probabilities"] = {f"Group {j+1}": float(p) for j, p in enumerate(probabilities[i])}

        # --- Step 3: Generating Final Outputs ---
        print("\n--- Generating Final Outputs ---")
        json_fname = f"{args.method}_k{args.k}_info.json"
        with open(results_dir / json_fname, 'w') as f: json.dump(foreground_metadata, f, indent=2)
        print(f"Saved patch info to: {results_dir / json_fname}")

        print("Creating colored footprint visualization...")
        downscale = max(1, original_image_size[0] // VISUALIZATION_MAX_WIDTH)
        thumb_size = (original_image_size[0] // downscale, original_image_size[1] // downscale)
        base_thumb = full_res_image.resize(thumb_size, Image.Resampling.LANCZOS).convert("RGBA")
        overlay = Image.new("RGBA", base_thumb.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for meta in foreground_metadata:
            coords = meta["coordinates"]
            color = get_group_color(meta["group_id"] - 1, args.k)
            thumb_coords = [coords['left']//downscale, coords['top']//downscale, coords['right']//downscale, coords['bottom']//downscale]
            draw.rectangle(thumb_coords, fill=color)
        composite_img = Image.alpha_composite(base_thumb, overlay).convert("RGB")
        img_h = composite_img.height
        font_size = max(16, int(img_h * 0.025)); padding = int(img_h * 0.05); available_height = img_h - (2 * padding)
        while True:
            font = get_font(font_size)
            box_size = int(font_size * 1.5); margin = int(font_size * 0.5)
            total_legend_height = (args.k * box_size) + ((args.k - 1) * margin)
            if total_legend_height <= available_height or font_size <= 10: break
            font_size -= 1
        text_bbox = font.getbbox(f"Group {args.k}"); text_width = text_bbox[2] - text_bbox[0]
        legend_width = margin + box_size + margin + text_width + margin
        final_img = Image.new("RGB", (composite_img.width + legend_width, img_h), "white")
        final_img.paste(composite_img, (0, 0))
        draw_legend = ImageDraw.Draw(final_img)
        start_y = (img_h - total_legend_height) // 2
        for i in range(args.k):
            color = get_group_color(i, args.k, alpha=255)
            box_y = start_y + i * (box_size + margin)
            box_x = composite_img.width + margin
            draw_legend.rectangle([box_x, box_y, box_x + box_size, box_y + box_size], fill=color)
            text_label = f"Group {i + 1}"
            text_bbox = font.getbbox(text_label); text_h = text_bbox[3] - text_bbox[1]
            text_y = box_y + (box_size - text_h) // 2
            draw_legend.text((box_x + box_size + margin, text_y), text_label, fill="black", font=font)
        footprint_fname = f"{args.method}_k{args.k}_clustering_results.jpg"
        final_img.save(results_dir / footprint_fname, quality=95)
        print(f"Saved footprint to: {results_dir / footprint_fname}")

        print("\n--- Saving Representative Patches ---")
        grouped_patches_dir = results_dir / "grouped_patches"
        patches_by_group = {i: [] for i in range(1, args.k + 1)}
        for meta in foreground_metadata: patches_by_group[meta['group_id']].append(meta)

        for group_id, group_meta_list in sorted(patches_by_group.items()):
            num_to_sample = min(args.n_patches, len(group_meta_list))
            if num_to_sample == 0: continue
            print(f"  > Group {group_id}: Sampling {num_to_sample} of {len(group_meta_list)} patches.")
            sampled_meta = random.sample(group_meta_list, num_to_sample)
            group_output_dir = grouped_patches_dir / f"Group_{group_id}"
            group_output_dir.mkdir(parents=True, exist_ok=True)

            for meta in sampled_meta:
                coords = meta["coordinates"]
                patch_img = full_res_image.crop((coords['left'], coords['top'], coords['right'], coords['bottom']))

                patch_jpg_path = group_output_dir / Path(meta["patch_filename"]).name.replace('.npy', '.jpg')
                patch_img.save(patch_jpg_path)

    finally:
        if full_res_image: full_res_image.close()
    
    print(f"\nProcessing complete for {sample_dir.name}.")

if __name__ == "__main__":
    main()
