import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

PHYSICAL_PATCH_SIZE_UM = 128.0
VISUALIZATION_MAX_WIDTH = 2048 # Max width of output visualization
ROI_DETECTION_THUMBNAIL_SIZE = 2048 # Size of thumbnail for HSV thresholding
FOREGROUND_COLOR = (65, 105, 225, 140)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect foreground tissue patches using adaptive HSV thresholding."
    )
    parser.add_argument("--sample_dir", required=True, type=str, help="Path to the sample directory.")
    parser.add_argument(
        "--min_foreground_percentage",
        type=float,
        default=0.01,
        help="The minimum percentage of a patch that must be foreground to be included."
    )
    return parser.parse_args()

def get_hsv_thresholds_from_thumbnail(pil_thumbnail_image: Image.Image) -> tuple:
    """Calculates adaptive HSV thresholds using Otsu's method on non-zero pixels."""
    hsv_image = cv2.cvtColor(np.array(pil_thumbnail_image), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)

    h_nz, s_nz, v_nz = h[h > 0], s[s > 0], v[v > 0]

    hthresh = threshold_otsu(h_nz) if h_nz.size > 10 and np.std(h_nz) > 1 else 90.0
    sthresh = threshold_otsu(s_nz) if s_nz.size > 10 and np.std(s_nz) > 1 else 20.0
    vthresh = threshold_otsu(v_nz) if v_nz.size > 10 and np.std(v_nz) > 1 else 220.0

    min_hsv = np.array([hthresh, sthresh, 70], np.uint8)
    max_hsv = np.array([179, 255, vthresh], np.uint8)
    
    print(f"  Adaptive HSV thresholds: MinH={min_hsv[0]:.0f}, MinS={min_hsv[1]:.0f}, MaxV={max_hsv[2]:.0f}")
    return min_hsv, max_hsv

def detect_and_visualize_foreground(sample_dir: Path, min_fg_percent: float):
    sample_id = sample_dir.name
    print(f"\n{'='*40}\nDetecting foreground for sample: {sample_id}")

    data_dir = sample_dir / "data"
    image_path = data_dir / "histology.tif"
    mpp_path = data_dir / "pixel-size.txt"
    
    if not all([image_path.exists(), mpp_path.exists()]):
        print(f"  Error: Missing required files in '{data_dir}'. Aborting.")
        return

    wsi = None
    try:
        wsi = Image.open(image_path)
        img_w, img_h = wsi.size

        with open(mpp_path, 'r') as f:
            mpp = float(f.read().strip())
        patch_size_px = int(round(PHYSICAL_PATCH_SIZE_UM / mpp))
        print(f"  - Image: {img_w}x{img_h} | MPP: {mpp:.4f} | Patch Size: {patch_size_px}x{patch_size_px} px")

        print("Analyzing thumbnail to determine tissue area...")
        thumbnail = wsi.copy()
        thumbnail.thumbnail((ROI_DETECTION_THUMBNAIL_SIZE, ROI_DETECTION_THUMBNAIL_SIZE))
        min_hsv, max_hsv = get_hsv_thresholds_from_thumbnail(thumbnail)

        num_patches_x = img_w // patch_size_px
        num_patches_y = img_h // patch_size_px
        total_patches = num_patches_x * num_patches_y
        pbar = tqdm(total=total_patches, desc=f"Filtering patches")

        foreground_metadata = []
        patch_id_counter = 0
        for y_idx in range(num_patches_y):
            for x_idx in range(num_patches_x):
                patch_id_counter += 1
                top = y_idx * patch_size_px
                left = x_idx * patch_size_px
                right = left + patch_size_px
                bottom = top + patch_size_px
                
                patch_pil = wsi.crop((left, top, right, bottom))
                patch_hsv = cv2.cvtColor(np.array(patch_pil), cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(patch_hsv, min_hsv, max_hsv)
                
                foreground_percentage = cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])
                
                if foreground_percentage >= min_fg_percent:
                    patch_info = {
                        "id": patch_id_counter,
                        "coordinates": {"top": top, "left": left, "bottom": bottom, "right": right}
                    }
                    foreground_metadata.append(patch_info)
                pbar.update(1)
        pbar.close()
        
        print(f"Identified {len(foreground_metadata)} foreground patches.")

        json_path = sample_dir / "foreground_patches.json"
        with open(json_path, 'w') as f:
            json.dump(foreground_metadata, f, indent=2)
        print(f"Saved foreground patch data to: {json_path}")
        
        print("Creating segmentation visualization...")
        downscale = max(1, img_w // VISUALIZATION_MAX_WIDTH)
        thumb_size = (img_w // downscale, img_h // downscale)
        base_thumb = wsi.resize(thumb_size, Image.Resampling.LANCZOS).convert("RGBA")
        overlay = Image.new("RGBA", base_thumb.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        for meta in foreground_metadata:
            coords = meta["coordinates"]
            thumb_coords = [
                coords['left'] // downscale, 
                coords['top'] // downscale, 
                coords['right'] // downscale, 
                coords['bottom'] // downscale
            ]
            draw.rectangle(thumb_coords, fill=FOREGROUND_COLOR)
        
        composite_img = Image.alpha_composite(base_thumb, overlay).convert("RGB")
        segmentation_path = sample_dir / "foreground_segmentation.jpg"
        composite_img.save(segmentation_path, quality=95)
        print(f"Saved segmentation visualization to: {segmentation_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if wsi:
            wsi.close()
        print(f"\nFinished processing {sample_id}.")
        print("="*40)

if __name__ == "__main__":
    args = parse_args()
    sample_directory = Path(args.sample_dir)
    if not sample_directory.is_dir():
        print(f"ERROR: Directory not found: '{sample_directory}'")
        exit(1)
    
    detect_and_visualize_foreground(sample_directory, args.min_foreground_percentage)