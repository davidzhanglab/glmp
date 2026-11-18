import argparse
import os
import json
import logging
import time
from pathlib import Path
import sys
import getpass

import google.generativeai as genai
from PIL import Image

def natural_sort_key_group(group_name):
    parts = group_name.split('_'); return int(parts[-1]) if len(parts) > 1 and parts[0].lower() == "group" else group_name
    
def find_image_files(image_dir):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    return image_files

def call_gemini_with_retry(model, content_parts, generation_config, safety_settings, max_retries = 3): 
    """Makes a call to the Gemini model with exponential backoff for retries."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(content_parts, generation_config=generation_config, safety_settings=safety_settings)
            return response.text
        except Exception as e:
            logging.error(f"Gemini API Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                logging.warning(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate structured descriptions for clustered image patches.")
    parser.add_argument("--sample_dir", required=True, type=Path, help="Path to the main sample directory.")
    parser.add_argument("--analysis_name", required=True, type=str, help="Name of the analysis, corresponding to the results folder prefix.")
    parser.add_argument("--prompt_txt_path", required=True, type=Path, help="Path to text file with the main instructions.")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of times to run the Gemini analysis.")
    parser.add_argument("--max_retries", type=int, default=3, help="Number of retries for each run.")
    parser.add_argument("--model_name", default="gemini-2.5-pro", help="Gemini model to use.")
    parser.add_argument("--max_images_per_group", type=int, default=25, help="Maximum number of images to use per group.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Model top_p.")
    parser.add_argument("--top_k", type=int, default=1, help="Model top_k.")
    parser.add_argument("--max_output_tokens", type=int, default=8192, help="Maximum number of output tokens.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    
    sample_dir = args.sample_dir
    results_dir = sample_dir / f"{args.analysis_name}_results"
    grouped_patches_dir = results_dir / "grouped_patches"

    if not grouped_patches_dir.is_dir():
        logging.error(f"Input directory for groups not found: {grouped_patches_dir}"); sys.exit(1)

    try:
        with open(args.prompt_txt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {args.prompt_txt_path}"); sys.exit(1)

    group_subfolders = [d.name for d in grouped_patches_dir.iterdir() if d.is_dir() and d.name.lower().startswith("group_")]
    group_subfolders.sort(key=natural_sort_key_group)
    if not group_subfolders:
        logging.error(f"No 'Group_*' subfolders found in {grouped_patches_dir}. Aborting."); sys.exit(1)
    
    all_content_parts = [prompt_text]
    for group_name in group_subfolders:
        group_path = grouped_patches_dir / group_name
        group_id_num = group_name.split('_')[-1]
        group_intro = f"\nThe subsequent image patches pertain to Group {group_id_num}."
        all_content_parts.append(group_intro)
        image_paths = find_image_files(group_path)[:args.max_images_per_group]
        for img_path in image_paths:
            try:
                all_content_parts.append(Image.open(img_path))
            except Exception as e:
                logging.error(f"Failed to load image {img_path}: {e}")
    
    num_images = sum(isinstance(part, Image.Image) for part in all_content_parts)
    logging.info(f"Content prepared with 1 prompt, {len(all_content_parts) - num_images - 1} group headers, and {num_images} images.")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.info("GOOGLE_API_KEY environment variable not set.")
        try:
            api_key = getpass.getpass("Please enter your Google AI API key: ")
        except Exception as e:
            logging.error(f"Could not read API key from prompt: {e}")
            sys.exit(1)
    if not api_key:
        logging.error("No API key provided. Aborting.")
        sys.exit(1)

    try:
        genai.configure(api_key=api_key)
        logging.info("Successfully configured the Google AI (genai) API.")
    except Exception as e:
        logging.error(f"Failed to configure the Google AI API. Is the key valid? Error: {e}")
        sys.exit(1)

    logging.info(f"Using Google AI model: {args.model_name}")


    safety_settings = {
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    }
    model = genai.GenerativeModel(args.model_name)
    generation_config = genai.GenerationConfig(
    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, max_output_tokens=args.max_output_tokens, response_mime_type="application/json"
    )
    
    llm_analysis_dir = results_dir / "Gemini_2.5_Pro_Output"
    for i in range(1, args.num_runs + 1):
        run_output_dir = llm_analysis_dir / f"run_{i:02d}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = run_output_dir / "pathology_summary.json"

        if output_json_path.exists():
            logging.info(f"Output for run {i} already exists. Skipping.")
            continue
        
        logging.info(f"--- Starting Run {i}/{args.num_runs} ---")
        response_text = call_gemini_with_retry(model, all_content_parts, generation_config, safety_settings, args.max_retries)
        
        if response_text:
            try:
                parsed_json = json.loads(response_text)
                with open(output_json_path, "w", encoding='utf-8') as f:
                    json.dump(parsed_json, f, indent=2, ensure_ascii=False)
                logging.info(f"Success! Run {i} summary saved to {output_json_path}")
            except json.JSONDecodeError:
                logging.error(f"Run {i} failed: Response was not valid JSON.")
        else:
            logging.error(f"Failed to get a valid response for run {i} after {args.max_retries} attempts.")

    logging.info(f"All runs for sample {sample_dir.name} completed.")
    
if __name__ == "__main__":
    main()
