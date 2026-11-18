import argparse
import json
import os
import sys
import time
from pathlib import Path
import getpass
import numpy as np
from tqdm import tqdm
from google import genai
from google.genai import types

def safe_embed(client: genai.Client, model_name: str, content: str, task_type: str, max_retries=10, initial_delay=1):
    """
    Generate a text embedding with exponential backoff for rate limits.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model=model_name,
                contents=content,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return response.embeddings[0].values
        except Exception as e:
            is_429_error = '429' in str(e) or 'resource has been exhausted' in str(e).lower()
            if is_429_error and attempt < max_retries:
                print(f"Resource Exhausted error, retry {attempt}/{max_retries} in {delay}s…")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Failed embedding after {attempt} attempts. Error: {e}")
                raise
    raise Exception("Embedding failed after all retries.")


def generate_embeddings_for_file(client: genai.Client, model_name: str, task_type: str, input_json_path: Path, output_json_path: Path):
    """Generates and saves embeddings for a single pathology_summary.json file."""
    if output_json_path.exists():
        print(f"Overwriting existing (potentially empty) file: {output_json_path}")
        os.remove(output_json_path)
    print(f"Generating embeddings for {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = {}

    for group, fields in data.items():
        text = "\n".join(f"{k}: {v}" for k, v in fields.items())
        result[group] = safe_embed(client, model_name, text, task_type)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"Saved embeddings to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate weighted patch embeddings from MLLM group summaries.")
    parser.add_argument("--sample_dir", required=True, type=Path, help="Path to the main sample directory.")
    parser.add_argument("--analysis_name", required=True, type=str, help="Name of the analysis, corresponding to the results folder prefix.")
    parser.add_argument("--task_type", default="RETRIEVAL_DOCUMENT", help="Task type for the embedding model.")
    parser.add_argument("--model", "-m", default="gemini-embedding-exp-03-07",
                        help="Text embedding model name.")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Please enter your Google AI API key: ")
    if not api_key:
        parser.error("A Google API key is required.")
    client = genai.Client(api_key=api_key)

    analysis_results_dir = args.sample_dir / f"{args.analysis_name}_results"
    llm_analysis_path = analysis_results_dir / "Gemini_2.5_Pro_Output"
    final_embeddings_dir = analysis_results_dir / "Gemini_2.5_Pro_embeddings"
    final_embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    if not llm_analysis_path.is_dir():
        print(f"Error: LLM Analysis directory not found: {llm_analysis_path}")
        sys.exit(1)

    search_pattern = f"{args.analysis_name}_k*_info.json"
    found_files = list(analysis_results_dir.glob(search_pattern))
    if not found_files:
        print(f"Error: Could not find patch info JSON file in {analysis_results_dir} matching pattern '{search_pattern}'")
        sys.exit(1)
    patch_info_json_path = found_files[0]
    print(f"\nUsing patch probability file: {patch_info_json_path}")
    
    print(f"\n: Generating group embeddings for each run ---")
    run_dirs = sorted([d for d in llm_analysis_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    for run_dir in run_dirs:
        input_json = run_dir / "pathology_summary.json"
        output_json = run_dir / "pathology_summary_embeddings.json"
        if input_json.exists():
            try:
                generate_embeddings_for_file(client, args.model, args.task_type, input_json, output_json)
            except Exception as e:
                print(f"Could not process {input_json}. Error: {e}")

    accumulator = {}
    for run_dir in run_dirs:
        embedding_file = run_dir / "pathology_summary_embeddings.json"
        if not embedding_file.exists():
            print(f"Warning: Embedding file not found for {run_dir.name}, skipping.")
            continue
        with open(embedding_file, 'r') as f:
            data = json.load(f)
        for group, emb_list in data.items():
            if group not in accumulator:
                accumulator[group] = []
            accumulator[group].append(np.array(emb_list))
            
    averaged_prototypes = {group: np.mean(np.array(vectors), axis=0).tolist() for group, vectors in accumulator.items() if vectors}

    print(f"\n: Saving weighted embeddings to {final_embeddings_dir} ---")
    with open(patch_info_json_path, 'r') as f:
        patch_metadata = json.load(f)

    prototypes_np = {k: np.array(v) for k, v in averaged_prototypes.items()}
    embedding_dim = len(next(iter(prototypes_np.values())))

    for patch_meta in tqdm(patch_metadata, desc="Saving weighted MLLM embeddings"):
        output_stem = Path(patch_meta['patch_filename']).stem
        probabilities = patch_meta.get('probabilities', {})
        weighted_vector = np.zeros(embedding_dim, dtype=np.float32)
        for group_name, prob in probabilities.items():
            if group_name in prototypes_np:
                weighted_vector += prototypes_np[group_name] * prob
        output_path = final_embeddings_dir / f"{output_stem}.npy"
        np.save(output_path, weighted_vector)

    print(f"\nSuccessfully saved {len(patch_metadata)} weighted MLLM embeddings.")
    
if __name__ == "__main__":
    main()
