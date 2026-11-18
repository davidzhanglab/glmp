#!/bin/bash

# GLMP Pipeline Runner

set -e  # Exit on any error

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

: "${HF_TOKEN:?Set HF_TOKEN to your Hugging Face token}"
: "${GOOGLE_API_KEY:?Set GOOGLE_API_KEY to your Google API key}"

echo "=========================================="
echo "GLMP Pipeline Runner"
echo "=========================================="

# Prepare Sample Data
echo "Preparing sample data..."
folder="TXG_0001879_07142022"
source_url="https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif"
target_path="Samples/${folder}/data/histology.tif"

mkdir -p "$(dirname "$target_path")"
wget --quiet --show-progress "$source_url" -O "$target_path"

# Add Pixel Size (MPP) Information
echo "Adding pixel size information..."
MPP="0.3638" # Provide the MPP value for the WSI
echo "$MPP" > "Samples/${folder}/data/pixel-size.txt"

# Tissue patch grouping
echo "Grouping tissue patches..."
MIN_FOREGROUND_PERCENTAGE="0.001"

python detect_foreground.py \
    --sample_dir "Samples/${folder}" \
    --min_foreground_percentage "$MIN_FOREGROUND_PERCENTAGE"

HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
NUM_CLUSTERS=10
NUM_SAMPLES_PER_CLUSTER=25
MODEL_NAME="${MODEL_NAME:-paige-ai/Virchow2}"

python extract_and_cluster.py \
       --sample_dir "Samples/${folder}" \
       --login_token "$HUGGING_FACE_HUB_TOKEN" \
       --model_name "$MODEL_NAME" \
       --k "$NUM_CLUSTERS" \
       --n_patches "$NUM_SAMPLES_PER_CLUSTER"

# Generate Biological Descriptions with MLLM
echo "Generating biological descriptions with MLLM..."
# Set the name of the analysis, which should match the prefix of the clustering results folder.
ANALYSIS_NAME="mymethod"
# Set the number of times to repeat the MLLM analysis.
NUM_RUNS=10

python run_mllm.py \
    --sample_dir "Samples/${folder}" \
    --analysis_name "$ANALYSIS_NAME" \
    --prompt_txt_path "prompt.txt" \
    --num_runs "$NUM_RUNS"

# Generate Weighted MLLM Embeddings
echo "Generating weighted MLLM embeddings..."
python generate_mllm_embeddings.py \
    --sample_dir "Samples/${folder}" \
    --analysis_name "$ANALYSIS_NAME"

echo "Pipeline completed successfully!"
