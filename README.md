# GLMP: General-purpose LLM-mediated Pathology model

Demo for **LLM-mediated pathology models for robust cross-institution generalization**.

GLMP generates patch embeddings through intermediate text descriptions:

1. Group tissue patches using pathology foundation model features.
2. Generate biology-focused descriptions of representative patches with an MLLM.
3. Encode the descriptions with a text embedding model.
4. Compute patch embeddings as weighted sums of cluster embeddings using soft cluster memberships.

## Run the demo

Install the dependencies and configure Hugging Face and Google API credentials:

```bash
pip install -r requirements.txt
export HF_TOKEN="your-hugging-face-token"
export GOOGLE_API_KEY="your-google-api-key"
./run.sh
```

The demo uses Virchow2, Gemini 2.5 Pro, and Gemini Embedding on a breast tissue slide. The runner downloads the slide and saves results under `Samples/`. The description prompt is in `prompt.txt`.
