#!/bin/bash
echo "Starting vLLM server..."

# Path to our model inside the container
MODEL_PATH="/workspace/Final_output/Model"

# Run the vLLM API server with explicit tokenizer
python3 -m vllm.entrypoints.api_server \
    --model $MODEL_PATH \
    --tokenizer "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --port 8000 \
    --max-model-len 2048
