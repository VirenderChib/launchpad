import json
import os
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from dotenv import load_dotenv


def tokenize_dataset(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print(f"Loading tokenizer for {model_name}...")
    load_dotenv()  # Make sure to load environment variables
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    print(f"HF Token available: {'Yes' if HF_TOKEN else 'No'}")
    
    if not HF_TOKEN:
        raise ValueError("HUGGINGFACE_TOKEN not found in environment variables. Please set it in .env file")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=HF_TOKEN,trust_remote_code=True )
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")

    # Get absolute paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    split_dir = os.path.join(project_root, "data/processed/splits")
    tokenized_dir = os.path.join(project_root, "data/processed/tokenized")
    os.makedirs(tokenized_dir, exist_ok=True)
    
    print(f"Using splits directory: {split_dir}")
    print(f"Using tokenized directory: {tokenized_dir}")

    for split in ["train", "validation", "test"]:
        path = os.path.join(split_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"Skipping {split}, file not found.")
            continue

        print(f"🚀 Tokenizing {split} split...")
        encoded_data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                record = json.loads(line)
                text = record["prompt"] + "\n" + record["response"]
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=1024,
                    padding=False
                )
                encoded_data.append(tokens)

        torch.save(encoded_data, os.path.join(tokenized_dir, f"{split}.pt"))
        print(f"Saved tokenized {split} data -> {os.path.join(tokenized_dir, f'{split}.pt')}")

    print("Tokenization complete!")
