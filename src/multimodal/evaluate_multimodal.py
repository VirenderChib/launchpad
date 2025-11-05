import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from .vision_encoder import VisionEncoder
import evaluate
import math

def evaluate_model():
    print("🚀 Starting model evaluation...")

    # ---- PATHS ----
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(root, "multimodal_data", "mm_cleaned.jsonl")
    model_path = os.path.join(root, "outputs", "mm_finetuned")

    # ---- LOAD MODEL + TOKENIZER ----
    print("📦 Loading model and tokenizer...")
    vision_encoder = VisionEncoder()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    # ---- METRICS ----
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    print("📊 Metrics initialized (BLEU, ROUGE, Perplexity).")

    # ---- LOAD DATA ----
    print(f"🧠 Loading evaluation data from: {data_path}")
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f][:20]  # evaluate only on 20 samples for speed

    predictions, references, losses = [], [], []

    # ---- EVALUATION LOOP ----
    for ex in tqdm(data, desc="Evaluating"):
        # 1️⃣ Encode image
        image_emb = vision_encoder.encode_image(ex["image"])

        # 2️⃣ Create multimodal input prompt
        prompt = "Describe this image in technical context."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 3️⃣ Generate model output
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(ex["caption"])

        # 4️⃣ Compute loss for perplexity
        with torch.no_grad():
            labels = tokenizer(ex["caption"], return_tensors="pt").input_ids.to(model.device)
            loss = model(input_ids=inputs["input_ids"], labels=labels).loss
            losses.append(loss.item())

    # ---- METRIC COMPUTATION ----
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
    perplexity = math.exp(sum(losses) / len(losses))

    # ---- PRINT RESULTS ----
    print("\n📈 Evaluation Results:")
    print(f"ROUGE: {rouge_score}")
    print(f"BLEU: {bleu_score}")
    print(f"Perplexity: {perplexity:.4f}")

    print("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_model()
