import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# ======= CONFIGURATION =======
# Path to your *final fine-tuned multimodal model* (merged with LoRA if applicable)
MODEL_PATH = "./outputs/mm_finetuned"
# Output folder for the quantized model
QUANTIZED_MODEL_DIR = "./outputs/mm_finetuned_gptq"
# Quantization parameters
BITS = 4
GROUP_SIZE = 128
DESC_ACT = False
USE_TRITON = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Representative calibration texts (can be short since multimodal)
CALIB_TEXTS = [
    "Describe the image in detail.",
    "Write a caption for the given image.",
    "What objects are visible in this scene?",
    "Summarize the visual context in one sentence."
]
CALIB_SAMPLES = 128
MAX_LENGTH = 128
# ==============================================


# -------- Load tokenizer --------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# -------- Prepare calibration examples --------
def prepare_calib_examples(tokenizer, texts, num_samples=128, max_length=128):
    texts = (texts * ((num_samples // len(texts)) + 1))[:num_samples]
    examples = []
    for t in texts:
        enc = tokenizer(t, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # 1D tensors
        examples.append(enc)
    return examples

examples = prepare_calib_examples(tokenizer, CALIB_TEXTS, CALIB_SAMPLES, MAX_LENGTH)

# -------- Quantization Config --------
quantize_config = BaseQuantizeConfig(
    bits=BITS,
    group_size=GROUP_SIZE,
    desc_act=DESC_ACT,
)

# -------- Run GPTQ Quantization --------
print(f"🚀 Starting GPTQ quantization ({BITS}-bit)...")
quantizer = AutoGPTQForCausalLM.from_pretrained(
    MODEL_PATH,
    quantize_config,
    device_map="auto",
)
quantizer.quantize(examples=examples, use_triton=USE_TRITON)

# -------- Save Quantized Model --------
os.makedirs(QUANTIZED_MODEL_DIR, exist_ok=True)
quantizer.save_pretrained(QUANTIZED_MODEL_DIR)
tokenizer.save_pretrained(QUANTIZED_MODEL_DIR)
print(f"✅ Quantized model saved at: {QUANTIZED_MODEL_DIR}")

# -------- Compare Non-Quantized vs Quantized --------

def measure_inference(model, tokenizer, prompt, device="cuda", n_tokens=30, n_trials=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Warmup
    _ = model.generate(**inputs, max_new_tokens=n_tokens)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(n_trials):
        start = time.time()
        _ = model.generate(**inputs, max_new_tokens=n_tokens)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    return avg_time


prompt = "Describe the image in one short sentence."

print("\n⏱️ Measuring inference speed...")

# 1️⃣ Load original model
print("Loading original model...")
orig_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

# Measure speed
orig_time = measure_inference(orig_model, tokenizer, prompt)
orig_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None
print(f"🧠 Original model - avg inference time: {orig_time:.2f}s, memory used: {orig_mem:.2f} GB" if orig_mem else f"🧠 Original model - avg inference time: {orig_time:.2f}s")

# 2️⃣ Load quantized model
print("\nLoading quantized model...")
qmodel = AutoGPTQForCausalLM.from_quantized(QUANTIZED_MODEL_DIR, device_map="auto")
q_time = measure_inference(qmodel, tokenizer, prompt)
q_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None
print(f"⚡ Quantized model - avg inference time: {q_time:.2f}s, memory used: {q_mem:.2f} GB" if q_mem else f"⚡ Quantized model - avg inference time: {q_time:.2f}s")

# 3️⃣ Print comparison
speedup = orig_time / q_time if q_time > 0 else 1.0
print("\n📊 COMPARISON SUMMARY")
print(f"  ➤ Inference Speedup: {speedup:.2f}x faster")
if orig_mem and q_mem:
    print(f"  ➤ Memory Reduction: {(1 - q_mem/orig_mem) * 100:.1f}% less VRAM")

# Optional: quick qualitative output
print("\n🧩 Sample Output Check (quantized model):")
inputs = tokenizer(prompt, return_tensors="pt").to(qmodel.device)
outputs = qmodel.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
