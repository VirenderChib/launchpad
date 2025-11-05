import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from datasets import Dataset
from transformers import BitsAndBytesConfig as bnb
import transformers

# ✅ Small runtime performance boost
torch.backends.cudnn.benchmark = True

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Model selection (keep same)
model_name = "meta-llama/Llama-2-7b-hf"
print(f"Loading model: {model_name}")

# ✅ BitsAndBytesConfig optimized for T4 GPU
# Using bfloat16 for compute precision (faster than fp16 on T4)
bnb_config = bnb(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # ✅ speed optimization
    llm_int4_threshold=6.0,
    llm_int4_has_fp16_weight=False
)

# ✅ Flash attention commented (since not stable on Colab)
# ✅ Added 'trust_remote_code=True' for LLaMA compatibility
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# ✅ LoRA Configuration (same)
print("Applying LoRA configuration...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ✅ Enable gradient checkpointing (saves VRAM)
model.gradient_checkpointing_enable()

# Load tokenized data
print("Loading tokenized data...")
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tokenized_dir = os.path.join(project_root, "data/processed/tokenized")
print(f"Loading data from: {tokenized_dir}")

with torch.serialization.safe_globals([transformers.tokenization_utils_base.BatchEncoding]):
    train_data = torch.load(os.path.join(tokenized_dir, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(tokenized_dir, "validation.pt"), weights_only=False)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# ✅ Add labels
def add_labels(example):
    example["labels"] = example["input_ids"][:]  # shallow copy
    return example

train_dataset = train_dataset.map(add_labels)
val_dataset = val_dataset.map(add_labels)

# ✅ For faster testing, optionally reduce dataset size (comment out later)
# train_dataset = train_dataset.select(range(4000))  # 20% subset for testing
# val_dataset = val_dataset.select(range(800))

# ✅ Training arguments tuned for Colab T4
training_args = TrainingArguments(
    output_dir="outputs/",
    per_device_train_batch_size=1,         # Safe for T4
    gradient_accumulation_steps=2,         # Reduced for speed
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=False,                            # ✅ Avoid mixed-precision instability on T4
    bf16=True,                             # ✅ Use bfloat16 instead (faster)
    save_total_limit=1,
    logging_dir="logs/",
    logging_steps=25,
    evaluation_strategy="no",              # ✅ Disable eval to save time
    save_steps=200,                        # Less frequent checkpoint saves
    report_to="none",
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting fine-tuning...")
trainer.train()

# Save fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN
)

print("Saving fine-tuned model...")
model.save_pretrained("outputs/fine_tuned_llama")
tokenizer.save_pretrained("outputs/fine_tuned_llama")

print("✅ Fine-tuning completed successfully!")
