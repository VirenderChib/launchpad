import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
from datasets import Dataset
from transformers import BitsAndBytesConfig as bnb
from transformers.training_args import TrainingArguments



load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading model: {model_name}")


bnb_config = bnb(
    load_in_4bit=True,          # or load_in_4bit=True if preferred
    llm_int4_threshold=6.0,
    llm_int4_has_fp16_weight=False
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN,  # replaced deprecated use_auth_token with token
    quantization_config=bnb_config,
    device_map="auto",
    # use_flash_attention_2=True   # flash attention included
)


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
#model.gradient_checkpointing_enable()  # gradient checkpointing for memory efficiency

print("Checking trainable parameters after applying LoRA:")
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")


print("Loading tokenized data...")
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tokenized_dir = os.path.join(project_root, "data/processed/tokenized")

print(f"Loading data from: {tokenized_dir}")

import transformers
with torch.serialization.safe_globals([transformers.tokenization_utils_base.BatchEncoding]):
    train_data = torch.load(os.path.join(tokenized_dir, "train.pt"), weights_only=False)
    val_data = torch.load(os.path.join(tokenized_dir, "validation.pt"), weights_only=False)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

def convert_to_tensors(example):
    example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.long)
    example["attention_mask"] = torch.tensor(example["attention_mask"], dtype=torch.long)
    example["labels"] = example["input_ids"].detach().clone()  # labels = input_ids for causal LM
    return example

train_dataset = train_dataset.map(convert_to_tensors)
val_dataset = val_dataset.map(convert_to_tensors)


training_args = TrainingArguments(
    output_dir="outputs/",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    save_total_limit=2,
    logging_dir="logs/",
    logging_steps=10,
#    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
#    deepspeed="ds_config.json",
    report_to="none"
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


print("Starting fine-tuning...")
trainer.train()


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN  # replaced use_auth_token with token
)


print("Saving fine-tuned model...")
model.save_pretrained("outputs/fine_tuned_llama")
tokenizer.save_pretrained("outputs/fine_tuned_llama")

print("Fine-tuning completed successfully!")
