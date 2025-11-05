import os
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from .data_loader import load_multimodal_datasets

def train_multimodal():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root, "mm_updated", "processed")
    output_dir = os.path.join(root, "outputs/mm_finetuned")

    train_ds, val_ds, _ = load_multimodal_datasets(data_dir)

    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # LoRA setup
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        num_train_epochs=1,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Multimodal fine-tuning complete.")
