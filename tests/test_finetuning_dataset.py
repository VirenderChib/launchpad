import torch, os
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

#def test_model_loads_correctly():
 #   model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
  #  assert model is not None

#def test_lora_applied_correctly():
 #   model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
  #  lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
   # model = get_peft_model(model, lora_cfg)
    #assert any("lora" in n.lower() for n, _ in model.named_parameters())

def test_tensor_conversion_function():
    from src.training.training_lora import convert_to_tensors
    record = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    result = convert_to_tensors(record)
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["labels"], torch.Tensor)

#def test_trainer_initialization():
 #   model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
  #  dummy_ds = Dataset.from_dict({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]], "labels": [[1, 2, 3]]})
   # args = TrainingArguments(output_dir="test_out", num_train_epochs=1, per_device_train_batch_size=1)
    #trainer = Trainer(model=model, args=args, train_dataset=dummy_ds)
    #assert trainer.args.output_dir == "test_out"
