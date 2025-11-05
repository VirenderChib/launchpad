from datasets import Dataset
import os, json
from tqdm import tqdm
from src.data_prep.cleaning import clean_text





def format_data(filtered_ds):
    formatted_records = []
    print("Formatting dataset entries...")

    for ex in tqdm(filtered_ds):
        instruction = clean_text(ex.get("instruction", ""))
        input_text = clean_text(ex.get("input", ""))
        response = clean_text(ex.get("output", ""))

        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

        formatted_records.append({
            "instruction": instruction,
            "input": input_text,
            "response": response,
            "prompt": prompt
        })

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/codealpaca_tech_formatted.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for record in formatted_records:
            f.write(json.dumps(record) + "\n")

    print(f"Saved formatted dataset to {output_path}")
    return formatted_records
