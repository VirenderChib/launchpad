import json
import os
from sklearn.model_selection import train_test_split

def split_dataset(formatted_data_path="data/processed/codealpaca_tech_formatted.jsonl"):
    print("Splitting dataset into train/val/test...")

    with open(formatted_data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    train, temp = train_test_split(data, test_size=0.1, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    output_dir = "data/processed/splits"
    os.makedirs(output_dir, exist_ok=True)

    for name, dataset in [("train", train), ("validation", val), ("test", test)]:
        out_path = os.path.join(output_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {name} set to {out_path} ({len(dataset)} records)")

    return output_dir
