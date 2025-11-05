import os
import json
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import requests
from io import BytesIO


import time

def save_image_from_url(url, path, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(path)
            return True
        except Exception as e:
            print(f"[Attempt {attempt+1}] Failed to save image {url}: {e}")
            time.sleep(2)  # brief pause before retry
    return False



def load_and_filter_coco(limit=8000, img_save_dir="data/mm_updated/images"):
    os.makedirs(img_save_dir, exist_ok=True)
    print("Loading COCO-Caption2017 dataset...")

    dataset = load_dataset("lmms-lab/COCO-Caption2017", split="val", streaming=True)
    filtered_data = []
    count = 0

    for i, sample in enumerate(dataset):
        text = sample.get("question", "")
        # Optionally append first answer if available
        answers = sample.get("answer", [])
        if answers and isinstance(answers, list):
            text += " " + answers[0]

        image_url = sample.get("coco_url")
        if not image_url:
            continue

        img_path = os.path.join(img_save_dir, f"{i}.jpg")
        if not save_image_from_url(image_url, img_path):
            continue

        filtered_data.append({
            "id": str(i),
            "instruction": text.strip(),
            "response": "",
            "image": os.path.abspath(img_path)
        })

        count += 1
        if count % 200 == 0:
            print(f"Collected {count} samples...")

        if count >= limit:
            break

    print(f"\n✅ Collected {len(filtered_data)} samples (no keyword filtering).")
    return filtered_data


def save_splits(data, output_dir="data/mm_updated/processed"):
    os.makedirs(output_dir, exist_ok=True)
    if len(data) == 0:
        print("Warning: No data to split and save.")
        return

    train, temp = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    for name, split_data in zip(["train", "val", "test"], [train, val, test]):
        out_path = os.path.join(output_dir, f"{name}_mm.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split_data:
                item["id"] = str(item["id"])
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(split_data)} samples to {out_path}")


def prepare_multimodal_dataset():
    data = load_and_filter_coco(limit=4000)
    save_splits(data)


if __name__ == "__main__":
    prepare_multimodal_dataset()
