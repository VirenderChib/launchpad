# main.py
"""
Master data-prep pipeline:
inspect -> filter -> cleaning -> format -> split -> tokenize

Assumes the following modules exist under src/data_prep/:
 - inspect_dataset.inspect_dataset()
 - filter_dataset.filter_dataset()
 - cleaning.clean_text()
 - format_data.format_data(filtered_ds)
 - split_dataset.split_dataset(formatted_data_path=...)
 - tokenize_dataset.tokenize_dataset(model_name=...)
"""

import os
import sys
from typing import Iterable, List, Dict
from dotenv import load_dotenv
load_dotenv()

# Ensure project root is on path (so src.* imports work)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import pipeline functions
from src.data_prep.inspect_dataset import inspect_dataset
from src.data_prep.filter_dataset import filter_dataset
from src.data_prep.format_data import format_data
from src.data_prep.split_dataset import split_dataset
from src.data_prep.tokenize_dataset import tokenize_dataset
from src.data_prep.cleaning import clean_text


def to_pylist(dataset_obj) -> List[Dict]:
    """
    Convert a HF Dataset (or similar) to a plain Python list of dicts.
    If already a list, return it.
    """
    try:
        # Hugging Face dataset has .to_list() or can be iterated
        if hasattr(dataset_obj, "to_list"):
            return dataset_obj.to_list()
        # some Dataset objects are dict-like with 'data' or accessible by index
        if hasattr(dataset_obj, "__len__") and hasattr(dataset_obj, "__getitem__"):
            return [dataset_obj[i] for i in range(len(dataset_obj))]
    except Exception:
        pass

    # If it's already a list, assume it's the desired format
    if isinstance(dataset_obj, list):
        return dataset_obj

    # Fallback: try to iterate
    try:
        return list(dataset_obj)
    except Exception:
        raise ValueError("Unable to convert dataset object to list of dicts.")


def apply_cleaning(records: Iterable[Dict]) -> List[Dict]:
    """
    Apply clean_text to the relevant text fields of each record.
    Returns a list of cleaned records (dicts with same keys).
    """
    cleaned = []
    for ex in records:
        # ensure we work with dict; if it's a HF dataset item this will be a dict-like
        rec = dict(ex)
        # apply cleaning to expected fields; safe fallback if missing
        rec["instruction"] = clean_text(rec.get("instruction", "") or "")
        rec["input"] = clean_text(rec.get("input", "") or "")
        rec["output"] = clean_text(rec.get("output", "") or "")
        # Keep other keys if present and create 'prompt' later in format_data
        cleaned.append(rec)
    return cleaned


def main():
    print("\n🚀 Starting Full Data Preparation Pipeline (inspect → filter → cleaning → format → split → tokenize)\n")

    # STEP 1: Inspect dataset (loads and shows samples)
    print("📘 Step 1: Inspect dataset")
    ds = inspect_dataset()
    print("✅ inspect_dataset() finished.\n")

    # STEP 2: Filter dataset (keeps tech-related examples)
    print("🔍 Step 2: Filter dataset")
    filtered = filter_dataset()  # your filter function loads the dataset internally
    # Convert to plain python list for in-memory processing
    filtered_list = to_pylist(filtered)
    print(f"✅ filter_dataset() returned {len(filtered_list)} records.\n")

    # STEP 3: Cleaning — apply cleaning.clean_text to each record
    print("🧹 Step 3: Cleaning dataset")
    cleaned_list = apply_cleaning(filtered_list)
    print(f"✅ Cleaning complete. Sample cleaned record keys: {list(cleaned_list[0].keys()) if cleaned_list else 'EMPTY'}\n")

    # STEP 4: Format data (creates prompt/response structure and writes formatted jsonl)
    print("🧩 Step 4: Formatting dataset into instruction–response prompts")
    formatted = format_data(cleaned_list)  # format_data expects an iterable of records
    # format_data returns formatted_records (list) by your implementation
    print(f"✅ Formatting complete. Formatted records: {len(formatted)}\n")

    # STEP 5: Split formatted data into train/validation/test jsonl files
    print("✂️ Step 5: Splitting dataset into train/validation/test")
    splits_dir = split_dataset()  # default path used inside split_dataset
    print(f"✅ Splitting complete. Files saved to: {splits_dir}\n")

    # STEP 6: Tokenize the split files to produce .pt files
    print("🔠 Step 6: Tokenizing split files (this may take time)")
    # choose the HF-compatible model name here
    model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenize_dataset(model_name=model_name)
    print("✅ Tokenization complete — .pt files should be in data/processed/tokenized/\n")

    print("🎉 All pipeline steps completed. Data is ready for training.\n")


if __name__ == "__main__":
    main()
