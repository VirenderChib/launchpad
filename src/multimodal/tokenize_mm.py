import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

# CONFIG — update if needed
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(ROOT, "mm_updated", "processed")   # contains train_mm.jsonl et al
OUTPUT_DIR = os.path.join(ROOT, "mm_updated", "processed")  # will write train_mm.pt etc
VISION_ENCODER = "openai/clip-vit-base-patch32"
TEXT_TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# limit per-split (None -> process all). Useful for quick debugging.
LIMIT_PER_SPLIT = None  # e.g. 2000 or None

# max text length to tokenize
MAX_TEXT_LEN = 256

# filenames expected (from previous filter stage)
SPLIT_FILES = {
    "train": "/home/virenderchib/Desktop/llama_chatbot/data/mm_updated/processed/train_mm.jsonl",
    "validation": "/home/virenderchib/Desktop/llama_chatbot/data/mm_updated/processed/val_mm.jsonl",   # change if your file is named validation_mm.jsonl
    "test": "/home/virenderchib/Desktop/llama_chatbot/data/mm_updated/processed/test_mm.jsonl"
}

# fields produced in each saved record (must match how training collator expects them)
# We'll save pixel_values (C,H,W), input_ids (list), attention_mask (list)
def process_split(split_name, in_fname, out_fname, tokenizer, vision_proc, limit=None):
    in_path = os.path.join(INPUT_DIR, in_fname)
    out_path = os.path.join(OUTPUT_DIR, out_fname)
    records = []

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"{in_path} not found. Make sure the JSONL split exists.")

    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if limit is not None:
        lines = lines[:limit]

    print(f"[{split_name}] processing {len(lines)} examples from {in_path} ...")
    for line in tqdm(lines):
        try:
            obj = json.loads(line)
            # Expecting obj keys: "image" (path or url) and "instruction"/"response" or "instruction" & "response"
            # we previously saved items with keys: 'instruction' and 'response' (or 'caption')
            image_ref = obj.get("image") or obj.get("image_path") or obj.get("img")
            # Prefer response if exists, else caption, else use instruction as fallback
            caption = obj.get("response") or obj.get("caption") or obj.get("instruction") or ""
            if image_ref is None or not caption:
                continue

            # ----- process image -----
            # image_ref may be a local path or URL. We assume local path in your repo or Drive.
            # If URL, CLIPImageProcessor accepts PIL.Image (we would need to download) — keep local paths for now.
            try:
                img = Image.open(image_ref).convert("RGB")
            except Exception as e_img:
                # If path is relative, try resolving relative to repo root or mm_updated/raw/images
                alt_path = os.path.join(ROOT, "mm_updated", "raw", os.path.basename(image_ref))
                try:
                    img = Image.open(alt_path).convert("RGB")
                except Exception:
                    # skip problematic image
                    # print(f"Skipping image {image_ref}: {e_img}")
                    continue

            pixel_values = vision_proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)  # (C,H,W) CPU tensor

            # ----- tokenize caption -----
            tok = tokenizer(
                caption,
                truncation=True,
                padding="max_length",
                max_length=MAX_TEXT_LEN,
                return_attention_mask=True,
            )

            rec = {
                "pixel_values": pixel_values,               # torch.FloatTensor (C,H,W) on CPU
                "input_ids": tok["input_ids"],              # list[int]
                "attention_mask": tok["attention_mask"],    # list[int]
                # optionally store raw text and image ref for debugging
                "caption": caption,
                "image_ref": image_ref
            }
            records.append(rec)
        except Exception as e:
            # skip bad lines and continue
            # optionally log e
            continue

    # Save as .pt list-of-dicts; each pixel_values is CPU tensor
    torch.save(records, out_path)
    print(f"[{split_name}] Saved {len(records)} records -> {out_path}")


def main(limit_per_split=LIMIT_PER_SPLIT):
    # load processors/tokenizer
    print("Loading tokenizer and vision processor...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_TOKENIZER)
    vision_proc = CLIPImageProcessor.from_pretrained(VISION_ENCODER)

    # create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # process each split
    # Note: adjust names if your filter saved different filenames (e.g. validation_mm.jsonl vs val_mm.jsonl)
    for split, fname in SPLIT_FILES.items():
        outname = f"{split}_mm.pt" if split != "validation" else "validation_mm.pt"
        # Ensure we map our save-naming to what data_loader expects: train_mm.pt, validation_mm.pt, test_mm.pt
        if split == "train":
            outname = "train_mm.pt"
        elif split == "validation":
            outname = "validation_mm.pt"
        elif split == "test":
            outname = "test_mm.pt"
        process_split(split, fname, outname, tokenizer, vision_proc, limit=limit_per_split)


if __name__ == "__main__":
    # Pass an integer to limit samples per split for fast debug, e.g. main(limit_per_split=100)
    main()
