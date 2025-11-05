from datasets import load_dataset

def filter_dataset(config=None):
    print("Loading dataset for filtering")
    ds = load_dataset("sahil2801/CodeAlpaca-20k")["train"]

    TECH_KEYWORDS = [
        "python","docker","kubernetes","linux","bash","ssh","server","error","stacktrace",
        "permission","aws","azure","gcp","api","install","configure","vm","memory","network",
        "cloud","deployment","database","debug","log","system","package"
    ]

    def is_tech(text):
        if not text:
            return False
        text = text.lower()
        return any(k in text for k in TECH_KEYWORDS)

    filtered_ds = ds.filter(lambda x: is_tech(x["instruction"]) or is_tech(x["output"]))
    print(f"Filtered dataset size: {len(filtered_ds)} / {len(ds)}")

    output_path = "data/processed/codealpaca_tech_filtered"
    filtered_ds.to_json(f"{output_path}.json")
    print(f" Saved filtered dataset to {output_path}.json")

    return filtered_ds
