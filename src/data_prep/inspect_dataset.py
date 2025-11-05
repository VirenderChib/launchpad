from datasets import load_dataset
import random

def inspect_dataset(config=None):
    print("🔍 Loading dataset...")
    dataset = load_dataset("sahil2801/CodeAlpaca-20k")

    print(dataset)
    print("\n📑 Columns:", dataset["train"].column_names)

    print("\n🧠 Sample examples:")
    for i in random.sample(range(len(dataset["train"])), 3):
        print(f"\nExample {i}:")
        print("Instruction:", dataset["train"][i]['instruction'])
        print("Input:", dataset["train"][i]['input'])
        print("Output:", dataset["train"][i]['output'])

    return dataset
