import os, json
from src.data_prep.format_data import format_data

def test_format_data_output_structure(tmp_path):
    mock_data = [
        {"instruction": "Write a function", "input": "Python", "output": "def hello(): pass"},
        {"instruction": "Explain AI", "input": "", "output": "Artificial Intelligence"}
    ]
    result = format_data(mock_data)
    assert isinstance(result, list)
    assert all("prompt" in rec for rec in result)
    assert "### Instruction" in result[0]["prompt"]
    assert "### Response" in result[1]["prompt"]

def test_format_data_creates_file():
    path = "data/processed/codealpaca_tech_formatted.jsonl"
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
    assert all(k in first_line for k in ["instruction", "input", "response", "prompt"])
