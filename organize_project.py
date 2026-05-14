import os
import shutil
from pathlib import Path

# Define target directories and their corresponding files
organization_map = {
    "checkpoints/bert": [
        "model.safetensors", 
        "config.json", 
        "tokenizer_config.json", 
        "tokenizer.json"
    ],
    "checkpoints": [
        "bilstm_best.pt", 
        "vocab.json", 
        "bilstm_history.json"
    ],
    "data": ["preprocessing.py"],
    "configs": ["config.yaml"],
    "results": ["bilstm_training_curves.png"]
}

root = Path(".")

# 1. Create all necessary folders
for folder in organization_map.keys():
    (root / folder).mkdir(parents=True, exist_ok=True)
    print(f"Verified folder: {folder}")

# 2. Move specifically mapped files
for folder, files in organization_map.items():
    for filename in files:
        src = root / filename
        if src.exists():
            shutil.move(src, root / folder / filename)
            print(f"Moved: {filename} -> {folder}/")

# 3. Move all remaining .json metrics to results/
# This catches bert_test_metrics.json, baseline_test_metrics.json, etc.
for json_file in root.glob("*.json"):
    # Prevent moving files that belong in bert/ or checkpoints/
    protected = ["model.safetensors", "config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "bilstm_history.json"]
    if json_file.name not in protected:
        shutil.move(json_file, root / "results" / json_file.name)
        print(f"Moved metric: {json_file.name} -> results/")

print("\nProject organization complete!")