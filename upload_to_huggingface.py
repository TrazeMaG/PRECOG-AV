"""
Upload PRECOG weights, labels, and dataset CSV to HuggingFace.
Run once. Requires: huggingface-cli login
"""
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()

uploads = [
    # (repo_id, repo_type, local_path, filename_on_hf)
    ("TrazeMaG/PRECOG-SENSE",   "model",   "D:/precog/models/sense_best.pt",      "sense_best.pt"),
    ("TrazeMaG/PRECOG-HERALD",  "model",   "D:/precog/models/herald_v2_best.pt",  "herald_v2_best.pt"),
    ("TrazeMaG/PRECOG-Labels",  "dataset", "D:/precog/danger_labels.csv",         "danger_labels.csv"),
    ("TrazeMaG/PRECOG-Labels",  "dataset", "D:/precog/precog_dataset.csv",        "precog_dataset.csv"),
]

for repo_id, repo_type, local_path, filename in uploads:
    print(f"\nUploading {filename} → {repo_id}")
    try:
        create_repo(repo_id, repo_type=repo_type,
                    exist_ok=True, private=False)
    except Exception as e:
        print(f"  Repo note: {e}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"  Done: {filename}")

print("\nAll uploads complete.")
print("SENSE:  https://huggingface.co/TrazeMaG/PRECOG-SENSE")
print("HERALD: https://huggingface.co/TrazeMaG/PRECOG-HERALD")
print("Labels: https://huggingface.co/datasets/TrazeMaG/PRECOG-Labels")