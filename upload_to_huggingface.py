"""
PRECOG — Upload weights, labels, dataset and model cards to HuggingFace
Username: Trazemag
Run after: hf auth login
"""
from huggingface_hub import HfApi, create_repo
import os

api      = HfApi()
HF_USER  = "Trazemag"

# ── Create repos ───────────────────────────────────────────────────────────────
repos = [
    (f"{HF_USER}/PRECOG-SENSE",   "model"),
    (f"{HF_USER}/PRECOG-HERALD",  "model"),
    (f"{HF_USER}/PRECOG-Labels",  "dataset"),
]

print("Creating repositories...")
for repo_id, repo_type in repos:
    try:
        create_repo(repo_id, repo_type=repo_type,
                    exist_ok=True, private=False)
        print(f"  OK: {repo_id}")
    except Exception as e:
        print(f"  Note ({repo_id}): already exists or no permission to create")

# ── Upload model weights and data files ────────────────────────────────────────
uploads = [
    (f"{HF_USER}/PRECOG-SENSE",  "model",   "D:/precog/models/sense_best.pt",      "sense_best.pt"),
    (f"{HF_USER}/PRECOG-HERALD", "model",   "D:/precog/models/herald_v2_best.pt",  "herald_v2_best.pt"),
    (f"{HF_USER}/PRECOG-Labels", "dataset", "D:/precog/danger_labels.csv",         "danger_labels.csv"),
    (f"{HF_USER}/PRECOG-Labels", "dataset", "D:/precog/precog_dataset.csv",        "precog_dataset.csv"),
]

print("\nUploading weights and data...")
for repo_id, repo_type, local_path, filename in uploads:
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {local_path}")
        continue
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  {filename} ({size_mb:.1f}MB) → {repo_id} ...", end=" ", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")

# ── Upload model cards (README.md for each repo) ───────────────────────────────
card_uploads = [
    (f"{HF_USER}/PRECOG-SENSE",  "model",   "D:/precog/hf_sense_card.md",  "README.md"),
    (f"{HF_USER}/PRECOG-HERALD", "model",   "D:/precog/hf_herald_card.md", "README.md"),
    (f"{HF_USER}/PRECOG-Labels", "dataset", "D:/precog/hf_labels_card.md", "README.md"),
]

print("\nUploading model cards...")
for repo_id, repo_type, local_path, filename in card_uploads:
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {local_path}")
        continue
    print(f"  README → {repo_id} ...", end=" ", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"""
All uploads complete. Public URLs:
  SENSE:  https://huggingface.co/{HF_USER}/PRECOG-SENSE
  HERALD: https://huggingface.co/{HF_USER}/PRECOG-HERALD
  Labels: https://huggingface.co/datasets/{HF_USER}/PRECOG-Labels
  GitHub: https://github.com/TrazeMaG/PRECOG-AV
""")