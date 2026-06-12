"""
PRECOG configuration — edit paths before running any script.
"""
import os

# ── Base directory ─────────────────────────────────────────────────────────────
BASE_DIR     = os.environ.get("PRECOG_BASE", "D:/precog")

# ── Data paths ─────────────────────────────────────────────────────────────────
FEATURES_DIR = os.path.join(BASE_DIR, "features")
LABELS_DIR   = os.path.join(BASE_DIR, "labels_all")
MODELS_DIR   = os.path.join(BASE_DIR, "models")

# ── Key files ──────────────────────────────────────────────────────────────────
DANGER_CSV   = os.path.join(BASE_DIR, "danger_labels.csv")
DATASET_CSV  = os.path.join(BASE_DIR, "precog_dataset.csv")
OBJ_CSV      = os.path.join(BASE_DIR, "object_features.csv")
NORM_STATS   = os.path.join(BASE_DIR, "obj_norm_stats.json")

# ── Benchmark paths ────────────────────────────────────────────────────────────
CCD_DIR      = os.path.join(BASE_DIR, "benchmarks", "CCD")
DAD_DIR      = os.path.join(BASE_DIR, "benchmarks", "DAD")

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_SENSE_REPO  = "TrazeMaG/PRECOG-SENSE"
HF_HERALD_REPO = "TrazeMaG/PRECOG-HERALD"
HF_LABELS_REPO = "TrazeMaG/PRECOG-Labels"

# ── Training ───────────────────────────────────────────────────────────────────
DEVICE       = "cuda"
BATCH_SIZE   = 256
N_FRAMES     = 5