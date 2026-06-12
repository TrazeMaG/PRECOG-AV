import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

LABELS_DIR = "D:/precog/labels_all"
DANGER_CSV = "D:/precog/danger_labels.csv"
OUTPUT_CSV = "D:/precog/danger_frame_index.csv"

danger_df  = pd.read_csv(DANGER_CSV)
danger_ids = set(danger_df[danger_df["is_danger"]==1]["clip_id"].astype(str))
print(f"Target: {len(danger_ids):,} danger clips")

# ── Step 1: scan filenames only (fast) ────────────────────────────────────────
print("Scanning filenames to build clip → path map...")
clip_to_path = {}
for chunk_name in tqdm(sorted(os.listdir(LABELS_DIR)), desc="Scanning"):
    chunk_path = os.path.join(LABELS_DIR, chunk_name)
    if not os.path.isdir(chunk_path):
        continue
    for fname in os.listdir(chunk_path):
        if not fname.endswith(".parquet"):
            continue
        clip_id = fname.split(".")[0]
        if clip_id in danger_ids:
            clip_to_path[clip_id] = os.path.join(chunk_path, fname)

print(f"Found {len(clip_to_path):,} danger parquet files")

# ── Step 2: load ONLY danger parquets ─────────────────────────────────────────
results = []
ROAD_USERS = {"automobile", "person", "bicyclist", "motorcycle"}

for clip_id, ppath in tqdm(clip_to_path.items(), desc="Processing"):
    try:
        df = pd.read_parquet(ppath,
             columns=["timestamp_us", "center_x", "center_y", "label_class"])
        df = df[df["label_class"].isin(ROAD_USERS)].copy()
        if len(df) == 0:
            continue
        df["dist"] = np.sqrt(df["center_x"]**2 + df["center_y"]**2)
        timestamps = sorted(df["timestamp_us"].unique())

        first_danger_ts = None
        for ts in timestamps:
            frame = df[df["timestamp_us"] == ts]
            if ((frame["dist"] < 8.0) &
                (frame["center_y"].abs() < 1.5) &
                (frame["center_x"] > 1)).any():
                first_danger_ts = ts
                break

        if first_danger_ts is not None:
            idx = timestamps.index(first_danger_ts)
            tta = (timestamps[-1] - first_danger_ts) / 1e6
            results.append({
                "clip_id":           clip_id,
                "first_danger_frame": idx,
                "total_frames":      len(timestamps),
                "time_to_danger_s":  round(tta, 2),
            })
    except Exception:
        pass

out = pd.DataFrame(results)
out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(out):,} clips → {OUTPUT_CSV}")
print(f"\nTime-to-danger statistics:")
print(out["time_to_danger_s"].describe().round(2))
print(f"\nMean TTA: {out['time_to_danger_s'].mean():.2f}s — PhysicalAI-AV mTTA ground truth")