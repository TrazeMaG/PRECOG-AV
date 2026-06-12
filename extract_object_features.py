import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

LABELS_DIR   = "D:/precog/labels_all"
FEATURES_DIR = "D:/precog/features"
OUTPUT_CSV   = "D:/precog/object_features.csv"

ROAD_USERS = {"automobile", "person", "bicyclist", "motorcycle"}
READ_COLS  = ["timestamp_us", "track_id", "center_x", "center_y", "label_class"]

# Step 1 — find all clip_ids that have camera features
print("Scanning camera feature files...")
camera_clips = set()
for path in glob.glob(os.path.join(FEATURES_DIR, "*.npz")):
    d = np.load(path, allow_pickle=True)
    camera_clips.update(str(c) for c in d["clip_ids"])
print(f"Clips with camera features: {len(camera_clips):,}")

# Step 2 — map each clip_id to its obstacle parquet path
print("Mapping clip IDs to obstacle files...")
clip_to_path = {}
for chunk_name in sorted(os.listdir(LABELS_DIR)):
    chunk_path = os.path.join(LABELS_DIR, chunk_name)
    if not os.path.isdir(chunk_path):
        continue
    for fname in os.listdir(chunk_path):
        if not fname.endswith(".parquet"):
            continue
        cid = fname.split(".")[0]
        if cid in camera_clips:
            clip_to_path[cid] = os.path.join(chunk_path, fname)

print(f"Matched: {len(clip_to_path):,} clips")

# Step 3 — compute 7 object statistics per clip
results = []

for clip_id, ppath in tqdm(clip_to_path.items(), desc="Extracting"):
    try:
        df = pd.read_parquet(ppath, columns=READ_COLS)
        df = df[df["label_class"].isin(ROAD_USERS)].copy()

        if len(df) == 0:
            results.append({
                "clip_id": clip_id, "n_road_users": 0, "min_dist_m": 99.0,
                "n_close_15m": 0, "n_in_path": 0, "mean_nearest_m": 99.0,
                "has_pedestrian": 0, "max_density": 0
            })
            continue

        df["dist"] = np.sqrt(df["center_x"]**2 + df["center_y"]**2)

        results.append({
            "clip_id":       clip_id,
            # How many unique objects were tracked
            "n_road_users":  int(df["track_id"].nunique()),
            # Closest any object ever got to ego
            "min_dist_m":    round(float(df["dist"].min()), 2),
            # Objects that came within 15m
            "n_close_15m":   int(df[df["dist"] < 15]["track_id"].nunique()),
            # Any object directly in the ego's path (centre lane)
            "n_in_path":     int(((df["center_y"].abs() < 1.5) &
                                   (df["center_x"] > 1)).any()),
            # Average per-frame nearest object distance
            "mean_nearest_m": round(float(
                df.groupby("timestamp_us")["dist"].min().mean()), 2),
            # Was a pedestrian present
            "has_pedestrian": int("person" in df["label_class"].values),
            # Busiest single frame — max simultaneous objects
            "max_density":   int(
                df.groupby("timestamp_us")["track_id"].nunique().max()),
        })

    except Exception:
        pass

df_out = pd.DataFrame(results).set_index("clip_id")
df_out.to_csv(OUTPUT_CSV)
print(f"\nSaved {len(df_out):,} clips → {OUTPUT_CSV}")
print("\nFeature summary:")
print(df_out.describe().round(2))