import pandas as pd
import numpy as np
import os
import zipfile
from huggingface_hub import list_repo_files, hf_hub_download

# Get all obstacle zip files
files = list(list_repo_files(
    "nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset"
))
obstacle_zips = sorted([f for f in files if "obstacle.offline" in f])
print(f"Total ZIP chunks to process: {len(obstacle_zips)}")

extract_dir = "D:/precog/labels_all"
os.makedirs(extract_dir, exist_ok=True)

all_results = []

for i, zip_file in enumerate(obstacle_zips):
    print(f"\nChunk {i+1}/{len(obstacle_zips)}: {zip_file}")

    path = hf_hub_download(
        repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
        repo_type="dataset",
        filename=zip_file
    )

    chunk_dir = os.path.join(extract_dir, f"chunk_{i:04d}")
    os.makedirs(chunk_dir, exist_ok=True)

    with zipfile.ZipFile(path, 'r') as z:
        z.extractall(chunk_dir)

    parquet_files = [f for f in os.listdir(chunk_dir) if f.endswith(".parquet")]

    for clip_file in parquet_files:
        clip_id = clip_file.split(".")[0]
        df = pd.read_parquet(os.path.join(chunk_dir, clip_file))
        df = df.sort_values(["track_id", "timestamp_us"]).reset_index(drop=True)
        df = df[df["label_class"].isin(["automobile", "person", "bicyclist", "motorcycle"])]

        if len(df) == 0:
            all_results.append({"clip_id": clip_id, "is_danger": False, "min_ttc": None})
            continue

        df["distance_m"] = np.sqrt(df["center_x"]**2 + df["center_y"]**2)

        dt = df.groupby("track_id")["timestamp_us"].diff() / 1e6
        raw_vx = df.groupby("track_id")["center_x"].diff() / dt
        raw_vy = df.groupby("track_id")["center_y"].diff() / dt
        df["vx"] = raw_vx.groupby(df["track_id"]).transform(lambda x: x.rolling(5, min_periods=1).mean())
        df["vy"] = raw_vy.groupby(df["track_id"]).transform(lambda x: x.rolling(5, min_periods=1).mean())

        df["closing_speed"] = -(df["vx"] * df["center_x"] + df["vy"] * df["center_y"]) / \
                               df["distance_m"].clip(lower=0.1)
        df["ttc"] = df["distance_m"] / df["closing_speed"].clip(lower=0.01)
        df.loc[df["closing_speed"] <= 2.0, "ttc"] = 999

        danger = df[
            (df["ttc"] < 3) &
            (df["distance_m"] < 20) &
            (df["closing_speed"] > 4) &
            (df["center_x"] > 1) &
            (df["center_y"].abs() < 1.5)
        ]

        min_ttc   = danger["ttc"].min() if len(danger) > 0 else None
        is_danger = len(danger) >= 3

        all_results.append({
            "clip_id":   clip_id,
            "is_danger": is_danger,
            "min_ttc":   round(min_ttc, 2) if min_ttc else None
        })

    print(f"  Clips processed so far: {len(all_results)}")

# Save labels
df_labels = pd.DataFrame(all_results)
df_labels.to_csv("D:/precog/danger_labels.csv", index=False)

print(f"\n=== FINAL RESULTS ===")
print(f"Total clips:     {len(df_labels)}")
print(f"Dangerous:       {df_labels['is_danger'].sum()} ({df_labels['is_danger'].mean()*100:.1f}%)")
print(f"Safe:            {(~df_labels['is_danger']).sum()}")
print(f"Labels saved to: D:/precog/danger_labels.csv")