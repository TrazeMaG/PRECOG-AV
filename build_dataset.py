import pandas as pd
from huggingface_hub import hf_hub_download

# Load danger labels we just mined
labels = pd.read_csv("D:/precog/danger_labels.csv")
labels = labels.set_index("clip_id")

# Load metadata
path1 = hf_hub_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset", filename="metadata/data_collection.parquet"
)
path2 = hf_hub_download(
    repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
    repo_type="dataset", filename="metadata/feature_presence.parquet"
)

dc = pd.read_parquet(path1)
fp = pd.read_parquet(path2)

# Radar available = any radar sensor present
radar_cols = [c for c in fp.columns if c.startswith("radar")]
fp["radar_available"] = fp[radar_cols].any(axis=1)

# Merge everything
dataset = labels.join(dc[["country", "month", "hour_of_day"]], how="left")
dataset = dataset.join(fp[["radar_available", "lidar_top_360fov"]], how="left")
dataset = dataset.rename(columns={"lidar_top_360fov": "lidar_available"})

# Geographic split — train on 20 countries, val on 3, test on 2
val_countries  = {"Austria", "Finland", "Portugal"}
test_countries = {"Greece", "Bulgaria"}

def assign_split(country):
    if country in test_countries:  return "test"
    if country in val_countries:   return "val"
    return "train"

dataset["split"] = dataset["country"].apply(assign_split)

# Save
dataset.to_csv("D:/precog/precog_dataset.csv")

print("=== PRECOG TRAINING DATASET ===")
print(f"Total clips:     {len(dataset):,}")
print(f"\nDanger breakdown:")
print(dataset["is_danger"].value_counts())
print(f"\nSplit breakdown:")
print(dataset["split"].value_counts())
print(f"\nRadar available:")
print(dataset["radar_available"].value_counts())
print(f"\nTop 10 countries:")
print(dataset["country"].value_counts().head(10))
print(f"\nSaved to: D:/precog/precog_dataset.csv")