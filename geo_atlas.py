import pandas as pd
import numpy as np
import os

DANGER_CSV  = "D:/precog/danger_labels.csv"
DATASET_CSV = "D:/precog/precog_dataset.csv"

# ── Load and inspect both files ───────────────────────────────────────────────
print("Loading files...")
danger  = pd.read_csv(DANGER_CSV)
dataset = pd.read_csv(DATASET_CSV)

print(f"danger_labels  columns: {danger.columns.tolist()}")
print(f"precog_dataset columns: {dataset.columns.tolist()}")
print(f"danger_labels  shape:   {danger.shape}")
print(f"precog_dataset shape:   {dataset.shape}")

# ── Build working dataframe ───────────────────────────────────────────────────
# Find the clip id column in each file
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]

did_col = find_col(danger,  ["clip_id", "id", "clip"])
ds_col  = find_col(dataset, ["clip_id", "id", "clip"])

print(f"\nUsing id columns: danger={did_col}, dataset={ds_col}")

# Rename for merge
danger  = danger.rename(columns={did_col: "clip_id"})
dataset = dataset.rename(columns={ds_col: "clip_id"})

# If precog_dataset already has is_danger, use it directly
if "is_danger" in dataset.columns:
    print("is_danger already in precog_dataset — using directly")
    df = dataset.copy()
else:
    print("Merging danger labels into dataset...")
    danger_col = find_col(danger, ["is_danger", "danger", "label"])
    danger = danger.rename(columns={danger_col: "is_danger"})
    df = dataset.merge(danger[["clip_id", "is_danger"]], on="clip_id", how="left")
    df["is_danger"] = df["is_danger"].fillna(0).astype(int)

# Find country and split columns
country_col = find_col(df, ["country", "Country", "nation"])
split_col   = find_col(df, ["split", "Split", "set"])

print(f"Country col: {country_col}, Split col: {split_col}")
print(f"\nTotal clips: {len(df):,}")
print(f"Danger clips: {df['is_danger'].sum():,} ({df['is_danger'].mean()*100:.2f}%)")

# ── Geographic danger atlas ───────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"GEOGRAPHIC DANGER ATLAS — danger rate by country")
print(f"{'='*62}")
print(f"{'Country':<26} {'Total':>8} {'Danger':>8} {'Rate':>7}  Chart")
print(f"{'-'*62}")

geo = df.groupby(country_col).agg(
    total=("clip_id", "count"),
    danger=("is_danger", "sum")
).reset_index()
geo["rate"] = geo["danger"] / geo["total"] * 100
geo = geo.sort_values("rate", ascending=False)

for _, row in geo.iterrows():
    bar = "█" * max(1, int(row["rate"] / 0.4))
    print(f"{str(row[country_col]):<26} {row['total']:>8,} {row['danger']:>8,} "
          f"{row['rate']:>6.1f}%  {bar}")

# ── Hour of day ───────────────────────────────────────────────────────────────
hour_col = None
for c in ["hour_of_day", "hour", "Hour", "time_of_day"]:
    if c in df.columns:
        hour_col = c
        break

if hour_col:
    print(f"\n{'='*55}")
    print(f"HOUR OF DAY DANGER ANALYSIS")
    print(f"{'='*55}")
    print(f"{'Hour':>6} {'Total':>8} {'Danger':>8} {'Rate':>7}  Chart")
    print(f"{'-'*55}")
    hour = df.groupby(hour_col).agg(
        total=("clip_id", "count"),
        danger=("is_danger", "sum")
    ).reset_index()
    hour["rate"] = hour["danger"] / hour["total"] * 100
    hour = hour.sort_values(hour_col)
    for _, row in hour.iterrows():
        bar = "█" * max(1, int(row["rate"] / 0.25))
        print(f"{int(row[hour_col]):>6} {row['total']:>8,} {row['danger']:>8,} "
              f"{row['rate']:>6.1f}%  {bar}")
    peak_hour = hour.loc[hour["rate"].idxmax()]
    safe_hour = hour.loc[hour["rate"].idxmin()]
    print(f"\nMost dangerous hour: {int(peak_hour[hour_col])}:00 "
          f"({peak_hour['rate']:.1f}% danger rate)")
    print(f"Safest hour:         {int(safe_hour[hour_col])}:00 "
          f"({safe_hour['rate']:.1f}% danger rate)")
else:
    print("\nhour_of_day column not found in dataset")

# ── Split breakdown ───────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"GEOGRAPHIC SPLIT BREAKDOWN")
print(f"{'='*50}")
split_stats = df.groupby(split_col).agg(
    total=("clip_id", "count"),
    danger=("is_danger", "sum")
).reset_index()
split_stats["rate"] = (split_stats["danger"] / split_stats["total"] * 100).round(2)
print(split_stats.to_string(index=False))

# ── Key findings summary ──────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"KEY FINDINGS FOR PAPER")
print(f"{'='*55}")
top3 = geo.head(3)
bot3 = geo.tail(3)
print(f"Top 3 most dangerous countries:")
for _, r in top3.iterrows():
    print(f"  {r[country_col]}: {r['rate']:.1f}% danger rate")
print(f"\nTop 3 safest countries:")
for _, r in bot3.iterrows():
    print(f"  {r[country_col]}: {r['rate']:.1f}% danger rate")
print(f"\nDanger rate range: {geo['rate'].min():.1f}% — {geo['rate'].max():.1f}%")
print(f"Ratio most/least dangerous: {geo['rate'].max()/max(geo['rate'].min(),0.01):.1f}x")