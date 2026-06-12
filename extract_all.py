import cv2
import torch
import timm
import numpy as np
import pandas as pd
import zipfile
import os
import socket
import time
from torchvision import transforms
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

socket.setdefaulttimeout(120)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CAMERA          = "camera_front_wide_120fov"
FRAMES_PER_CLIP = 5
OUTPUT_DIR      = "D:/precog/features"
TMP_DIR         = "D:/precog/tmp_camera"
DATASET_CSV     = "D:/precog/precog_dataset.csv"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

print(f"Loading ViT-B/16 on {DEVICE}...")
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
model.eval().to(DEVICE)

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = pd.read_csv(DATASET_CSV, index_col="clip_id")
print(f"Dataset clips: {len(dataset):,}")

print("Fetching file list...")
all_files   = list(list_repo_files("nvidia/PhysicalAI-Autonomous-Vehicles", repo_type="dataset"))
camera_zips = sorted([f for f in all_files if f"camera/{CAMERA}" in f])
print(f"Total chunks: {len(camera_zips)}")

def extract_frames(video_path, n=5):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    targets = set(int(x) for x in np.linspace(0, total - 1, n))
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in targets:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
        if len(frames) == n:
            break
    cap.release()
    return frames if len(frames) == n else None

def frames_to_features(frames):
    tensors = torch.stack([preprocess(f) for f in frames]).to(DEVICE)
    with torch.no_grad():
        return model(tensors).cpu().numpy()

def free_gb():
    import shutil
    return shutil.disk_usage("D:/").free / 1e9

print("\nStarting extraction...\n")

for i, zip_file in enumerate(camera_zips):
    chunk_name = f"chunk_{i:04d}"
    save_path  = os.path.join(OUTPUT_DIR, f"{chunk_name}.npz")

    if os.path.exists(save_path):
        if i % 50 == 0:
            print(f"[{i+1}/{len(camera_zips)}] skipping done chunks...")
        continue

    print(f"[{i+1}/{len(camera_zips)}] {chunk_name} — free space: {free_gb():.1f} GB")

    # ── Download with automatic retry ─────────────────────────────────────────
    cached_path = None
    for attempt in range(5):
        try:
            cached_path = hf_hub_download(
                "nvidia/PhysicalAI-Autonomous-Vehicles",
                repo_type="dataset",
                filename=zip_file
            )
            break
        except Exception as e:
            if attempt == 4:
                print(f"  Skipping {chunk_name} after 5 failed attempts")
                break
            print(f"  Download stalled, retrying ({attempt+1}/5)...")
            time.sleep(10)

    if cached_path is None:
        continue
    # ──────────────────────────────────────────────────────────────────────────

    with zipfile.ZipFile(cached_path, "r") as z:
        mp4_files = [n for n in z.namelist() if n.endswith(".mp4")]
        z.extractall(TMP_DIR, members=mp4_files)

    try:
        os.remove(cached_path)
    except Exception:
        pass

    chunk_features, chunk_ids, chunk_labels = [], [], []

    for mp4_name in tqdm(mp4_files, desc=chunk_name, leave=False):
        clip_id    = mp4_name.split(".")[0]
        video_path = os.path.join(TMP_DIR, mp4_name)

        if clip_id not in dataset.index:
            if os.path.exists(video_path):
                os.remove(video_path)
            continue

        frames = extract_frames(video_path, FRAMES_PER_CLIP)
        if os.path.exists(video_path):
            os.remove(video_path)
        if frames is None:
            continue

        feats     = frames_to_features(frames)
        label_row = dataset.loc[clip_id]

        chunk_features.append(feats)
        chunk_ids.append(clip_id)
        chunk_labels.append({
            "is_danger":       int(label_row["is_danger"]),
            "radar_available": int(label_row["radar_available"]),
            "split":           str(label_row["split"]),
            "country":         str(label_row["country"]),
        })

    if chunk_features:
        np.savez_compressed(
            save_path,
            features  = np.array(chunk_features, dtype=np.float32),
            clip_ids  = np.array(chunk_ids),
            is_danger = np.array([l["is_danger"]       for l in chunk_labels], dtype=np.int8),
            radar_lbl = np.array([l["radar_available"]  for l in chunk_labels], dtype=np.int8),
            splits    = np.array([l["split"]            for l in chunk_labels]),
            countries = np.array([l["country"]          for l in chunk_labels]),
        )
        print(f"  Saved {len(chunk_features)} clips → {chunk_name}.npz")
    else:
        print(f"  No matching clips in {chunk_name}")

print(f"\nExtraction complete.")
print(f"Total chunks saved: {len(os.listdir(OUTPUT_DIR))}")