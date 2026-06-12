"""
PRECOG — YOLOv8 Object Feature Extractor
Runs detection at full resolution then extracts 7 normalised features.
"""
import numpy as np
import pandas as pd
import json
import os
from ultralytics import YOLO

ROAD_CLASSES = {0:"person", 1:"bicycle", 2:"car",
                3:"motorcycle", 5:"bus", 7:"truck"}
OBJ_HEIGHT   = {"car":1.5,"truck":3.0,"bus":3.2,
                "person":1.7,"bicycle":1.0,"motorcycle":1.2}

def bbox_dist(h_px, cls_name, img_h, focal_px=554.0):
    real_h = OBJ_HEIGHT.get(cls_name, 1.5)
    f      = focal_px * (img_h / 1080.0)
    return float(min((real_h * f) / max(h_px, 1), 100.0))

def load_norm_stats(obj_csv="D:/precog/object_features.csv",
                    cache="D:/precog/obj_norm_stats.json"):
    if os.path.exists(cache):
        with open(cache) as f:
            s = json.load(f)
        return s["mean"], s["std"]
    print("Computing normalisation stats...")
    df   = pd.read_csv(obj_csv, index_col="clip_id")
    cols = ["n_road_users","min_dist_m","n_close_15m","n_in_path",
            "mean_nearest_m","has_pedestrian","max_density"]
    for c in ["n_road_users","min_dist_m","n_close_15m","mean_nearest_m"]:
        df[c] = np.log1p(df[c])
    mean = df[cols].mean().to_dict()
    std  = df[cols].std().clip(lower=1e-6).to_dict()
    with open(cache,"w") as f:
        json.dump({"mean":mean,"std":std},f,indent=2)
    return mean, std

def extract_features(model, frames_full_res, mean_stats, std_stats,
                     conf_threshold=0.25):
    """
    Args:
        frames_full_res: list of (H, W, 3) uint8 RGB — ORIGINAL resolution
                         Do NOT resize before passing here.
    Returns:
        np.ndarray (7,) normalised object features
    """
    all_dets = []

    for frame in frames_full_res:
        H, W = frame.shape[:2]
        # Run at original resolution for best detection quality
        results = model(frame, verbose=False, imgsz=640,
                        classes=list(ROAD_CLASSES.keys()))
        if not results or results[0].boxes is None:
            all_dets.append([])
            continue
        boxes = results[0].boxes
        frame_dets = []
        for i in range(len(boxes)):
            cid  = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            if cid not in ROAD_CLASSES or conf < conf_threshold:
                continue
            cls_name = ROAD_CLASSES[cid]
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cx   = (x1 + x2) / 2 / W
            cy   = (y1 + y2) / 2 / H
            h_px = y2 - y1
            dist = bbox_dist(h_px, cls_name, H)
            frame_dets.append({"cls":cls_name,"cx":cx,"cy":cy,"dist":dist})
        all_dets.append(frame_dets)

    flat = [d for fr in all_dets for d in fr]

    raw = {
        "n_road_users":   len(flat),
        "min_dist_m":     min((d["dist"] for d in flat), default=99.0),
        "n_close_15m":    sum(1 for d in flat if d["dist"] < 15.0),
        "n_in_path":      int(any(
                              0.3 < d["cx"] < 0.7 and d["cy"] > 0.4
                              for d in flat)),
        "mean_nearest_m": float(np.mean(
                              [min((d["dist"] for d in fr), default=99.0)
                               for fr in all_dets])) if all_dets else 99.0,
        "has_pedestrian": int(any(d["cls"]=="person" for d in flat)),
        "max_density":    max((len(fr) for fr in all_dets), default=0),
    }

    for c in ["n_road_users","min_dist_m","n_close_15m","mean_nearest_m"]:
        raw[c] = np.log1p(raw[c])

    cols = ["n_road_users","min_dist_m","n_close_15m","n_in_path",
            "mean_nearest_m","has_pedestrian","max_density"]
    return np.array([(raw[c]-mean_stats[c])/std_stats[c]
                     for c in cols], dtype=np.float32)


if __name__ == "__main__":
    import cv2, glob

    print("Loading YOLOv8n...")
    model  = YOLO("yolov8n.pt")
    mean_s, std_s = load_norm_stats()

    clips = glob.glob("D:/precog/demo_clips/danger/*.mp4")
    if not clips:
        clips = glob.glob("D:/precog/demo_clips/safe/*.mp4")

    if not clips:
        print("No demo clips found. Testing on synthetic frame...")
        # Create a synthetic road scene (grey road with coloured rectangles)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[400:] = [80, 80, 80]   # grey road
        cv2.rectangle(frame, (400,300),(600,450),(0,0,200),-1)   # blue car
        cv2.rectangle(frame, (700,280),(950,440),(200,0,0),-1)   # red car
        frames = [frame] * 5
    else:
        cap    = cv2.VideoCapture(clips[0])
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for idx in np.linspace(0, total-1, 5, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, f = cap.read()
            if ret:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        print(f"Testing on: {os.path.basename(clips[0])} "
              f"({len(frames)} frames, {frames[0].shape})")

    feats = extract_features(model, frames, mean_s, std_s)

    cols = ["n_road_users","min_dist_m","n_close_15m","n_in_path",
            "mean_nearest_m","has_pedestrian","max_density"]
    print(f"\nYOLOv8 object features (normalised):")
    for c, v in zip(cols, feats):
        print(f"  {c:<20}: {v:+.4f}")

    print("\nYOLOv8 integration working correctly.")