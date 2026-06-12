"""
Causal inference — feeds frames progressively and records
when PRECOG first crosses the danger threshold.
Shows how early PRECOG fires vs when danger visually appears.
"""
import cv2
import torch
import numpy as np
import sys
sys.path.insert(0, "D:/precog")
from precog_pipeline import PRECOG

CLIP       = "D:/precog/demo_clips/danger/3b938e26-4310-42b7-a0fe-5573e6b9214d.mp4"
DANGER_FRAME = 175   # from danger_frame_index
THRESHOLD    = 0.55
WINDOW       = 5     # frames to average for stable signal

precog  = PRECOG()
preprocess = precog.preprocess

cap   = cv2.VideoCapture(CLIP)
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames_t = []
scores   = []
first_trigger = None

print(f"Clip: {total} frames at {fps:.0f}fps")
print(f"Danger starts at frame {DANGER_FRAME} ({DANGER_FRAME/fps:.1f}s)")
print(f"Threshold: {THRESHOLD}")
print(f"\nRunning causal inference...\n")
print(f"{'Frame':>6} {'Time':>7} {'Score':>8}  Signal")
print("-" * 45)

for frame_idx in range(total):
    ret, f = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    frames_t.append(precog.preprocess(rgb))

    # Keep rolling window of last WINDOW frames
    if len(frames_t) > WINDOW:
        frames_t.pop(0)

    if len(frames_t) < WINDOW:
        continue

    with torch.no_grad():
        from precog_pipeline import DEVICE
        vit_feats   = precog.vit(
            torch.stack(frames_t).to(DEVICE)).cpu()
        feat_t      = vit_feats.unsqueeze(0).to(DEVICE)
        obj_zeros   = torch.zeros(1, 7).to(DEVICE)
        danger_prob = float(torch.sigmoid(
            precog.herald(feat_t, obj_zeros)).item())

    scores.append((frame_idx, danger_prob))
    t = frame_idx / fps

    if frame_idx % 15 == 0 or danger_prob > THRESHOLD:
        bar    = "█" * int(danger_prob * 20)
        marker = ""
        if frame_idx == DANGER_FRAME:
            marker = "  ← ACTUAL DANGER START"
        elif danger_prob > THRESHOLD and first_trigger is None:
            first_trigger = (frame_idx, t, danger_prob)
            marker = "  ← PRECOG FIRES HERE"
        print(f"{frame_idx:>6} {t:>6.1f}s {danger_prob:>8.3f}  {bar}{marker}")

cap.release()

print(f"\n{'='*55}")
print(f"CAUSAL INFERENCE SUMMARY")
print(f"{'='*55}")
print(f"Actual danger frame:  {DANGER_FRAME} ({DANGER_FRAME/fps:.1f}s)")
if first_trigger:
    f_idx, f_t, f_score = first_trigger
    gap = (DANGER_FRAME / fps) - f_t
    print(f"PRECOG first trigger: frame {f_idx} ({f_t:.1f}s) "
          f"— score {f_score:.3f}")
    if gap > 0:
        print(f"Early warning:        {gap:.1f}s BEFORE visible danger")
        print(f"\nFor paper/demo: PRECOG fires {gap:.1f}s before the "
              f"danger condition is reached.")
    else:
        print(f"Trigger AFTER danger start ({abs(gap):.1f}s late)")
else:
    print(f"PRECOG did not trigger above {THRESHOLD} on this clip")

import numpy as np
all_scores = [s for _, s in scores]
print(f"\nScore distribution:")
print(f"  Mean:  {np.mean(all_scores):.3f}")
print(f"  Max:   {np.max(all_scores):.3f}")
print(f"  Pre-danger mean (frames 0-{DANGER_FRAME}): "
      f"{np.mean([s for i,s in scores if i < DANGER_FRAME]):.3f}")
print(f"  Post-danger mean (frames {DANGER_FRAME}+): "
      f"{np.mean([s for i,s in scores if i >= DANGER_FRAME]):.3f}")