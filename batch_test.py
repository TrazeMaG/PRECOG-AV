import glob, sys
sys.path.insert(0, "D:/precog")
from precog_pipeline import PRECOG
import numpy as np

precog = PRECOG()

danger_clips = sorted(glob.glob("D:/precog/demo_clips/danger/*.mp4"))
safe_clips   = sorted(glob.glob("D:/precog/demo_clips/safe/*.mp4"))

print(f"Testing {len(danger_clips)} danger + {len(safe_clips)} safe clips\n")

scores_d, scores_s = [], []

for clip in danger_clips:
    r = precog.run(clip)
    if "error" not in r:
        scores_d.append(r["danger_prob"])

for clip in safe_clips:
    r = precog.run(clip)
    if "error" not in r:
        scores_s.append(r["danger_prob"])

thresh = 0.4
tp = sum(1 for s in scores_d if s > thresh)
tn = sum(1 for s in scores_s if s <= thresh)
fp = len(scores_s) - tn
fn = len(scores_d) - tp

prec = tp / max(tp+fp, 1)
rec  = tp / max(tp+fn, 1)
f1   = 2*prec*rec / max(prec+rec, 1e-6)

print(f"{'='*50}")
print(f"PRECOG PIPELINE — QUALITATIVE TEST")
print(f"{'='*50}")
print(f"Danger clips:  {len(scores_d)}  |  Safe clips: {len(scores_s)}")
print(f"Threshold:     {thresh}")
print(f"")
print(f"True Positives:  {tp}/{len(scores_d)} danger clips caught")
print(f"True Negatives:  {tn}/{len(scores_s)} safe clips correct")
print(f"")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1:        {f1:.3f}")
print(f"")
print(f"Danger score stats:")
print(f"  Danger clips — mean: {np.mean(scores_d):.3f}  "
      f"max: {max(scores_d):.3f}  min: {min(scores_d):.3f}")
if scores_s:
    print(f"  Safe clips   — mean: {np.mean(scores_s):.3f}  "
          f"max: {max(scores_s):.3f}  min: {min(scores_s):.3f}")