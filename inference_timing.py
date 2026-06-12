import torch
import torch.nn as nn
import numpy as np
import time
import glob
import os
import timm
from torchvision import transforms

FEATURES_DIR = "D:/precog/features"
MODEL_DIR    = "D:/precog/models"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_FRAMES     = 5
N_RUNS       = 100

# Load SENSE
class SENSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64),  nn.GELU(), nn.Linear(64, 1))
    def forward(self, x): return torch.sigmoid(self.net(x)).squeeze(-1)

# Load HERALD
class HERALDv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token  = nn.Parameter(torch.randn(1, 1, 768))
        self.pos_embed  = nn.Embedding(N_FRAMES + 1, 768)
        layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=4, dim_feedforward=768*2,
            dropout=0.3, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.cam_norm   = nn.LayerNorm(768)
        self.obj_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(768+128, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x, obj):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.shape[1], device=x.device)
        x   = x + self.pos_embed(pos)
        x   = self.cam_norm(self.transformer(x))
        cam = x[:, 0]
        o   = self.obj_encoder(obj)
        return self.head(torch.cat([cam, o], dim=1)).squeeze(-1)

print("Loading models...")
sense  = SENSEModel().to(DEVICE)
herald = HERALDv2().to(DEVICE)
sense.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "sense_best.pt"), map_location=DEVICE))
herald.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "herald_v2_best.pt"), map_location=DEVICE))
sense.eval()
herald.eval()

# Load ViT for feature extraction timing
print("Loading ViT-B/16...")
vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
vit.eval().to(DEVICE)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load sample features for SENSE + HERALD timing
sample_path = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.npz")))[0]
data        = np.load(sample_path, allow_pickle=True)
feat_sample = torch.tensor(data["features"][:1], dtype=torch.float32)
T = feat_sample.shape[1]
if T < N_FRAMES:
    feat_sample = torch.cat([feat_sample,
        torch.zeros(1, N_FRAMES-T, 768)], dim=1)
elif T > N_FRAMES:
    feat_sample = feat_sample[:, :N_FRAMES]
feat_sample = feat_sample.to(DEVICE)
obj_sample  = torch.zeros(1, 7).to(DEVICE)

# Dummy frame for ViT timing (224x224 RGB)
dummy_frames = torch.randn(N_FRAMES, 3, 224, 224).to(DEVICE)

print(f"\nWarming up GPU...")
for _ in range(10):
    with torch.no_grad():
        avg = feat_sample.mean(dim=1)
        sense(avg)
        herald(feat_sample, obj_sample)
        vit(dummy_frames)

print(f"Timing {N_RUNS} runs...\n")

# ── Time 1: ViT feature extraction (N frames) ──────────────────────────────
times_vit = []
for _ in range(N_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = vit(dummy_frames)
    torch.cuda.synchronize()
    times_vit.append((time.perf_counter() - t0) * 1000)

# ── Time 2: SENSE inference ───────────────────────────────────────────────────
times_sense = []
for _ in range(N_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        avg = feat_sample.mean(dim=1)
        sense(avg)
    torch.cuda.synchronize()
    times_sense.append((time.perf_counter() - t0) * 1000)

# ── Time 3: HERALD inference ──────────────────────────────────────────────────
times_herald = []
for _ in range(N_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        herald(feat_sample, obj_sample)
    torch.cuda.synchronize()
    times_herald.append((time.perf_counter() - t0) * 1000)

# ── Time 4: Full pipeline (ViT + SENSE + HERALD) ─────────────────────────────
times_full = []
for _ in range(N_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        feats = vit(dummy_frames).unsqueeze(0)
        avg   = feats.mean(dim=1)
        sense_score = sense(avg)
        herald(feats, obj_sample)
    torch.cuda.synchronize()
    times_full.append((time.perf_counter() - t0) * 1000)

print(f"{'='*55}")
print(f"INFERENCE TIMING RESULTS (RTX 4060 Laptop, CUDA)")
print(f"{'='*55}")
print(f"{'Component':<35} {'Mean':>8} {'Std':>7} {'Min':>7}")
print(f"{'-'*55}")
for name, times in [
    ("ViT-B/16 feature extraction (5 frames)", times_vit),
    ("SENSE sensor health prediction",         times_sense),
    ("HERALD danger prediction",               times_herald),
    ("Full PRECOG pipeline",                   times_full),
]:
    print(f"{name:<35} {np.mean(times):>7.1f}ms "
          f"{np.std(times):>6.1f}ms {np.min(times):>6.1f}ms")

total = np.mean(times_full)
fps   = 1000 / total
print(f"\nFull pipeline: {total:.1f}ms per clip = {fps:.1f} FPS")
print(f"Required for real-time AV (10Hz): 100ms per clip")
if total < 100:
    print(f"✓ REAL-TIME CAPABLE — {100/total:.1f}x faster than required")
else:
    print(f"✗ Below real-time threshold — optimisation needed")

print(f"\nFor paper: PRECOG processes at {fps:.0f} FPS on a single RTX 4060 GPU")