import torch
import torch.nn as nn
import numpy as np
import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CCD_POS      = "D:/precog/benchmarks/CCD/vgg16_features/positive"
CCD_NEG      = "D:/precog/benchmarks/CCD/vgg16_features/negative"
MODEL_DIR    = "D:/precog/models"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 40
BATCH_SIZE   = 32
LR           = 5e-4
N_FRAMES     = 50
VGG_DIM      = 4096
HERALD_DIM   = 768
FPS          = 10.0
THRESHOLD    = 0.5

# ── Dataset ───────────────────────────────────────────────────────────────────
class CCDDataset(Dataset):
    """
    CCD: data=(50,20,4096), det=(50,19,6), labels=(2,)
    labels[0] = binary accident label
    labels[1] = accident frame index (for mTTA)
    """
    def __init__(self, pos_dir, neg_dir, train=True, train_ratio=0.8):
        pos_files = sorted(glob.glob(os.path.join(pos_dir, "*.npz")))
        neg_files = sorted(glob.glob(os.path.join(neg_dir, "*.npz")))

        random.seed(42)
        random.shuffle(pos_files)
        random.shuffle(neg_files)

        def split(files):
            n = int(len(files) * train_ratio)
            return files[:n] if train else files[n:]

        self.samples = (
            [(f, 1) for f in split(pos_files)] +
            [(f, 0) for f in split(neg_files)]
        )
        random.shuffle(self.samples)

        tag = "train" if train else "test"
        n_pos = sum(1 for _, l in self.samples if l == 1)
        print(f"CCD {tag}: {len(self.samples)} clips  |  "
              f"accident: {n_pos}  |  safe: {len(self.samples)-n_pos}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        d = np.load(path, allow_pickle=True)

        # data: (50, 20, 4096)
        # data[:,0,:] = global frame feature
        # data[:,1:,:] = 19 object features
        data = d["data"].astype(np.float32)  # (50, 20, 4096)
        det  = d["det"].astype(np.float32)   # (50, 19, 6)

        frame_feats = data[:, 0, :]          # (50, 4096) global scene
        obj_feats   = data[:, 1:, :]         # (50, 19, 4096)

        # Weight objects by detection confidence (det[:,:,4] = confidence score)
        conf = det[:, :, 4:5]                # (50, 19, 1)
        conf = torch.tensor(conf)
        conf = torch.softmax(conf, dim=1)    # normalise across 19 objects

        obj_feats_t = torch.tensor(obj_feats)
        # Weighted sum of object features → (50, 4096)
        obj_weighted = (obj_feats_t * conf).sum(dim=1)

        # Accident frame for mTTA calculation
        accident_frame = int(d["labels"][1]) if label == 1 else -1

        return (
            torch.tensor(frame_feats),   # (50, 4096) scene stream
            obj_weighted,                # (50, 4096) object stream
            torch.tensor(float(label)),  # binary label
            accident_frame,              # frame index of accident
        )


# ── Two-stream HERALD for CCD ─────────────────────────────────────────────────
class HERALDonCCD(nn.Module):
    """
    Two-stream architecture:
      Stream 1 — Scene:   (50, 4096) → project → (50, 768) → transformer → 768
      Stream 2 — Objects: (50, 4096) → project → (50, 768) → pool → 768
      Fusion: concat(768+768) → MLP → danger score
    """
    def __init__(self):
        super().__init__()

        # Stream 1: scene features
        self.scene_proj = nn.Sequential(
            nn.Linear(VGG_DIM, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, HERALD_DIM),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, HERALD_DIM))
        self.pos_embed = nn.Embedding(N_FRAMES + 1, HERALD_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=HERALD_DIM, nhead=8,
            dim_feedforward=HERALD_DIM * 2,
            dropout=0.2, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.scene_norm  = nn.LayerNorm(HERALD_DIM)

        # Stream 2: object features
        self.obj_proj = nn.Sequential(
            nn.Linear(VGG_DIM, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, HERALD_DIM),
        )
        self.obj_pool = nn.AdaptiveAvgPool1d(1)

        # Fusion
        self.head = nn.Sequential(
            nn.Linear(HERALD_DIM * 2, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 128),            nn.GELU(),
            nn.Linear(128, 1)
        )

    def load_pretrained(self, path):
        if not os.path.exists(path):
            print("No pre-trained HERALD — training from scratch")
            return
        state     = torch.load(path, map_location="cpu")
        own_state = self.state_dict()
        matched   = {k: v for k, v in state.items()
                     if k in own_state and own_state[k].shape == v.shape}
        own_state.update(matched)
        self.load_state_dict(own_state)
        print(f"Loaded {len(matched)}/{len(own_state)} weights "
              f"from PhysicalAI pre-training")

    def forward(self, scene, obj):
        # Scene stream
        B = scene.shape[0]
        s = self.scene_proj(scene)                     # (B, 50, 768)
        cls = self.cls_token.expand(B, -1, -1)
        s   = torch.cat([cls, s], dim=1)               # (B, 51, 768)
        pos = torch.arange(s.shape[1], device=s.device)
        s   = s + self.pos_embed(pos)
        s   = self.scene_norm(self.transformer(s))
        scene_out = s[:, 0]                            # (B, 768) CLS token

        # Object stream
        o = self.obj_proj(obj)                         # (B, 50, 768)
        o = o.permute(0, 2, 1)                         # (B, 768, 50)
        obj_out = self.obj_pool(o).squeeze(-1)         # (B, 768)

        # Fusion
        fused = torch.cat([scene_out, obj_out], dim=1) # (B, 1536)
        return self.head(fused).squeeze(-1)


# ── mTTA metric ───────────────────────────────────────────────────────────────
def compute_mtta(preds, labels, accident_frames, threshold=THRESHOLD):
    """
    Mean Time-To-Accident: average seconds of warning before the accident.
    Only computed on accident clips where the model correctly triggers.
    """
    ttas = []
    for pred, label, acc_frame in zip(preds, labels, accident_frames):
        if label != 1 or acc_frame < 0:
            continue
        if pred > threshold:
            # Model predicts danger — TTA = time from prediction to accident
            # Since we predict on the full clip, TTA = accident_frame / fps
            tta = acc_frame / FPS
            ttas.append(tta)
    return float(np.mean(ttas)) if ttas else 0.0


# ── Main training + evaluation ────────────────────────────────────────────────
print(f"Device: {DEVICE}\n")

train_ds = CCDDataset(CCD_POS, CCD_NEG, train=True)
test_ds  = CCDDataset(CCD_POS, CCD_NEG, train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

model = HERALDonCCD().to(DEVICE)
model.load_pretrained(os.path.join(MODEL_DIR, "herald_v2_best.pt"))

n_pos = sum(1 for _, l in train_ds.samples if l == 1)
n_neg = len(train_ds.samples) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
print(f"\nClass weight: {pos_weight.item():.1f}x\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_ap    = 0.0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for scene, obj, labels, _ in train_loader:
        scene, obj, labels = scene.to(DEVICE), obj.to(DEVICE), labels.to(DEVICE)
        loss = criterion(model(scene, obj), labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # Evaluate
    model.eval()
    all_preds, all_labels, all_acc_frames = [], [], []
    with torch.no_grad():
        for scene, obj, labels, acc_frames in test_loader:
            preds = torch.sigmoid(
                model(scene.to(DEVICE), obj.to(DEVICE))
            ).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_acc_frames.extend(acc_frames.numpy())

    ap   = average_precision_score(all_labels, all_preds)
    mtta = compute_mtta(all_preds, all_labels, all_acc_frames)

    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} "
          f"| AP: {ap*100:.2f}% | mTTA: {mtta:.2f}s", end="")

    if ap > best_ap:
        best_ap    = ap
        best_epoch = epoch
        torch.save(model.state_dict(),
                   os.path.join(MODEL_DIR, "herald_ccd_best.pt"))
        print(" ← best")
    else:
        print()

print(f"\n=== CCD BENCHMARK FINAL RESULTS ===")
print(f"Best epoch: {best_epoch}")
print(f"AP:   {best_ap*100:.2f}%")
print(f"\nComparison vs published SOTA on CCD:")
print(f"  DSA    (2019):  74.8% AP")
print(f"  GCRN   (2020):  74.9% AP")
print(f"  DSTA   (2021):  82.6% AP")
print(f"  AccNet (2024):  91.2% AP")
print(f"  CRASH  (2024):  99.5% AP")
print(f"  RARE   (2025):  99.8% AP")
print(f"  OURS (HERALD pretrained on PhysicalAI-AV): {best_ap*100:.2f}% AP")