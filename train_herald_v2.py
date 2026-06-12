import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_DIR = "D:/precog/features"
OBJ_CSV      = "D:/precog/object_features.csv"
MODEL_DIR    = "D:/precog/models"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 50
BATCH_SIZE   = 256
LR           = 5e-4
N_FRAMES     = 5
OBJ_DIM      = 7

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load object features ──────────────────────────────────────────────────────
print("Loading object features...")
obj_df = pd.read_csv(OBJ_CSV, index_col="clip_id")

# Log-normalise skewed columns
for col in ["n_road_users", "min_dist_m", "n_close_15m", "mean_nearest_m"]:
    obj_df[col] = np.log1p(obj_df[col])

# Standard normalise all 7 features using training set stats
obj_cols = ["n_road_users", "min_dist_m", "n_close_15m", "n_in_path",
            "mean_nearest_m", "has_pedestrian", "max_density"]

obj_mean = obj_df[obj_cols].mean()
obj_std  = obj_df[obj_cols].std().clip(lower=1e-6)
obj_df[obj_cols] = (obj_df[obj_cols] - obj_mean) / obj_std

print(f"Object features loaded: {len(obj_df):,} clips")

# ── Load camera features ──────────────────────────────────────────────────────
print("Loading camera features...")
splits_data = {"train": [], "val": [], "test": []}

for path in tqdm(sorted(glob.glob(os.path.join(FEATURES_DIR, "*.npz")))):
    data      = np.load(path, allow_pickle=True)
    feats     = data["features"]
    danger    = data["is_danger"]
    radar     = data["radar_lbl"]
    splits    = data["splits"]
    countries = data["countries"]
    clip_ids  = data["clip_ids"]

    for i in range(len(feats)):
        d = int(danger[i])
        if d < 0:
            continue
        cid = str(clip_ids[i])
        if cid not in obj_df.index:
            continue

        obj_feat = obj_df.loc[cid, obj_cols].values.astype(np.float32)

        entry = {
            "feat":    feats[i],        # (n_frames, 768)
            "obj":     obj_feat,        # (7,)
            "danger":  d,
            "radar":   int(radar[i]),
            "country": str(countries[i]),
            "split":   str(splits[i]),
        }
        s = str(splits[i])
        if s in splits_data:
            splits_data[s].append(entry)

for s, items in splits_data.items():
    n_d = sum(x["danger"] for x in items)
    print(f"{s:5s}: {len(items):6,} clips  |  danger: {n_d:4,} ({n_d/max(len(items),1)*100:.1f}%)")

# ── Dataset ───────────────────────────────────────────────────────────────────
class HERALDDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        feat = torch.tensor(item["feat"], dtype=torch.float32)

        T = feat.shape[0]
        if T < N_FRAMES:
            feat = torch.cat([feat, torch.zeros(N_FRAMES - T, 768)], dim=0)
        elif T > N_FRAMES:
            feat = feat[:N_FRAMES]

        return (feat,
                torch.tensor(item["obj"],    dtype=torch.float32),
                torch.tensor(item["danger"], dtype=torch.float32),
                torch.tensor(item["radar"],  dtype=torch.long),
                item["country"])

def make_loader(items, shuffle=True):
    return DataLoader(HERALDDataset(items), batch_size=BATCH_SIZE,
                      shuffle=shuffle, num_workers=0)

train_loader = make_loader(splits_data["train"], shuffle=True)
val_loader   = make_loader(splits_data["val"],   shuffle=False)
test_loader  = make_loader(splits_data["test"],  shuffle=False)

# ── Two-stream model ──────────────────────────────────────────────────────────
class HERALDv2(nn.Module):
    """
    Two-stream danger prediction:
      Stream 1 — Camera:  (n_frames, 768) → Transformer → 768-dim
      Stream 2 — Objects: (7,) → MLP → 128-dim
      Fusion:   concat(768 + 128) → MLP head → danger probability
    """
    def __init__(self, n_frames=5, embed_dim=768, obj_dim=7,
                 n_heads=4, dropout=0.3):
        super().__init__()

        # Camera transformer stream
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Embedding(n_frames + 1, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.cam_norm = nn.LayerNorm(embed_dim)

        # Object feature stream
        self.obj_encoder = nn.Sequential(
            nn.Linear(obj_dim, 64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 128),      nn.GELU(),
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(embed_dim + 128, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),              nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, cam_feats, obj_feats):
        # Camera stream
        B = cam_feats.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, cam_feats], dim=1)
        pos = torch.arange(x.shape[1], device=x.device)
        x   = x + self.pos_embed(pos)
        x   = self.cam_norm(self.transformer(x))
        cam_out = x[:, 0]                        # CLS → (B, 768)

        # Object stream
        obj_out = self.obj_encoder(obj_feats)    # (B, 128)

        # Fusion
        fused = torch.cat([cam_out, obj_out], dim=1)  # (B, 896)
        return self.head(fused).squeeze(-1)

# ── Training setup ────────────────────────────────────────────────────────────
n_danger  = sum(x["danger"] for x in splits_data["train"])
n_safe    = len(splits_data["train"]) - n_danger
pos_weight = torch.tensor([n_safe / max(n_danger, 1)], device=DEVICE)
print(f"\nClass weight: {pos_weight.item():.1f}x on danger")

model     = HERALDv2().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {DEVICE}\n")

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    preds, labels, radars, countries = [], [], [], []

    with torch.no_grad():
        for cam, obj, y, r, c in loader:
            p = torch.sigmoid(model(cam.to(DEVICE), obj.to(DEVICE))).cpu().numpy()
            preds.extend(p)
            labels.extend(y.numpy())
            radars.extend(r.numpy())
            countries.extend(c)

    y = np.array(labels)
    p = np.array(preds)
    r = np.array(radars)

    if len(set(y)) < 2:
        return {}

    auc = roc_auc_score(y, p)
    ap  = average_precision_score(y, p)
    f1  = f1_score(y, (p > 0.5).astype(int), zero_division=0)

    # Geographic generalisation gap
    val_c  = {"Austria", "Finland", "Portugal"}
    test_c = {"Greece", "Bulgaria"}
    seen   = np.array([c not in val_c and c not in test_c for c in countries])
    ggg = 0.0
    if seen.sum() > 10 and (~seen).sum() > 10:
        try:
            ggg = roc_auc_score(y[seen], p[seen]) - roc_auc_score(y[~seen], p[~seen])
        except Exception:
            pass

    # Sensor degradation robustness
    sdr = 0.0
    on, off = r == 1, r == 0
    if on.sum() > 10 and off.sum() > 10:
        try:
            sdr = roc_auc_score(y[on], p[on]) - roc_auc_score(y[off], p[off])
        except Exception:
            pass

    return {"auc": auc, "ap": ap, "f1": f1, "ggg": ggg, "sdr": sdr}

# ── Training loop ─────────────────────────────────────────────────────────────
best_auc   = 0.0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for cam, obj, y, _, _ in train_loader:
        cam, obj, y = cam.to(DEVICE), obj.to(DEVICE), y.to(DEVICE)
        loss = criterion(model(cam, obj), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    m = evaluate(val_loader)

    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
          f"AUC: {m.get('auc',0):.4f} | AP: {m.get('ap',0):.4f} | "
          f"F1: {m.get('f1',0):.4f}", end="")

    if m.get("auc", 0) > best_auc:
        best_auc   = m["auc"]
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "herald_v2_best.pt"))
        print(" ← best")
    else:
        print()

# ── Final test ────────────────────────────────────────────────────────────────
print(f"\nBest epoch: {best_epoch} (Val AUC: {best_auc:.4f})")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "herald_v2_best.pt")))
t = evaluate(test_loader)

print(f"""
=== HERALD v2 TEST RESULTS (two-stream: camera + object) ===
AUC-ROC:                        {t['auc']:.4f}
Average Precision:              {t['ap']:.4f}
F1 Score:                       {t['f1']:.4f}
Geographic Generalisation Gap:  {t['ggg']:.4f}
Sensor Degradation Robustness:  {t['sdr']:.4f}

Model saved: {MODEL_DIR}/herald_v2_best.pt
""")