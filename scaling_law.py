import torch
import torch.nn as nn
import numpy as np
import glob
import os
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import pandas as pd

FEATURES_DIR = "D:/precog/features"
OBJ_CSV      = "D:/precog/object_features.csv"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 30
BATCH_SIZE   = 256
LR           = 5e-4
N_FRAMES     = 5
FRACTIONS    = [0.10, 0.25, 0.50, 0.75, 1.00]

print("Loading object features...")
obj_df = pd.read_csv(OBJ_CSV, index_col="clip_id")
for col in ["n_road_users", "min_dist_m", "n_close_15m", "mean_nearest_m"]:
    obj_df[col] = np.log1p(obj_df[col])
obj_cols = ["n_road_users", "min_dist_m", "n_close_15m", "n_in_path",
            "mean_nearest_m", "has_pedestrian", "max_density"]
obj_df[obj_cols] = (obj_df[obj_cols] - obj_df[obj_cols].mean()) / obj_df[obj_cols].std().clip(lower=1e-6)

print("Loading all features...")
all_train, all_val, all_test = [], [], []

for path in tqdm(sorted(glob.glob(os.path.join(FEATURES_DIR, "*.npz")))):
    data     = np.load(path, allow_pickle=True)
    feats    = data["features"]
    danger   = data["is_danger"]
    splits   = data["splits"]
    clip_ids = data["clip_ids"]
    for i in range(len(feats)):
        d = int(danger[i])
        if d < 0: continue
        cid = str(clip_ids[i])
        obj = obj_df.loc[cid, obj_cols].values.astype(np.float32) if cid in obj_df.index else np.zeros(7, dtype=np.float32)
        entry = {"feat": feats[i], "obj": obj, "danger": d}
        s = str(splits[i])
        if s == "val":   all_val.append(entry)
        elif s == "test": all_test.append(entry)
        else:            all_train.append(entry)

print(f"Full train: {len(all_train):,}  Val: {len(all_val):,}  Test: {len(all_test):,}")

class HERALDDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        feat = torch.tensor(item["feat"], dtype=torch.float32)
        T = feat.shape[0]
        if T < N_FRAMES: feat = torch.cat([feat, torch.zeros(N_FRAMES-T, 768)], dim=0)
        elif T > N_FRAMES: feat = feat[:N_FRAMES]
        return feat, torch.tensor(item["obj"], dtype=torch.float32), torch.tensor(item["danger"], dtype=torch.float32)

class HERALDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.pos_embed = nn.Embedding(N_FRAMES + 1, 768)
        layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=4, dim_feedforward=768*2,
            dropout=0.3, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(768)
        self.obj_enc = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 128), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(768+128, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x, obj):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.shape[1], device=x.device)
        x   = x + self.pos_embed(pos)
        x   = self.norm(self.transformer(x))
        cam = x[:, 0]
        o   = self.obj_enc(obj)
        return self.head(torch.cat([cam, o], dim=1)).squeeze(-1)

val_loader  = DataLoader(HERALDDataset(all_val),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(HERALDDataset(all_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

results = []

for frac in FRACTIONS:
    random.seed(42)
    n_clips   = int(len(all_train) * frac)
    subset    = random.sample(all_train, n_clips)
    n_danger  = sum(x["danger"] for x in subset)
    n_safe    = len(subset) - n_danger
    approx_clips = n_clips

    print(f"\n{'='*55}")
    print(f"Fraction {frac*100:.0f}%: {n_clips:,} clips ({n_danger:,} danger)")
    print(f"{'='*55}")

    pw    = torch.tensor([n_safe / max(n_danger, 1)], device=DEVICE)
    model = HERALDModel().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)
    loader = DataLoader(HERALDDataset(subset), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    best_auc, best_state = 0.0, None
    for epoch in range(1, EPOCHS+1):
        model.train()
        for feat, obj, lbl in loader:
            feat, obj, lbl = feat.to(DEVICE), obj.to(DEVICE), lbl.to(DEVICE)
            loss = crit(model(feat, obj), lbl)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        model.eval()
        preds, lbls = [], []
        with torch.no_grad():
            for feat, obj, lbl in val_loader:
                p = torch.sigmoid(model(feat.to(DEVICE), obj.to(DEVICE))).cpu().numpy()
                preds.extend(p); lbls.extend(lbl.numpy())
        auc = roc_auc_score(lbls, preds)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{EPOCHS} | Val AUC: {auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for feat, obj, lbl in test_loader:
            p = torch.sigmoid(model(feat.to(DEVICE), obj.to(DEVICE))).cpu().numpy()
            preds.extend(p); lbls.extend(lbl.numpy())

    test_auc = roc_auc_score(lbls, preds)
    test_ap  = average_precision_score(lbls, preds)
    results.append({"frac": frac, "clips": n_clips, "danger": int(n_danger),
                    "val_auc": best_auc, "test_auc": test_auc, "test_ap": test_ap})
    print(f"  → Test AUC: {test_auc:.4f}  AP: {test_ap:.4f}")

print(f"\n{'='*65}")
print(f"DATA SCALING LAW RESULTS")
print(f"{'='*65}")
print(f"{'Fraction':>10} {'Clips':>10} {'Danger':>8} {'Val AUC':>10} {'Test AUC':>10} {'AP':>8}")
print(f"{'-'*65}")
for r in results:
    print(f"{r['frac']*100:>9.0f}% {r['clips']:>10,} {r['danger']:>8,} "
          f"{r['val_auc']:>10.4f} {r['test_auc']:>10.4f} {r['ap']:>8.4f}")

print(f"\nKey finding: Is AUC still rising at 100%? ", end="")
if len(results) >= 2 and results[-1]['test_auc'] > results[-2]['test_auc']:
    print("YES — more data would still improve performance.")
else:
    print("Plateau reached at 100%.")