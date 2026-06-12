import torch
import torch.nn as nn
import numpy as np
import glob
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm

FEATURES_DIR = "D:/precog/features"
OBJ_CSV      = "D:/precog/object_features.csv"
MODEL_DIR    = "D:/precog/models"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 30
BATCH_SIZE   = 256
LR           = 5e-4
N_FRAMES     = 5

import pandas as pd
obj_df = pd.read_csv(OBJ_CSV, index_col="clip_id")
for col in ["n_road_users", "min_dist_m", "n_close_15m", "mean_nearest_m"]:
    obj_df[col] = np.log1p(obj_df[col])
obj_cols = ["n_road_users", "min_dist_m", "n_close_15m", "n_in_path",
            "mean_nearest_m", "has_pedestrian", "max_density"]
obj_df[obj_cols] = (obj_df[obj_cols] - obj_df[obj_cols].mean()) / obj_df[obj_cols].std().clip(lower=1e-6)

print("Loading features...")
splits_data = {"train": [], "val": [], "test": []}
for path in tqdm(sorted(glob.glob(os.path.join(FEATURES_DIR, "*.npz")))):
    data    = np.load(path, allow_pickle=True)
    feats   = data["features"]
    danger  = data["is_danger"]
    radar   = data["radar_lbl"]
    splits  = data["splits"]
    countries = data["countries"]
    clip_ids  = data["clip_ids"]
    for i in range(len(feats)):
        d = int(danger[i])
        if d < 0: continue
        cid = str(clip_ids[i])
        obj = obj_df.loc[cid, obj_cols].values.astype(np.float32) if cid in obj_df.index else np.zeros(7, dtype=np.float32)
        entry = {"feat": feats[i], "obj": obj, "danger": d,
                 "radar": int(radar[i]), "split": str(splits[i])}
        s = str(splits[i])
        if s in splits_data: splits_data[s].append(entry)

class HERALDDataset(Dataset):
    def __init__(self, items, use_obj=True):
        self.items   = items
        self.use_obj = use_obj
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        feat = torch.tensor(item["feat"], dtype=torch.float32)
        T = feat.shape[0]
        if T < N_FRAMES: feat = torch.cat([feat, torch.zeros(N_FRAMES-T, 768)], dim=0)
        elif T > N_FRAMES: feat = feat[:N_FRAMES]
        obj = torch.tensor(item["obj"], dtype=torch.float32) if self.use_obj else torch.zeros(7)
        return feat, obj, torch.tensor(item["danger"], dtype=torch.float32)

# ── Camera-only model ─────────────────────────────────────────────────────────
class HERALDCameraOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.pos_embed = nn.Embedding(N_FRAMES + 1, 768)
        layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=4, dim_feedforward=768*2,
            dropout=0.3, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x, obj=None):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.shape[1], device=x.device)
        x   = x + self.pos_embed(pos)
        x   = self.norm(self.transformer(x))
        return self.head(x[:, 0]).squeeze(-1)

# ── Camera + Object model ─────────────────────────────────────────────────────
class HERALDCameraObj(nn.Module):
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
        obj = self.obj_enc(obj)
        return self.head(torch.cat([cam, obj], dim=1)).squeeze(-1)

# ── Camera + Object + SENSE signal ───────────────────────────────────────────
sense_model = None
def load_sense():
    global sense_model
    class SENSE(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, 64),  nn.GELU(), nn.Linear(64, 1))
        def forward(self, x): return torch.sigmoid(self.net(x)).squeeze(-1)
    sense_model = SENSE().to(DEVICE)
    sense_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "sense_best.pt"), map_location=DEVICE))
    sense_model.eval()

class HERALDFull(nn.Module):
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
            nn.Linear(768+128+1, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x, obj):
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        inp = torch.cat([cls, x], dim=1)
        pos = torch.arange(inp.shape[1], device=inp.device)
        inp = inp + self.pos_embed(pos)
        inp = self.norm(self.transformer(inp))
        cam = inp[:, 0]
        obj_out = self.obj_enc(obj)
        avg_feat = x.mean(dim=1)
        with torch.no_grad():
            sense_score = sense_model(avg_feat).unsqueeze(1)
        fused = torch.cat([cam, obj_out, sense_score], dim=1)
        return self.head(fused).squeeze(-1)

def train_eval(model_class, name, use_obj=True):
    print(f"\n{'='*50}")
    print(f"ABLATION: {name}")
    print(f"{'='*50}")

    train_ds = HERALDDataset(splits_data["train"], use_obj=use_obj)
    val_ds   = HERALDDataset(splits_data["val"],   use_obj=use_obj)
    test_ds  = HERALDDataset(splits_data["test"],  use_obj=use_obj)

    n_danger = sum(x["danger"] for x in splits_data["train"])
    n_safe   = len(splits_data["train"]) - n_danger
    pw = torch.tensor([n_safe / max(n_danger,1)], device=DEVICE)

    model = model_class().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_auc, best_state = 0.0, None
    for epoch in range(1, EPOCHS+1):
        model.train()
        for feat, obj, lbl in train_loader:
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
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:02d}/{EPOCHS} | Val AUC: {auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for feat, obj, lbl in test_loader:
            p = torch.sigmoid(model(feat.to(DEVICE), obj.to(DEVICE))).cpu().numpy()
            preds.extend(p); lbls.extend(lbl.numpy())

    test_auc = roc_auc_score(lbls, preds)
    test_ap  = average_precision_score(lbls, preds)
    test_f1  = f1_score(lbls, (np.array(preds)>0.5).astype(int), zero_division=0)

    print(f"\n  {name} TEST RESULTS:")
    print(f"  AUC: {test_auc:.4f}  |  AP: {test_ap:.4f}  |  F1: {test_f1:.4f}")
    return {"name": name, "auc": test_auc, "ap": test_ap, "f1": test_f1}

load_sense()

results = []
results.append(train_eval(HERALDCameraOnly,  "A: Camera only",              use_obj=False))
results.append(train_eval(HERALDCameraObj,   "B: Camera + Object",          use_obj=True))
results.append(train_eval(HERALDFull,        "C: Camera + Object + SENSE",  use_obj=True))

print(f"\n{'='*60}")
print(f"ABLATION STUDY SUMMARY")
print(f"{'='*60}")
print(f"{'Variant':<30} {'AUC':>8} {'AP':>8} {'F1':>8}")
print(f"{'-'*60}")
for r in results:
    print(f"{r['name']:<30} {r['auc']:>8.4f} {r['ap']:>8.4f} {r['f1']:>8.4f}")
print(f"\nImprovements:")
if len(results) >= 2:
    print(f"  +Object:  AUC {results[1]['auc']-results[0]['auc']:+.4f}  AP {results[1]['ap']-results[0]['ap']:+.4f}")
if len(results) >= 3:
    print(f"  +SENSE:   AUC {results[2]['auc']-results[1]['auc']:+.4f}  AP {results[2]['ap']-results[1]['ap']:+.4f}")
    print(f"  Total:    AUC {results[2]['auc']-results[0]['auc']:+.4f}  AP {results[2]['ap']-results[0]['ap']:+.4f}")