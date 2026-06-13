"""
DriveBench: General-Purpose Driving Scene Encoder
Multi-task pre-training on 298,326 clips across 25 countries.
Like ImageNet weights — but for driving.

Architecture: Shared TransformerEncoder → 256-dim DriveBench embedding
Tasks: Danger, Geographic Region, Time of Day, Radar Health, Traffic Density

Nikhil Upadhyay | Independent Researcher | Dublin Business School
Paper 2 — targeting IEEE T-ITS / ICRA 2027
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr
sys.path.insert(0, "D:/precog")

# ── Config ─────────────────────────────────────────────────────────────────────
FEATURES_DIR  = "D:/precog/features"
DATASET_CSV   = "D:/precog/precog_dataset.csv"
MODELS_DIR    = "D:/precog/models"
OUTPUT_MODEL  = f"{MODELS_DIR}/drivebench_best.pt"
OUTPUT_EMBEDS = "D:/precog/drivebench_embeddings.npz"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512
EPOCHS     = 30
LR         = 3e-4
EMBED_DIM  = 256
N_FRAMES   = 5
N_REGIONS  = 6

# ── Geographic region mapping ──────────────────────────────────────────────────
# Groups 25 countries into 6 regions — captures geographic driving patterns
COUNTRY_TO_REGION = {
    # 0 Western Europe
    "Germany":0,"France":0,"Netherlands":0,"Belgium":0,
    "Switzerland":0,"Austria":0,"Luxembourg":0,
    # 1 Nordic
    "Finland":1,"Norway":1,"Sweden":1,"Denmark":1,
    # 2 Southern Europe
    "Italy":2,"Portugal":2,"Spain":2,"Greece":2,"Croatia":2,
    # 3 Eastern Europe
    "Bulgaria":3,"Estonia":3,"Latvia":3,"Lithuania":3,
    "Poland":3,"Czech Republic":3,"Romania":3,
    # 4 UK & Ireland
    "United Kingdom":4,"Ireland":4,
    # 5 Other / North America / Asia
    "United States":5,"Canada":5,"Japan":5,"Singapore":5,
}
REGION_NAMES = ["W.Europe","Nordic","S.Europe","E.Europe","UK/Ireland","Other"]

def hour_to_tod(hour):
    """Time-of-day: morning / midday-peak / evening / night"""
    if   6  <= hour <= 10: return 0  # morning commute
    elif 11 <= hour <= 15: return 1  # midday-peak (highest danger — from atlas)
    elif 16 <= hour <= 20: return 2  # evening
    else:                  return 3  # night

TOD_NAMES = ["Morning","Midday-Peak","Evening","Night"]

# ── Feature cache ──────────────────────────────────────────────────────────────
class FeatureCache:
    """Lazy-loads feature chunks from disk, caches in memory by chunk id."""
    def __init__(self, features_dir):
        self.dir   = features_dir
        self._cache = {}
        self._index = {}   # clip_id → (chunk_path, row_index)
        self._built = False

    def build_index(self, clip_ids):
        """Scan all chunks once and build clip_id → location mapping."""
        if self._built: return
        print("Building feature index (scanning chunks)...")
        chunks = sorted([f for f in os.listdir(self.dir) if f.endswith(".npz")])
        found = 0
        target = set(clip_ids)
        for chunk_file in chunks:
            path = os.path.join(self.dir, chunk_file)
            try:
                data = np.load(path, allow_pickle=True)
                ids  = data["clip_ids"] if "clip_ids" in data else data.get("ids", [])
                for i, cid in enumerate(ids):
                    cid_str = str(cid)
                    if cid_str in target:
                        self._index[cid_str] = (path, i)
                        found += 1
            except Exception as e:
                continue
            if found >= len(target): break
        print(f"  Indexed {found}/{len(target)} clips")
        self._built = True

    def get(self, clip_id):
        if clip_id not in self._index:
            return None
        path, idx = self._index[clip_id]
        if path not in self._cache:
            data = np.load(path, allow_pickle=True)
            key  = "features" if "features" in data else list(data.keys())[0]
            self._cache[path] = data[key]
        feat = self._cache[path][idx]  # (5, 768) or (N_frames, 768)
        # Ensure shape is (N_FRAMES, 768)
        if feat.ndim == 1:
            feat = feat.reshape(1, -1).repeat(N_FRAMES, axis=0)
        elif feat.shape[0] > N_FRAMES:
            feat = feat[:N_FRAMES]
        elif feat.shape[0] < N_FRAMES:
            pad = np.zeros((N_FRAMES - feat.shape[0], feat.shape[1]))
            feat = np.vstack([feat, pad])
        return feat.astype(np.float32)

# ── Dataset ────────────────────────────────────────────────────────────────────
class DriveBenchDataset(Dataset):
    def __init__(self, df, cache):
        self.df    = df.reset_index(drop=True)
        self.cache = cache

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        feat = self.cache.get(str(row["clip_id"]))
        if feat is None:
            feat = np.zeros((N_FRAMES, 768), dtype=np.float32)

        danger  = float(row.get("is_danger", 0))
        radar   = float(row.get("radar_available", 1))
        region  = int(COUNTRY_TO_REGION.get(str(row.get("country","Unknown")), 5))
        hour    = int(row.get("hour_of_day", 12)) if pd.notna(row.get("hour_of_day")) else 12
        tod     = hour_to_tod(hour)
        # Traffic: normalize object count (from object features if available, else 0.5)
        traffic = float(row.get("n_objects_norm", 0.5))
        # TTC regression target: clip to [0, 30], normalise to [0, 1]
        ttc_raw = row.get("min_ttc", 30.0)
        ttc     = float(np.clip(ttc_raw if pd.notna(ttc_raw) else 30.0, 0, 30) / 30.0)

        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(danger,  dtype=torch.float32),
            torch.tensor(region,  dtype=torch.long),
            torch.tensor(tod,     dtype=torch.long),
            torch.tensor(radar,   dtype=torch.float32),
            torch.tensor(traffic, dtype=torch.float32),
            torch.tensor(ttc,     dtype=torch.float32),
        )

# ── Model ──────────────────────────────────────────────────────────────────────
class DriveBenchModel(nn.Module):
    """
    Shared TransformerEncoder backbone → 256-dim DriveBench embedding
    Five multi-task heads force the embedding to capture:
      danger context, geographic patterns, temporal patterns,
      sensor context, and traffic density simultaneously.
    The 256-dim embedding IS the contribution — usable downstream
    like ImageNet features, but for driving scenes.
    """
    def __init__(self, embed_dim=EMBED_DIM, n_frames=N_FRAMES, n_regions=N_REGIONS):
        super().__init__()

        # ── Shared backbone ───────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.pos_embed = nn.Embedding(n_frames + 1, 768)
        layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.norm        = nn.LayerNorm(768)

        # ── Projection → DriveBench embedding ────────────────────────
        self.projector = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # ── Multi-task heads ─────────────────────────────────────────
        self.danger_head  = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.region_head  = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, n_regions))
        self.tod_head     = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 4))
        self.radar_head   = nn.Sequential(nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.traffic_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, 1))
        self.ttc_head     = nn.Sequential(nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def encode(self, x):
        B   = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.shape[1], device=x.device)
        x   = x + self.pos_embed(pos)
        x   = self.norm(self.transformer(x))
        return self.projector(x[:, 0])  # CLS token → embedding

    def forward(self, x):
        emb = self.encode(x)
        return {
            "embedding": emb,
            "danger":    self.danger_head(emb).squeeze(-1),
            "region":    self.region_head(emb),
            "tod":       self.tod_head(emb),
            "radar":     self.radar_head(emb).squeeze(-1),
            "traffic":   self.traffic_head(emb).squeeze(-1),
            "ttc":       self.ttc_head(emb).squeeze(-1),
        }

# ── Loss ───────────────────────────────────────────────────────────────────────
class DriveBenchLoss(nn.Module):
    """Weighted multi-task loss — weights tuned to balance task difficulties."""
    def __init__(self, danger_pos_weight=10.0):
        super().__init__()
        pw = torch.tensor([danger_pos_weight])
        self.bce_danger  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.ce_region   = nn.CrossEntropyLoss()
        self.ce_tod      = nn.CrossEntropyLoss()
        self.bce_radar   = nn.BCEWithLogitsLoss()
        self.huber       = nn.HuberLoss(delta=0.1)

        # Task weights — danger and radar most important
        self.W = {"danger":2.5, "region":1.0, "tod":0.8,
                  "radar":1.5, "traffic":0.6, "ttc":0.5}

    def forward(self, preds, targets):
        d, reg, tod, rad, traf, ttc = targets
        losses = {
            "danger":  self.bce_danger(preds["danger"], d),
            "region":  self.ce_region(preds["region"], reg),
            "tod":     self.ce_tod(preds["tod"], tod),
            "radar":   self.bce_radar(preds["radar"], rad),
            "traffic": self.huber(torch.sigmoid(preds["traffic"]), traf),
            "ttc":     self.huber(torch.sigmoid(preds["ttc"]),     ttc),
        }
        total = sum(self.W[k] * v for k, v in losses.items())
        return total, losses

# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds = {k: [] for k in ["danger","region","tod","radar","traffic","ttc"]}
    all_true  = {k: [] for k in ["danger","region","tod","radar","traffic","ttc"]}
    total_loss = 0.0

    for batch in loader:
        x, danger, region, tod, radar, traffic, ttc = [b.to(device) for b in batch]
        out = model(x)
        targets = (danger, region, tod, radar, traffic, ttc)
        loss, _ = criterion(out, targets)
        total_loss += loss.item()

        all_preds["danger"].extend(torch.sigmoid(out["danger"]).cpu().numpy())
        all_preds["region"].extend(out["region"].argmax(1).cpu().numpy())
        all_preds["tod"].extend(out["tod"].argmax(1).cpu().numpy())
        all_preds["radar"].extend(torch.sigmoid(out["radar"]).cpu().numpy())
        all_preds["traffic"].extend(torch.sigmoid(out["traffic"]).cpu().numpy())
        all_preds["ttc"].extend(torch.sigmoid(out["ttc"]).cpu().numpy())

        all_true["danger"].extend(danger.cpu().numpy())
        all_true["region"].extend(region.cpu().numpy())
        all_true["tod"].extend(tod.cpu().numpy())
        all_true["radar"].extend(radar.cpu().numpy())
        all_true["traffic"].extend(traffic.cpu().numpy())
        all_true["ttc"].extend(ttc.cpu().numpy())

    d_auc  = roc_auc_score(all_true["danger"],  all_preds["danger"])
    r_acc  = accuracy_score(all_true["region"],  all_preds["region"])
    t_acc  = accuracy_score(all_true["tod"],     all_preds["tod"])
    ra_auc = roc_auc_score(all_true["radar"],    all_preds["radar"])
    def safe_r(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6: return 0.0
        try: return float(pearsonr(a, b)[0])
        except: return 0.0

    tr_r  = safe_r(all_true["traffic"], all_preds["traffic"])
    ttc_r = safe_r(all_true["ttc"],     all_preds["ttc"])

    composite = float(np.nanmean([d_auc, r_acc, t_acc, ra_auc,
                                   max(tr_r, 0), max(ttc_r, 0)]))

    metrics = {
        "danger_auc": d_auc, "region_acc": r_acc, "tod_acc": t_acc,
        "radar_auc": ra_auc, "traffic_r": tr_r, "ttc_r": ttc_r,
        "composite": composite, "loss": total_loss/len(loader)
    }
    return metrics

# ── Export embeddings ──────────────────────────────────────────────────────────
@torch.no_grad()
def export_embeddings(model, loader, device, path):
    """Export 256-dim DriveBench embedding for every clip — the ImageNet equivalent."""
    model.eval()
    embeddings, clip_ids = [], []
    for batch in loader:
        x = batch[0].to(device)
        emb = model.encode(x).cpu().numpy()
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    np.savez_compressed(path, embeddings=embeddings)
    print(f"Exported {len(embeddings):,} embeddings → {path}")
    print(f"Shape: {embeddings.shape}  — 298k × 256-dim DriveBench vectors")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("DriveBench: General-Purpose Driving Scene Encoder")
    print("Multi-task pre-training on 298,326 clips — 25 countries")
    print("=" * 65)

    # Load metadata
    print("\nLoading dataset CSV...")
    df = pd.read_csv(DATASET_CSV)
    print(f"  Total clips: {len(df):,}")

    # Add normalized traffic column (proxy from dataset stats)
    if "n_objects" in df.columns:
        df["n_objects_norm"] = (df["n_objects"] / df["n_objects"].max()).fillna(0.5)
    else:
        df["n_objects_norm"] = 0.5

    # Build feature cache + index
    cache = FeatureCache(FEATURES_DIR)
    cache.build_index(df["clip_id"].astype(str).tolist())

    # Splits
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Datasets and loaders
    train_ds = DriveBenchDataset(train_df, cache)
    val_ds   = DriveBenchDataset(val_df,   cache)
    test_ds  = DriveBenchDataset(test_df,  cache)
    all_ds   = DriveBenchDataset(df,       cache)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    all_loader   = DataLoader(all_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model     = DriveBenchModel().to(DEVICE)
    criterion = DriveBenchLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDriveBench parameters: {total_params:,}")

    # Training
    best_composite = 0.0
    print(f"\nTraining on {DEVICE.upper()} | {EPOCHS} epochs | batch {BATCH_SIZE}\n")
    print(f"{'Epoch':>5} {'Loss':>8} {'DangAUC':>9} {'RegAcc':>8} {'TodAcc':>8} {'RadAUC':>9} {'TrafR':>7} {'Comp':>7}")
    print("-" * 68)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            x, danger, region, tod, radar, traffic, ttc = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            out = model(x)
            loss, _ = criterion(out, (danger, region, tod, radar, traffic, ttc))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validate every epoch
        m = evaluate(model, val_loader, criterion, DEVICE)

        print(f"{epoch:>5} {avg_loss:>8.4f} {m['danger_auc']:>9.4f} {m['region_acc']:>8.4f} "
              f"{m['tod_acc']:>8.4f} {m['radar_auc']:>9.4f} {m['traffic_r']:>7.4f} {m['composite']:>7.4f}  "
              f"({time.time()-t0:.0f}s)")

        if m["composite"] > best_composite:
            best_composite = m["composite"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": m,
                "embed_dim": EMBED_DIM,
                "n_regions": N_REGIONS,
                "region_names": REGION_NAMES,
                "tod_names": TOD_NAMES,
            }, OUTPUT_MODEL)
            print(f"       *** Best composite {best_composite:.4f} — saved ***")

    # Test evaluation
    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS (Greece + Bulgaria — never seen in training)")
    ckpt = torch.load(OUTPUT_MODEL)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, criterion, DEVICE)
    print(f"  Danger AUC:       {test_m['danger_auc']:.4f}")
    print(f"  Region Accuracy:  {test_m['region_acc']:.4f}")
    print(f"  Time-of-Day Acc:  {test_m['tod_acc']:.4f}")
    print(f"  Radar AUC:        {test_m['radar_auc']:.4f}")
    print(f"  Traffic Pearson r:{test_m['traffic_r']:.4f}")
    print(f"  TTC Pearson r:    {test_m['ttc_r']:.4f}")
    print(f"  Composite Score:  {test_m['composite']:.4f}")

    # Export all 298k embeddings
    print(f"\nExporting DriveBench-256 embeddings for all {len(df):,} clips...")
    export_embeddings(model, all_loader, DEVICE, OUTPUT_EMBEDS)

    print("\n" + "=" * 65)
    print("DriveBench training complete.")
    print(f"Model:      {OUTPUT_MODEL}")
    print(f"Embeddings: {OUTPUT_EMBEDS}")
    print("These 256-dim vectors are the DriveBench contribution —")
    print("the first general driving scene encoder at 25-country scale.")
    print("=" * 65)

    # Save results summary
    results = {"val": ckpt["val_metrics"], "test": test_m,
                "total_clips": len(df), "embed_dim": EMBED_DIM}
    with open(f"{MODELS_DIR}/drivebench_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {MODELS_DIR}/drivebench_results.json")

if __name__ == "__main__":
    main()