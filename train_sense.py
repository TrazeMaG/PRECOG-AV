import torch
import torch.nn as nn
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_DIR = "D:/precog/features"
MODEL_DIR    = "D:/precog/models"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 30
BATCH_SIZE   = 512
LR           = 1e-3

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load all features ─────────────────────────────────────────────────────────
print("Loading features from disk...")

train_feats, train_labels = [], []
val_feats,   val_labels   = [], []
test_feats,  test_labels  = [], []

npz_files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.npz")))
print(f"Found {len(npz_files)} chunk files")

for path in tqdm(npz_files, desc="Loading"):
    data = np.load(path, allow_pickle=True)

    feats  = data["features"]   # (n_clips, n_frames, 768)
    labels = data["radar_lbl"]  # (n_clips,) — 1 = radar present, 0 = absent
    splits = data["splits"]     # (n_clips,) — train/val/test

    for i in range(len(feats)):
        f = feats[i].mean(axis=0)   # average frames → (768,)
        l = int(labels[i])
        s = str(splits[i])

        if l < 0:                   # unknown label, skip
            continue

        if s == "val":
            val_feats.append(f)
            val_labels.append(l)
        elif s == "test":
            test_feats.append(f)
            test_labels.append(l)
        else:
            train_feats.append(f)
            train_labels.append(l)

train_feats  = np.array(train_feats,  dtype=np.float32)
val_feats    = np.array(val_feats,    dtype=np.float32)
test_feats   = np.array(test_feats,   dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.float32)
val_labels   = np.array(val_labels,   dtype=np.float32)
test_labels  = np.array(test_labels,  dtype=np.float32)

print(f"\nTrain: {len(train_feats):,} clips")
print(f"Val:   {len(val_feats):,} clips")
print(f"Test:  {len(test_feats):,} clips")
print(f"Radar present in train: {train_labels.mean()*100:.1f}%")

# ── Dataset ───────────────────────────────────────────────────────────────────
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features)
        self.y = torch.tensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FeatureDataset(train_feats, train_labels),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(FeatureDataset(val_feats,   val_labels),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(FeatureDataset(test_feats,  test_labels),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
class SENSEModel(nn.Module):
    """
    Predicts radar sensor availability from averaged ViT frame features.
    Input:  (batch, 768) — scene feature vector
    Output: (batch,)     — radar confidence score in [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

model     = SENSEModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCELoss()

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on: {DEVICE}\n")

# ── Training loop ─────────────────────────────────────────────────────────────
best_auc = 0.0

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Validate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            pred = model(X.to(DEVICE)).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(y.numpy())

    val_auc = roc_auc_score(all_labels, all_preds)
    val_ap  = average_precision_score(all_labels, all_preds)
    avg_loss = train_loss / len(train_loader)

    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}", end="")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "sense_best.pt"))
        print(" ← best")
    else:
        print()

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "sense_best.pt")))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        pred = model(X.to(DEVICE)).cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(y.numpy())

test_auc = roc_auc_score(all_labels, all_preds)
test_ap  = average_precision_score(all_labels, all_preds)
preds_bin = (np.array(all_preds) > 0.5).astype(int)
accuracy  = (preds_bin == np.array(all_labels, dtype=int)).mean()

print(f"\n=== SENSE TEST RESULTS ===")
print(f"AUC-ROC:  {test_auc:.4f}")
print(f"Avg Prec: {test_ap:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"\nModel saved to: {MODEL_DIR}/sense_best.pt")