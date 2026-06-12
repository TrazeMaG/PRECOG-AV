"""
PRECOG Full Inference Pipeline
Input:  any video file
Output: SENSE score, HERALD danger probability, VOICE alert
Uses camera-only HERALD (AUC 0.8351) for deployment on arbitrary video.
Oracle results (AUC 0.8805) reported in paper using ground truth labels.
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import timm
import os
from torchvision import transforms

MODEL_DIR = "D:/precog/models"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_FRAMES  = 5

class SENSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64),  nn.GELU(), nn.Linear(64, 1))
    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

class HERALDv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token   = nn.Parameter(torch.randn(1, 1, 768))
        self.pos_embed   = nn.Embedding(N_FRAMES + 1, 768)
        layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=4, dim_feedforward=1536,
            dropout=0.3, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.cam_norm    = nn.LayerNorm(768)
        self.obj_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 128), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(896, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.GELU(), nn.Linear(64, 1))
    def forward(self, x, obj):
        B   = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.shape[1], device=x.device)
        x   = x + self.pos_embed(pos)
        x   = self.cam_norm(self.transformer(x))
        cam = x[:, 0]
        o   = self.obj_encoder(obj)
        return self.head(torch.cat([cam, o], dim=1)).squeeze(-1)

class PRECOG:
    def __init__(self):
        print("Loading PRECOG pipeline...")

        # ViT feature extractor
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0)
        self.vit.eval().to(DEVICE)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

        # SENSE
        self.sense = SENSEModel().to(DEVICE)
        self.sense.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "sense_best.pt"),
            map_location=DEVICE))
        self.sense.eval()

        # HERALD (camera-only path — obj_zeros passed at inference)
        self.herald = HERALDv2().to(DEVICE)
        self.herald.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, "herald_v2_best.pt"),
            map_location=DEVICE))
        self.herald.eval()

        print(f"PRECOG ready on {DEVICE}")

    def extract_frames(self, video_path, n=N_FRAMES):
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None
        frames = []
        for idx in np.linspace(0, total - 1, n, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, f = cap.read()
            if ret:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                frames.append(self.preprocess(rgb))
        cap.release()
        if len(frames) < n:
            return None
        return torch.stack(frames)

    def run(self, video_path):
        tensors = self.extract_frames(video_path)
        if tensors is None:
            return {"error": "Could not read video"}

        with torch.no_grad():
            # Camera features via ViT
            vit_feats = self.vit(tensors.to(DEVICE)).cpu()  # (N, 768)

            # SENSE: sensor health from average frame feature
            avg_feat    = vit_feats.mean(dim=0, keepdim=True).to(DEVICE)
            sense_score = float(self.sense(avg_feat).item())

            # HERALD: camera-only danger prediction
            # Object stream set to zeros — camera features carry the signal
            feat_t      = vit_feats.unsqueeze(0).to(DEVICE)  # (1, N, 768)
            obj_zeros   = torch.zeros(1, 7).to(DEVICE)
            danger_logit = self.herald(feat_t, obj_zeros)
            danger_prob  = float(torch.sigmoid(danger_logit).item())

        alert = self._voice(sense_score, danger_prob)

        return {
            "sense_score":  round(sense_score, 3),
            "danger_prob":  round(danger_prob, 3),
            "radar_status": "OPERATIONAL" if sense_score > 0.5 else "DEGRADED",
            "danger_level": ("CRITICAL" if danger_prob > 0.7 else
                             "WARNING"  if danger_prob > 0.4 else "NOMINAL"),
            "alert":        alert,
        }

    def _voice(self, sense, danger):
        if danger > 0.7:
            level = "CRITICAL"
        elif danger > 0.4:
            level = "WARNING"
        else:
            return ("NOMINAL: No immediate danger detected. "
                    "Monitoring active.")

        parts = [level]
        parts.append(f"danger probability {danger*100:.0f}%")

        if sense < 0.3:
            parts.append(
                f"radar DEGRADED (confidence {sense:.2f}) — "
                f"operating on camera-only mode")
        elif sense < 0.6:
            parts.append(
                f"radar REDUCED (confidence {sense:.2f})")

        parts.append("Recommend caution and speed reduction.")
        return ". ".join(parts)


if __name__ == "__main__":
    import glob

    precog = PRECOG()

    danger_clips = sorted(
        glob.glob("D:/precog/demo_clips/danger/*.mp4"))[:3]
    safe_clips   = sorted(
        glob.glob("D:/precog/demo_clips/safe/*.mp4"))[:2]
    test_clips   = danger_clips + safe_clips

    if not test_clips:
        print("No demo clips found yet.")
    else:
        correct = 0
        for clip in test_clips:
            folder = "DANGER" if "danger" in clip else "SAFE"
            result = precog.run(clip)
            predicted = result["danger_level"]
            match = "✓" if (
                (folder == "DANGER" and predicted in ["CRITICAL","WARNING"]) or
                (folder == "SAFE"   and predicted == "NOMINAL")
            ) else "✗"
            print(f"{match} [{folder}] "
                  f"danger={result['danger_prob']:.3f}  "
                  f"sense={result['sense_score']:.3f}  "
                  f"→ {predicted}")
            if match == "✓":
                correct += 1
        print(f"\nCorrect: {correct}/{len(test_clips)}")
        print(f"\nSample alert:")
        print(f"  {precog.run(test_clips[0])['alert']}")