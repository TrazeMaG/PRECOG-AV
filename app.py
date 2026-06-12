"""
PRECOG — HuggingFace Spaces Demo
Proactive Risk and Environmental Cognition for Autonomous Vehicles
MAGWOLF Research | Dublin Business School
"""
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import cv2
import timm
import os
import tempfile
import traceback
import imageio
from torchvision import transforms
from huggingface_hub import hf_hub_download

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_FRAMES = 5

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
        return self.head(torch.cat(
            [x[:, 0], self.obj_encoder(obj)], dim=1)).squeeze(-1)

# ── Load models ────────────────────────────────────────────────────────────────
print(f"Loading PRECOG on {DEVICE}...")

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

vit = timm.create_model(
    "vit_base_patch16_224", pretrained=True, num_classes=0)
vit.eval().to(DEVICE)

sense_model  = SENSEModel().to(DEVICE)
herald_model = HERALDv2().to(DEVICE)

def load_weights(model, local_path, hf_repo, hf_filename):
    if os.path.exists(local_path):
        model.load_state_dict(torch.load(local_path, map_location=DEVICE))
        print(f"  Loaded {hf_filename} from local")
    else:
        print(f"  Downloading {hf_filename} from {hf_repo}...")
        path = hf_hub_download(hf_repo, hf_filename)
        model.load_state_dict(torch.load(path, map_location=DEVICE))

load_weights(sense_model,
    "D:/precog/models/sense_best.pt",
    "MAGWOLF/PRECOG-SENSE", "sense_best.pt")
load_weights(herald_model,
    "D:/precog/models/herald_v2_best.pt",
    "MAGWOLF/PRECOG-HERALD", "herald_v2_best.pt")

sense_model.eval()
herald_model.eval()
print("PRECOG ready.")

# ── Overlay ────────────────────────────────────────────────────────────────────
def annotate_frame(frame, danger_prob, sense_score, W, H):
    frame = frame.copy()
    if danger_prob > 0.7:
        color, level_txt = (0, 0, 220),   "CRITICAL"
    elif danger_prob > 0.4:
        color, level_txt = (0, 140, 255), "WARNING"
    else:
        color, level_txt = (0, 200, 80),  "NOMINAL"

    sense_color = (0, 200, 80) if sense_score > 0.5 else (0, 0, 220)
    border      = 10 if danger_prob > 0.4 else 3
    cv2.rectangle(frame, (0, 0), (W-1, H-1), color, border)

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (295, 172), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (8, 8), (295, 172), color, 2)

    cv2.putText(frame, f"PRECOG  {level_txt}",
        (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"DANGER  {danger_prob*100:.0f}%",
        (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.rectangle(frame, (16, 80), (283, 96), (50, 50, 50), -1)
    cv2.rectangle(frame, (16, 80),
        (16 + int(danger_prob * 267), 96), color, -1)

    cv2.putText(frame, f"SENSE   {sense_score:.2f}",
        (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.rectangle(frame, (16, 128), (283, 144), (50, 50, 50), -1)
    cv2.rectangle(frame, (16, 128),
        (16 + int(sense_score * 267), 144), sense_color, -1)

    cv2.putText(frame,
        "RADAR: NOMINAL" if sense_score > 0.5 else "RADAR: DEGRADED",
        (16, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sense_color, 1)
    return frame

# ── Video writer using imageio ─────────────────────────────────────────────────
def write_mp4(frames_bgr, fps):
    tmp  = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = tmp.name
    tmp.close()
    writer = imageio.get_writer(
        path, fps=max(float(fps), 1.0),
        codec="libx264", quality=8,
        ffmpeg_params=["-pix_fmt", "yuv420p"])
    for f in frames_bgr:
        writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    writer.close()
    return path

# ── Main inference ─────────────────────────────────────────────────────────────
def run_precog(video_path):
    if video_path is None:
        return None, "—", "—", "Please upload a video clip first."

    try:
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 10.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if total == 0 or W == 0 or H == 0:
            return None, "—", "—", "Could not read video file."

        # Sample N frames for inference
        cap      = cv2.VideoCapture(video_path)
        frames_t = []
        for idx in np.linspace(0, total-1, N_FRAMES, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, f = cap.read()
            if ret:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                frames_t.append(preprocess(rgb))
        cap.release()

        if len(frames_t) < N_FRAMES:
            return None, "—", "—", "Not enough frames in video."

        # Model inference
        with torch.no_grad():
            vit_feats   = vit(torch.stack(frames_t).to(DEVICE)).cpu()
            avg         = vit_feats.mean(0, keepdim=True).to(DEVICE)
            sense_score = float(sense_model(avg).item())
            feat_t      = vit_feats.unsqueeze(0).to(DEVICE)
            obj_zeros   = torch.zeros(1, 7).to(DEVICE)
            danger_prob = float(torch.sigmoid(
                herald_model(feat_t, obj_zeros)).item())

        # Build annotated video
        cap         = cv2.VideoCapture(video_path)
        output_bgr  = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output_bgr.append(
                annotate_frame(frame, danger_prob, sense_score, W, H))
        cap.release()

        out_path = write_mp4(output_bgr, fps)
        alert    = build_voice(sense_score, danger_prob)

        return (out_path,
                f"{sense_score:.3f}",
                f"{danger_prob:.3f}",
                alert)

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return None, "Error", "Error", f"Exception:\n{tb}"

def build_voice(sense, danger):
    if danger > 0.7:
        lvl = "🔴 CRITICAL"
    elif danger > 0.4:
        lvl = "🟡 WARNING"
    else:
        return "🟢 NOMINAL — No immediate danger detected. Monitoring active."
    parts = [f"{lvl} — Danger probability {danger*100:.0f}%."]
    if sense < 0.3:
        parts.append(
            f"⚠ Radar DEGRADED (SENSE {sense:.2f}) — "
            f"operating on camera-only mode.")
    elif sense < 0.6:
        parts.append(f"⚠ Radar REDUCED (SENSE {sense:.2f}).")
    else:
        parts.append(f"✓ Sensor health nominal (SENSE {sense:.2f}).")
    parts.append("Recommend caution and speed reduction.")
    return " ".join(parts)

# ── UI ─────────────────────────────────────────────────────────────────────────
css = """
#title    { text-align:center; font-size:2rem; font-weight:700;
            margin-bottom:0.2rem; }
#subtitle { text-align:center; color:#888; margin-bottom:1.5rem;
            font-size:0.9rem; line-height:1.7; }
footer    { display:none !important; }
"""

with gr.Blocks(title="PRECOG — AV Danger Anticipation") as demo:
    gr.HTML("""
        <h1 id='title'>PRECOG</h1>
        <p id='subtitle'>
            Proactive Risk and Environmental Cognition for Autonomous Vehicles
            &nbsp;·&nbsp; MAGWOLF Research &nbsp;·&nbsp;
            Dublin Business School<br>
            <small>
            298,326 clips · 25 countries ·
            CCD: 99.95% AP ·
            DAD: mTTA 3.83s (best published) ·
            30 FPS on consumer GPU
            </small>
        </p>""")

    with gr.Row():
        with gr.Column(scale=1):
            video_in  = gr.Video(
                label="Upload dashcam / AV clip", height=320)
            run_btn   = gr.Button("▶  Run PRECOG", variant="primary")

            with gr.Row():
                sense_out  = gr.Textbox(
                    label="SENSE — Radar health (0=degraded · 1=nominal)",
                    interactive=False)
                danger_out = gr.Textbox(
                    label="HERALD — Danger probability",
                    interactive=False)

            alert_out = gr.Textbox(
                label="VOICE — Natural language alert",
                lines=3, interactive=False)

        with gr.Column(scale=1):
            video_out = gr.Video(
                label="PRECOG output (annotated)", height=320)

    gr.HTML("""
        <hr style='margin:1.5rem 0;border-color:#333'>
        <div style='display:flex;justify-content:space-around;
                    flex-wrap:wrap;gap:1rem;text-align:center;
                    color:#aaa;font-size:0.82rem;padding-bottom:1rem'>
            <div><b style='color:#fff'>SENSE</b><br>
                 Camera-only radar health · AUC 1.00</div>
            <div><b style='color:#fff'>HERALD</b><br>
                 Danger anticipation · AUC 0.88</div>
            <div><b style='color:#fff'>VOICE</b><br>
                 Natural language alerts</div>
            <div><b style='color:#fff'>Scale</b><br>
                 25 countries · 1,700 hours</div>
        </div>""")

    run_btn.click(
        fn=run_precog,
        inputs=[video_in],
        outputs=[video_out, sense_out, danger_out, alert_out])

if __name__ == "__main__":
    demo.launch(share=True, css=css)