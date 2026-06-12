# PRECOG: Proactive Risk and Environmental Cognition for Autonomous Vehicles

**Nikhil Upadhyay** | Independent Researcher | Dublin Business School

[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/TrazeMaG/PRECOG-AV)
[![Weights](https://img.shields.io/badge/Weights-HuggingFace-yellow)](https://huggingface.co/TrazeMaG)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

PRECOG is a proactive danger anticipation system for autonomous vehicles trained on
**298,326 real driving clips across 25 countries** — the largest scale ever used in
accident anticipation research. It detects road hazards, calculates risk, and averts
danger before threats become geometrically measurable.

---

## Results

### PhysicalAI-AV Benchmark (Our Dataset)

| Module | Metric | Value |
|--------|--------|-------|
| SENSE | Test AUC | **1.0000** |
| SENSE | Accuracy | **99.71%** |
| HERALD v2 | Test AUC | **0.8805** |
| HERALD v2 | Average Precision | **0.2593** |
| HERALD v2 | Geographic Gap (GGG) | **−0.018** (better on unseen countries) |
| HERALD v2 | Sensor Degradation Robustness | −0.119 (radar-agnostic) |

### Standard Benchmarks

| Benchmark | Metric | PRECOG | Previous Best |
|-----------|--------|--------|---------------|
| CCD | Average Precision | **99.95%** | 99.80% (RARE) |
| CCD | mTTA | **4.25s** | — |
| DAD | mTTA | **3.83s** | 3.16s (LATTE) |
| DAD | Average Precision | 54.82% | — |

### Additional Experiments

| Experiment | Key Finding |
|-----------|-------------|
| Ablation | Camera 0.835 → +Object 0.875 → +SENSE 0.879 AUC |
| Scaling Law | Peaks at 50% data (138k clips) — architecture bottleneck identified |
| Geographic Atlas | Italy 9.9% vs Estonia 2.1% danger rate — 4.7× ratio across 25 countries |
| Hour of Day | Peak danger 13:00–15:00, safest 04:00 |
| Sensor Degradation | SENSE: 0.996 → 0.012 when radar fails (98.8% sensitivity) |
| Inference Speed | 30 FPS on RTX 4060 Laptop — 3× faster than real-time AV requirement |
| SENSE Leakage Validation | 0% country prediction accuracy — genuine signal confirmed |
| Causal Inference mTTA | PRECOG fires ~5.7s before TTC danger condition on test clips |
| Camera-Only Performance | 94.8% of full-model AUC with camera alone |
| PhysicalAI-AV mTTA | Mean 11.18s danger window across 2,177 labelled clips |

---

## Architecture
Camera Frames
│
▼
ViT-B/16 ──────────────────────────┐
│                              │
▼                              ▼
SENSE                          HERALD v2
(Radar Health)              (Danger Anticipation)
AUC 1.00                       AUC 0.88
│                              │
└──────────────┬───────────────┘
▼
VOICE
(Language Alert)

**SENSE** predicts radar sensor health from camera-only features using a 4-layer MLP
trained on 276,445 clips across 20 countries. AUC 1.0000 on held-out test countries
(Greece, Bulgaria) with 0% geographic leakage confirmed.

**HERALD v2** is a two-stream transformer combining ViT-B/16 camera features and
object proximity statistics to anticipate danger. Trained on 276,445 clips, evaluated
on 6,602 clips from held-out countries never seen during training.

**VOICE** generates natural language safety alerts using rule-based logic over SENSE
and HERALD outputs. Alpamayo (NVIDIA) integration planned as future work.

---

## Dataset

This project uses the [NVIDIA PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset (gated access — request at HuggingFace).

**Geographic split:**
- Train: 20 countries, 275,573 clips
- Val: Austria, Finland, Portugal — 16,151 clips
- Test: Greece, Bulgaria — 6,602 clips

**Danger labels:** 14,192 clips labelled dangerous via TTC-based physics mining.
Labels available at [TrazeMaG/PRECOG-Labels](https://huggingface.co/datasets/TrazeMaG/PRECOG-Labels).

---

## Installation

```bash
git clone https://github.com/TrazeMaG/PRECOG-AV
cd PRECOG-AV
conda create -n precog python=3.11
conda activate precog
pip install -r requirements.txt
```

---

## Reproducing Results

### 1. Download dataset
Request access at https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

### 2. Extract features
```bash
python src/data/extract_all.py           # ViT-B/16 features (~3,146 chunks)
python src/data/extract_object_features.py  # Object proximity statistics
```

### 3. Mine danger labels
```bash
python src/data/mine_all.py              # TTC-based danger labelling
python src/data/build_dataset.py         # Geographic train/val/test splits
```

### 4. Train models
```bash
python src/models/train_sense.py         # Train SENSE (radar health)
python src/models/train_herald_v2.py     # Train HERALD (danger anticipation)
```

### 5. Run benchmarks
```bash
python src/benchmarks/setup_benchmark.py # CCD setup
python src/benchmarks/fix_ccd_mtta.py    # CCD mTTA (causal inference)
python src/benchmarks/dad_twostream.py   # DAD two-stream evaluation
```

### 6. Run experiments
```bash
python src/experiments/ablation.py         # Ablation study
python src/experiments/scaling_law.py      # Data scaling law
python src/experiments/geo_atlas.py        # Geographic danger atlas
python src/experiments/sensor_degradation.py  # Sensor robustness
python src/experiments/inference_timing.py    # Real-time speed test
```

### 7. Run inference
```bash
python src/inference/precog_pipeline.py  # End-to-end inference on any video
```

### 8. Launch demo
```bash
python src/demo/app.py                   # Gradio demo (local)
```

---

## Pre-trained Weights

| Model | HuggingFace | Size |
|-------|-------------|------|
| SENSE | [TrazeMaG/PRECOG-SENSE](https://huggingface.co/TrazeMaG/PRECOG-SENSE) | 2.1 MB |
| HERALD v2 | [TrazeMaG/PRECOG-HERALD](https://huggingface.co/TrazeMaG/PRECOG-HERALD) | 37 MB |

```python
from huggingface_hub import hf_hub_download
sense_path  = hf_hub_download("TrazeMaG/PRECOG-SENSE",  "sense_best.pt")
herald_path = hf_hub_download("TrazeMaG/PRECOG-HERALD", "herald_v2_best.pt")
```

---

## Configuration

Set your data paths in `config.py` before running any script:

```python
FEATURES_DIR = "/path/to/precog/features"
MODELS_DIR   = "/path/to/precog/models"
DATA_DIR     = "/path/to/precog"
```

---

## Citation

```bibtex
@misc{upadhyay2026precog,
  title   = {PRECOG: Proactive Risk and Environmental Cognition 
             for Autonomous Vehicles -- A 298,326-Clip Multi-Country Study},
  author  = {Upadhyay, Nikhil},
  year    = {2026},
  url     = {https://github.com/TrazeMaG/PRECOG-AV}
}
```

---

## Acknowledgements

Built on the [NVIDIA PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
dataset. ViT-B/16 features from [timm](https://github.com/huggingface/pytorch-image-models).
Object detection via [YOLOv8](https://github.com/ultralytics/ultralytics).

---

*Independent research | Dublin Business School | 2026*
