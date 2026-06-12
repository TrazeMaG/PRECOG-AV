\---

language: en

tags:

&#x20; - autonomous-vehicles

&#x20; - sensor-health

&#x20; - radar

&#x20; - computer-vision

&#x20; - safety

license: mit

\---



\# PRECOG-SENSE: Camera-Only Radar Health Prediction



\*\*Author:\*\* Nikhil Upadhyay | Independent Researcher | Dublin Business School  

\*\*Project:\*\* \[PRECOG-AV](https://github.com/TrazeMaG/PRECOG-AV)



\## Model Description



SENSE is a lightweight 4-layer MLP that predicts radar sensor availability from

ViT-B/16 camera features alone. It is Module 1 of the PRECOG danger anticipation

system for autonomous vehicles.



\*\*Key result:\*\* AUC 1.0000 on held-out test countries (Greece, Bulgaria) with

0% geographic leakage confirmed — the signal is genuine.



\## Performance



| Metric | Value |

|--------|-------|

| Test AUC | \*\*1.0000\*\* |

| Test Accuracy | \*\*99.71%\*\* |

| Val AUC | 0.9649 |

| Parameters | 541,569 |



\## Training Data



Trained on 276,445 clips from 20 countries using the

\[NVIDIA PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) dataset.

Geographic split: train on 20 countries, validate on Austria/Finland/Portugal,

test on Greece/Bulgaria (unseen during training).



\## Usage



```python

import torch

import torch.nn as nn

from huggingface\_hub import hf\_hub\_download



class SENSEModel(nn.Module):

&#x20;   def \_\_init\_\_(self):

&#x20;       super().\_\_init\_\_()

&#x20;       self.net = nn.Sequential(

&#x20;           nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.3),

&#x20;           nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),

&#x20;           nn.Linear(256, 64),  nn.GELU(), nn.Linear(64, 1))

&#x20;   def forward(self, x):

&#x20;       return torch.sigmoid(self.net(x)).squeeze(-1)



path  = hf\_hub\_download("Trazemag/PRECOG-SENSE", "sense\_best.pt")

model = SENSEModel()

model.load\_state\_dict(torch.load(path, map\_location="cpu"))

model.eval()



\# Input: ViT-B/16 feature vector (768-dim) from any camera frame

\# Output: scalar in \[0,1] — near 1.0 = radar present, near 0.0 = radar absent

```



\## Citation



```bibtex

@misc{upadhyay2026precog,

&#x20; title  = {PRECOG: Proactive Risk and Environmental Cognition for Autonomous Vehicles},

&#x20; author = {Upadhyay, Nikhil},

&#x20; year   = {2026},

&#x20; url    = {https://github.com/TrazeMaG/PRECOG-AV}

}

```

