\---

language: en

tags:

&#x20; - autonomous-vehicles

&#x20; - danger-anticipation

&#x20; - transformer

&#x20; - computer-vision

&#x20; - safety

license: mit

\---



\# PRECOG-HERALD: Proactive Danger Anticipation



\*\*Author:\*\* Nikhil Upadhyay | Independent Researcher | Dublin Business School  

\*\*Project:\*\* \[PRECOG-AV](https://github.com/TrazeMaG/PRECOG-AV)



\## Model Description



HERALD v2 is a two-stream transformer that anticipates danger in driving scenes

from ViT-B/16 camera features. It is Module 2 of the PRECOG system.

Trained on 276,445 clips across 20 countries — the largest scale ever used in

accident anticipation research.



\## Performance



\### PhysicalAI-AV (Our Benchmark — 25 countries)



| Metric | Value |

|--------|-------|

| Test AUC | \*\*0.8805\*\* |

| Average Precision | \*\*0.2593\*\* |

| Geographic Gap (GGG) | \*\*−0.018\*\* (better on unseen countries) |

| Sensor Degradation Robustness | −0.119 (radar-agnostic) |



\### Standard Benchmarks



| Benchmark | Metric | PRECOG | Previous Best |

|-----------|--------|--------|---------------|

| CCD | AP | \*\*99.95%\*\* | 99.80% (RARE) |

| CCD | mTTA | \*\*4.25s\*\* | — |

| DAD | mTTA | \*\*3.83s\*\* | 3.16s (LATTE) |



\## Usage



```python

import torch

import torch.nn as nn

from huggingface\_hub import hf\_hub\_download



class HERALDv2(nn.Module):

&#x20;   def \_\_init\_\_(self, n\_frames=5):

&#x20;       super().\_\_init\_\_()

&#x20;       self.cls\_token   = nn.Parameter(torch.randn(1, 1, 768))

&#x20;       self.pos\_embed   = nn.Embedding(n\_frames + 1, 768)

&#x20;       layer = nn.TransformerEncoderLayer(

&#x20;           d\_model=768, nhead=4, dim\_feedforward=1536,

&#x20;           dropout=0.3, batch\_first=True, norm\_first=True)

&#x20;       self.transformer = nn.TransformerEncoder(layer, num\_layers=2)

&#x20;       self.cam\_norm    = nn.LayerNorm(768)

&#x20;       self.obj\_encoder = nn.Sequential(

&#x20;           nn.Linear(7, 64), nn.GELU(), nn.Dropout(0.3),

&#x20;           nn.Linear(64, 128), nn.GELU())

&#x20;       self.head = nn.Sequential(

&#x20;           nn.Linear(896, 256), nn.GELU(), nn.Dropout(0.3),

&#x20;           nn.Linear(256, 64),  nn.GELU(), nn.Linear(64, 1))

&#x20;   def forward(self, x, obj):

&#x20;       B   = x.shape\[0]

&#x20;       cls = self.cls\_token.expand(B, -1, -1)

&#x20;       x   = torch.cat(\[cls, x], dim=1)

&#x20;       pos = torch.arange(x.shape\[1], device=x.device)

&#x20;       x   = x + self.pos\_embed(pos)

&#x20;       x   = self.cam\_norm(self.transformer(x))

&#x20;       return self.head(torch.cat(

&#x20;           \[x\[:, 0], self.obj\_encoder(obj)], dim=1)).squeeze(-1)



path  = hf\_hub\_download("Trazemag/PRECOG-HERALD", "herald\_v2\_best.pt")

model = HERALDv2()

model.load\_state\_dict(torch.load(path, map\_location="cpu"))

model.eval()



\# x:   (1, N\_FRAMES, 768) — ViT-B/16 features per frame

\# obj: (1, 7)             — object proximity statistics (zeros = camera-only)

\# output: danger probability in \[0, 1]

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

