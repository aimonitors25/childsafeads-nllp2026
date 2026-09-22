"""Is QLoRA-ST3's lead on the disclosure flags the model, the seeds, or the input?

QLoRA-ST3 differs from ENC-L1 in more than the model: it reads the level-2
fields (title, description, paid-promotion label) plus a rubric of the
taxonomy's definitions, and Table 7 averages three QLoRA seeds against six
encoder seeds. Two controls, both on dev at a flat 0.5 with no constraint layer
(the setting of Table 7):
  * equal seed counts: encoder seeds 0-2 against the three QLoRA seeds;
  * input: the one encoder run trained on the level-2 view, against the spread
    of single transcript-only encoder runs and single QLoRA runs.
The model and its rubric are not separated by anything here.

Usage:
    python src/analysis/qlora_controls.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core"))
from data import ST3, load_split
import metrics

W = ROOT / "work"
dv = load_split("dev")
idx = {x["instanceID"]: i for i, x in enumerate(dv)}
gold = [x["labels"]["st3"] for x in dv]
FLAGS = ("undisclosed_advertising", "inadequate_disclosure")


def p3(name):
    z = np.load(W / name, allow_pickle=True)
    p = np.zeros((len(dv), 8))
    p[[idx[i] for i in z["ids"].tolist()]] = z["p3"]
    return p


def score(p):
    pred = [[c for j, c in enumerate(ST3) if p[i, j] >= .5] or ["no_flag"] for i in range(len(dv))]
    per = metrics.per_label_f1(gold, pred, ST3)
    return (metrics._multilabel(gold, pred, ST3),) + tuple(per[f] for f in FLAGS)


enc4 = [p3(f"probs_dev_L1_ModernBERT-large_len1024_seed{s}.npz") for s in range(6)]
enc6 = [p3(f"probs_dev_L1_ModernBERT-large_len1024_seed{s}_ep6.npz") for s in range(6)]
qlora = [p3(f"probs_dev_L2_Qwen2.5-7B-Instruct_len1024_seed{s}_qlora.npz") for s in range(3)]
enc_l2 = p3("probs_dev_L2_ModernBERT-large_len2048_seed0.npz")

print(f"{'':32s} {'ST3':>6} {'undisc':>7} {'inadeq':>7}")
for name, p in (("ENC-L1 6 ep, 6 seeds (Table 7)", np.mean(enc6, 0)),
                ("ENC-L1 6 ep, seeds 0-2", np.mean(enc6[:3], 0)),
                ("QLoRA-ST3, 3 seeds (Table 7)", np.mean(qlora, 0)),
                ("ENC at level 2, one run", enc_l2)):
    print(f"{name:32s} " + " ".join(f"{v:7.3f}" for v in score(p)))
for name, runs in (("single ENC-L1 runs (12)", enc4 + enc6), ("single QLoRA runs (3)", qlora)):
    s = np.array([score(p) for p in runs])
    print(f"{name:32s} " + " ".join(f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(s.min(0), s.max(0))))
