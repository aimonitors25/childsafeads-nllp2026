"""Encoder vs TF-IDF vs their 50/50 blend on the 5-fold cross-validation (Section 7.2).

Scores are means over the five channel-grouped folds of train+dev (2,857
out-of-fold instances). Pooling the folds instead gives lower ST1 numbers,
because ST1 is macro-averaged over the labels present in the reference and the
two `other` instances fall in only two folds.

Also answers Appendix D's question: how much of the blend's ST2 gain is just
recalibration? The encoder's own per-label thresholds are fitted on the four
training folds and applied to the held-out fold; the result depends on the
threshold grid, so three grids are reported.

Usage:
    python src/analysis/st2_blend_cv.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core"))
from data import ST1, ST2, load_split
import metrics

W = ROOT / "work"
pool = load_split("train") + load_split("dev")
idx = {x["instanceID"]: i for i, x in enumerate(pool)}
g1 = [x["labels"]["st1"] for x in pool]
g2 = [x["labels"]["st2"] for x in pool]

e1, e2 = np.zeros((len(pool), 5)), np.zeros((len(pool), 12))
fold = np.full(len(pool), -1)
for k in range(5):
    z = np.load(W / "cv" / f"oof_L4_ModernBERT-large_len2048_ep4_lr2e-05_seed0_f{k}of5.npz", allow_pickle=True)
    ix = [idx[i] for i in z["ids"].tolist()]
    e1[ix], e2[ix], fold[ix] = z["p1"], z["p2"], k


def classical(task, ncol):
    out = np.zeros((len(pool), ncol))
    z = np.load(W / f"classical_oof_{task}.npz", allow_pickle=True)
    out[[idx[i] for i in z["ids"].tolist()]] = z["p"]
    return out


c1, c2 = classical("st1", 5), classical("st2", 12)
FOLDS = [[i for i in range(len(pool)) if fold[i] == k] for k in range(5)]


def st1(P, ix):
    gold = [g1[i] for i in ix]
    return f1_score(gold, [ST1[int(np.argmax(P[i]))] for i in ix],
                    labels=sorted(set(gold)), average="macro", zero_division=0)


def st2(P, ix, th=None):
    th = np.full(12, 0.5) if th is None else th
    pred = [[c for j, c in enumerate(ST2) if P[i, j] >= th[j]] or [ST2[int(np.argmax(P[i] - th))]] for i in ix]
    return metrics._multilabel([g2[i] for i in ix], pred, ST2)


for name, fn, E, C in (("ST1", st1, e1, c1), ("ST2", st2, e2, c2)):
    B = (E + C) / 2
    enc, cls, bl = ([fn(P, f) for f in FOLDS] for P in (E, C, B))
    print(f"{name}  encoder {np.mean(enc):.3f}  TF-IDF {np.mean(cls):.3f}  blend {np.mean(bl):.3f}"
          f"  (blend beats encoder on {sum(b > e for b, e in zip(bl, enc))}/5 folds)")

Y = MultiLabelBinarizer(classes=ST2).fit_transform(g2)
base = np.mean([st2(e2, f) for f in FOLDS])
blend = np.mean([st2((e2 + c2) / 2, f) for f in FOLDS])
for step in (0.05, 0.025, 0.01):
    G = np.arange(0.05, 0.9501, step)
    res = []
    for k, va in enumerate(FOLDS):
        tr = np.array([i for i in range(len(pool)) if fold[i] != k])
        th = np.array([max(G, key=lambda t: f1_score(Y[tr, j], e2[tr, j] >= t, zero_division=0))
                       for j in range(12)])
        res.append(st2(e2, va, th))
    print(f"encoder, thresholds fitted on training folds (grid step {step}): ST2 {np.mean(res):.3f}"
          f" = {(np.mean(res) - base) / (blend - base):.0%} of the blend's gain")
