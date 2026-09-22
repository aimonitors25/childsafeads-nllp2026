"""Do tuned ST3 thresholds transfer? Per-label vs global vs joint coordinate ascent.

Each scheme is fitted on one channel-grouped half of dev and scored on the other,
over 20 random halves, against a flat 0.5 on the same halves. ST3 probabilities
are the six-seed, four-epoch L1 encoder (the arm these schemes were tried on);
the rules applied are joint_tune.decide_st3 (paid-promotion flag, short transcript).
Reports the gain on the fitting half (what tuning appears to buy) and on the
held-out half (what it buys on unseen channels).

Usage:
    python src/analysis/threshold_schemes.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core")); sys.path.insert(0, str(ROOT / "src" / "explored"))
from data import ST3, load_split, _g
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import joint_tune as jt
dv = load_split("dev"); idx = {x["instanceID"]: i for i, x in enumerate(dv)}
p3 = np.zeros((len(dv), 8)); n = 0
for s in range(6):
    z = np.load(ROOT / f"work/probs_dev_L1_ModernBERT-large_len1024_seed{s}.npz", allow_pickle=True)
    p3[[idx[i] for i in z["ids"].tolist()]] += z["p3"]; n += 1
p3 /= n
gold = [x["labels"]["st3"] for x in dv]
Y = MultiLabelBinarizer(classes=ST3).fit_transform(gold)
chan = np.array([_g(x, "channel_context", "channelID") for x in dv]); chans = sorted(set(chan))
G = np.arange(0.05, 0.96, 0.025)
def score(ix, th): return jt.st3_macro(p3[ix], [dv[i] for i in ix], [gold[i] for i in ix], th)
def per_label(ix):
    th = np.full(8, .5)
    for j in range(8):
        th[j] = max(G, key=lambda t: f1_score(Y[ix, j], p3[ix, j] >= t, zero_division=0))
    return th
def glob(ix): t = max(G, key=lambda t: score(ix, np.full(8, t))); return np.full(8, t)
def joint(ix): return np.asarray(jt.coordinate_ascent(p3[ix], [dv[i] for i in ix], [gold[i] for i in ix])[0])
res = {k: ([], []) for k in ("per-label", "global", "joint")}
for seed in range(20):
    r = np.random.default_rng(seed); pick = set(r.choice(chans, len(chans) // 2, replace=False))
    A = [i for i in range(len(dv)) if chan[i] in pick]; B = [i for i in range(len(dv)) if chan[i] not in pick]
    base = np.full(8, .5)
    for k, f in (("per-label", per_label), ("global", glob), ("joint", joint)):
        th = f(A)
        res[k][0].append(score(A, th) - score(A, base)); res[k][1].append(score(B, th) - score(B, base))
for k, (ins, out) in res.items():
    out = np.array(out)
    print(f"{k:10s} in-sample {np.mean(ins):+.3f}   held-out {out.mean():+.3f} (sd {out.std():.3f}), wins {int((out > 0).sum())}/20")
