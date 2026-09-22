"""Routing vs ensembling for the ST3 hybrid, and the submission-4 -> 5 ST3 change (Sections 7.2, 7.4).

All ST3 scores use the shipped constraint layer on dev. Reports the full-dev
score of ENC-L1 alone and of QLoRA-ST3 averaged into all eight flags or only the
two disclosure flags, then the gain over ENC-L1 on 20 random channel-grouped
halves of dev (mean, sd, halves improved). The adoption criterion needs 18/20.
The last block checks the whole ST3 change of submission 5 against submission 4.

Usage:
    python src/analysis/st3_blend_routing.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core"))
from data import ST3, load_split, clean_st3, _g
from rules import is_short
import metrics
dv = load_split("dev"); idx = {x["instanceID"]: i for i, x in enumerate(dv)}; gold = [x["labels"]["st3"] for x in dv]
def P(f):
    z = np.load(f, allow_pickle=True); p = np.zeros((len(dv), 8)); p[[idx[i] for i in z["ids"].tolist()]] = z["p3"]; return p
enc = np.mean([P(ROOT / f"work/probs_dev_L1_ModernBERT-large_len1024_seed{s}_ep6.npz") for s in range(6)], 0)
ql = np.mean([P(ROOT / f"work/probs_dev_L2_Qwen2.5-7B-Instruct_len1024_seed{s}_qlora.npz") for s in range(3)], 0)
D = [ST3.index("undisclosed_advertising"), ST3.index("inadequate_disclosure")]
disc = enc.copy(); disc[:, D] = (enc[:, D] + ql[:, D]) / 2
allf = (enc + ql) / 2
def decide(b3, i):
    x = dv[i]; st3 = set(c for j, c in enumerate(ST3) if b3[i, j] >= .5)
    if b3[i, ST3.index("inadequate_disclosure")] >= .40: st3.add("inadequate_disclosure")
    else: st3.discard("inadequate_disclosure")
    if _g(x, "video_context", "official_disclosure") == "true": st3 -= {"undisclosed_advertising"}
    if {"undisclosed_advertising", "inadequate_disclosure"} <= st3: st3 -= {"inadequate_disclosure"}
    if is_short(x): st3 = {"insufficient_context"}
    elif b3[i, ST3.index("no_flag")] >= 0.6: st3 = {"no_flag"}
    return clean_st3(sorted(st3))
def st3(b3, ix): return metrics._multilabel([gold[i] for i in ix], [decide(b3, i) for i in ix], ST3)
ALL = range(len(dv))
print(f"full dev ST3: encoder {st3(enc, ALL):.4f}  all-flags blend {st3(allf, ALL):.4f}  disclosure-only blend {st3(disc, ALL):.4f}")
chan = np.array([_g(x, "channel_context", "channelID") for x in dv]); chans = sorted(set(chan))
for scheme in ("held-out half (one half per draw)",):
    dA, dD, base = [], [], []
    for seed in range(20):
        pick = set(np.random.default_rng(seed).choice(chans, len(chans) // 2, replace=False))
        B = [i for i in ALL if chan[i] not in pick]
        e = st3(enc, B); base.append(e); dA.append(st3(allf, B) - e); dD.append(st3(disc, B) - e)
    dA, dD = np.array(dA), np.array(dD)
    print(f"20 halves: encoder mean {np.mean(base):.4f}")
    print(f"  all-flags blend       {dA.mean():+.4f}  sd {dA.std():.3f}  wins {int((dA>0).sum())}/20")
    print(f"  disclosure-only blend {dD.mean():+.4f}  sd {dD.std():.3f}  wins {int((dD>0).sum())}/20")

# whole ST3 change of submission 5 against submission 4 (4-epoch encoder, no_flag rule only)
e4 = np.mean([P(ROOT / f"work/probs_dev_L1_ModernBERT-large_len1024_seed{s}.npz") for s in range(6)], 0)
def decide4(i):
    x = dv[i]; st3 = set(c for j, c in enumerate(ST3) if e4[i, j] >= .5)
    if _g(x, "video_context", "official_disclosure") == "true": st3 -= {"undisclosed_advertising"}
    u, a = ST3.index("undisclosed_advertising"), ST3.index("inadequate_disclosure")
    if {"undisclosed_advertising", "inadequate_disclosure"} <= st3:   # default tie-break: keep the higher
        st3 -= {"inadequate_disclosure"} if e4[i, u] >= e4[i, a] else {"undisclosed_advertising"}
    if is_short(x): st3 = {"insufficient_context"}
    elif e4[i, ST3.index("no_flag")] >= 0.6: st3 = {"no_flag"}
    return clean_st3(sorted(st3))
s4 = lambda ix: metrics._multilabel([gold[i] for i in ix], [decide4(i) for i in ix], ST3)
print(f"\nsubmission 4 -> 5 ST3 on full dev: {s4(ALL):.4f} -> {st3(disc, ALL):.4f} ({st3(disc, ALL) - s4(ALL):+.4f})")
d = []
for seed in range(20):
    pick = set(np.random.default_rng(seed).choice(chans, len(chans) // 2, replace=False))
    B = [i for i in ALL if chan[i] not in pick]; d.append(st3(disc, B) - s4(B))
d = np.array(d); print(f"  over 20 halves: {d.mean():+.4f}, sd {d.std():.3f}, improved {int((d > 0).sum())}/20")
