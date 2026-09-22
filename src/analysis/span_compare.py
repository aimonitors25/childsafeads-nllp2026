"""Paired comparison of the st3_evidence span arm against its own baseline.

Both arms are the same code, same seed, same channel-grouped folds, differing
only in --span-weight, so every fold is a matched pair.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, ST3_FAMILY, load_split, clean_st3, _g
from rules import is_short
import metrics

W = Path(__file__).resolve().parents[2] / "work" / "cv"
pool = load_split("train") + load_split("dev")
by_id = {x["instanceID"]: x for x in pool}
import os
SEED = os.environ.get("SEED", "0")
ARMS = ({"baseline": "_base28", "span": "_span05"} if SEED == "0"
        else {"baseline": "_base28s1", "span": "_span05s1"})
SEEDN = SEED


def decide(ids, probs, no_flag_t=0.6):
    preds = {}
    for i in ids:
        x = by_id[i]
        p1, p2, p3 = probs[i]
        st2 = [c for j, c in enumerate(ST2) if p2[j] >= .5] or [ST2[int(np.argmax(p2))]]
        st3 = set(c for j, c in enumerate(ST3) if p3[j] >= .5)
        if _g(x, "video_context", "official_disclosure") == "true":
            st3 -= {"undisclosed_advertising"}
        if is_short(x):
            st3 = {"insufficient_context"}
        elif no_flag_t and p3[ST3.index("no_flag")] >= no_flag_t:
            st3 = {"no_flag"}
        preds[i] = {"st1": ST1[int(np.argmax(p1))], "st2": st2, "st3": clean_st3(sorted(st3))}
    return preds


P, F = {}, defaultdict(lambda: defaultdict(list))
for arm, tag in ARMS.items():
    P[arm] = {}
    for f in range(5):
        z = np.load(W / f"oof_L1_ModernBERT-large_len1024_ep4_lr2e-05_seed{SEEDN}_f{f}of5{tag}.npz",
                    allow_pickle=True)
        for k, i in enumerate(z["ids"].tolist()):
            P[arm][i] = (z["p1"][k], z["p2"][k], z["p3"][k])
            F[arm][f].append(i)

print(f"Pooled over {len(P['baseline'])} out-of-fold instances (train+dev, 5 channel-grouped folds)\n")
print(f"{'arm':<12}{'ST1':>8}{'ST2':>8}{'ST3':>8}{'ST3-fam':>9}{'mean':>8}")
S = {}
for arm in ARMS:
    ids = list(P[arm])
    S[arm] = metrics.score([by_id[i] for i in ids], decide(ids, P[arm]))
    s = S[arm]
    print(f"{arm:<12}{s['st1']:>8.3f}{s['st2']:>8.3f}{s['st3']:>8.3f}"
          f"{s['st3_family']:>9.3f}{s['mean']:>8.3f}")
d = {k: S["span"][k] - S["baseline"][k] for k in ("st1", "st2", "st3", "st3_family", "mean")}
print(f"{'delta':<12}{d['st1']:>+8.3f}{d['st2']:>+8.3f}{d['st3']:>+8.3f}"
      f"{d['st3_family']:>+9.3f}{d['mean']:>+8.3f}")

print("\nPaired per fold — the consistency signal that predicted transfer in this project:")
print(f"{'metric':<12}" + "".join(f"{'f'+str(k):>9}" for k in range(5)) + f"{'mean':>9}{'sd':>8}{'won':>6}")
for key in ("st3", "st3_family", "st1", "st2", "mean"):
    ds = []
    for f in range(5):
        ids = F["baseline"][f]
        a = metrics.score([by_id[i] for i in ids], decide(ids, P["span"]))[key]
        b = metrics.score([by_id[i] for i in ids], decide(ids, P["baseline"]))[key]
        ds.append(a - b)
    print(f"{key:<12}" + "".join(f"{x:>+9.3f}" for x in ds) +
          f"{np.mean(ds):>+9.3f}{np.std(ds, ddof=1):>8.3f}{sum(x > 0 for x in ds):>4}/5")

print("\nPer-flag ST3 F1, pooled — where the auxiliary supervision actually lands:")
ids = list(P["baseline"])
gold = [by_id[i]["labels"]["st3"] for i in ids]
pf = {}
for arm in ARMS:
    pr = decide(ids, P[arm])
    pf[arm] = metrics.per_label_f1(gold, [pr[i]["st3"] for i in ids], ST3)
print(f"{'flag':<40}{'support':>8}{'baseline':>10}{'span':>8}{'delta':>9}")
for k in ST3:
    n = sum(1 for g in gold if k in g)
    print(f"{k:<40}{n:>8}{pf['baseline'][k]:>10.3f}{pf['span'][k]:>8.3f}"
          f"{pf['span'][k] - pf['baseline'][k]:>+9.3f}")
