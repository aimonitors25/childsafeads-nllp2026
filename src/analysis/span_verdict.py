"""Verdict on st3_evidence span supervision, across both seeds.

Ten matched pairs: two seeds x five channel-grouped folds, baseline and treatment
differing only in --span-weight. Also reports seed-to-seed variance of the
*baseline alone*, which turns out to be the finding.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g
from rules import is_short
import metrics

W = Path(__file__).resolve().parents[2] / "work" / "cv"
pool = load_split("train") + load_split("dev")
by_id = {x["instanceID"]: x for x in pool}
TAG = {(0, "baseline"): "_base28", (0, "span"): "_span05",
       (1, "baseline"): "_base28s1", (1, "span"): "_span05s1"}


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


P, FOLD = {}, {}
for (seed, arm), tag in TAG.items():
    P[seed, arm], FOLD[seed, arm] = {}, {}
    for f in range(5):
        z = np.load(W / f"oof_L1_ModernBERT-large_len1024_ep4_lr2e-05_seed{seed}"
                        f"_f{f}of5{tag}.npz", allow_pickle=True)
        FOLD[seed, arm][f] = z["ids"].tolist()
        for k, i in enumerate(z["ids"].tolist()):
            P[seed, arm][i] = (z["p1"][k], z["p2"][k], z["p3"][k])

sc = {}
for key in P:
    ids = list(P[key])
    sc[key] = metrics.score([by_id[i] for i in ids], decide(ids, P[key]))

print("Pooled out-of-fold scores, 2,857 instances per cell\n")
print(f"{'':<22}{'ST1':>8}{'ST2':>8}{'ST3':>8}{'ST3-fam':>9}{'mean':>8}")
for seed in (0, 1):
    for arm in ("baseline", "span"):
        s = sc[seed, arm]
        print(f"seed {seed} {arm:<14}{s['st1']:>8.3f}{s['st2']:>8.3f}{s['st3']:>8.3f}"
              f"{s['st3_family']:>9.3f}{s['mean']:>8.3f}")
    d = {k: sc[seed, "span"][k] - sc[seed, "baseline"][k] for k in sc[seed, "span"]}
    print(f"{'  delta':<22}{d['st1']:>+8.3f}{d['st2']:>+8.3f}{d['st3']:>+8.3f}"
          f"{d['st3_family']:>+9.3f}{d['mean']:>+8.3f}\n")

print("All ten matched pairs (2 seeds x 5 folds), treatment minus baseline:")
print(f"{'metric':<12}" + "".join(f"{'s'+str(s)+'f'+str(f):>8}" for s in (0, 1) for f in range(5))
      + f"{'mean':>9}{'sd':>8}{'won':>7}")
for key in ("st3", "st3_family", "st1", "st2", "mean"):
    ds = []
    for seed in (0, 1):
        for f in range(5):
            ids = FOLD[seed, "baseline"][f]
            ds.append(metrics.score([by_id[i] for i in ids], decide(ids, P[seed, "span"]))[key] -
                      metrics.score([by_id[i] for i in ids], decide(ids, P[seed, "baseline"]))[key])
    print(f"{key:<12}" + "".join(f"{x:>+8.3f}" for x in ds) +
          f"{np.mean(ds):>+9.3f}{np.std(ds, ddof=1):>8.3f}{sum(x > 0 for x in ds):>5}/10")

print("\nThe control that matters: the SAME baseline config, two seeds.")
print(f"{'metric':<14}{'seed 0':>9}{'seed 1':>9}{'seed swing':>12}{'span effect (s0)':>18}")
for key in ("st1", "st2", "st3", "st3_family", "mean"):
    a, b = sc[0, "baseline"][key], sc[1, "baseline"][key]
    eff = sc[0, "span"][key] - sc[0, "baseline"][key]
    print(f"{key:<14}{a:>9.3f}{b:>9.3f}{b - a:>+12.3f}{eff:>+18.3f}")

print("\nPer-flag ST3 F1 — the seed-0 headline against its own replication:")
ids = list(P[0, "baseline"])
gold = [by_id[i]["labels"]["st3"] for i in ids]
pf = {k: metrics.per_label_f1(gold, [decide(ids, P[k])[i]["st3"] for i in ids], ST3) for k in P}
print(f"{'flag':<38}{'s0 base':>9}{'s0 span':>9}{'s0 Δ':>8}"
      f"{'s1 base':>9}{'s1 span':>9}{'s1 Δ':>8}")
for k in ST3:
    print(f"{k:<38}{pf[0,'baseline'][k]:>9.3f}{pf[0,'span'][k]:>9.3f}"
          f"{pf[0,'span'][k] - pf[0,'baseline'][k]:>+8.3f}"
          f"{pf[1,'baseline'][k]:>9.3f}{pf[1,'span'][k]:>9.3f}"
          f"{pf[1,'span'][k] - pf[1,'baseline'][k]:>+8.3f}")
