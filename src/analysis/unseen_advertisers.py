"""Does the system generalise to advertisers it never saw, or memorise brands?

Dev is split by whether an instance's outbound domain occurs anywhere in train
(the splits are channel-disjoint but not advertiser-disjoint). ST1/ST2 come from
the six-seed ENC-L4 at 0.5; the TF-IDF blend is left out because its test-time
fit includes dev. ST3 is scored for each arm, under two rule sets: the Aug-14
rules (paid-promotion flag, short transcript, no_flag at 0.6) and the shipped
constraint layer (adds inadequate_disclosure at 0.40 and the disclosure
tie-break). The last row of the shipped block is the submitted ST3 system.

The two halves differ in label mix, so the differences are observations, not a
controlled comparison; the script prints the mix alongside.

Usage:
    python src/analysis/unseen_advertisers.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g, domain
from rules import is_short
import metrics

W = ROOT / "work"
tr, dv = load_split("train"), load_split("dev")
idx = {x["instanceID"]: i for i, x in enumerate(dv)}


def avg(pats):
    p = [np.zeros((len(dv), k)) for k in (5, 12, 8)]
    n = 0
    for f in pats:
        if f.exists():
            z = np.load(f, allow_pickle=True)
            ix = [idx[i] for i in z["ids"].tolist()]
            for k, key in enumerate(("p1", "p2", "p3")):
                p[k][ix] += z[key]
            n += 1
    assert n, pats
    return [q / n for q in p]


a1, a2, _ = avg([W / f"probs_dev_L4_ModernBERT-large_len2048_seed{s}.npz" for s in range(6)])
enc4 = avg([W / f"probs_dev_L1_ModernBERT-large_len1024_seed{s}.npz" for s in range(6)])[2]
enc6 = avg([W / f"probs_dev_L1_ModernBERT-large_len1024_seed{s}_ep6.npz" for s in range(6)])[2]
qlora = avg([W / f"probs_dev_L2_Qwen2.5-7B-Instruct_len1024_seed{s}_qlora.npz" for s in range(3)])[2]
# Same combination as build_st3_hybrid.py: disclosure flags averaged, the rest from the encoder.
hyb = enc6.copy()
for f in ("undisclosed_advertising", "inadequate_disclosure"):
    j = ST3.index(f)
    hyb[:, j] = (enc6[:, j] + qlora[:, j]) / 2
ARMS = {"ENC-L1, 4 epochs": enc4, "ENC-L1, 6 epochs": enc6, "QLoRA-ST3 alone": qlora,
        "hybrid (submitted)": hyb}


def decide(b3, shipped):
    pr = {}
    for i, x in enumerate(dv):
        s2 = [c for j, c in enumerate(ST2) if a2[i, j] >= .5] or [ST2[int(np.argmax(a2[i]))]]
        s3 = set(c for j, c in enumerate(ST3) if b3[i, j] >= .5)
        if shipped:
            if b3[i, ST3.index("inadequate_disclosure")] >= .40:
                s3.add("inadequate_disclosure")
            else:
                s3.discard("inadequate_disclosure")
        if _g(x, "video_context", "official_disclosure") == "true":
            s3 -= {"undisclosed_advertising"}
        if shipped and {"undisclosed_advertising", "inadequate_disclosure"} <= s3:
            s3 -= {"inadequate_disclosure"}
        if is_short(x):
            s3 = {"insufficient_context"}
        elif b3[i, ST3.index("no_flag")] >= 0.6:
            s3 = {"no_flag"}
        pr[x["instanceID"]] = {"st1": ST1[int(np.argmax(a1[i]))], "st2": s2,
                               "st3": clean_st3(sorted(s3))}
    return pr


seen = {domain(x) for x in tr}
A = [x for x in dv if domain(x) in seen]
B = [x for x in dv if domain(x) not in seen]
print(f"dev: advertiser seen in train n={len(A)}, unseen n={len(B)}\n")
for shipped in (False, True):
    print("rules:", "shipped constraint layer" if shipped else "Aug-14 rules")
    for name, b3 in ARMS.items():
        pr = decide(b3, shipped)
        sa, sb = metrics.score(A, pr), metrics.score(B, pr)
        print(f"  {name:20s} ST3 seen {sa['st3']:.3f} unseen {sb['st3']:.3f} ({sb['st3'] - sa['st3']:+.3f})"
              f"   ST1 {sb['st1'] - sa['st1']:+.3f}   ST2 {sb['st2'] - sa['st2']:+.3f}")
    print()
print(f"{'label mix':38s} {'seen':>7} {'unseen':>7}")
for f in ST3:
    print(f"{f:38s} {np.mean([f in x['labels']['st3'] for x in A]):7.1%} "
          f"{np.mean([f in x['labels']['st3'] for x in B]):7.1%}")
