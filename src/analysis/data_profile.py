"""Full descriptive profile of the ChildSafeAds release.

Everything reported in `documents/data_analysis.md` is computed here.
Run: python src/analysis/data_profile.py > work/data_profile.txt
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.data import (ST1, ST2, ST3, ST3_FAMILY, DEV_DIR, TEST_DIR, load, domain)

SPLITS = {"train": DEV_DIR / "train.jsonl", "dev": DEV_DIR / "dev.jsonl",
          "test": TEST_DIR / "test.jsonl"}
D = {k: load(v) for k, v in SPLITS.items()}
OUT = {}


def g(inst, group, field, default=""):
    return ((inst.get(group) or {}).get(field) or default)


def words(s):
    return len(s.split())


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def head(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------- 1 integrity
def json_ok(line):
    try:
        json.loads(line)
        return True
    except Exception:
        return False


head("1. FILE INTEGRITY")
for name, path in SPLITS.items():
    raw = path.read_text(encoding="utf-8")
    lines = [l for l in raw.splitlines() if l.strip()]
    naive = sum(1 for l in lines if json_ok(l))
    print(f"{name:6s} bytes={len(raw):>9,} physical_lines={len(lines):>4} "
          f"json.loads_ok={naive:>4} raw_decode_objects={len(D[name]):>4} "
          f"lost_by_naive_parse={len(D[name]) - naive}")
    ids = [i.get("instanceID") for i in D[name]]
    print(f"       unique instanceIDs={len(set(ids))}  duplicates={len(ids) - len(set(ids))}")
    keys = Counter(k for i in D[name] for k in i)
    print(f"       top-level keys: {dict(keys)}")


# ------------------------------------------------------------ 2 field coverage
head("2. FIELD COVERAGE BY ACCESS LEVEL (% of split with non-empty field)")
FIELDS = [("L1", "transcript", "text"), ("L2", "video_context", "title"),
          ("L2", "video_context", "description"), ("L2", "video_context", "official_disclosure"),
          ("L3", "channel_context", "channel_name"), ("L4", "product_page", "raw_url"),
          ("L4", "product_page", "resolved_url"), ("L4", "product_page", "page_title"),
          ("L4", "product_page", "text")]
print(f"{'level':5s} {'field':38s} " + " ".join(f"{s:>16s}" for s in D))
for lvl, grp, fld in FIELDS:
    row = []
    for s, insts in D.items():
        n = sum(1 for i in insts if g(i, grp, fld).strip())
        row.append(f"{n:>6}/{len(insts):<4} {100*n/len(insts):5.1f}%")
    print(f"{lvl:5s} {grp + '.' + fld:38s} " + " ".join(f"{r:>16s}" for r in row))

print("\nofficial_disclosure value distribution:")
for s, insts in D.items():
    c = Counter(g(i, "video_context", "official_disclosure") or "<empty/unknown>" for i in insts)
    print(f"  {s:6s} " + "  ".join(f"{k}={v} ({100*v/len(insts):.1f}%)" for k, v in c.most_common()))

print("\nProduct-page attrition (L4 is the expensive level — how often does it pay off?):")
for s, insts in D.items():
    has_url = [i for i in insts if g(i, "product_page", "raw_url").strip()]
    resolved = [i for i in has_url if g(i, "product_page", "resolved_url").strip()]
    texty = [i for i in has_url if words(g(i, "product_page", "text")) >= 20]
    print(f"  {s:6s} url {len(has_url):>4}/{len(insts)}  resolved {len(resolved):>4}  "
          f"page_text>=20w {len(texty):>4} ({100*len(texty)/len(insts):.1f}% of split)")

# --------------------------------------------------------------- 3 text length
head("3. TEXT LENGTH (words) AND SEGMENT DURATION (seconds)")
print(f"{'split':6s} {'field':22s} {'n>0':>5} {'mean':>7} {'p10':>6} {'p50':>6} {'p90':>6} {'p99':>7} {'max':>7}")
for s, insts in D.items():
    for lbl, grp, fld in [("transcript", "transcript", "text"),
                          ("description", "video_context", "description"),
                          ("page_text", "product_page", "text")]:
        v = [words(g(i, grp, fld)) for i in insts]
        nz = [x for x in v if x > 0]
        print(f"{s:6s} {lbl:22s} {len(nz):>5} {sum(v)/len(v):>7.0f} {pct(v,10):>6} "
              f"{pct(v,50):>6} {pct(v,90):>6} {pct(v,99):>7} {max(v):>7}")
    dur = []
    for i in insts:
        try:
            dur.append(float(g(i, "transcript", "segment_end", "0")) -
                       float(g(i, "transcript", "segment_start", "0")))
        except ValueError:
            pass
    print(f"{s:6s} {'segment_seconds':22s} {len(dur):>5} {sum(dur)/len(dur):>7.0f} "
          f"{pct(dur,10):>6.0f} {pct(dur,50):>6.0f} {pct(dur,90):>6.0f} {pct(dur,99):>7.0f} {max(dur):>7.0f}")

print("\nShort/empty transcripts (the insufficient_context signature):")
for s, insts in D.items():
    e = sum(1 for i in insts if words(g(i, "transcript", "text")) == 0)
    u30 = sum(1 for i in insts if words(g(i, "transcript", "text")) < 30)
    print(f"  {s:6s} empty={e:>3} (<30 words)={u30:>3}")

# ------------------------------------------------------------ 4 label profile
LAB = {s: [i.get("labels") or {} for i in insts] for s, insts in D.items() if s != "test"}
head("4. LABEL DISTRIBUTIONS (test labels are withheld)")

print("ST1 — commercial type (single label)")
print(f"{'label':32s} " + " ".join(f"{s:>18s}" for s in LAB))
for k in ST1:
    row = []
    for s, ls in LAB.items():
        n = sum(1 for l in ls if l.get("st1") == k)
        row.append(f"{n:>5} ({100*n/len(ls):5.1f}%)")
    print(f"{k:32s} " + " ".join(f"{r:>18s}" for r in row))
for s, ls in LAB.items():
    print(f"  {s}: unseen/other st1 values = "
          f"{Counter(l.get('st1') for l in ls if l.get('st1') not in ST1)}")

for task, keys in (("ST2 — product category (multi-label)", ST2),
                   ("ST3 — compliance flags (multi-label)", ST3)):
    print(f"\n{task}")
    print(f"{'label':40s} " + " ".join(f"{s:>18s}" for s in LAB))
    tk = "st2" if keys is ST2 else "st3"
    for k in keys:
        row = []
        for s, ls in LAB.items():
            n = sum(1 for l in ls if k in (l.get(tk) or []))
            row.append(f"{n:>5} ({100*n/len(ls):5.1f}%)")
        print(f"{k:40s} " + " ".join(f"{r:>18s}" for r in row))
    for s, ls in LAB.items():
        card = [len(l.get(tk) or []) for l in ls]
        print(f"  {s}: labels/instance mean={sum(card)/len(card):.2f} "
              f"dist={dict(sorted(Counter(card).items()))}")

print("\nST3 family view (disclosure / content / product):")
for s, ls in LAB.items():
    c = Counter()
    for l in ls:
        for f in {ST3_FAMILY[x] for x in (l.get("st3") or []) if x in ST3_FAMILY}:
            c[f] += 1
    print(f"  {s:6s} " + "  ".join(f"{k}={v} ({100*v/len(ls):.1f}%)" for k, v in c.most_common()))

print("\nExclusivity check (no_flag / insufficient_context should stand alone):")
for s, ls in LAB.items():
    bad = sum(1 for l in ls if len(set(l.get("st3") or [])) > 1 and
              ({"no_flag", "insufficient_context"} & set(l.get("st3") or [])))
    empt = sum(1 for l in ls if not (l.get("st3") or []))
    print(f"  {s:6s} violations={bad}  empty_st3={empt}")

# ---------------------------------------------------- 5 macro-F1 leverage
head("5. METRIC LEVERAGE — what each class is worth on the dev leaderboard")
print("mean_macro_f1 = (macroF1(ST1) + macroF1(ST2) + macroF1(ST3)) / 3")
print(f"{'task':5s} {'classes':>7} {'weight of one class in the final score':>42s}")
for t, keys in (("ST1", ST1), ("ST2", ST2), ("ST3", ST3)):
    print(f"{t:5s} {len(keys):>7} {1/(3*len(keys)):>41.3%}")
print("\nDev support vs. score weight — the rarest classes carry the same weight as the largest:")
for t, keys, tk in (("ST1", ST1, "st1"), ("ST2", ST2, "st2"), ("ST3", ST3, "st3")):
    ls = LAB["dev"]
    sup = {k: (sum(1 for l in ls if l.get("st1") == k) if tk == "st1"
               else sum(1 for l in ls if k in (l.get(tk) or []))) for k in keys}
    thin = {k: v for k, v in sorted(sup.items(), key=lambda x: x[1]) if v <= 20}
    print(f"  {t}: dev support <=20 -> {thin}  (each still worth {1/(3*len(keys)):.1%})")

# -------------------------------------------------------- 6 label structure
head("6. LABEL STRUCTURE — how the three sub-tasks constrain each other")
tr = LAB["train"]
print("ST2 -> ST1 determinism on train (is ST1 a function of the category?):")
for k in ST2:
    c = Counter(l.get("st1") for l in tr if k in (l.get("st2") or []))
    if sum(c.values()) >= 10:
        top, n = c.most_common(1)[0]
        print(f"  {k:22s} n={sum(c.values()):>4}  {top:28s} {n:>4}/{sum(c.values())} "
              f"({100*n/sum(c.values()):5.1f}%)")

print("\nST2 co-occurrence on train (pairs seen >= 15 times):")
pairs = Counter()
for l in tr:
    ks = sorted(set(l.get("st2") or []))
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            pairs[(ks[a], ks[b])] += 1
for (a, b), n in pairs.most_common():
    if n >= 15:
        print(f"  {a:20s} + {b:20s} {n:>4}")

print("\nST3 co-occurrence on train (pairs seen >= 10 times):")
pairs = Counter()
for l in tr:
    ks = sorted(set(l.get("st3") or []))
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            pairs[(ks[a], ks[b])] += 1
for (a, b), n in pairs.most_common():
    if n >= 10:
        print(f"  {a:38s} + {b:38s} {n:>4}")

# ------------------------------------------------- 7 disclosure constraint
head("7. THE DISCLOSURE CONSTRAINT (official_disclosure vs ST3 disclosure flags)")
for s, insts in D.items():
    if s == "test":
        continue
    print(f"\n{s}:")
    for val in ("true", "false", ""):
        sub = [i for i in insts if (g(i, "video_context", "official_disclosure") or "") == val]
        if not sub:
            continue
        und = sum(1 for i in sub if "undisclosed_advertising" in ((i["labels"] or {}).get("st3") or []))
        inad = sum(1 for i in sub if "inadequate_disclosure" in ((i["labels"] or {}).get("st3") or []))
        nof = sum(1 for i in sub if "no_flag" in ((i["labels"] or {}).get("st3") or []))
        print(f"  disclosure={val or '<unknown>':<10} n={len(sub):>5}  "
              f"undisclosed_advertising={und:>4} ({100*und/len(sub):5.1f}%)  "
              f"inadequate_disclosure={inad:>4} ({100*inad/len(sub):5.1f}%)  no_flag={nof:>4}")

print("\n#ad / sponsor keyword in the description vs undisclosed_advertising (train):")
KW = re.compile(r"#ad\b|#sponsor|sponsored|paid promotion|paid partnership|in partnership with",
                re.I)
for s in ("train", "dev"):
    sub = D[s]
    for hit in (True, False):
        q = [i for i in sub if bool(KW.search(g(i, "video_context", "description"))) is hit]
        if not q:
            continue
        und = sum(1 for i in q if "undisclosed_advertising" in ((i["labels"] or {}).get("st3") or []))
        print(f"  {s:6s} description_keyword={str(hit):5s} n={len(q):>5} "
              f"undisclosed={und:>4} ({100*und/len(q):5.1f}%)")

# --------------------------------------------------------- 8 domain leakage
head("8. BRAND / DOMAIN LEAKAGE ACROSS CHANNEL-DISJOINT SPLITS")
dom = {s: [domain(i) for i in insts] for s, insts in D.items()}
train_dom = {d for d in dom["train"] if d}
for s in ("dev", "test"):
    have = [d for d in dom[s] if d]
    seen = [d for d in have if d in train_dom]
    print(f"  {s:5s} instances_with_domain={len(have):>4}/{len(D[s])}  "
          f"domain_also_in_train={len(seen):>4} = {100*len(seen)/len(D[s]):.1f}% of split, "
          f"{100*len(seen)/max(1,len(have)):.1f}% of those with a domain")
    print(f"        unique domains={len(set(have))}  of which new to train={len(set(have) - train_dom)}")

print("\nTop 15 train domains:")
for d, n in Counter(d for d in dom["train"] if d).most_common(15):
    c = Counter(l.get("st1") for i, l in zip(D["train"], LAB["train"]) if domain(i) == d)
    top, tn = c.most_common(1)[0]
    print(f"  {d:34s} n={n:>4}  dominant st1={top:28s} {100*tn/n:5.1f}%")

print("\nDomain -> ST1 purity on train:")
by = defaultdict(Counter)
for i, l in zip(D["train"], LAB["train"]):
    if domain(i):
        by[domain(i)][l.get("st1")] += 1
for thresh in (1, 3, 5):
    sel = {d: c for d, c in by.items() if sum(c.values()) >= thresh}
    pure = sum(1 for c in sel.values() if len(c) == 1)
    cov = sum(sum(c.values()) for c in sel.values())
    acc = sum(c.most_common(1)[0][1] for c in sel.values()) / max(1, cov)
    print(f"  domains with >= {thresh} train instances: {len(sel):>4}  "
          f"single-ST1 {pure:>4} ({100*pure/len(sel):5.1f}%)  "
          f"majority-vote accuracy over their {cov} instances = {100*acc:.1f}%")

print("\nMemorisation ceiling: dev instances whose domain is in train, scored by train majority vote")
for tk, keys in (("st1", ST1), ("st2", ST2)):
    hit = tot = 0
    for i, l in zip(D["dev"], LAB["dev"]):
        d = domain(i)
        if d and d in by:
            tot += 1
            if tk == "st1":
                hit += (by[d].most_common(1)[0][0] == l.get("st1"))
    if tk == "st1":
        print(f"  ST1 exact match on the {tot} covered dev instances = {100*hit/max(1,tot):.1f}% "
              f"({100*hit/len(D['dev']):.1f}% of the whole dev split)")

# ----------------------------------------------------------- 9 channels
head("9. CHANNELS")
ch = {s: [g(i, "channel_context", "channelID") for i in insts] for s, insts in D.items()}
for s in D:
    c = Counter(x for x in ch[s] if x)
    sizes = sorted(c.values(), reverse=True)
    print(f"  {s:6s} channels={len(c):>4}  instances/channel mean={sum(sizes)/len(sizes):.1f} "
          f"median={pct(sizes,50)} max={sizes[0]}  top-10 channels cover "
          f"{100*sum(sizes[:10])/len(D[s]):.1f}% of the split")
print("  split disjointness: " + ", ".join(
    f"{a}&{b}={len(set(ch[a]) & set(ch[b]) - {''})}" for a, b in
    (("train", "dev"), ("train", "test"), ("dev", "test"))))

# ------------------------------------------------------- 10 st3_evidence
head("10. st3_evidence — the unused rationale supervision")
for s, ls in LAB.items():
    ev = [e for l in ls for e in (l.get("st3_evidence") or [])]
    withev = sum(1 for l in ls if l.get("st3_evidence"))
    print(f"\n  {s:6s} spans={len(ev)}  instances_with_evidence={withev}/{len(ls)} "
          f"({100*withev/len(ls):.1f}%)")
    print(f"         spans per flag: {dict(Counter(e.get('flag') for e in ev).most_common())}")
    qw = [words(e.get("quote", "")) for e in ev]
    if qw:
        print(f"         quote length words: mean={sum(qw)/len(qw):.1f} p50={pct(qw,50)} p90={pct(qw,90)}")
    loc = Counter()
    for i, l in zip(D[s], ls):
        for e in (l.get("st3_evidence") or []):
            q = (e.get("quote") or "").strip().lower()
            if not q:
                continue
            where = [n for n, t in (("transcript", g(i, "transcript", "text")),
                                    ("description", g(i, "video_context", "description")),
                                    ("page", g(i, "product_page", "text")))
                     if q in t.lower()]
            loc[where[0] if where else "NOT FOUND"] += 1
    print(f"         located in: {dict(loc.most_common())}")

# ------------------------------------------------------ 11 train/dev drift
head("11. TRAIN -> DEV DRIFT (is dev a fair proxy for test?)")
print(f"{'label':44s} {'train%':>8} {'dev%':>8} {'abs diff':>9}")
rows = []
for tk, keys in (("st1", ST1), ("st2", ST2), ("st3", ST3)):
    for k in keys:
        f = lambda ls: (sum(1 for l in ls if l.get("st1") == k) if tk == "st1"
                        else sum(1 for l in ls if k in (l.get(tk) or []))) / len(ls) * 100
        a, b = f(LAB["train"]), f(LAB["dev"])
        rows.append((f"{tk}:{k}", a, b, abs(a - b)))
for n, a, b, d in sorted(rows, key=lambda r: -r[3])[:12]:
    print(f"{n:44s} {a:>8.1f} {b:>8.1f} {d:>9.1f}")
print(f"  mean absolute train->dev shift over all {len(rows)} classes = "
      f"{sum(r[3] for r in rows)/len(rows):.2f} points")

print("\nCoverage drift on the fields a system depends on (dev vs test):")
for lvl, grp, fld in FIELDS:
    a = 100 * sum(1 for i in D["dev"] if g(i, grp, fld).strip()) / len(D["dev"])
    b = 100 * sum(1 for i in D["test"] if g(i, grp, fld).strip()) / len(D["test"])
    if abs(a - b) >= 1.0:
        print(f"  {grp + '.' + fld:38s} dev {a:5.1f}%  test {b:5.1f}%  diff {b-a:+.1f}")

# --------------------------------------------- 12 rare-class signatures
head("12. RARE-CLASS SIGNATURES (can the tail be reached by structure alone?)")
print("st1 == 'none' — what do those product pages look like? (train, n=%d)"
      % sum(1 for l in LAB["train"] if l.get("st1") == "none"))
for i, l in list(zip(D["train"], LAB["train"])):
    if l.get("st1") == "none":
        print(f"    {domain(i)[:32]:34s} page_words={words(g(i,'product_page','text')):>5} "
              f"title={g(i,'product_page','page_title')[:52]!r}")

print("\ninsufficient_context vs transcript length:")
for s, ls in LAB.items():
    ic = [(i, l) for i, l in zip(D[s], ls) if "insufficient_context" in (l.get("st3") or [])]
    if not ic:
        continue
    w = [words(g(i, "transcript", "text")) for i, _ in ic]
    u30 = sum(1 for x in w if x < 30)
    base = sum(1 for i in D[s] if words(g(i, "transcript", "text")) < 30)
    print(f"  {s:6s} n={len(ic):>3}  transcript words: {sorted(w)}")
    print(f"         {u30}/{len(ic)} have <30 words; but {base} instances in the split are <30 words "
          f"-> precision of the length rule = {100*u30/max(1,base):.1f}%")

print("\nhfss_food_marketing / age_restricted / gambling — domains carrying them (train):")
for flag, tk in (("hfss_food_marketing", "st3"), ("age_restricted_or_prohibited_product", "st3"),
                 ("gambling", "st2"), ("toys", "st2")):
    c = Counter(domain(i) for i, l in zip(D["train"], LAB["train"])
                if flag in ((l.get(tk) or []) if tk != "st1" else [l.get("st1")]))
    print(f"  {flag:38s} {len(c)} domains, top: "
          + ", ".join(f"{d}×{n}" for d, n in c.most_common(6)))

print("\ninadequate_disclosure vs where the disclosure sits (train):")
for lo, hi in ((0, 50), (50, 150), (150, 400), (400, 10**6)):
    sub = [(i, l) for i, l in zip(D["train"], LAB["train"])
           if lo <= words(g(i, "video_context", "description")) < hi]
    if not sub:
        continue
    n = sum(1 for _, l in sub if "inadequate_disclosure" in (l.get("st3") or []))
    print(f"  description {lo:>4}-{hi if hi < 10**6 else '+':>5} words  n={len(sub):>5}  "
          f"inadequate_disclosure={n:>4} ({100*n/len(sub):5.1f}%)")

# ------------------------------------------------ 13 near-duplicate content
head("13. REPEATED CONTENT")
for fld, grp in (("transcript", "transcript"), ("page_text", "product_page")):
    key = "text"
    allt = {s: [re.sub(r"\W+", " ", g(i, grp, key).lower()).strip()[:400] for i in insts]
            for s, insts in D.items()}
    tr = Counter(t for t in allt["train"] if len(t) > 80)
    dup_in_train = sum(v - 1 for v in tr.values() if v > 1)
    for s in ("dev", "test"):
        hit = sum(1 for t in allt[s] if len(t) > 80 and t in tr)
        print(f"  {fld:11s} {s:5s}: {hit}/{len(D[s])} ({100*hit/len(D[s]):.1f}%) share a first-400-char "
              f"prefix with a train instance   [{dup_in_train} repeats inside train]")

# -------------------------------------- 14 what each data level actually buys
head("14. WHAT EACH ACCESS LEVEL BUYS — TF-IDF + logistic regression, train -> dev")
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import f1_score
    from core.data import view

    ytr1 = [l.get("st1") for l in LAB["train"]]
    yde1 = [l.get("st1") for l in LAB["dev"]]
    def mat(ls, keys, tk):
        return np.array([[1 if k in (l.get(tk) or []) else 0 for k in keys] for l in ls])
    ytr2, yde2 = mat(LAB["train"], ST2, "st2"), mat(LAB["dev"], ST2, "st2")
    ytr3, yde3 = mat(LAB["train"], ST3, "st3"), mat(LAB["dev"], ST3, "st3")

    print(f"{'level':22s} {'ST1':>7} {'ST2':>7} {'ST3':>7} {'mean':>7}")
    names = {1: "L1 transcript", 2: "L2 +video meta", 3: "L3 +channel", 4: "L4 +product page"}
    for lvl in (1, 2, 3, 4):
        Xtr = [view(i, level=lvl) for i in D["train"]]
        Xde = [view(i, level=lvl) for i in D["dev"]]
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000)
        A, B = v.fit_transform(Xtr), v.transform(Xde)
        lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        m1 = lr().fit(A, ytr1)
        f1 = f1_score(yde1, m1.predict(B), average="macro", zero_division=0)
        f2 = f1_score(yde2, OneVsRestClassifier(lr()).fit(A, ytr2).predict(B),
                      average="macro", zero_division=0)
        f3 = f1_score(yde3, OneVsRestClassifier(lr()).fit(A, ytr3).predict(B),
                      average="macro", zero_division=0)
        print(f"{names[lvl]:22s} {f1:>7.3f} {f2:>7.3f} {f3:>7.3f} {(f1+f2+f3)/3:>7.3f}")

    print("\nMajority-class / trivial floor on dev:")
    maj = Counter(ytr1).most_common(1)[0][0]
    f1 = f1_score(yde1, [maj] * len(yde1), average="macro", zero_division=0)
    f2 = f1_score(yde2, np.tile((ytr2.mean(0) > .5).astype(int), (len(yde2), 1)),
                  average="macro", zero_division=0)
    f3 = f1_score(yde3, np.tile((ytr3.mean(0) > .5).astype(int), (len(yde3), 1)),
                  average="macro", zero_division=0)
    print(f"{'always-majority':22s} {f1:>7.3f} {f2:>7.3f} {f3:>7.3f} {(f1+f2+f3)/3:>7.3f}")

    print("\nPer-class dev F1 at L4 (where the macro average is actually lost):")
    Xtr = [view(i, level=4) for i in D["train"]]
    Xde = [view(i, level=4) for i in D["dev"]]
    v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000)
    A, B = v.fit_transform(Xtr), v.transform(Xde)
    lr = lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    p1 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(A, ytr1).predict(B)
    for k, s in zip(ST1, f1_score(yde1, p1, average=None, labels=ST1, zero_division=0)):
        print(f"  ST1 {k:42s} F1={s:.3f}  dev_support={sum(1 for y in yde1 if y == k)}")
    for keys, ytr, yde, t in ((ST2, ytr2, yde2, "ST2"), (ST3, ytr3, yde3, "ST3")):
        P = OneVsRestClassifier(lr()).fit(A, ytr).predict(B)
        for j, k in enumerate(keys):
            print(f"  {t} {k:42s} F1={f1_score(yde[:, j], P[:, j], zero_division=0):.3f}  "
                  f"dev_support={int(yde[:, j].sum())}")
except ImportError as e:
    print("  scikit-learn unavailable, skipping:", e)
