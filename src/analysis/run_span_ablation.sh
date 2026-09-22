#!/usr/bin/env bash
# Paired A/B for st3_evidence span supervision, on the 2,857-instance
# channel-grouped CV instrument. Baseline and treatment differ in exactly one
# flag and share the seed, the folds and the code path.
set -u
cd "$(dirname "$0")/../models"
for f in 0 1 2 3 4; do
  for arm in base span; do
    if [ "$arm" = base ]; then W=0.0; T=_base28; else W=0.5; T=_span05; fi
    out=../../work/cv/log_L1_${arm}_f${f}.txt
    echo "=== fold $f arm $arm (span-weight $W) $(date +%H:%M:%S) ==="
    python3 encoder.py --level 1 --maxlen 1024 --epochs 4 --seed 0 \
      --cv-folds 5 --cv-fold "$f" --span-weight "$W" --tag "$T" > "$out" 2>&1
    grep -E "^(span supervision|fold |epoch 4)" "$out"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
