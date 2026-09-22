#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../models"
for f in 0 1 2 3 4; do
  for arm in base span; do
    if [ "$arm" = base ]; then W=0.0; T=_base28s1; else W=0.5; T=_span05s1; fi
    out=../../work/cv/log_L1_${arm}_s1_f${f}.txt
    echo "=== fold $f arm $arm seed 1 $(date +%H:%M:%S) ==="
    python3 encoder.py --level 1 --maxlen 1024 --epochs 4 --seed 1 \
      --cv-folds 5 --cv-fold "$f" --span-weight "$W" --tag "$T" > "$out" 2>&1
    grep -E "^fold " "$out"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
