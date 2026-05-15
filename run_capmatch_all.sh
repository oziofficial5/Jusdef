#!/usr/bin/env bash

set -e

for SEED in 42 43 44; do
  echo "=== Starting capmatch seed ${SEED} ==="
  python scripts/train_jusdef_capmatch.py \
    --seed "${SEED}" \
    --epochs 200 \
    --capmatch \
    --tag full
  echo "=== Finished seed ${SEED} ==="
done
