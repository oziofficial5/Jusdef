#!/bin/bash
set -e
cd ~/Jusdef
source .venv/bin/activate
echo "Started: $(date)"
df -h .


echo "=== M4: R-GCN ==="
for SEED in 42 43 44; do
    python scripts/train_rgcn.py --seed $SEED
done

echo "=== M6a: JusDef full ==="
for SEED in 42 43 44; do
    python scripts/train_jusdef.py --seed $SEED --tag full
done

echo "=== M6b: Ablations ==="
python scripts/train_jusdef.py --seed 42 --tag no_dmp --no_dmp
python scripts/train_jusdef.py --seed 42 --tag no_auth --no_authority
python scripts/train_jusdef.py --seed 42 --tag no_dmp_no_auth --no_dmp --no_authority

echo "=== DONE: $(date) ==="
ls -la outputs/logs/*.json
