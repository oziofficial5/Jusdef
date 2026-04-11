"""Smoke test for JusDef losses."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.train.losses import JusDefLoss

criterion = JusDefLoss(lambda1=0.1, lambda2=0.1)

# Fake data
scores = torch.randn(4, 100, requires_grad=True)
targets = torch.zeros(4, 100)
targets[0, [1, 5, 10]] = 1.0
targets[1, [2, 8]] = 1.0
seen_mask = torch.ones(100, dtype=torch.bool)
seen_mask[80:] = False

# Stage 1: cls only
loss1 = criterion(scores, targets, seen_mask, stage=1)
print(f"Stage 1 loss: {loss1.item():.4f}")
assert loss1.item() > 0, "FAILED: Stage 1 loss should be positive"
assert not torch.isnan(loss1), "FAILED: NaN in stage 1"

# Stage 2: + onto
conc_embs = torch.randn(10, 512)
label_adj = torch.eye(100) * 0.1
loss2 = criterion(scores, targets, seen_mask,
                  conc_embs=conc_embs, label_adj=label_adj, stage=2)
print(f"Stage 2 loss: {loss2.item():.4f}")
assert loss2.item() > 0, "FAILED: Stage 2 loss should be positive"
assert not torch.isnan(loss2), "FAILED: NaN in stage 2"

# Stage 3: + defeat
active = torch.randn(5, 512)
defeated = torch.randn(3, 512)
loss3 = criterion(scores, targets, seen_mask,
                  conc_embs=conc_embs, label_adj=label_adj,
                  active_embs=active, defeated_embs=defeated, stage=3)
print(f"Stage 3 loss: {loss3.item():.4f}")
assert loss3.item() > 0, "FAILED: Stage 3 loss should be positive"
assert not torch.isnan(loss3), "FAILED: NaN in stage 3"

# Backward test
loss3.backward()
print("Backward: OK")

# Edge case: no defeated messages
loss_no_defeat = criterion(scores, targets, seen_mask,
                           conc_embs=conc_embs, label_adj=label_adj,
                           active_embs=active,
                           defeated_embs=torch.zeros(0, 512), stage=3)
print(f"No defeated msgs loss: {loss_no_defeat.item():.4f}")

print("\nAll loss smoke tests PASSED")