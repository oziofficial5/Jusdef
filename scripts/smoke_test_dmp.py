"""
Smoke test for DMP layer.

Tests:
  1. Proposition 2: all AFF + equal priority → no defeats → same as standard attention
  2. OVR defeats NEG
  3. Higher priority wins at same operator level
  4. Forward + backward pass without NaN
  5. Empty input handling
  6. Active/defeated embedding separation for L_defeat
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.dmp_layer import DMPLayer, compute_defeat_mask

print("=== Test 1: Proposition 2 (all AFF, equal priority → no defeats) ===")
operators = torch.zeros(5, dtype=torch.long)   # all AFF
priorities = torch.ones(5) * 0.5               # equal
dst_nodes = torch.zeros(5, dtype=torch.long)   # all target same node

mask = compute_defeat_mask(operators, priorities, dst_nodes)
print(f"  Mask: {mask.tolist()}")
assert (mask == 1.0).all(), "FAILED: No defeats should occur with all AFF + equal priority"
print("  PASSED")

print("\n=== Test 2: OVR defeats NEG ===")
operators2 = torch.tensor([1, 3])  # NEG=1, OVR=3
priorities2 = torch.tensor([0.5, 0.5])
dst2 = torch.zeros(2, dtype=torch.long)

mask2 = compute_defeat_mask(operators2, priorities2, dst2)
print(f"  Mask: {mask2.tolist()}")
assert mask2[0] == 0.0, "FAILED: NEG should be defeated by OVR"
assert mask2[1] == 1.0, "FAILED: OVR should remain active"
print("  PASSED")

print("\n=== Test 3: Higher priority wins at same operator ===")
operators3 = torch.tensor([1, 1])  # both NEG
priorities3 = torch.tensor([0.3, 0.8])  # second has higher priority
dst3 = torch.zeros(2, dtype=torch.long)

mask3 = compute_defeat_mask(operators3, priorities3, dst3)
print(f"  Mask: {mask3.tolist()}")
assert mask3[0] == 0.0, "FAILED: Lower priority should be defeated"
assert mask3[1] == 1.0, "FAILED: Higher priority should remain active"
print("  PASSED")

print("\n=== Test 4: Different dst nodes → no defeat across nodes ===")
operators4 = torch.tensor([1, 3])  # NEG, OVR
priorities4 = torch.tensor([0.5, 0.5])
dst4 = torch.tensor([0, 1])  # different target nodes

mask4 = compute_defeat_mask(operators4, priorities4, dst4)
print(f"  Mask: {mask4.tolist()}")
assert (mask4 == 1.0).all(), "FAILED: Different dst nodes should not defeat each other"
print("  PASSED")

print("\n=== Test 5: DMPLayer forward + backward ===")
layer = DMPLayer(in_dim=512, out_dim=512, temperature=5.0)

src_embs = torch.randn(8, 512)
dst_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
ops = torch.tensor([0, 1, 2, 0, 3, 0, 1, 2])
pris = torch.rand(8)

concept_ids = dst_ids  # grouping key for defeat

out = layer(src_embs, dst_ids, ops, pris, concept_ids, num_dst=3)
print(f"  Output shape: {out.shape}")
assert out.shape == (3, 512), f"FAILED: Expected (3, 512), got {out.shape}"
assert not torch.isnan(out).any(), "FAILED: NaN in output"

out.sum().backward()
print("  Forward + backward: OK")
print("  PASSED")

print("\n=== Test 6: Empty input ===")
empty_out = layer(
    torch.zeros(0, 512),
    torch.zeros(0, dtype=torch.long),
    torch.zeros(0, dtype=torch.long),
    torch.zeros(0),
    torch.zeros(0, dtype=torch.long),  # empty concept_ids
    num_dst=3,
)
print(f"  Empty output shape: {empty_out.shape}")
assert empty_out.shape == (3, 512), "FAILED: Empty input should produce zeros"
print("  PASSED")

print("\n=== Test 7: Active/defeated embedding separation ===")
concept_ids = dst_ids  # reuse the same grouping key
active, defeated = layer.get_active_defeated_embeddings(
    src_embs, dst_ids, ops, pris, concept_ids
)
print(f"  Active: {active.shape}, Defeated: {defeated.shape}")
total = active.shape[0] + defeated.shape[0]
assert total == 8, f"FAILED: Active + defeated should equal total edges ({total} != 8)"
print("  PASSED")

print("\n=== All DMP smoke tests PASSED ===")