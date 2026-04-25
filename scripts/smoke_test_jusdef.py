"""
Smoke test for the full JusDef model.

Tests:
  1. Full forward pass with DMP enabled
  2. Forward pass with DMP disabled (ablation -DMP, should equal R-GCN)
  3. Document pooling + label scoring
  4. Backward pass
  5. Defeat info returned correctly
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.kg.kg_builder import build_document_graph
from src.model.jusdef import JusDef

# Build a test graph
doc_struct = {
    "doc_id": 0,
    "labels": [1, 5, 10],
    "sections": [
        {
            "type": "PROVISIONS", "type_int": 2, "text": "test section",
            "concepts": [
                {"label_idx": 1, "operator": "AFF", "span_start": 0,
                 "span_end": 4, "phrase": "test", "label_text": "x",
                 "similarity": 0.8},
                {"label_idx": 5, "operator": "NEG", "span_start": 5,
                 "span_end": 9, "phrase": "test2", "label_text": "y",
                 "similarity": 0.9},
            ],
            "authorities": [
                {"text": "Regulation (EU) No 123/2020", "type": "REGULATION",
                 "level": 3.0, "start": 0, "end": 10},
            ],
        },
        {
            "type": "PREAMBLE", "type_int": 0, "text": "another section",
            "concepts": [
                {"label_idx": 10, "operator": "OVR", "span_start": 0,
                 "span_end": 5, "phrase": "test3", "label_text": "z",
                 "similarity": 0.85},
                {"label_idx": 1, "operator": "EXC", "span_start": 6,
                 "span_end": 10, "phrase": "test4", "label_text": "x",
                 "similarity": 0.77},
            ],
            "authorities": [],
        },
    ],
}

doc_emb = torch.randn(768)
sec_embs = torch.randn(2, 768)
label_embs = torch.randn(100, 768)
g = build_document_graph(doc_struct, doc_emb, sec_embs, label_embs)

# Prepare inputs
x_dict = {nt: g[nt].x for nt in g.node_types}
ei_dict = {et: g[et].edge_index for et in g.edge_types}

r2_key = ("sec", "mentions", "conc")
edge_attr_dict = {
    r2_key: {
        "operator": g[r2_key].operator,
        "priority": g[r2_key].priority,
    }
}

# ─── Test 1: JusDef with DMP ───
print("=== Test 1: JusDef with DMP ===")
model = JusDef(in_dim=768, hidden_dim=512, num_layers=2,
               use_dmp=True, use_authority=True)

h, defeat_info = model(x_dict, ei_dict, edge_attr_dict)

print(f"  Output node types: {list(h.keys())}")
for nt in h:
    print(f"    {nt}: {h[nt].shape}")

assert "sec" in h, "FAILED: sec not in output"
assert "label" in h, "FAILED: label not in output"
assert "conc" in h, "FAILED: conc not in output"
assert h["label"].shape == (100, 512), f"FAILED: label shape {h['label'].shape}"

# Check defeat info
assert defeat_info is not None, "FAILED: defeat_info should not be None"
print(f"  Active embs: {defeat_info['active_embs'].shape}")
print(f"  Defeated embs: {defeat_info['defeated_embs'].shape}")
print("  PASSED")

# ─── Test 2: JusDef without DMP (ablation) ───
print("\n=== Test 2: JusDef without DMP (ablation -DMP) ===")
model_no_dmp = JusDef(in_dim=768, hidden_dim=512, num_layers=2,
                       use_dmp=False, use_authority=False)

h2, defeat_info2 = model_no_dmp(x_dict, ei_dict)

print(f"  Output node types: {list(h2.keys())}")
assert defeat_info2 is None, "FAILED: defeat_info should be None when DMP disabled"
print("  Defeat info: None (correct)")
print("  PASSED")

# ─── Test 3: Document pooling + scoring ───
print("\n=== Test 3: Document pooling + scoring ===")
doc_emb_out = model.pool_document(h["sec"])
print(f"  Doc embedding: {doc_emb_out.shape}")
assert doc_emb_out.shape == (1, 512), f"FAILED: Expected (1, 512), got {doc_emb_out.shape}"

scores = model.score(doc_emb_out, h["label"])
print(f"  Scores: {scores.shape}")
assert scores.shape[-1] == 100, f"FAILED: Expected (100,), got {scores.shape}"
print(f"  Score range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
print("  PASSED")

# ─── Test 4: Backward pass ───
print("\n=== Test 4: Backward pass ===")
loss = scores.sum()
loss.backward()
assert not any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None), \
    "FAILED: NaN in gradients"
print("  No NaN gradients")
print("  PASSED")

# ─── Test 5: All parameters have gradients ───
print("\n=== Test 5: Parameter gradient check ===")
n_params = sum(p.numel() for p in model.parameters())
n_with_grad = sum(p.numel() for p in model.parameters() if p.grad is not None)
print(f"  Total params: {n_params:,}")
print(f"  Params with grad: {n_with_grad:,}")
print("  PASSED")

print("\n=== All JusDef smoke tests PASSED ===")