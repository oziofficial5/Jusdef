import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.kg.kg_builder import build_document_graph
from src.model.baselines import RGCN

# 1. Build a fake document graph
doc_struct = {
    "doc_id": 0,
    "labels": [1, 5, 10],
    "sections": [
        {
            "type": "PROVISIONS",
            "type_int": 2,
            "text": "test section",
            "concepts": [
                {"label_idx": 1, "operator": "AFF", "span_start": 0, "span_end": 4, "phrase": "test", "label_text": "x", "similarity": 0.8},
                {"label_idx": 5, "operator": "NEG", "span_start": 5, "span_end": 9, "phrase": "test2", "label_text": "y", "similarity": 0.9},
            ],
            "authorities": [
                {"text": "Regulation (EU) No 123/2020", "type": "REGULATION", "level": 3.0, "start": 0, "end": 10},
            ],
        },
        {
            "type": "PREAMBLE",
            "type_int": 0,
            "text": "another section",
            "concepts": [
                {"label_idx": 10, "operator": "OVR", "span_start": 0, "span_end": 5, "phrase": "test3", "label_text": "z", "similarity": 0.85},
            ],
            "authorities": [],
        },
    ],
}

doc_emb = torch.randn(768)
sec_embs = torch.randn(2, 768)
label_embs = torch.randn(100, 768)

# Test kg_builder
print("=== KG Builder Test ===")
g = build_document_graph(doc_struct, doc_emb, sec_embs, label_embs)

print(f"Node types: {g.node_types}")
for nt in g.node_types:
    print(f"  {nt}: {g[nt].x.shape}")

print(f"Edge types: {g.edge_types}")
for et in g.edge_types:
    print(f"  {et}: {g[et].edge_index.shape[1]} edges")

r2 = g["sec", "mentions", "conc"]
print(f"  r2 operators: {r2.operator.tolist()}")
print(f"  r2 priorities: {r2.priority.tolist()}")
print(f"Target y shape: {g.y.shape}, active: {g.y.sum().int().item()}")

# Test RGCN forward pass
print("\n=== RGCN Forward Test ===")
model = RGCN(in_dim=768, hidden_dim=512, out_dim=512)
x_dict = {nt: g[nt].x for nt in g.node_types}
ei_dict = {et: g[et].edge_index for et in g.edge_types if g[et].edge_index.size(1) > 0}
h = model(x_dict, ei_dict)
for nt in h:
    print(f"  {nt}: {h[nt].shape}")

# Test scoring
doc_emb_out = h["sec"].mean(dim=0, keepdim=True)
scores = model.score(doc_emb_out, h["label"]).squeeze(0)
print(f"Scores shape: {scores.shape}")
print(f"Score range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")

# Test backward
scores.sum().backward()
print("Backward pass: OK")

print("\nAll smoke tests PASSED")