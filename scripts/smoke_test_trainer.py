"""Smoke test for JusDef trainer."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import traceback
from src.kg.kg_builder import build_document_graph
from src.model.jusdef import JusDef
from src.train.trainer import train_jusdef, forward_one_graph

def make_fake_graph(doc_id, labels, ops):
    doc_struct = {
        "doc_id": doc_id,
        "labels": labels,
        "sections": [
            {
                "type": "PROVISIONS", "type_int": 2, "text": "test",
                "concepts": [
                    {"label_idx": labels[0], "operator": ops[0],
                     "span_start": 0, "span_end": 4, "phrase": "a",
                     "label_text": "x", "similarity": 0.8},
                    {"label_idx": labels[1] if len(labels) > 1 else labels[0],
                     "operator": ops[1] if len(ops) > 1 else "AFF",
                     "span_start": 5, "span_end": 9, "phrase": "b",
                     "label_text": "y", "similarity": 0.9},
                ],
                "authorities": [
                    {"text": "Reg 123/2020", "type": "REGULATION",
                     "level": 3.0, "start": 0, "end": 10},
                ],
            },
        ],
    }
    return build_document_graph(
        doc_struct, torch.randn(768), torch.randn(1, 768), torch.randn(100, 768)
    )

print("=== Building fake graphs ===")
g1 = make_fake_graph(0, [1, 5], ["AFF", "NEG"])
g2 = make_fake_graph(1, [10, 20], ["EXC", "OVR"])
print(f"  g1: {g1.node_types}")
print(f"  g2: {g2.node_types}")

print("\n=== Step 1: forward_one_graph ===")
try:
    model = JusDef(in_dim=768, hidden_dim=64, num_layers=1,
                   dropout=0.1, temperature=5.0,
                   use_dmp=True, use_authority=True)
    scores, defeat_info, conc_embs = forward_one_graph(model, g1, "cpu")
    print(f"  Scores: {scores.shape}")
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 2: train_jusdef (3 epochs) ===")
config = {
    "seed": 42,
    "epochs": 3,
    "lr": 1e-3,
    "hidden_dim": 64,
    "num_layers": 1,
    "dropout": 0.1,
    "temperature": 5.0,
    "lambda1": 0.1,
    "lambda2": 0.1,
    "patience": 10,
    "stage1_end": 1,
    "stage2_end": 2,
    "use_dmp": True,
    "use_authority": True,
    "train_graphs": [g1, g2],
    "val_graphs": [g1],
    "test_graphs": [g2],
    "seen_labels": list(range(80)),
    "unseen_labels": list(range(80, 100)),
    "label_adj": torch.eye(100),
    "checkpoint_path": "outputs/checkpoints/smoke_jusdef.pt",
}

try:
    results = train_jusdef(config)
    print(f"\n  Results: {results}")
    if results is None:
        print("  FAILED: returned None")
        sys.exit(1)
    print("  PASSED")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== All trainer smoke tests PASSED ===")