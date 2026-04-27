"""Detailed eval of LegalBERT Cosine baseline."""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from sklearn.metrics import f1_score
from src.model.baselines import tune_threshold

print('Loading...', flush=True)

exc_data = json.load(open('data/annotations/exception_labels.json'))
exc_idx = exc_data['exception_override_labels']
seen = list(range(80))
unseen = list(range(80, 100))

# Load test embeddings + label embeddings
test_embs = torch.load('data/processed/embeddings/test_doc_embs.pt', map_location='cpu')
label_embs = torch.load('data/processed/embeddings/label_embs.pt', map_location='cpu')
print(f'Test docs: {test_embs.shape}, Labels: {label_embs.shape}', flush=True)

# Load test targets
test_graphs = torch.load('data/processed/graphs/test_graphs.pt', map_location='cpu')
targets = torch.stack([g.y for g in test_graphs]).numpy()
print(f'Targets: {targets.shape}', flush=True)

# Cosine similarity
doc_norm = F.normalize(test_embs, dim=-1)
label_norm = F.normalize(label_embs, dim=-1)
scores = (doc_norm @ label_norm.T).numpy()

best_t, _ = tune_threshold(scores, targets)
preds = (scores >= best_t).astype(int)

print(f'\n=== LegalBERT Cosine ===')
print(f'  macro:  {f1_score(targets, preds, average="macro", zero_division=0):.4f}')
print(f'  micro:  {f1_score(targets, preds, average="micro", zero_division=0):.4f}')
print(f'  seen:   {f1_score(targets[:,seen], preds[:,seen], average="macro", zero_division=0):.4f}')
print(f'  unseen: {f1_score(targets[:,unseen], preds[:,unseen], average="macro", zero_division=0):.4f}')
print(f'  exc:    {f1_score(targets[:,exc_idx], preds[:,exc_idx], average="macro", zero_division=0):.4f}')
print(f'  threshold: {best_t:.2f}')
