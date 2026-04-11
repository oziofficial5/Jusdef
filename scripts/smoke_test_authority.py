"""Smoke test for AuthorityScorer."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.authority_scorer import AuthorityScorer

scorer = AuthorityScorer(type_emb_dim=8)

# 5 messages with different authority types
auth_types = torch.tensor([0, 1, 2, 3, 4])  # REG, DIR, DEC, ART, CASE
levels = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0])
recency = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])

priorities = scorer(auth_types, levels, recency)
print(f"Priorities: {priorities.tolist()}")
print(f"Shape: {priorities.shape}")  # (5,)

# Gradient test
priorities.sum().backward()
print("Backward: OK")

# Edge case: empty input
empty_priorities = scorer(
    torch.zeros(0, dtype=torch.long),
    torch.zeros(0),
    torch.zeros(0),
)
print(f"Empty input shape: {empty_priorities.shape}")  # (0,)

print("\nAuthority scorer smoke test PASSED")