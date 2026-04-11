"""
Authority priority scorer for JusDef.

Computes π(m) = w_a^T [e_type(a), ℓ(a), t(a)]

This encodes:
  - lex superior: higher-level authority wins (Treaty > Regulation > Directive)
  - lex posterior: more recent authority wins

All weights are learned. Lex specialis (subject-matter overlap) is not
captured by this linear model and is left to future work.

Reference: JusDef paper Section 4.3, Equation 1
"""
import torch
import torch.nn as nn


class AuthorityScorer(nn.Module):
    """
    Linear authority priority scorer.

    Input per message: [type_embedding(8), level(1), recency(1)] = 10 dims
    Output: scalar priority score π(m)

    Authority types (from kg_builder.py):
      0 = REGULATION
      1 = DIRECTIVE
      2 = DECISION
      3 = ARTICLE
      4 = CASE
      5 = UNKNOWN (padding)
    """
    NUM_AUTH_TYPES = 6

    def __init__(self, type_emb_dim=8):
        super().__init__()
        self.type_embedding = nn.Embedding(self.NUM_AUTH_TYPES, type_emb_dim)
        # Input: [type_emb(8) | level(1) | recency(1)] = 10 dims
        self.scorer = nn.Linear(type_emb_dim + 2, 1)

    def forward(self, auth_type, level, recency):
        """
        Args:
            auth_type: tensor (E,) int — authority type index per edge
            level: tensor (E,) float [0, 3] — hierarchical level
            recency: tensor (E,) float [0, 1] — temporal recency

        Returns:
            priorities: tensor (E,) float — learned priority scores
        """
        # Clamp auth_type to valid range
        auth_type = auth_type.clamp(0, self.NUM_AUTH_TYPES - 1)

        type_emb = self.type_embedding(auth_type)  # (E, 8)
        features = torch.cat([
            type_emb,
            level.unsqueeze(-1),
            recency.unsqueeze(-1),
        ], dim=-1)  # (E, 10)

        return self.scorer(features).squeeze(-1)  # (E,)