"""
Baseline models for JusDef.

Baseline 1: LegalBERT cosine similarity (zero-shot, no training)
Baseline 2: R-GCN on heterogeneous legal KG (standard aggregation, no defeat)

Edge type names match kg_builder.py exactly:
  ("doc", "has_section", "sec")
  ("sec", "mentions", "conc")
  ("sec", "cites", "auth")
  ("label", "maps_to", "conc")
  ("label", "parent_of", "label")
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, Linear


# ═══════════════════════════════════════════════════════════
# Baseline 1: LegalBERT Cosine Similarity (zero-shot)
# ═══════════════════════════════════════════════════════════

def legalbert_cosine_baseline(doc_embs, label_embs):
    """
    Zero-shot classification by cosine similarity.
    Args:
        doc_embs: tensor [N_docs, d]
        label_embs: tensor [N_labels, d]
    Returns:
        scores: numpy array [N_docs, N_labels]
    """
    doc_embs = doc_embs.float()
    label_embs = label_embs.float()
    doc_norm = F.normalize(doc_embs, p=2, dim=1)
    label_norm = F.normalize(label_embs, p=2, dim=1)
    scores = doc_norm @ label_norm.T
    return scores.cpu().numpy()


def tune_threshold(scores, targets, thresholds=None):
    """
    Find the threshold that maximizes Macro-F1.
    """
    if thresholds is None:
        thresholds = np.arange(-0.2, 0.9, 0.02)
    best_f1 = -1.0
    best_thresh = 0.5
    for t in thresholds:
        preds = (scores >= t).astype(int)
        macro = f1_score(targets, preds, average="macro", zero_division=0)
        if macro > best_f1:
            best_f1 = macro
            best_thresh = t
    return float(best_thresh), float(best_f1)


# ═══════════════════════════════════════════════════════════
# Baseline 2: R-GCN on Heterogeneous Legal KG
# ═══════════════════════════════════════════════════════════

class RGCN(nn.Module):
    """
    R-GCN baseline on the JusDef heterogeneous legal KG.

    Uses the SAME graph as JusDef but with standard message passing.
    No defeat, no operator conditioning, no authority scoring.

    Edge type names match kg_builder.py exactly.
    """

    def __init__(self, in_dim=768, hidden_dim=512, out_dim=512,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projections (one per node type, 768 -> hidden_dim)
        node_types = ["doc", "sec", "conc", "auth", "label"]
        self.input_proj = nn.ModuleDict({
            nt: Linear(in_dim, hidden_dim) for nt in node_types
        })

        # Message passing layers — edge names match kg_builder.py
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ("doc", "has_section", "sec"): SAGEConv(hidden_dim, hidden_dim),
                ("sec", "mentions", "conc"): SAGEConv(hidden_dim, hidden_dim),
                ("sec", "cites", "auth"): SAGEConv(hidden_dim, hidden_dim),
                ("label", "maps_to", "conc"): SAGEConv(hidden_dim, hidden_dim),
                ("label", "parent_of", "label"): SAGEConv(hidden_dim, hidden_dim),
            }, aggr="sum")
            self.convs.append(conv)

        # Output projection
        self.out_proj = Linear(hidden_dim, out_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the heterogeneous GNN.

        Args:
            x_dict: dict of {node_type: tensor [N_nodes, in_dim]}
            edge_index_dict: dict of {edge_type: tensor [2, N_edges]}

        Returns:
            h: dict of {node_type: tensor [N_nodes, out_dim]}
        """
        # Input projection
        h = {}
        for nt in x_dict:
            if nt in self.input_proj:
                h[nt] = F.relu(self.input_proj[nt](x_dict[nt]))
            else:
                h[nt] = x_dict[nt]

        # Message passing
        for conv in self.convs:
            valid_edges = {}
            for k, v in edge_index_dict.items():
                if k in conv.convs and v.size(1) > 0:
                    valid_edges[k] = v

            if valid_edges:
                new_h = conv(h, valid_edges)
                for k in new_h:
                    h[k] = self.dropout(F.relu(new_h[k]))

        # Output projection
        h = {k: self.out_proj(v) for k, v in h.items()}

        return h

    def score(self, doc_embs, label_embs):
        """
        Cosine similarity scoring between doc and label embeddings.

        Args:
            doc_embs: tensor [N_docs, out_dim]
            label_embs: tensor [N_labels, out_dim]
        Returns:
            scores: tensor [N_docs, N_labels]
        """
        doc_norm = F.normalize(doc_embs, dim=-1)
        label_norm = F.normalize(label_embs, dim=-1)
        return doc_norm @ label_norm.T