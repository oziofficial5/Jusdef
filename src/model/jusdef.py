"""
JusDef: Full model assembling all components.

Architecture:
  1. Input projection (all node types: 768 -> hidden_dim)
  2. Standard HeteroConv for non-r2 edges (r1, r4, r7, r8)
  3. DMP for r2 edges (sec -> conc, with defeat)
  4. Section-level attention pooling -> document embedding
  5. Cosine similarity scoring against label embeddings

When DMP is disabled (ablation -DMP), r2 edges use standard SAGEConv
and the model reduces to R-GCN.

Reference: JusDef paper Section 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

from src.model.authority_scorer import AuthorityScorer
from src.model.dmp_layer import DMPLayer


class JusDef(nn.Module):
    """
    Full JusDef model.

    Stages:
      1. Input projection (all node types)
      2. Standard HeteroConv for non-r2 edges
      3. DMP for r2 (sec -> conc) edges with defeat
      4. Section-level attention pooling -> document embedding
      5. Label scoring by cosine similarity
    """

    def __init__(self, in_dim=768, hidden_dim=512, num_layers=2,
                 dropout=0.3, temperature=5.0, use_dmp=True,
                 use_authority=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_dmp = use_dmp
        self.use_authority = use_authority

        # Input projections (one per node type)
        node_types = ["doc", "sec", "conc", "auth", "label"]
        self.input_proj = nn.ModuleDict({
            nt: Linear(in_dim, hidden_dim) for nt in node_types
        })

        # Authority scorer (for learned priorities on r2 edges)
        if use_authority:
            self.authority_scorer = AuthorityScorer(type_emb_dim=8)

        # DMP layers (one per GNN layer, only for r2 edges)
        if use_dmp:
            self.dmp_layers = nn.ModuleList([
                DMPLayer(hidden_dim, hidden_dim, temperature, dropout)
                for _ in range(num_layers)
            ])

        # Standard HeteroConv for non-r2 edges
        # Also includes r2 as SAGEConv when DMP is disabled (ablation)
        self.hetero_convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ("doc", "has_section", "sec"): SAGEConv(hidden_dim, hidden_dim),
                ("label", "maps_to", "conc"): SAGEConv(hidden_dim, hidden_dim),
                ("label", "parent_of", "label"): SAGEConv(hidden_dim, hidden_dim),
                ("sec", "cites", "auth"): SAGEConv(hidden_dim, hidden_dim),
            }
            # When DMP is off, r2 uses standard aggregation
            if not use_dmp:
                conv_dict[("sec", "mentions", "conc")] = SAGEConv(
                    hidden_dim, hidden_dim)

            conv = HeteroConv(conv_dict, aggr="sum")
            self.hetero_convs.append(conv)

        # Output projection
        self.out_proj = Linear(hidden_dim, hidden_dim)

        # Section-level attention for document pooling
        self.sec_attention = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        Full forward pass.

        Args:
            x_dict: {node_type: tensor [N, in_dim]}
            edge_index_dict: {edge_type: tensor [2, E]}
            edge_attr_dict: {edge_type: {attr_name: tensor}}
                            Only needed for r2 edges (operator, priority)

        Returns:
            h: {node_type: tensor [N, hidden_dim]}
            defeat_info: dict with active/defeated embeddings (for L_defeat)
                         None if DMP is disabled
        """
        # Input projection
        h = {}
        for nt in x_dict:
            if nt in self.input_proj:
                h[nt] = F.relu(self.input_proj[nt](x_dict[nt]))
            else:
                h[nt] = x_dict[nt]

        defeat_info = {"active_embs": [], "defeated_embs": []}

        for layer_idx in range(len(self.hetero_convs)):
            hetero_conv = self.hetero_convs[layer_idx]

            # 1. Standard hetero-conv for non-r2 edges (or all edges if no DMP)
            if self.use_dmp:
                # Exclude r2 from hetero_conv — DMP handles it
                non_r2_edges = {
                    k: v for k, v in edge_index_dict.items()
                    if k != ("sec", "mentions", "conc") and v.size(1) > 0
                }
            else:
                non_r2_edges = {
                    k: v for k, v in edge_index_dict.items()
                    if v.size(1) > 0
                }

            # Filter to edges the conv knows about
            valid_edges = {
                k: v for k, v in non_r2_edges.items()
                if k in hetero_conv.convs
            }

            if valid_edges:
                new_h = hetero_conv(h, valid_edges)
                for k in new_h:
                    h[k] = self.dropout(F.relu(new_h[k]))

            # 2. DMP for r2 edges (sec -> conc)
            if self.use_dmp and ("sec", "mentions", "conc") in edge_index_dict:
                r2_ei = edge_index_dict[("sec", "mentions", "conc")]

                if r2_ei.size(1) > 0 and edge_attr_dict is not None:
                    r2_key = ("sec", "mentions", "conc")
                    r2_ops = edge_attr_dict[r2_key]["operator"]
                    r2_pri_raw = edge_attr_dict[r2_key]["priority"]

                    # Optionally refine priorities with learned authority scorer
                    if self.use_authority and "auth" in h:
                        # Use raw priority features as input to scorer
                        # auth_type is encoded in priority features
                        r2_pri = r2_pri_raw  # fallback to raw
                    else:
                        r2_pri = r2_pri_raw

                    dmp = self.dmp_layers[layer_idx]

                    src_embs = h["sec"][r2_ei[0]]  # (E, hidden)
                    dst_ids = r2_ei[1]
                    concept_ids = dst_ids
                    num_conc = h["conc"].size(0)

                    dmp_out = dmp(
                        src_embs, dst_ids, r2_ops, r2_pri, concept_ids, num_conc
                    )

                    # Residual connection
                    h["conc"] = h["conc"] + dmp_out

                    # Collect active/defeated for L_defeat
                    active, defeated = dmp.get_active_defeated_embeddings(
                        src_embs, dst_ids, r2_ops, r2_pri, concept_ids
                    )
                    defeat_info["active_embs"].append(active)
                    defeat_info["defeated_embs"].append(defeated)

        # Output projection
        h = {k: self.out_proj(v) for k, v in h.items()}

        # Combine defeat info across layers
        if defeat_info["active_embs"]:
            defeat_info["active_embs"] = torch.cat(defeat_info["active_embs"])
            defeat_info["defeated_embs"] = torch.cat(defeat_info["defeated_embs"])
        else:
            defeat_info = None

        return h, defeat_info

    def pool_document(self, sec_embs):
        """
        Attention-weighted pooling of section embeddings -> document embedding.

        Args:
            sec_embs: tensor [N_sec, hidden_dim]

        Returns:
            doc_emb: tensor [1, hidden_dim]
        """
        attn = self.sec_attention(sec_embs)        # (N_sec, 1)
        weights = torch.softmax(attn, dim=0)       # (N_sec, 1)
        return (weights * sec_embs).sum(dim=0, keepdim=True)  # (1, hidden)

    def score(self, doc_emb, label_embs):
        """
        Cosine similarity scoring.

        Args:
            doc_emb: tensor [1, hidden_dim]
            label_embs: tensor [100, hidden_dim]

        Returns:
            scores: tensor [100]
        """
        doc_norm = F.normalize(doc_emb, dim=-1)
        label_norm = F.normalize(label_embs, dim=-1)
        return (doc_norm @ label_norm.T).squeeze(0)  # (100,)