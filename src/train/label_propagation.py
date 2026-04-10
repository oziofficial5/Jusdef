"""
C&S-style label propagation over EuroVoc label graph.

Given label embeddings and an adjacency matrix from the EuroVoc hierarchy,
propagate label information so that similar/related labels get closer embeddings.
This helps zero-shot prediction by spreading signal from seen to unseen labels.
"""
import torch


def cs_label_propagation(label_embs, label_adj, alpha=0.5, num_steps=3):
    """
    C&S-style propagation over EuroVoc label graph.

    h_y^(t+1) = alpha * h_y^(0) + (1 - alpha) * A_norm @ h_y^(t)

    Args:
        label_embs: tensor [N_labels, dim] — original label embeddings
        label_adj: tensor [N_labels, N_labels] — normalized adjacency matrix
                   (from build_label_adjacency_matrix in label_loader.py)
        alpha: float — how much to keep original embedding (0=full propagation, 1=no change)
        num_steps: int — number of propagation rounds

    Returns:
        tensor [N_labels, dim] — propagated label embeddings
    """
    h0 = label_embs.clone()
    h = label_embs.clone()

    for _ in range(num_steps):
        h = alpha * h0 + (1 - alpha) * (label_adj @ h)

    return h