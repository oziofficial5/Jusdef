"""
Defeasible Message Passing (DMP) layer for JusDef.

This is the core contribution. DMP differs from standard GNN aggregation
in one key way: messages can be DEFEATED (excluded from aggregation)
based on operator precedence and authority priority.

Key concepts:
  - Each r2 edge (sec -> conc) carries an operator (AFF/NEG/EXC/OVR)
    and an authority priority score
  - Message m_i defeats m_j if:
    (1) op_precedence(m_i) > op_precedence(m_j), OR
    (2) same precedence AND priority(m_i) > priority(m_j)
    AND both target the same concept node
  - Defeated messages are excluded via a binary mask
  - A Straight-Through Estimator (STE) makes the discrete mask differentiable

When no defeats occur (all AFF, equal priorities), DMP reduces exactly
to standard relational attention — making JusDef a strict generalisation.

Reference: JusDef paper Section 4.3, Equations 1-3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Operator precedence: higher number = stronger operator
# OVR > EXC > NEG > AFF
OPERATOR_PRECEDENCE = {0: 0, 1: 1, 2: 2, 3: 3}  # AFF=0, NEG=1, EXC=2, OVR=3


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for differentiable hard defeat masking.

    Forward pass: hard binary mask (0 or 1)
      - 1 if the message is active (undefeated)
      - 0 if the message is defeated

    Backward pass: smooth sigmoid gradient
      - Approximates the gradient of the hard mask with a sigmoid
      - Temperature controls sharpness of the approximation

    This lets gradients flow through the discrete defeat decision,
    which is essential for end-to-end training.

    Reference: STE
    """
    @staticmethod
    def forward(ctx, priority_gap, temperature=5.0):
        """
        Args:
            priority_gap: tensor (E,) — for each message, the gap between
                          its strongest defeater's priority and its own.
                          Negative or -inf means no defeater exists.
            temperature: float — controls sigmoid sharpness in backward

        Returns:
            mask: tensor (E,) — 1.0 if active, 0.0 if defeated
        """
        ctx.save_for_backward(priority_gap)
        ctx.temperature = temperature
        # Hard mask: active if no defeater is stronger (gap <= 0)
        return (priority_gap <= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """Sigmoid approximation for gradient."""
        priority_gap, = ctx.saved_tensors
        tau = ctx.temperature
        # d/dx sigmoid(-tau * x)
        sig = torch.sigmoid(-tau * priority_gap)
        sigmoid_grad = tau * sig * (1 - sig)
        return grad_output * sigmoid_grad, None  # None for temperature


def compute_defeat_mask(operators, priorities, dst_nodes, temperature=5.0):
    """
    Compute which messages are defeated and which are active.

    For each message m_i, check all other messages m_j targeting the
    same concept node. If m_j has higher operator precedence, or same
    precedence but higher priority, then m_j defeats m_i.

    Args:
        operators: tensor (E,) int — operator type per edge (0=AFF..3=OVR)
        priorities: tensor (E,) float — authority priority per edge
        dst_nodes: tensor (E,) int — target concept node per edge
        temperature: float — STE temperature

    Returns:
        mask: tensor (E,) float — 1.0=active, 0.0=defeated
    """
    E = operators.size(0)
    if E == 0:
        return torch.ones(0, device=operators.device)

    op_prec = operators.float()  # Use int value as precedence rank

    # For each message i, find the max "defeat strength" from any j
    # priority_gap[i] = max over all j that defeat i of (priority_j - priority_i)
    # If no j defeats i, gap stays at -inf → mask = 1 (active)
    priority_gap = torch.full((E,), -1e9, device=operators.device)

    # Group edges by destination node for efficiency
    unique_dsts = dst_nodes.unique()

    for dst in unique_dsts:
        # Find all edges targeting this concept node
        mask = (dst_nodes == dst)
        indices = mask.nonzero(as_tuple=True)[0]

        if indices.size(0) <= 1:
            continue  # Only one message — no defeat possible

        # For each pair (i, j) in this group, check if j defeats i
        for idx_pos_i in range(indices.size(0)):
            i = indices[idx_pos_i]
            for idx_pos_j in range(indices.size(0)):
                if idx_pos_i == idx_pos_j:
                    continue
                j = indices[idx_pos_j]

                op_i = op_prec[i].item()
                op_j = op_prec[j].item()
                pr_i = priorities[i].item()
                pr_j = priorities[j].item()

                # j defeats i if:
                #   op_j > op_i (stronger operator), OR
                #   op_j == op_i AND pr_j > pr_i (same operator, higher priority)
                j_defeats_i = (op_j > op_i) or (op_j == op_i and pr_j > pr_i)

                if j_defeats_i:
                    # Gap must be positive when defeat occurs
                    # Operator precedence contributes a base gap
                    op_gap = op_j - op_i  # >= 0 when op defeats
                    pr_gap = pr_j - pr_i  # can be 0 when priorities equal
                    gap = op_gap + pr_gap + 0.01  # ensure strictly positive
                    if gap > priority_gap[i]:
                        priority_gap[i] = gap

    # Apply STE for differentiability
    mask = StraightThroughEstimator.apply(priority_gap, temperature)
    return mask


class DMPLayer(nn.Module):
    """
    Defeasible Message Passing layer.

    Only operates on r2 (sec → conc) edges. All other edge types
    use standard HeteroConv aggregation in the JusDef model.

    Architecture per layer:
      1. Apply operator-specific weight matrices W_omega to source embeddings
      2. Compute defeat mask via STE
      3. Mask out defeated messages
      4. Attention-weighted aggregation of active messages per concept node

    When all operators are AFF and priorities are equal, no defeats occur,
    and DMP reduces to standard attention aggregation (Proposition 2).

    Reference: JusDef paper Section 4.3, Equation 2
    """
    NUM_OPERATORS = 4  # AFF=0, NEG=1, EXC=2, OVR=3

    def __init__(self, in_dim=512, out_dim=512, temperature=5.0, dropout=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.temperature = temperature

        # One weight matrix per operator type
        self.W_op = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(self.NUM_OPERATORS)
        ])

        # Attention scoring for active messages
        self.attn = nn.Linear(out_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src_embs, dst_node_ids, operators, priorities, num_dst):
        """
        Args:
            src_embs: tensor (E, in_dim) — source node embeddings for each r2 edge
            dst_node_ids: tensor (E,) — which concept node each message targets
            operators: tensor (E,) int — operator type per edge
            priorities: tensor (E,) float — authority priority per edge
            num_dst: int — total number of concept nodes

        Returns:
            out: tensor (num_dst, out_dim) — updated concept node embeddings
        """
        E = src_embs.size(0)
        device = src_embs.device

        if E == 0:
            return torch.zeros(num_dst, self.out_dim, device=device)

        # Step 1: Apply operator-specific weight matrices
        msg = torch.zeros(E, self.out_dim, device=device)
        for op_idx in range(self.NUM_OPERATORS):
            op_mask = (operators == op_idx)
            if op_mask.any():
                msg[op_mask] = self.W_op[op_idx](src_embs[op_mask])

        # Step 2: Compute defeat mask
        defeat_mask = compute_defeat_mask(
            operators, priorities, dst_node_ids, self.temperature
        )  # (E,) — 1=active, 0=defeated

        # Step 3: Apply mask to messages
        active_msg = msg * defeat_mask.unsqueeze(-1)  # (E, out_dim)

        # Step 4: Attention-weighted aggregation per destination node
        attn_scores = self.attn(active_msg).squeeze(-1)  # (E,)

        out = torch.zeros(num_dst, self.out_dim, device=device)

        for v in range(num_dst):
            edge_mask = (dst_node_ids == v)
            if not edge_mask.any():
                continue

            # Prefer active messages; fallback to all if everything is defeated
            active_edge_mask = edge_mask & (defeat_mask > 0.5)
            if not active_edge_mask.any():
                active_edge_mask = edge_mask

            # Softmax attention over selected messages
            scores_v = attn_scores[active_edge_mask]
            weights = torch.softmax(scores_v, dim=0).unsqueeze(-1)  # (K, 1)
            msgs_v = active_msg[active_edge_mask]  # (K, out_dim)
            out[v] = (weights * msgs_v).sum(dim=0)

        return self.dropout(out)

    def get_active_defeated_embeddings(self, src_embs, dst_node_ids,
                                       operators, priorities):
        """
        Helper for L_defeat loss: return separate tensors of active
        and defeated message embeddings.

        Returns:
            active_embs: tensor (P, out_dim)
            defeated_embs: tensor (Q, out_dim)
        """
        E = src_embs.size(0)
        if E == 0:
            return torch.zeros(0, self.out_dim), torch.zeros(0, self.out_dim)

        # Transform messages
        msg = torch.zeros(E, self.out_dim, device=src_embs.device)
        for op_idx in range(self.NUM_OPERATORS):
            op_mask = (operators == op_idx)
            if op_mask.any():
                msg[op_mask] = self.W_op[op_idx](src_embs[op_mask])

        # Compute mask
        defeat_mask = compute_defeat_mask(
            operators, priorities, dst_node_ids, self.temperature
        )

        active = msg[defeat_mask > 0.5]
        defeated = msg[defeat_mask <= 0.5]

        return active, defeated