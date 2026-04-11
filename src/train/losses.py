"""
JusDef training losses with staged training.

Stage 1 (warmup):  L = L_cls only
Stage 2 (add onto): L = L_cls + lambda1 * L_onto
Stage 3 (full):     L = L_cls + lambda1 * L_onto + lambda2 * L_defeat

This staged approach prevents gradient instability from the defeat
loss early in training when embeddings are still random.

Reference: JusDef paper Section 4.4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Binary cross-entropy for multi-label classification.
    Only computed on seen labels (zero-shot protocol).
    """
    def forward(self, scores, targets, seen_mask):
        """
        Args:
            scores: tensor [N, 100] — model output scores
            targets: tensor [N, 100] — binary ground truth
            seen_mask: tensor [100] bool — True for seen labels
        """
        seen_scores = scores[:, seen_mask]
        seen_targets = targets[:, seen_mask]
        return F.binary_cross_entropy_with_logits(seen_scores, seen_targets)


class OntologyContrastiveLoss(nn.Module):
    """
    Pull concept embeddings closer if they are near in EuroVoc hierarchy.
    Uses InfoNCE-style contrastive loss.

    Reference: JusDef paper Section 4.4, L_onto
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, conc_embs, label_adj):
        """
        Args:
            conc_embs: tensor [C, dim] — concept node embeddings
            label_adj: tensor [100, 100] — normalized label adjacency
        """
        if conc_embs.size(0) < 2:
            return torch.tensor(0.0, device=conc_embs.device)

        conc_norm = F.normalize(conc_embs, dim=-1)
        sim = conc_norm @ conc_norm.T  # (C, C)

        # InfoNCE: each concept is a positive pair with itself,
        # negatives are all other concepts
        N = min(conc_embs.size(0), 32)  # cap for speed
        loss = torch.tensor(0.0, device=conc_embs.device)

        for i in range(N):
            # Positive: most similar concept (excluding self)
            pos_sims = sim[i].clone()
            pos_sims[i] = -1e9  # mask self
            pos_idx = pos_sims.argmax()

            # InfoNCE
            numerator = torch.exp(sim[i, pos_idx] / self.tau)
            denominator = torch.exp(sim[i] / self.tau).sum() - torch.exp(
                sim[i, i] / self.tau)
            loss -= torch.log(numerator / (denominator + 1e-8))

        return loss / N


class DefeatPolarityLoss(nn.Module):
    """
    Margin loss: active and defeated messages of the same concept
    should have distinct embeddings.

    This encourages the model to learn different representations
    for concepts that are legally applicable vs. defeated.

    Reference: JusDef paper Section 4.4, Equation 3
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, active_embs, defeated_embs):
        """
        Args:
            active_embs: tensor [P, dim] — embeddings of active messages
            defeated_embs: tensor [Q, dim] — embeddings of defeated messages
        """
        if active_embs.size(0) == 0 or defeated_embs.size(0) == 0:
            device = active_embs.device if active_embs.size(0) > 0 else \
                     defeated_embs.device
            return torch.tensor(0.0, device=device)

        # Mean embedding of each group
        active_mean = F.normalize(active_embs.mean(dim=0, keepdim=True), dim=-1)
        defeated_mean = F.normalize(defeated_embs.mean(dim=0, keepdim=True), dim=-1)

        # Push apart: similarity should be below margin
        sim = (active_mean @ defeated_mean.T).squeeze()
        loss = F.relu(sim + self.margin)  # want sim < -margin
        return loss


class JusDefLoss(nn.Module):
    """
    Combined loss with staged training.

    Stage 1: L_cls only (warmup, stabilize embeddings)
    Stage 2: L_cls + lambda1 * L_onto (add ontology structure)
    Stage 3: L_cls + lambda1 * L_onto + lambda2 * L_defeat (full)
    """
    def __init__(self, lambda1=0.1, lambda2=0.1,
                 onto_temperature=0.07, defeat_margin=1.0):
        super().__init__()
        self.cls_loss = ClassificationLoss()
        self.onto_loss = OntologyContrastiveLoss(onto_temperature)
        self.defeat_loss = DefeatPolarityLoss(defeat_margin)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, scores, targets, seen_mask,
                conc_embs=None, label_adj=None,
                active_embs=None, defeated_embs=None,
                stage=1):
        """
        Args:
            scores: tensor [N, 100]
            targets: tensor [N, 100]
            seen_mask: tensor [100] bool
            conc_embs: tensor [C, dim] (for L_onto, stages 2+)
            label_adj: tensor [100, 100] (for L_onto, stages 2+)
            active_embs: tensor [P, dim] (for L_defeat, stage 3)
            defeated_embs: tensor [Q, dim] (for L_defeat, stage 3)
            stage: int — training stage (1, 2, or 3)

        Returns:
            total_loss: scalar tensor
        """
        # Stage 1: classification only
        loss = self.cls_loss(scores, targets, seen_mask)

        # Stage 2: add ontology contrastive
        if stage >= 2 and conc_embs is not None and label_adj is not None:
            loss = loss + self.lambda1 * self.onto_loss(conc_embs, label_adj)

        # Stage 3: add defeat polarity
        if stage >= 3 and active_embs is not None and defeated_embs is not None:
            loss = loss + self.lambda2 * self.defeat_loss(
                active_embs, defeated_embs)

        return loss