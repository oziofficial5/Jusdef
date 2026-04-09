import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


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