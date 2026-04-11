"""
Unified evaluation metrics for all JusDef models.

Computes: Macro-F1, Micro-F1, Macro-F1_unseen, Macro-F1_exc
Reusable across cosine, C&S, R-GCN, and JusDef.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score


def compute_all_metrics(scores, targets, seen_labels, unseen_labels,
                        exc_labels=None, threshold=None):
    """
    Compute all evaluation metrics for a model's predictions.

    Args:
        scores: numpy array [N_docs, N_labels] — raw model scores
        targets: numpy array [N_docs, N_labels] — binary ground truth
        seen_labels: list of int — indices of seen labels
        unseen_labels: list of int — indices of unseen labels
        exc_labels: list of int or None — indices of exception-dependent labels
        threshold: float or None — if None, tunes on the provided data

    Returns:
        dict with all metrics
    """
    from src.model.baselines import tune_threshold

    # Tune or use provided threshold
    if threshold is None:
        threshold, _ = tune_threshold(scores, targets)

    preds = (scores >= threshold).astype(int)

    # Overall metrics
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)

    # Seen-only metrics
    seen_idx = list(seen_labels)
    macro_f1_seen = f1_score(
        targets[:, seen_idx], preds[:, seen_idx],
        average="macro", zero_division=0)

    # Unseen-only metrics
    unseen_idx = list(unseen_labels)
    macro_f1_unseen = f1_score(
        targets[:, unseen_idx], preds[:, unseen_idx],
        average="macro", zero_division=0)

    results = {
        "threshold": round(float(threshold), 4),
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "macro_f1_seen": round(float(macro_f1_seen), 4),
        "macro_f1_unseen": round(float(macro_f1_unseen), 4),
    }

    # Exception-label metrics
    if exc_labels is not None and len(exc_labels) > 0:
        macro_f1_exc = f1_score(
            targets[:, exc_labels], preds[:, exc_labels],
            average="macro", zero_division=0)
        results["macro_f1_exc"] = round(float(macro_f1_exc), 4)

    return results


def load_exception_labels(path="data/annotations/exception_labels.json"):
    """Load exception label indices from annotation file."""
    p = Path(path)
    if not p.exists():
        print(f"  Warning: {path} not found, skipping Macro-F1_exc")
        return None
    with open(p, "r") as f:
        data = json.load(f)
    return data.get("exception_override_labels", None)


def print_results(results, model_name="Model"):
    """Pretty-print results dict."""
    print(f"\n{'=' * 50}")
    print(f"RESULTS: {model_name}")
    print(f"{'=' * 50}")
    for k, v in results.items():
        print(f"  {k}: {v}")


def save_results(results, path):
    """Save results dict to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {p}")