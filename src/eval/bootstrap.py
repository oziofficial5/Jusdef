"""
Paired bootstrap significance testing.

Tests whether model A is significantly better than model B
on a given metric, using paired bootstrap resampling.
"""
import numpy as np
from sklearn.metrics import f1_score


def paired_bootstrap_significance(scores_a, scores_b, targets,
                                  threshold_a=0.5, threshold_b=0.5,
                                  metric="macro_f1",
                                  n_bootstrap=10000, alpha=0.05):
    """
    Paired bootstrap significance test.

    Args:
        scores_a: numpy array [N, L] — scores from model A
        scores_b: numpy array [N, L] — scores from model B
        targets: numpy array [N, L] — binary ground truth
        threshold_a: float — decision threshold for model A
        threshold_b: float — decision threshold for model B
        metric: str — "macro_f1" or "micro_f1"
        n_bootstrap: int — number of bootstrap samples
        alpha: float — significance level

    Returns:
        dict with base_diff, p_value, significant, confidence_interval
    """
    N = targets.shape[0]

    preds_a = (scores_a >= threshold_a).astype(int)
    preds_b = (scores_b >= threshold_b).astype(int)

    def compute_metric(tgt, pred):
        avg = "macro" if metric == "macro_f1" else "micro"
        return f1_score(tgt, pred, average=avg, zero_division=0)

    base_a = compute_metric(targets, preds_a)
    base_b = compute_metric(targets, preds_b)
    base_diff = base_a - base_b

    # Bootstrap
    diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(N, N, replace=True)
        diff = (compute_metric(targets[idx], preds_a[idx]) -
                compute_metric(targets[idx], preds_b[idx]))
        diffs.append(diff)

    diffs = np.array(diffs)

    # Two-tailed p-value: proportion of bootstrap diffs that have opposite sign
    if base_diff > 0:
        p_value = (diffs <= 0).mean()
    else:
        p_value = (diffs >= 0).mean()

    # Confidence interval
    ci_low = np.percentile(diffs, 100 * alpha / 2)
    ci_high = np.percentile(diffs, 100 * (1 - alpha / 2))

    return {
        "model_a_score": round(float(base_a), 4),
        "model_b_score": round(float(base_b), 4),
        "base_diff": round(float(base_diff), 4),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < alpha),
        "confidence_interval": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "n_bootstrap": n_bootstrap,
    }