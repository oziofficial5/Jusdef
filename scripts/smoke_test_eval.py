"""Smoke test for evaluation metrics and bootstrap."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.metrics import compute_all_metrics, load_exception_labels, print_results
from src.eval.bootstrap import paired_bootstrap_significance

# Fake data: 20 docs, 100 labels
np.random.seed(42)
scores = np.random.randn(20, 100) * 0.3
targets = (np.random.rand(20, 100) > 0.9).astype(int)

seen = list(range(80))
unseen = list(range(80, 100))
exc = load_exception_labels()

# Test compute_all_metrics
print("=== Metrics Test ===")
results = compute_all_metrics(scores, targets, seen, unseen, exc)
print_results(results, "Fake Model")

# Test bootstrap
print("\n=== Bootstrap Test ===")
scores_b = scores + np.random.randn(20, 100) * 0.1  # slightly different
sig = paired_bootstrap_significance(
    scores, scores_b, targets,
    threshold_a=results["threshold"],
    threshold_b=results["threshold"],
    metric="macro_f1",
    n_bootstrap=1000)

print(f"  p_value: {sig['p_value']}")
print(f"  significant: {sig['significant']}")
print(f"  diff: {sig['base_diff']}")

print("\nAll evaluation smoke tests PASSED")