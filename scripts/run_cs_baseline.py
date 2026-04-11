import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.baselines import legalbert_cosine_baseline
from src.train.label_propagation import cs_label_propagation
from src.preprocess.data_loader import (
    load_eurlex,
    make_seen_unseen_split,
    get_label_names,
)
from src.eval.metrics import (
    compute_all_metrics,
    load_exception_labels,
    print_results,
    save_results,
)


def labels_to_matrix(label_lists, n_labels=100):
    """Convert list of label index lists to a binary matrix [n_docs, n_labels]."""
    targets = np.zeros((len(label_lists), n_labels), dtype=int)
    for i, labels in enumerate(label_lists):
        for l in labels:
            if 0 <= l < n_labels:
                targets[i, l] = 1
    return targets


def main():
    parser = argparse.ArgumentParser(description="C&S label propagation baseline")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    emb_dir = Path("data/processed/embeddings")

    # Load embeddings
    val_doc_embs = torch.load(emb_dir / "validation_doc_embs.pt", map_location="cpu")
    test_doc_embs = torch.load(emb_dir / "test_doc_embs.pt", map_location="cpu")
    label_embs = torch.load(emb_dir / "label_embs.pt", map_location="cpu")

    if args.debug:
        val_doc_embs = val_doc_embs[:10]
        test_doc_embs = test_doc_embs[:10]

    print("Embeddings loaded")
    print(" validation_doc_embs:", val_doc_embs.shape)
    print(" test_doc_embs:", test_doc_embs.shape)
    print(" label_embs:", label_embs.shape)

    # Load EUR-LEX splits
    df_train = load_eurlex("train")
    df_val = load_eurlex("validation")
    df_test = load_eurlex("test")

    if args.debug:
        df_val = df_val.iloc[:10].reset_index(drop=True)
        df_test = df_test.iloc[:10].reset_index(drop=True)

    # Seen / unseen split (Yu protocol)
    seen, unseen = make_seen_unseen_split(df_train)

    # Ground-truth label matrices
    val_targets = labels_to_matrix(df_val["labels"].tolist(), n_labels=100)
    test_targets = labels_to_matrix(df_test["labels"].tolist(), n_labels=100)

    # Build adjacency: connect labels in the same EuroVoc domain (prefix-based)
    label_names = get_label_names()
    n_labels = len(label_names)
    label_adj = torch.zeros(n_labels, n_labels, dtype=torch.float32)

    # Group by first 2 characters of label name (proxy for domain)
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            if str(label_names[i])[:2] == str(label_names[j])[:2]:
                label_adj[i, j] = 1.0
                label_adj[j, i] = 1.0

    # Add self-loops and row-normalize
    label_adj += torch.eye(n_labels)
    row_sums = label_adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
    label_adj = label_adj / row_sums

    n_edges = int((label_adj > 0).sum().item() - n_labels)
    print(f"  Label graph: {n_edges} edges (domain-based grouping)")

    # C&S label propagation
    propagated_label_embs = cs_label_propagation(
        label_embs,
        label_adj,
        alpha=args.alpha,
        num_steps=args.steps,
    )

    # Cosine baseline with propagated labels
    val_scores = legalbert_cosine_baseline(val_doc_embs, propagated_label_embs)
    test_scores = legalbert_cosine_baseline(test_doc_embs, propagated_label_embs)

    # Shape sanity check — transpose if needed so both are [n_docs, n_labels]
    print("val_scores shape:", val_scores.shape)
    print("val_targets shape:", val_targets.shape)

    if val_scores.shape != val_targets.shape:
        if val_scores.shape[::-1] == val_targets.shape:
            val_scores = val_scores.T
            test_scores = test_scores.T
            print("Transposed scores to match targets shape.")
        else:
            raise ValueError(
                f"Scores/targets shape mismatch: scores {val_scores.shape}, "
                f"targets {val_targets.shape}"
            )

    # Shared evaluation: tune on val, report on test
    exc_labels = load_exception_labels()

    val_metrics = compute_all_metrics(
        val_scores, val_targets, seen, unseen, exc_labels
    )

    test_metrics = compute_all_metrics(
        test_scores, test_targets, seen, unseen, exc_labels,
        threshold=val_metrics["threshold"],
    )

    results = {
        "model": "cs_label_propagation",
        "alpha": args.alpha,
        "steps": args.steps,
        "n_val_docs": len(df_val),
        "n_test_docs": len(df_test),
        "val": val_metrics,
        "test": test_metrics,
    }

    print_results(test_metrics, "C&S Label Propagation")
    save_results(results, "outputs/logs/baseline_cs_label_propagation.json")


if __name__ == "__main__":
    main()