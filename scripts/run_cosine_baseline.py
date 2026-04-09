import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.baselines import legalbert_cosine_baseline, tune_threshold
from src.preprocess.data_loader import load_eurlex, make_seen_unseen_split


def labels_to_matrix(label_lists, n_labels=100):
    targets = np.zeros((len(label_lists), n_labels), dtype=int)
    for i, labels in enumerate(label_lists):
        for l in labels:
            if 0 <= l < n_labels:
                targets[i, l] = 1
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    emb_dir = Path("data/processed/embeddings")

    val_doc_embs = torch.load(emb_dir / "validation_doc_embs.pt", map_location="cpu")
    test_doc_embs = torch.load(emb_dir / "test_doc_embs.pt", map_location="cpu")
    label_embs = torch.load(emb_dir / "label_embs.pt", map_location="cpu")

    if args.debug:
        val_doc_embs = val_doc_embs[:100]
        test_doc_embs = test_doc_embs[:100]

    print("Embeddings loaded:")
    print("  validation_doc_embs:", val_doc_embs.shape)
    print("  test_doc_embs:", test_doc_embs.shape)
    print("  label_embs:", label_embs.shape)

    df_train = load_eurlex("train")
    df_val = load_eurlex("validation")
    df_test = load_eurlex("test")

    if args.debug:
        df_val = df_val.iloc[:100].reset_index(drop=True)
        df_test = df_test.iloc[:100].reset_index(drop=True)

    seen, unseen = make_seen_unseen_split(df_train)

    val_targets = labels_to_matrix(df_val["labels"].tolist(), n_labels=100)
    test_targets = labels_to_matrix(df_test["labels"].tolist(), n_labels=100)

    val_scores = legalbert_cosine_baseline(val_doc_embs, label_embs)
    test_scores = legalbert_cosine_baseline(test_doc_embs, label_embs)

    best_thresh, val_macro = tune_threshold(val_scores, val_targets)
    val_preds = (val_scores >= best_thresh).astype(int)
    val_micro = f1_score(val_targets, val_preds, average="micro", zero_division=0)

    test_preds = (test_scores >= best_thresh).astype(int)
    test_macro = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    test_micro = f1_score(test_targets, test_preds, average="micro", zero_division=0)

    results = {
        "best_threshold": best_thresh,
        "val_macro_f1": float(val_macro),
        "val_micro_f1": float(val_micro),
        "test_macro_f1": float(test_macro),
        "test_micro_f1": float(test_micro),
        "n_val_docs": len(df_val),
        "n_test_docs": len(df_test),
    }

    out_dir = Path("outputs/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_legalbert_cosine.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults:")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()