from datasets import load_dataset
import pandas as pd
from collections import Counter


def load_eurlex(split: str = "train") -> pd.DataFrame:
    ds = load_dataset("coastalcph/lex_glue", "eurlex", trust_remote_code=True)

    df = pd.DataFrame({
        "text": ds[split]["text"],
        "labels": ds[split]["labels"],
        "doc_id": list(range(len(ds[split])))
    })
    return df


def make_seen_unseen_split(train_df: pd.DataFrame, n_unseen: int = 20):
    label_counts = Counter()

    for labels in train_df["labels"]:
        for label in labels:
            label_counts[label] += 1

    sorted_by_freq = sorted(range(100), key=lambda x: label_counts.get(x, 0))

    unseen = sorted(sorted_by_freq[:n_unseen])
    seen = sorted(set(range(100)) - set(unseen))

    return seen, unseen


def get_label_names():
    ds = load_dataset("coastalcph/lex_glue", "eurlex", trust_remote_code=True)
    return ds["train"].features["labels"].feature.names