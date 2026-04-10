import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.data_loader import get_label_names
from src.preprocess.concept_linker import get_legalbert_model, embed_texts


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load label names from LexGLUE
    print("Loading label names from LexGLUE...")
    ds = load_dataset("coastalcph/lex_glue", "eurlex", trust_remote_code=True)
    label_names = get_label_names()

    # Turn label names into simple texts
    label_texts = [name.replace("_", " ").lower() for name in label_names]

    # 2) Load LegalBERT
    print("Loading LegalBERT...")
    tokenizer, model = get_legalbert_model(device)

    # 3) Embed label texts (use your local embed_texts signature)
    print("Embedding label texts...")
    label_embs = embed_texts(label_texts, tokenizer, model, device)

    # 4) Save to data/processed/embeddings/label_embs.pt
    out_dir = Path("data/processed/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "label_embs.pt"
    torch.save(label_embs, out_path)

    print(f"Saved label embeddings to {out_path}, shape {tuple(label_embs.shape)}")


if __name__ == "__main__":
    main()