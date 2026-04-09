import os
import sys
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.data_loader import load_eurlex
from src.preprocess.section_splitter import split_into_sections
from src.preprocess.concept_linker import get_legalbert_model


def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 16,
    device: str = "cpu",
    max_length: int = 512,
) -> torch.Tensor:
    """Encode a list of texts with LegalBERT and return CLS embeddings."""
    all_embs = []

    model.to(device)
    model.eval()

    dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding texts"):
            enc = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            # CLS token is at position 0
            cls_emb = outputs.last_hidden_state[:, 0, :].detach().cpu()
            all_embs.append(cls_emb)

    if not all_embs:
        # No texts, return empty tensor with correct dim
        return torch.empty(0, model.config.hidden_size)

    return torch.cat(all_embs, dim=0)


def process_split(
    split_name: str,
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 512,
    max_docs: int = None,
) -> Dict[str, Any]:
    """
    For a given split (train/validation/test):
    - compute document-level embeddings
    - compute section-level embeddings
    - store metadata (doc_id, labels, section types, mapping indices)
    """
    print(f"\nProcessing split: {split_name}")

    df = load_eurlex(split_name)

    if max_docs is not None:
        df = df.iloc[:max_docs].reset_index(drop=True)

    texts = df["text"].tolist()
    doc_ids = df["doc_id"].tolist()
    labels = df["labels"].tolist()

    # 1. Document-level embeddings
    print("Encoding full documents...")
    doc_embs = encode_texts(
        texts,
        tokenizer,
        model,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )  # [num_docs, hidden_dim]

    # 2. Section-level embeddings
    print("Splitting into sections...")
    all_section_texts: List[str] = []
    all_section_types: List[str] = []
    all_section_doc_indices: List[int] = []  # which doc each section belongs to

    for i, text in enumerate(tqdm(texts, desc="Splitting docs")):
        sections = split_into_sections(text)
        for sec in sections:
            all_section_texts.append(sec["text"])
            all_section_types.append(sec["type"])
            all_section_doc_indices.append(i)

    if len(all_section_texts) > 0:
        print("Encoding sections...")
        section_embs = encode_texts(
            all_section_texts,
            tokenizer,
            model,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )  # [num_sections, hidden_dim]
    else:
        print("No sections found; creating empty tensor.")
        section_embs = torch.empty((0, model.config.hidden_size))

    result: Dict[str, Any] = {
        "doc_ids": doc_ids,
        "labels": labels,
        "doc_embs": doc_embs,
        "section_embs": section_embs,
        "section_types": all_section_types,
        "section_doc_indices": all_section_doc_indices,
    }

    return result


def save_split_outputs(
    split_name: str,
    outputs: Dict[str, Any],
    out_dir: str = "data/processed/embeddings",
):
    os.makedirs(out_dir, exist_ok=True)

    # Document embeddings
    doc_path = os.path.join(out_dir, f"{split_name}_doc_embs.pt")
    torch.save(
        {
            "doc_ids": outputs["doc_ids"],
            "labels": outputs["labels"],
            "embeddings": outputs["doc_embs"],
        },
        doc_path,
    )
    print(f"Saved document embeddings to {doc_path}")

    # Section embeddings + metadata
    sec_path = os.path.join(out_dir, f"{split_name}_section_embs.pt")
    torch.save(
        {
            "doc_ids": outputs["doc_ids"],
            "section_embs": outputs["section_embs"],
            "section_types": outputs["section_types"],
            "section_doc_indices": outputs["section_doc_indices"],
        },
        sec_path,
    )
    print(f"Saved section embeddings to {sec_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to use: "cpu" or "cuda"',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length for LegalBERT",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, process only a small subset of docs per split",
    )
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size
    max_length = args.max_length
    debug = args.debug

    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}, max_length: {max_length}, debug={debug}")

    # Load LegalBERT using your helper
    tokenizer, model = get_legalbert_model(device=device)

    # EUR-LEX splits: "train", "validation", "test"
    for split in ["train", "validation", "test"]:
        if debug:
            # small subset for laptop testing
            max_docs = 50 if split == "train" else 10
        else:
            max_docs = None

        outputs = process_split(
            split_name=split,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            max_docs=max_docs,
        )
        save_split_outputs(split, outputs)


if __name__ == "__main__":
    main()