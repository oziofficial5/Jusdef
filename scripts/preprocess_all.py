"""
Full preprocessing pipeline: sections + concepts + operators + authorities.
Saves one pickle per split with all structured data.

Run:
    python scripts/preprocess_all.py
    python scripts/preprocess_all.py --debug  (laptop, 10 docs only)
"""
import os
import sys
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.data_loader import load_eurlex, get_label_names
from src.preprocess.label_loader import load_label_texts
from src.preprocess.section_splitter import split_into_sections
from src.preprocess.operator_detector import detect_operators_in_section
from src.preprocess.authority_extractor import extract_authorities
from src.preprocess.concept_linker import (
    get_legalbert_model,
    build_eurovoc_embeddings,
    link_concepts_to_eurovoc,
)


def process_one_doc(
    doc_idx,
    text,
    labels,
    eurovoc_embs,
    label_texts,
    tokenizer,
    model,
    device="cpu",
):
    """Process a single document through the full pipeline."""
    sections = split_into_sections(text)
    processed_sections = []

    for sec in sections:
        sec_text = sec["text"]

        # 1. Concept linking
        concepts = link_concepts_to_eurovoc(
            sec_text,
            eurovoc_embs,
            label_texts,
            tokenizer,
            model,
            threshold=0.75,
            device=device,
        )

        # 2. Operator detection for each concept
        if concepts:
            spans = [(c["span_start"], c["span_end"]) for c in concepts]
            ops = detect_operators_in_section(sec_text, spans)
            for c, op in zip(concepts, ops):
                c["operator"] = op

        # 3. Authority extraction
        authorities = extract_authorities(sec_text)

        # 4. Section type as int
        type_map = {
            "PREAMBLE": 0,
            "DEFINITIONS": 1,
            "PROVISIONS": 2,
            "PENALTIES": 3,
            "ANNEX": 4,
        }

        processed_sections.append(
            {
                "type": sec["type"],
                "type_int": type_map.get(sec["type"], 2),
                "text": sec_text[:500],  # truncate text to save space
                "concepts": concepts,
                "authorities": authorities,
            }
        )

    return {
        "doc_id": doc_idx,
        "labels": labels,
        "sections": processed_sections,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    label_names = get_label_names()
    label_texts = load_label_texts(label_names)
    tokenizer, model = get_legalbert_model(args.device)
    eurovoc_embs = build_eurovoc_embeddings(
        label_texts,
        tokenizer,
        model,
        device=args.device,
    )

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        print(f"\n{'=' * 50}")
        print(f"Processing {split}")
        print(f"{'=' * 50}")

        df = load_eurlex(split)
        if args.debug:
            df = df.iloc[:10].reset_index(drop=True)

        processed_docs = []
        concept_count = 0
        operator_counts = {"AFF": 0, "NEG": 0, "EXC": 0, "OVR": 0}

        for idx in tqdm(range(len(df)), desc=split):
            doc = process_one_doc(
                doc_idx=idx,
                text=df.iloc[idx]["text"],
                labels=df.iloc[idx]["labels"],
                eurovoc_embs=eurovoc_embs,
                label_texts=label_texts,
                tokenizer=tokenizer,
                model=model,
                device=args.device,
            )
            processed_docs.append(doc)

            # Stats
            for sec in doc["sections"]:
                for c in sec["concepts"]:
                    concept_count += 1
                    op = c.get("operator", "AFF")
                    operator_counts[op] = operator_counts.get(op, 0) + 1

        # Save
        out_path = out_dir / f"{split}_processed.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(processed_docs, f)

        print(f"Saved {len(processed_docs)} docs to {out_path}")
        print(f"  Total concepts: {concept_count}")
        print(f"  Operators: {operator_counts}")


if __name__ == "__main__":
    main()