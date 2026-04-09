import pickle

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


def main(doc_idx: int = 0):
    df = load_eurlex("train")
    label_names = get_label_names()
    label_texts = load_label_texts(label_names)

    tokenizer, model = get_legalbert_model("cpu")
    eurovoc_embs = build_eurovoc_embeddings(label_texts, tokenizer, model, device="cpu")

    text = df.iloc[doc_idx]["text"]
    sections = split_into_sections(text)

    processed_sections = []

    for sec in sections:
        sec_text = sec["text"]

        # Concept linking
        concepts = link_concepts_to_eurovoc(
            sec_text,
            eurovoc_embs,
            label_texts,
            tokenizer,
            model,
            threshold=0.75,
            device="cpu",
        )

        # Operator detection for each concept
        if concepts:
            spans = [(c["span_start"], c["span_end"]) for c in concepts]
            ops = detect_operators_in_section(sec_text, spans)
            for c, op in zip(concepts, ops):
                c["operator"] = op

        # Authority extraction
        authorities = extract_authorities(sec_text)

        processed_sections.append({
            "type": sec["type"],
            "text": sec_text,
            "concepts": concepts,
            "authorities": authorities,
        })

    out = {
        "doc_id": int(df.iloc[doc_idx]["doc_id"]),
        "labels": df.iloc[doc_idx]["labels"],
        "sections": processed_sections,
    }

    with open("outputs/smoke_doc.pkl", "wb") as f:
        pickle.dump(out, f)

    print(f"Saved outputs/smoke_doc.pkl for doc {doc_idx}")


if __name__ == "__main__":
    main(doc_idx=0)