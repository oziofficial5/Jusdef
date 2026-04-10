from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
import spacy
from torch.nn.functional import cosine_similarity


_LEGALBERT_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
_nlp = None
_tokenizer = None
_model = None


def get_spacy_nlp():
    global _nlp
    if _nlp is None:
        import en_core_web_lg
        _nlp = en_core_web_lg.load()
    return _nlp


def get_legalbert_model(device: str = "cpu"):
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_LEGALBERT_MODEL_NAME)
        _model = AutoModel.from_pretrained(_LEGALBERT_MODEL_NAME)
        _model.to(device)
        _model.eval()
    return _tokenizer, _model


def embed_texts(texts: List[str], tokenizer, model, device: str = "cpu", batch_size: int = 16) -> torch.Tensor:
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            outputs = model(**encoded)
            # CLS token embedding
            cls = outputs.last_hidden_state[:, 0, :]
            embs.append(cls.cpu())
    return torch.cat(embs, dim=0)


def extract_noun_phrases(text: str) -> List[Tuple[str, int, int]]:
    nlp = get_spacy_nlp()
    doc = nlp(text[:500000])
    phrases = []
    for np in doc.noun_chunks:
        span_text = np.text.strip()
        if len(span_text) < 3:
            continue
        phrases.append((span_text, np.start_char, np.end_char))
    return phrases


def build_eurovoc_embeddings(label_texts, tokenizer, model, device: str = "cpu") -> torch.Tensor:
    """
    label_texts is a dict: {idx: text} or a list of strings.
    Convert to a list in index order and embed.
    """
    if isinstance(label_texts, dict):
        texts = [label_texts[i] for i in range(len(label_texts))]
    else:
        texts = list(label_texts)
    return embed_texts(texts, tokenizer, model, device=device, batch_size=32)


def link_concepts_to_eurovoc(
    text: str,
    eurovoc_embs: torch.Tensor,
    label_texts: List[str],
    tokenizer,
    model,
    threshold: float = 0.75,
    device: str = "cpu",
) -> List[Dict]:
    phrases = extract_noun_phrases(text)
    if not phrases:
        return []

    phrase_texts = [p[0] for p in phrases]
    phrase_embs = embed_texts(phrase_texts, tokenizer, model, device=device)

    eurovoc_norm = torch.nn.functional.normalize(eurovoc_embs, dim=1)
    phrase_norm = torch.nn.functional.normalize(phrase_embs, dim=1)

    sims = phrase_norm @ eurovoc_norm.T

    results = []
    for i, (phrase, start, end) in enumerate(phrases):
        sim_row = sims[i]
        max_sim, max_idx = sim_row.max(dim=0)
        max_sim_val = max_sim.item()
        if max_sim_val >= threshold:
            label_idx = int(max_idx.item())
            results.append({
                "phrase": phrase,
                "span_start": start,
                "span_end": end,
                "label_idx": label_idx,
                "label_text": label_texts[label_idx],
                "similarity": max_sim_val,
            })
    return results