import torch
from rdflib import Graph
from rdflib.namespace import SKOS


def load_label_texts(label_names: list) -> dict:
    """
    Convert label names into clean text descriptions.
    Returns:
        {label_index: cleaned_label_text}
    """
    return {i: name.replace("_", " ").lower() for i, name in enumerate(label_names)}


def parse_eurovoc_hierarchy(rdf_path: str, label_names: list) -> dict:
    """
    Parse EuroVoc RDF and extract relationships among the labels
    that appear in EUR-LEX.

    Returns:
        hierarchy = {
            label_name: [connected_label_1, connected_label_2, ...]
        }
    """
    g = Graph()
    g.parse(rdf_path, format="xml")

    label_set = set(label_names)
    hierarchy = {name: [] for name in label_names}

    label_to_uri = {}

    # Map English prefLabels to URIs
    for s, p, o in g.triples((None, SKOS.prefLabel, None)):
        if str(getattr(o, "language", "")) == "en":
            name = str(o).lower().replace(" ", "_")
            if name in label_set:
                label_to_uri[name] = str(s)

    uri_to_label = {v: k for k, v in label_to_uri.items()}

    # Collect broader / narrower / related links
    relation_preds = [SKOS.broader, SKOS.narrower, SKOS.related]

    for rel in relation_preds:
        for s, p, o in g.triples((None, rel, None)):
            src = uri_to_label.get(str(s))
            tgt = uri_to_label.get(str(o))
            if src and tgt:
                hierarchy[src].append(tgt)

    # Remove duplicates
    for k in hierarchy:
        hierarchy[k] = sorted(list(set(hierarchy[k])))

    return hierarchy


def build_label_adjacency_matrix(hierarchy: dict, label_names: list) -> torch.Tensor:
    """
    Build a normalized adjacency matrix A_norm of shape (100, 100)
    for label propagation.

    A[i, j] = 1 if label i is connected to label j
    plus self-loops.
    """
    n = len(label_names)
    idx = {name: i for i, name in enumerate(label_names)}

    A = torch.zeros(n, n, dtype=torch.float)

    for src, targets in hierarchy.items():
        if src not in idx:
            continue
        for tgt in targets:
            if tgt in idx:
                i, j = idx[src], idx[tgt]
                A[i, j] = 1.0
                A[j, i] = 1.0  # make symmetric

    # Add self-loops
    A += torch.eye(n)

    # Row-normalize
    row_sums = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    A_norm = A / row_sums

    return A_norm