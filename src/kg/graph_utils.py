import torch
from torch_geometric.data import HeteroData


def validate_graph(data: HeteroData) -> bool:
    """
    Run sanity checks on a JusDef HeteroData graph.
    """
    checks = {
        "doc nodes exist": "doc" in data.node_types,
        "sec nodes exist": "sec" in data.node_types,
        "conc nodes exist": "conc" in data.node_types,
        "label nodes exist": "label" in data.node_types,
        "doc.x has 1 node": data["doc"].x.shape[0] == 1,
        "label.x has 100 nodes": data["label"].x.shape[0] == 100,
        "y shape": data.y.shape == torch.Size([100]),
        "no NaN in doc.x": not torch.isnan(data["doc"].x).any(),
        "r1 exists": ("doc", "has_section", "sec") in data.edge_types,
        "r2 exists": ("sec", "mentions", "conc") in data.edge_types,
        "r2 has operator": hasattr(data["sec", "mentions", "conc"], "operator"),
        "r2 has priority": hasattr(data["sec", "mentions", "conc"], "priority"),
    }

    if data["sec", "mentions", "conc"].edge_index.size(1) > 0:
        ops = data["sec", "mentions", "conc"].operator
        checks["r2 operator ids in 0-3"] = ops.max().item() <= 3
    else:
        checks["r2 operator ids in 0-3"] = True

    all_pass = True
    for name, ok in checks.items():
        if not ok:
            print("[FAIL]", name)
            all_pass = False
    return all_pass


def print_graph_stats(data: HeteroData) -> None:
    print("Node types:")
    for nt in data.node_types:
        print(" ", nt, data[nt].x.shape)

    print("Edge types:")
    for et in data.edge_types:
        n_edges = data[et].edge_index.size(1)
        print(" ", et, n_edges, "edges")

    if hasattr(data, "y"):
        print("Labels active:", int(data.y.sum().item()))