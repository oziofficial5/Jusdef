import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import f1_score
from src.model.baselines import tune_threshold

exc_labels = json.load(open("data/annotations/exception_labels.json"))
exc_set = set(exc_labels)
seen = list(range(80))
unseen = list(range(80, 100))
exc_idx = [i for i in range(100) if i in exc_set]

# Evaluate each saved checkpoint
for tag in ["full", "full_v2", "no_dmp"]:
    ckpt = f"outputs/checkpoints/jusdef_{tag}_s42.pt"
    if not os.path.exists(ckpt):
        print(f"  {tag}: checkpoint not found, skipping")
        continue
    
    print(f"\n=== {tag} ===")
    
    from src.model.jusdef import JusDef
    from src.train.trainer import forward_one_graph
    
    use_dmp = "no_dmp" not in tag
    model = JusDef(in_dim=768, hidden_dim=512, num_layers=2,
                   use_dmp=use_dmp, use_authority=use_dmp)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    
    test_graphs = torch.load("data/processed/graphs/test_graphs.pt", map_location="cpu")
    
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for g in test_graphs:
            scores, _, _ = forward_one_graph(model, g, "cpu")
            all_scores.append(scores.cpu().squeeze())
            all_targets.append(g.y.cpu())
    
    logits = torch.stack(all_scores).numpy()
    targets = torch.stack(all_targets).numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    
    best_t, _ = tune_threshold(probs, targets)
    preds = (probs >= best_t).astype(int)
    
    macro = f1_score(targets, preds, average="macro", zero_division=0)
    micro = f1_score(targets, preds, average="micro", zero_division=0)
    
    # Per-group
    macro_seen = f1_score(targets[:, seen], preds[:, seen], average="macro", zero_division=0)
    macro_unseen = f1_score(targets[:, unseen], preds[:, unseen], average="macro", zero_division=0)
    macro_exc = f1_score(targets[:, exc_idx], preds[:, exc_idx], average="macro", zero_division=0)
    
    print(f"  Macro-F1:       {macro:.4f}")
    print(f"  Micro-F1:       {micro:.4f}")
    print(f"  F1 (seen):      {macro_seen:.4f}")
    print(f"  F1 (unseen):    {macro_unseen:.4f}")
    print(f"  F1 (exception): {macro_exc:.4f}")
    print(f"  Threshold:      {best_t:.2f}")
