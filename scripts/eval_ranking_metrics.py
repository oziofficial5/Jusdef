"""Threshold-free ranking metrics for R-GCN and JusDef checkpoints.

Reviewer request (ANNPR camera-ready, R3 major 4): AUPRC, R-precision, and
per-label validation-tuned-threshold F1, so the head-to-head does not hinge on a
single global threshold. Also dumps raw score matrices for reuse
(Y_exc sensitivity, per-label analysis).

Outputs:
    outputs/scores/{tag}_{split}_logits.npy     [N, 100] raw logits
    outputs/scores/targets_{split}.npy          [N, 100] binary targets
    outputs/logs/ranking_metrics.json
"""
import os, sys, json, time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

from src.model.baselines import RGCN
from src.model.jusdef import JusDef
from src.train.trainer import forward_one_graph

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}", flush=True)

exc_idx = json.load(open("data/annotations/exception_labels.json"))["exception_override_labels"]
SEEN = list(range(80))
UNSEEN = list(range(80, 100))
SUBSETS = {"all": list(range(100)), "seen": SEEN, "unseen": UNSEEN, "exc": exc_idx}

os.makedirs("outputs/scores", exist_ok=True)


def score_jusdef(ckpt, hd, graphs):
    model = JusDef(in_dim=768, hidden_dim=hd, num_layers=2, use_dmp=True, use_authority=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(DEVICE).eval()
    logits, targets = [], []
    with torch.no_grad():
        for i, g in enumerate(graphs):
            if i % 1000 == 0:
                print(f"    {i}/{len(graphs)}", flush=True)
            s, _, _ = forward_one_graph(model, g, DEVICE)
            logits.append(s.detach().cpu().view(-1))
            targets.append(g.y.cpu())
    return torch.stack(logits).numpy(), torch.stack(targets).numpy()


def score_rgcn(ckpt, graphs):
    model = RGCN(in_dim=768, hidden_dim=512)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(DEVICE).eval()
    logits, targets = [], []
    with torch.no_grad():
        for i, g in enumerate(graphs):
            if i % 1000 == 0:
                print(f"    {i}/{len(graphs)}", flush=True)
            g = g.to(DEVICE)
            h = model(g.x_dict, g.edge_index_dict)
            doc_emb = h["sec"].mean(dim=0, keepdim=True)
            s = model.score(doc_emb, h["label"]).squeeze(0)
            logits.append(s.detach().cpu().view(-1))
            targets.append(g.y.cpu())
    return torch.stack(logits).numpy(), torch.stack(targets).numpy()


def auprc_subset(logits, targets, cols):
    vals = []
    for j in cols:
        if targets[:, j].sum() > 0:
            vals.append(average_precision_score(targets[:, j], logits[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


def r_precision(logits, targets):
    vals = []
    for i in range(logits.shape[0]):
        k = int(targets[i].sum())
        if k == 0:
            continue
        topk = np.argsort(-logits[i])[:k]
        vals.append(targets[i, topk].sum() / k)
    return float(np.mean(vals))


def per_label_threshold_f1(val_logits, val_targets, test_logits, test_targets):
    """Tune one threshold per label on validation, apply to test."""
    def sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
    vp, tp = sig(val_logits), sig(test_logits)
    n_labels = vp.shape[1]
    thresholds = np.full(n_labels, 0.5)
    grid = np.arange(0.01, 1.0, 0.01)
    for j in range(n_labels):
        y = val_targets[:, j]
        if y.sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y, (vp[:, j] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[j] = best_t
    preds = (tp >= thresholds[None, :]).astype(int)
    out = {}
    for name, cols in SUBSETS.items():
        out[name] = float(f1_score(test_targets[:, cols], preds[:, cols],
                                   average="macro", zero_division=0))
    return out


MODELS = [
    ("rgcn_s42", "rgcn", "outputs/checkpoints/best_rgcn_seed42.pt", 512),
    ("rgcn_s43", "rgcn", "outputs/checkpoints/best_rgcn_seed43.pt", 512),
    ("rgcn_s44", "rgcn", "outputs/checkpoints/best_rgcn_seed44.pt", 512),
    ("jusdef_s42", "jusdef", "outputs/checkpoints/jusdef_full_s42.pt", 512),
    ("jusdef_s43", "jusdef", "outputs/checkpoints/jusdef_full_s43.pt", 512),
    ("jusdef_s44", "jusdef", "outputs/checkpoints/jusdef_full_s44.pt", 512),
]

results = {}
for split in ["validation", "test"]:
    print(f"\nLoading {split} graphs...", flush=True)
    graphs = torch.load(f"data/processed/graphs/{split}_graphs.pt", map_location="cpu")
    print(f"  {len(graphs)} graphs", flush=True)
    for tag, kind, ckpt, hd in MODELS:
        out_log = f"outputs/scores/{tag}_{split}_logits.npy"
        if os.path.exists(out_log):
            print(f"  SKIP {tag} {split} (exists)", flush=True)
            continue
        if not os.path.exists(ckpt):
            print(f"  SKIP {tag}: no checkpoint", flush=True)
            continue
        print(f"  scoring {tag} on {split}...", flush=True)
        t0 = time.time()
        logits, targets = (score_rgcn(ckpt, graphs) if kind == "rgcn"
                           else score_jusdef(ckpt, hd, graphs))
        np.save(out_log, logits)
        tpath = f"outputs/scores/targets_{split}.npy"
        if not os.path.exists(tpath):
            np.save(tpath, targets)
        print(f"    done in {time.time()-t0:.0f}s", flush=True)
    del graphs

print("\nComputing metrics...", flush=True)
tt = np.load("outputs/scores/targets_test.npy")
vt = np.load("outputs/scores/targets_validation.npy")
for tag, kind, ckpt, hd in MODELS:
    lp = f"outputs/scores/{tag}_test_logits.npy"
    if not os.path.exists(lp):
        continue
    tl = np.load(lp)
    vl = np.load(f"outputs/scores/{tag}_validation_logits.npy")
    r = {"auprc": {name: auprc_subset(tl, tt, cols) for name, cols in SUBSETS.items()},
         "r_precision": r_precision(tl, tt),
         "per_label_thr_f1": per_label_threshold_f1(vl, vt, tl, tt)}
    results[tag] = r
    print(f"\n{tag}:", flush=True)
    print(f"  AUPRC       : " + " ".join(f"{k}={v:.4f}" for k, v in r["auprc"].items()), flush=True)
    print(f"  R-precision : {r['r_precision']:.4f}", flush=True)
    print(f"  perlab-F1   : " + " ".join(f"{k}={v:.4f}" for k, v in r["per_label_thr_f1"].items()), flush=True)

with open("outputs/logs/ranking_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved outputs/logs/ranking_metrics.json", flush=True)

print("\n=== Mean over seeds ===", flush=True)
for fam in ["rgcn", "jusdef"]:
    tags = [t for t in results if t.startswith(fam)]
    if not tags:
        continue
    print(f"{fam} (n={len(tags)}):", flush=True)
    for name in SUBSETS:
        vals = [results[t]["auprc"][name] for t in tags]
        print(f"  AUPRC {name}: {np.mean(vals):.4f} +- {np.std(vals):.4f}", flush=True)
    vals = [results[t]["r_precision"] for t in tags]
    print(f"  R-precision: {np.mean(vals):.4f} +- {np.std(vals):.4f}", flush=True)
    for name in SUBSETS:
        vals = [results[t]["per_label_thr_f1"][name] for t in tags]
        print(f"  perlab-F1 {name}: {np.mean(vals):.4f} +- {np.std(vals):.4f}", flush=True)
