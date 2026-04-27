"""Detailed evaluation of R-GCN checkpoints on test set."""
import os, sys, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import f1_score
from src.model.baselines import RGCN, tune_threshold

print('Loading...', flush=True)

exc_data = json.load(open('data/annotations/exception_labels.json'))
exc_idx = exc_data['exception_override_labels']
seen = list(range(80))
unseen = list(range(80, 100))

test_graphs = torch.load('data/processed/graphs/test_graphs.pt', map_location='cpu')
print(f'Loaded {len(test_graphs)} test graphs', flush=True)

results = []
for s in [42, 43, 44]:
    ckpt = f'outputs/checkpoints/best_rgcn_seed{s}.pt'
    if not os.path.exists(ckpt):
        print(f'SKIP seed {s}: not found', flush=True)
        continue
    
    print(f'\nEvaluating R-GCN seed {s}...', flush=True)
    model = RGCN(in_dim=768, hidden_dim=512)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    
    all_logits, all_targets = [], []
    with torch.no_grad():
        for i, g in enumerate(test_graphs):
            if i % 1000 == 0:
                print(f'  {i}/{len(test_graphs)}', flush=True)
            h = model(g.x_dict, g.edge_index_dict)
            doc_emb = h['sec'].mean(dim=0, keepdim=True)
            logits = model.score(doc_emb, h['label']).squeeze(0)
            all_logits.append(logits.cpu())
            all_targets.append(g.y.cpu())
    
    logits = torch.stack(all_logits).numpy()
    targets = torch.stack(all_targets).numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    best_t, _ = tune_threshold(probs, targets)
    preds = (probs >= best_t).astype(int)
    
    r = {
        'seed': s,
        'macro': float(f1_score(targets, preds, average='macro', zero_division=0)),
        'micro': float(f1_score(targets, preds, average='micro', zero_division=0)),
        'seen': float(f1_score(targets[:,seen], preds[:,seen], average='macro', zero_division=0)),
        'unseen': float(f1_score(targets[:,unseen], preds[:,unseen], average='macro', zero_division=0)),
        'exc': float(f1_score(targets[:,exc_idx], preds[:,exc_idx], average='macro', zero_division=0)),
        'threshold': float(best_t),
    }
    results.append(r)
    print(f'  macro={r["macro"]:.4f} micro={r["micro"]:.4f} seen={r["seen"]:.4f} unseen={r["unseen"]:.4f} exc={r["exc"]:.4f} t={r["threshold"]:.2f}', flush=True)

print('\n=== Mean ± Std (R-GCN, 3 seeds) ===', flush=True)
for m in ['macro', 'micro', 'seen', 'unseen', 'exc']:
    vals = [r[m] for r in results]
    print(f'  {m}: {np.mean(vals):.4f} ± {np.std(vals):.4f}', flush=True)

with open('outputs/logs/rgcn_detailed_eval.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to outputs/logs/rgcn_detailed_eval.json', flush=True)
