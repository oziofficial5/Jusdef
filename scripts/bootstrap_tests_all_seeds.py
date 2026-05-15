"""
Paired bootstrap significance tests for JusDef vs R-GCN comparisons across all seeds.

Computes per-seed p-values for:
  1. JusDef (full) vs R-GCN on F1(Y_exc)
  2. JusDef (full) vs R-GCN on F1(Y_s)
  3. JusDef (full) vs R-GCN on Macro-F1
  4. JusDef (full) vs JusDef (-DMP) on Macro-F1
  5. JusDef (full) vs JusDef (-DMP) on F1(Y_exc)
  6. JusDef (full) vs JusDef (-Authority) on Macro-F1
  7. JusDef (full) vs JusDef (-Authority) on F1(Y_exc)

Saves detailed per-seed results to outputs/logs/bootstrap_results_all_seeds.json
"""
import os
import sys
import json
import warnings

import torch
import numpy as np
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.baselines import RGCN, tune_threshold
from src.model.jusdef import JusDef
from src.train.trainer import forward_one_graph


print('Loading test graphs...', flush=True)
exc_data = json.load(open('data/annotations/exception_labels.json'))
exc_idx = exc_data['exception_override_labels']
seen = list(range(80))
unseen = list(range(80, 100))
all_idx = list(range(100))
test_graphs = torch.load('data/processed/graphs/test_graphs.pt', map_location='cpu')
print(f'Test graphs: {len(test_graphs)}', flush=True)


def get_predictions(ckpt, hd=512, dmp=True, auth=True, is_rgcn=False):
    """Returns (probs, targets) over the test set."""
    if is_rgcn:
        model = RGCN(in_dim=768, hidden_dim=hd)
    else:
        model = JusDef(
            in_dim=768,
            hidden_dim=hd,
            num_layers=2,
            use_dmp=dmp,
            use_authority=auth,
        )

    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for i, g in enumerate(test_graphs):
            if i % 1000 == 0:
                print(f'  progress: {i}/{len(test_graphs)}', flush=True)
            if is_rgcn:
                h = model(g.x_dict, g.edge_index_dict)
                doc_emb = h['sec'].mean(dim=0, keepdim=True)
                logits = model.score(doc_emb, h['label']).squeeze(0)
            else:
                logits, _, _ = forward_one_graph(model, g, 'cpu')
                logits = logits.cpu().view(-1)
            all_logits.append(logits.cpu())
            all_targets.append(g.y.cpu())

    logits = torch.stack(all_logits).numpy()
    targets = torch.stack(all_targets).numpy()
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    return probs, targets


def predict_with_threshold(probs, targets):
    best_t, _ = tune_threshold(probs, targets)
    preds = (probs >= best_t).astype(int)
    return preds, best_t


def f1_subset(targets, preds, label_idx, average='macro'):
    return f1_score(
        targets[:, label_idx],
        preds[:, label_idx],
        average=average,
        zero_division=0,
    )


def paired_bootstrap(preds_A, preds_B, targets, label_idx, n=10000, seed=42):
    """
    H0: model A and model B have equal F1 on the given label subset.
    Returns (mean_diff, p_value, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    n_docs = targets.shape[0]
    diffs = np.empty(n)

    for i in range(n):
        idx = rng.integers(0, n_docs, size=n_docs)
        f_A = f1_subset(targets[idx], preds_A[idx], label_idx)
        f_B = f1_subset(targets[idx], preds_B[idx], label_idx)
        diffs[i] = f_A - f_B

    mean_diff = diffs.mean()
    if mean_diff > 0:
        p = 2 * (diffs <= 0).mean()
    else:
        p = 2 * (diffs >= 0).mean()
    p = min(p, 1.0)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return mean_diff, p, ci_low, ci_high


def metric_value(targets, preds, metric_name, idx):
    if metric_name == 'Macro-F1':
        return f1_score(targets, preds, average='macro', zero_division=0)
    return f1_subset(targets, preds, idx)


seeds = [42, 43, 44]
all_results = []

for seed in seeds:
    print('\n' + '=' * 80, flush=True)
    print(f'SEED {seed}', flush=True)
    print('=' * 80, flush=True)

    ckpt_rgcn = f'outputs/checkpoints/best_rgcn_seed{seed}.pt'
    ckpt_full = f'outputs/checkpoints/jusdef_full_s{seed}.pt'
    ckpt_nodmp = f'outputs/checkpoints/jusdef_no_dmp_s{seed}.pt'
    ckpt_noauth = f'outputs/checkpoints/jusdef_no_auth_s{seed}.pt'

    needed = [ckpt_rgcn, ckpt_full, ckpt_nodmp, ckpt_noauth]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        print('Missing checkpoints:', flush=True)
        for m in missing:
            print(f'  {m}', flush=True)
        print(f'SKIP seed {seed}', flush=True)
        continue

    print(f'\nLoading R-GCN seed {seed}...', flush=True)
    probs_rgcn, targets = get_predictions(ckpt_rgcn, hd=512, is_rgcn=True)
    preds_rgcn, t_rgcn = predict_with_threshold(probs_rgcn, targets)
    print(f'  R-GCN threshold: {t_rgcn:.2f}', flush=True)

    print(f'\nLoading JusDef (full) seed {seed}...', flush=True)
    probs_full, _ = get_predictions(ckpt_full, hd=512, dmp=True, auth=True)
    preds_full, t_full = predict_with_threshold(probs_full, targets)
    print(f'  JusDef threshold: {t_full:.2f}', flush=True)

    print(f'\nLoading JusDef (-DMP) seed {seed}...', flush=True)
    probs_nodmp, _ = get_predictions(ckpt_nodmp, hd=512, dmp=False, auth=True)
    preds_nodmp, t_nodmp = predict_with_threshold(probs_nodmp, targets)
    print(f'  -DMP threshold: {t_nodmp:.2f}', flush=True)

    print(f'\nLoading JusDef (-Authority) seed {seed}...', flush=True)
    probs_noauth, _ = get_predictions(ckpt_noauth, hd=512, dmp=True, auth=False)
    preds_noauth, t_noauth = predict_with_threshold(probs_noauth, targets)
    print(f'  -Authority threshold: {t_noauth:.2f}', flush=True)

    tests = [
        ('JusDef (full) vs R-GCN',      'F1(Y_exc)', exc_idx, preds_full, preds_rgcn),
        ('JusDef (full) vs R-GCN',      'F1(Y_s)',   seen,    preds_full, preds_rgcn),
        ('JusDef (full) vs R-GCN',      'Macro-F1',  all_idx, preds_full, preds_rgcn),
        ('JusDef (full) vs -DMP',       'Macro-F1',  all_idx, preds_full, preds_nodmp),
        ('JusDef (full) vs -DMP',       'F1(Y_exc)', exc_idx, preds_full, preds_nodmp),
        ('JusDef (full) vs -Authority', 'Macro-F1',  all_idx, preds_full, preds_noauth),
        ('JusDef (full) vs -Authority', 'F1(Y_exc)', exc_idx, preds_full, preds_noauth),
    ]

    print('\n' + '-' * 80, flush=True)
    print(f'PAIRED BOOTSTRAP TESTS (n=10,000, bootstrap seed={seed})', flush=True)
    print('-' * 80, flush=True)

    for name, metric, idx, preds_A, preds_B in tests:
        diff, p, lo, hi = paired_bootstrap(preds_A, preds_B, targets, idx, n=10000, seed=seed)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        score_A = metric_value(targets, preds_A, metric, idx)
        score_B = metric_value(targets, preds_B, metric, idx)

        print(f'\n{name:35s} on {metric}', flush=True)
        print(f'  Score A:     {score_A:.4f}', flush=True)
        print(f'  Score B:     {score_B:.4f}', flush=True)
        print(f'  Mean diff:   {diff:+.4f}', flush=True)
        print(f'  95% CI:      [{lo:+.4f}, {hi:+.4f}]', flush=True)
        print(f'  p-value:     {p:.4f}  [{sig}]', flush=True)

        all_results.append({
            'seed': seed,
            'comparison': name,
            'metric': metric,
            'score_A': float(score_A),
            'score_B': float(score_B),
            'mean_diff': float(diff),
            'p_value': float(p),
            'ci_low': float(lo),
            'ci_high': float(hi),
            'significance': sig,
            'thresholds': {
                'rgcn': float(t_rgcn),
                'full': float(t_full),
                'no_dmp': float(t_nodmp),
                'no_authority': float(t_noauth),
            }
        })

out_path = 'outputs/logs/bootstrap_results_all_seeds.json'
os.makedirs('outputs/logs', exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print('\n' + '=' * 80, flush=True)
print(f'Saved per-seed results to {out_path}', flush=True)
print('Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant', flush=True)
