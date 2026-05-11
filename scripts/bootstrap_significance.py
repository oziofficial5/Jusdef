"""
Paired bootstrap significance tests for JusDef vs R-GCN comparisons.

Computes p-values for:
  1. JusDef (full) vs R-GCN on F1(Y_exc)
  2. JusDef (full) vs JusDef (-DMP) on Macro-F1
  3. JusDef (full) vs JusDef (-Authority) on Macro-F1
"""
import os, sys, json, torch, numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import f1_score
from src.model.baselines import RGCN, tune_threshold
from src.model.jusdef import JusDef
from src.train.trainer import forward_one_graph

# ───────── Data ─────────
print('Loading test graphs...', flush=True)
exc_data = json.load(open('data/annotations/exception_labels.json'))
exc_idx = exc_data['exception_override_labels']
seen = list(range(80))
unseen = list(range(80, 100))
test_graphs = torch.load('data/processed/graphs/test_graphs.pt', map_location='cpu')
print(f'Test graphs: {len(test_graphs)}', flush=True)


def get_predictions(model_loader_fn, ckpt, hd=512, dmp=True, auth=True, is_rgcn=False):
    """Returns (probs, targets) over the test set."""
    if is_rgcn:
        model = RGCN(in_dim=768, hidden_dim=hd)
    else:
        model = JusDef(in_dim=768, hidden_dim=hd, num_layers=2,
                       use_dmp=dmp, use_authority=auth)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for g in test_graphs:
            if is_rgcn:
                h = model(g.x_dict, g.edge_index_dict)
                doc_emb = h['sec'].mean(dim=0, keepdim=True)
                logits = model.score(doc_emb, h['label']).squeeze(0)
            else:
                logits, _, _ = forward_one_graph(model, g, 'cpu')
                logits = logits.cpu().view(-1)
            all_logits.append(logits.cpu() if is_rgcn else logits)
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
    return f1_score(targets[:, label_idx], preds[:, label_idx],
                    average=average, zero_division=0)


def paired_bootstrap(preds_A, preds_B, targets, label_idx, n=10000, seed=42):
    """
    H0: model A and model B have equal F1 on the given label subset.
    Returns (mean_diff, p_value, ci_low, ci_high).
    
    p_value is two-sided.
    """
    rng = np.random.default_rng(seed)
    n_docs = targets.shape[0]
    diffs = np.empty(n)

    for i in range(n):
        idx = rng.integers(0, n_docs, size=n_docs)  # resample documents
        f_A = f1_subset(targets[idx], preds_A[idx], label_idx)
        f_B = f1_subset(targets[idx], preds_B[idx], label_idx)
        diffs[i] = f_A - f_B

    mean_diff = diffs.mean()
    # Two-sided p-value via bootstrap CI
    if mean_diff > 0:
        p = 2 * (diffs <= 0).mean()
    else:
        p = 2 * (diffs >= 0).mean()
    p = min(p, 1.0)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return mean_diff, p, ci_low, ci_high


# ───────── Get predictions for all models ─────────
print('\nLoading R-GCN seed 42...', flush=True)
probs_rgcn, targets = get_predictions(None, 'outputs/checkpoints/best_rgcn_seed42.pt', hd=512, is_rgcn=True)
preds_rgcn, t_rgcn = predict_with_threshold(probs_rgcn, targets)
print(f'  R-GCN threshold: {t_rgcn:.2f}', flush=True)

print('\nLoading JusDef (full) seed 42...', flush=True)
probs_jusdef, _ = get_predictions(None, 'outputs/checkpoints/jusdef_full_s42.pt', hd=512, dmp=True, auth=True)
preds_jusdef, t_jusdef = predict_with_threshold(probs_jusdef, targets)
print(f'  JusDef threshold: {t_jusdef:.2f}', flush=True)

print('\nLoading JusDef (-DMP) seed 42...', flush=True)
probs_nodmp, _ = get_predictions(None, 'outputs/checkpoints/jusdef_no_dmp_s42.pt', hd=512, dmp=False, auth=True)
preds_nodmp, t_nodmp = predict_with_threshold(probs_nodmp, targets)
print(f'  -DMP threshold: {t_nodmp:.2f}', flush=True)

print('\nLoading JusDef (-Authority) seed 42...', flush=True)
probs_noauth, _ = get_predictions(None, 'outputs/checkpoints/jusdef_no_auth_s42.pt', hd=512, dmp=True, auth=False)
preds_noauth, t_noauth = predict_with_threshold(probs_noauth, targets)
print(f'  -Auth threshold: {t_noauth:.2f}', flush=True)

# ───────── Run bootstrap tests ─────────
print('\n' + '=' * 70, flush=True)
print('PAIRED BOOTSTRAP TESTS (n=10,000, seed=42)', flush=True)
print('=' * 70, flush=True)
all_idx = list(range(100))

tests = [
    ('JusDef (full) vs R-GCN',     'F1(Y_exc)', preds_jusdef, preds_rgcn,    exc_idx),
    ('JusDef (full) vs R-GCN',     'F1(Y_s)',   preds_jusdef, preds_rgcn,    seen),
    ('JusDef (full) vs R-GCN',     'Macro-F1',  preds_jusdef, preds_rgcn,    all_idx),
    ('JusDef (full) vs -DMP',      'Macro-F1',  preds_jusdef, preds_nodmp,   all_idx),
    ('JusDef (full) vs -DMP',      'F1(Y_exc)', preds_jusdef, preds_nodmp,   exc_idx),
    ('JusDef (full) vs -Authority','Macro-F1',  preds_jusdef, preds_noauth,  all_idx),
    ('JusDef (full) vs -Authority','F1(Y_exc)', preds_jusdef, preds_noauth,  exc_idx),
]

results = []
for name, metric, pA, pB, idx in tests:
    diff, p, lo, hi = paired_bootstrap(pA, pB, targets, idx)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f'\n{name:35s} on {metric}', flush=True)
    print(f'  Mean diff:   {diff:+.4f}', flush=True)
    print(f'  95% CI:      [{lo:+.4f}, {hi:+.4f}]', flush=True)
    print(f'  p-value:     {p:.4f}  [{sig}]', flush=True)
    results.append({
        'comparison': name, 'metric': metric,
        'mean_diff': float(diff), 'p_value': float(p),
        'ci_low': float(lo), 'ci_high': float(hi),
        'significance': sig,
    })

with open('outputs/logs/bootstrap_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '=' * 70, flush=True)
print('Saved to outputs/logs/bootstrap_results.json', flush=True)
print('Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant', flush=True)