"""Leave-one-label-out sensitivity of the Y_exc gap (ANNPR camera-ready, R3 major 6).

Requires score matrices dumped by eval_ranking_metrics.py in outputs/scores/.
Replicates the paper's evaluation protocol (global threshold per model/seed),
then recomputes the JusDef - R-GCN macro-F1 gap on Y_exc with each of the 21
labels removed in turn.
"""
import json
import numpy as np
from sklearn.metrics import f1_score

exc = json.load(open("data/annotations/exception_labels.json"))["exception_override_labels"]
tt = np.load("outputs/scores/targets_test.npy")


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def tune(probs, targets):
    best_t, best = 0.5, -1.0
    for t in np.arange(0.02, 0.5, 0.02):
        f = f1_score(targets, (probs >= t).astype(int), average="macro", zero_division=0)
        if f > best:
            best, best_t = f, t
    return best_t


deltas_full, deltas_loo = [], {}
for s in [42, 43, 44]:
    jp = sig(np.load(f"outputs/scores/jusdef_s{s}_test_logits.npy"))
    rp = sig(np.load(f"outputs/scores/rgcn_s{s}_test_logits.npy"))
    jpred = (jp >= tune(jp, tt)).astype(int)
    rpred = (rp >= tune(rp, tt)).astype(int)
    f_j = f1_score(tt[:, exc], jpred[:, exc], average="macro", zero_division=0)
    f_r = f1_score(tt[:, exc], rpred[:, exc], average="macro", zero_division=0)
    deltas_full.append(f_j - f_r)
    for drop in exc:
        cols = [c for c in exc if c != drop]
        d = (f1_score(tt[:, cols], jpred[:, cols], average="macro", zero_division=0)
             - f1_score(tt[:, cols], rpred[:, cols], average="macro", zero_division=0))
        deltas_loo.setdefault(drop, []).append(d)

print("full-set Delta exc per seed:", [round(d, 4) for d in deltas_full],
      "mean", round(float(np.mean(deltas_full)), 4))
loo_means = {k: float(np.mean(v)) for k, v in deltas_loo.items()}
vals = np.array(list(loo_means.values()))
print(f"leave-one-out Delta exc: min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f}")
print(f"all {len(vals)} jackknife deltas positive: {bool((vals > 0).all())}")

with open("outputs/logs/yexc_jackknife.json", "w") as f:
    json.dump({"full_set_per_seed": deltas_full, "loo_mean_per_dropped_label": loo_means}, f, indent=2)
print("saved outputs/logs/yexc_jackknife.json")
