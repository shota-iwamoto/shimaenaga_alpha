"""
Example 3: Cross-Validation & Hyperparameter Search
=====================================================
sklearn の cross_val_score / GridSearchCV と組み合わせる方法を示す.
Wine dataset (178 samples, 13 features, 3 classes) を使用.

This example demonstrates:
- 5-fold cross-validation with ShimaenagaClassifier
- Simple grid search over tier / num_tokens
- Feature importance proxy via permutation importance
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import sys, os, time

sys.path.insert(0, os.path.dirname(__file__))
from sbgbm import ShimaenagaClassifier

print("=" * 60)
print("Cross-Validation & Hyperparameter Search: Wine dataset")
print("=" * 60)

wine = load_wine()
X, y = wine.data.astype(np.float32), wine.target
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
print()

# ─── 5-fold Cross-Validation ──────────────────────────────────
print("--- 5-Fold Cross-Validation (Tier-1) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

clf_cv = ShimaenagaClassifier(
    num_class=3,
    tier=1,
    num_tokens=3,
    num_heads=2,
    d_attn=4,
    num_iterations=100,
    learning_rate=0.1,
    token_num_leaves=8,
    gate_num_leaves=4,
    min_data_in_leaf=3,
    seed=0,
)

scores = []
for fold, (tr, val) in enumerate(cv.split(X, y)):
    clf_cv.set_params(seed=fold)
    clf_cv.fit(X[tr], y[tr])
    acc = accuracy_score(y[val], clf_cv.predict(X[val]))
    scores.append(acc)
    print(f"  Fold {fold+1}: {acc:.4f}")

print(f"  Mean ± Std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
print()

# ─── Simple Grid Search ───────────────────────────────────────
print("--- Grid Search: tier × num_tokens ---")
grid = [
    dict(tier=0, num_tokens=1, num_heads=1),
    dict(tier=1, num_tokens=2, num_heads=1),
    dict(tier=1, num_tokens=3, num_heads=2),
    dict(tier=1, num_tokens=4, num_heads=2),
]

best_score, best_cfg = 0.0, None
for cfg in grid:
    fold_scores = []
    for fold, (tr, val) in enumerate(cv.split(X, y)):
        m = ShimaenagaClassifier(
            num_class=3,
            num_iterations=100,
            learning_rate=0.1,
            token_num_leaves=8,
            gate_num_leaves=4,
            min_data_in_leaf=3,
            seed=fold,
            **cfg,
        )
        m.fit(X[tr], y[tr])
        fold_scores.append(accuracy_score(y[val], m.predict(X[val])))
    mean = np.mean(fold_scores)
    tag = f"tier={cfg['tier']}, P={cfg['num_tokens']}, H={cfg['num_heads']}"
    print(f"  [{tag:30s}]  CV acc = {mean:.4f}")
    if mean > best_score:
        best_score, best_cfg = mean, cfg

print(f"\n  Best config: {best_cfg}  (CV acc = {best_score:.4f})")
print()

# ─── Permutation Importance (proxy) ──────────────────────────
print("--- Permutation Feature Importance (best model) ---")
clf_best = ShimaenagaClassifier(
    num_class=3,
    num_iterations=100,
    learning_rate=0.1,
    token_num_leaves=8,
    gate_num_leaves=4,
    min_data_in_leaf=3,
    seed=42,
    **best_cfg,
)
# Use the last fold's val set as held-out set for permutation
(tr, val) = list(cv.split(X, y))[-1]
clf_best.fit(X[tr], y[tr])
base_acc = accuracy_score(y[val], clf_best.predict(X[val]))

importance = []
X_val = X[val].copy()
for f in range(X.shape[1]):
    orig = X_val[:, f].copy()
    np.random.shuffle(X_val[:, f])
    acc_perm = accuracy_score(y[val], clf_best.predict(X_val))
    importance.append(base_acc - acc_perm)
    X_val[:, f] = orig

# Top-5 features
order = np.argsort(importance)[::-1]
for rank, fi in enumerate(order[:5]):
    print(f"  {rank+1}. {wine.feature_names[fi]:30s}  importance = {importance[fi]:+.4f}")

print("\nDone.")
