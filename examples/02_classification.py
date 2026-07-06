"""
Example 2: Classification — Binary and Multiclass
==================================================
Binary:     Breast Cancer Wisconsin (569 samples, 30 features)
Multiclass: Iris (150 samples, 4 features, 3 classes)

This example demonstrates:
- ShimaenagaClassifier for binary and multiclass tasks
- predict() and predict_proba()
- Accuracy / log-loss metrics
"""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sbgbm import ShimaenagaClassifier

# ─────────────────────────────────────────────────────────────
# 1. Binary Classification — Breast Cancer
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Binary Classification: Breast Cancer Wisconsin")
print("=" * 60)

bc = load_breast_cancer()
X, y = bc.data.astype(np.float32), bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {bc.target_names}  (0=malignant, 1=benign)")
print()

clf = ShimaenagaClassifier(
    num_class=1,       # binary
    tier=1,
    num_tokens=4,
    num_heads=2,
    d_attn=4,
    num_iterations=150,
    learning_rate=0.05,
    token_num_leaves=16,
    gate_num_leaves=8,
    min_data_in_leaf=5,
    lambda_v=1.0,
    seed=0,
)
clf.fit(X_train, y_train)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba[:, 1])
ll  = log_loss(y_test, y_proba)

print(f"  Accuracy : {acc:.4f}")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  Log-loss : {ll:.4f}")
print()

# ─────────────────────────────────────────────────────────────
# 2. Multiclass Classification — Iris
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Multiclass Classification: Iris (3 classes)")
print("=" * 60)

iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {iris.target_names.tolist()}")
print()

clf3 = ShimaenagaClassifier(
    num_class=3,       # 3-class softmax
    tier=1,
    num_tokens=2,      # 4 features → 2 tokens
    num_heads=1,
    d_attn=4,
    num_iterations=100,
    learning_rate=0.1,
    token_num_leaves=8,
    gate_num_leaves=4,
    min_data_in_leaf=3,
    lambda_v=1.0,
    seed=42,
)
clf3.fit(X_train, y_train)

y_pred3  = clf3.predict(X_test)
y_proba3 = clf3.predict_proba(X_test)

acc3 = accuracy_score(y_test, y_pred3)
ll3  = log_loss(y_test, y_proba3)

print(f"  Accuracy : {acc3:.4f}")
print(f"  Log-loss : {ll3:.4f}")
print()

# class-level breakdown
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred3, target_names=iris.target_names))

print("Done.")
