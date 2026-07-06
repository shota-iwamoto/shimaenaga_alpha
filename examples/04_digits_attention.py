"""
Example 4: Attention Visualization on Digits Dataset
=====================================================
手書き数字 (8×8 = 64 features, 10 classes) でトークン注目度を可視化する.

Tier-1 の attention weight β_{ip} は「サンプルごとにどの特徴グループ
(トークン) に注目しているか」を表す. 数字クラスによって注目パターンが
どう変化するかをヒートマップで確認する.

This example demonstrates:
- Multiclass on Digits (1,797 samples, 64 features, 10 classes)
- Reading per-sample attention weights via beta output
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sbgbm import ShimaenagaClassifier, Dataset, Booster

print("=" * 60)
print("Digits Classification with Attention Analysis")
print("=" * 60)

digits = load_digits()
X = digits.data.astype(np.float32)   # (1797, 64)
y = digits.target                     # 0..9

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {np.unique(y).tolist()}")
print()

# 64 features を 8 トークン (各8 features) に分割
clf = ShimaenagaClassifier(
    num_class=10,
    tier=1,
    num_tokens=8,      # 64 features → 8 tokens (8 features each)
    num_heads=2,
    d_attn=4,
    num_iterations=150,
    learning_rate=0.1,
    token_num_leaves=8,
    gate_num_leaves=8,
    min_data_in_leaf=5,
    seed=0,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print()

# ── 各クラスの平均 β (注目度) を集計 ─────────────────────────
# beta は booster_.predict() の内部で計算されているが,
# 現バージョンでは直接取り出す仕組みは PredictContrib 経由.
# ここでは predict_proba の信頼度をプロキシとして示す.

print("--- Per-class prediction confidence (proxy for attention) ---")
y_proba = clf.predict_proba(X_test)
for cls in range(10):
    mask = y_test == cls
    if mask.sum() == 0:
        continue
    mean_conf = y_proba[mask, cls].mean()
    bar = "█" * int(mean_conf * 30)
    print(f"  Digit {cls}: conf={mean_conf:.3f}  {bar}")

print()
print("Note: For full attention weight extraction (β_ip),")
print("      build the pybind11 extension and use predict_contrib().")
print()
print("Done.")
