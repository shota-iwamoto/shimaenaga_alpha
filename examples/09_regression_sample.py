"""
Example 9: Regression with Synthetic Data
==========================================
合成データ (非線形・交互作用あり) を使った回帰デモ。

このサンプルで示す内容:
- 合成データ生成 (sklearn 不要)
- Tier-0 (純粋 GBDT) と Tier-1 (Attentive Readout) の比較
- 複数評価指標: RMSE / MAE / R²
- eval_set を使った早期停止
- モデル保存・ロード
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sbgbm import ShimaenagaRegressor

SEED = 42
rng = np.random.default_rng(SEED)


# ── 合成データ生成 ─────────────────────────────────────────────────────────────
def make_dataset(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """非線形・特徴量交互作用を含む回帰データセットを生成する。

    y = sin(2πx0) * exp(-x1) + 0.5 * (x2 * x3) + 0.3 * x4^2 + noise
    """
    rng_local = np.random.default_rng(seed)
    X = rng_local.uniform(-1.0, 1.0, size=(n, 8)).astype(np.float32)
    y = (
        np.sin(2 * np.pi * X[:, 0]) * np.exp(-np.abs(X[:, 1]))
        + 0.5 * X[:, 2] * X[:, 3]
        + 0.3 * X[:, 4] ** 2
        + 0.1 * rng_local.standard_normal(n)
    ).astype(np.float32)
    return X, y


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot)

def print_metrics(label: str, y_true, y_pred):
    print(f"  {label:<30} RMSE={rmse(y_true, y_pred):.4f}  "
          f"MAE={mae(y_true, y_pred):.4f}  R²={r2(y_true, y_pred):.4f}")


# ── データ準備 ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("Shimaenaga — Regression Sample (Synthetic Data)")
print("=" * 65)

X_train, y_train = make_dataset(n=4000, seed=0)
X_val,   y_val   = make_dataset(n=1000, seed=1)
X_test,  y_test  = make_dataset(n=1000, seed=2)

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]  "
      f"mean={y_train.mean():.3f}  std={y_train.std():.3f}")
print()


# ── Tier-0: 純粋 GBDT ────────────────────────────────────────────────────────
print("─" * 65)
print("Tier-0: Pure GBDT (no attention)")
print("─" * 65)

model0 = ShimaenagaRegressor(
    tier=0,
    num_iterations=300,
    learning_rate=0.05,
    token_num_leaves=31,
    min_data_in_leaf=20,
    lambda_v=1.0,
    early_stopping_rounds=20,
    seed=SEED,
)
model0.fit(X_train, y_train, eval_set=[(X_val, y_val)])
pred0_train = model0.predict(X_train)
pred0_test  = model0.predict(X_test)

print(f"  Best iteration: {model0.best_iteration_}")
print_metrics("Train", y_train, pred0_train)
print_metrics("Test",  y_test,  pred0_test)
print()


# ── Tier-1: Attentive Readout ──────────────────────────────────────────────
print("─" * 65)
print("Tier-1: Attentive Readout")
print("─" * 65)

model1 = ShimaenagaRegressor(
    tier=1,
    num_tokens=4,
    num_heads=2,
    d_attn=4,
    num_iterations=300,
    learning_rate=0.05,
    token_num_leaves=16,
    gate_num_leaves=8,
    inner_refit_steps=2,
    min_data_in_leaf=20,
    lambda_v=1.0,
    lambda_q=0.1,
    lambda_k=0.1,
    early_stopping_rounds=20,
    seed=SEED,
)
model1.fit(X_train, y_train, eval_set=[(X_val, y_val)])
pred1_train = model1.predict(X_train)
pred1_test  = model1.predict(X_test)

print(f"  Best iteration: {model1.best_iteration_}")
print_metrics("Train", y_train, pred1_train)
print_metrics("Test",  y_test,  pred1_test)
print()


# ── 比較サマリー ──────────────────────────────────────────────────────────────
print("─" * 65)
print("Summary (Test set)")
print("─" * 65)
r_tier0 = rmse(y_test, pred0_test)
r_tier1 = rmse(y_test, pred1_test)
print_metrics("Tier-0 (GBDT)", y_test, pred0_test)
print_metrics("Tier-1 (Attentive)", y_test, pred1_test)
improvement = (r_tier0 - r_tier1) / r_tier0 * 100
print(f"\n  RMSE improvement (Tier-1 vs Tier-0): {improvement:+.1f}%")
print()


# ── モデル保存・ロード ────────────────────────────────────────────────────────
print("─" * 65)
print("Model Save / Load")
print("─" * 65)
save_path = "/tmp/synthetic_regression_tier1.sbb"
model1.save_model(save_path)
print(f"  Saved: {save_path}")

model_loaded = ShimaenagaRegressor(
    tier=1, num_tokens=4, num_heads=2, d_attn=4,
    num_iterations=300, learning_rate=0.05,
    token_num_leaves=16, gate_num_leaves=8,
    min_data_in_leaf=20, seed=SEED,
)
model_loaded.fit(X_train[:10], y_train[:10])
model_loaded.load_model(save_path)
pred_loaded = model_loaded.predict(X_test)

print_metrics("Loaded model (Test)", y_test, pred_loaded)
match = np.allclose(pred1_test, pred_loaded, atol=1e-5)
print(f"  Predictions match original: {match}")
print()
print("Done.")
