"""
Example 1: Regression with California Housing dataset
======================================================
California Housing: 20,640 samples, 8 numeric features
Target: median house value (×100,000 USD)

This example demonstrates:
- ShimaenagaRegressor (Tier-0 pure GBDT and Tier-1 with attention)
- Train/test split and RMSE evaluation
- Model save/load
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from sbgbm import ShimaenagaRegressor

print("=" * 60)
print("California Housing Regression Demo")
print("=" * 60)

# ── データロード ───────────────────────────────────────────────
data = fetch_california_housing()
X, y = data.data.astype(np.float32), data.target.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Features: {data.feature_names}")
print()

# ── Tier-0: pure GBDT (LightGBM 互換) ─────────────────────────
print("--- Tier-0: Pure GBDT (no attention) ---")
model0 = ShimaenagaRegressor(
    tier=0,
    num_iterations=200,
    learning_rate=0.1,
    token_num_leaves=31,
    min_data_in_leaf=20,
    lambda_v=1.0,
    seed=42,
)
model0.fit(X_train, y_train)
pred0 = model0.predict(X_test)
rmse0 = root_mean_squared_error(y_test, pred0)
print(f"  Test RMSE: {rmse0:.4f}")

# ── Tier-1: Attentive Readout ─────────────────────────────────
print()
print("--- Tier-1: Attentive Readout ---")
model1 = ShimaenagaRegressor(
    tier=1,
    num_tokens=4,        # 4 feature-group tokens
    num_heads=2,         # 2 attention heads
    d_attn=4,            # attention dimension
    num_iterations=200,
    learning_rate=0.1,
    token_num_leaves=16,
    gate_num_leaves=8,
    inner_refit_steps=2,
    min_data_in_leaf=20,
    lambda_v=1.0,
    lambda_q=0.1,
    lambda_k=0.1,
    seed=42,
)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
rmse1 = root_mean_squared_error(y_test, pred1)
print(f"  Test RMSE: {rmse1:.4f}")

print()
print(f"Improvement: {(rmse0 - rmse1) / rmse0 * 100:.1f}% (Tier-1 vs Tier-0)")

# ── モデル保存・ロード ─────────────────────────────────────────
print()
print("--- Model Save / Load ---")
model1.save_model("/tmp/california_tier1.sbb")
print("  Saved: /tmp/california_tier1.sbb")

model_loaded = ShimaenagaRegressor(
    tier=1, num_tokens=4, num_heads=2, d_attn=4,
    num_iterations=1, learning_rate=0.1,
    token_num_leaves=16, gate_num_leaves=8,
    min_data_in_leaf=20, seed=42,
)
# load_model needs an initialised booster first; this throwaway fit only
# allocates it (load_model overwrites the config + blocks from the file).
# Use >= min_data_in_leaf samples and a single iteration so the trees can
# actually split — a degenerate 10-sample fit would log a frozen metric.
model_loaded.fit(X_train[:200], y_train[:200])
model_loaded.load_model("/tmp/california_tier1.sbb")
pred_loaded = model_loaded.predict(X_test)
rmse_loaded = root_mean_squared_error(y_test, pred_loaded)
print(f"  Loaded model RMSE: {rmse_loaded:.4f} (should match {rmse1:.4f})")

print()
print("Done.")
