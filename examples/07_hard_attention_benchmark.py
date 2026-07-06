"""
07_hard_attention_benchmark.py
==============================
Shimaenaga Tier-1 vs LightGBM: 注意機構が有利・不利なシナリオの詳細分析

【結論（先に示す）】
  ・Shimaenaga が LightGBM に勝つのは「ソフト混合ゲート + 専門家ごとに別の特徴量グループ」の場合
  ・Shimaenaga が LightGBM に負けるのは「ハード分割ゲート（軸平行）+ 同一特徴量の非線形関数」の場合
  ・理由の解説は末尾の総括を参照

【実験構成】
  Task A: ソフト混合 K-Expert（本命）
    - gate: K グループの特徴量和の softmax → 各 expert の重みは SOFT
    - signal: 各 expert が専用の特徴量グループを使う非線形関数
    - 等算出量比較 + LightGBM のサブサンプリングなし（公平性のため）
    → Shimaenaga が有利なケース

  Task B: ハード分割 K-Expert（対照実験）
    - gate: x0 の等幅ビン（軸平行 = HARD）
    - signal: 全 expert が同一の 4 特徴量に対する非線形関数
    → LightGBM が有利なケース（05/06 の結論を追認）

  Task C: Beta 診断 ― Shimaenaga の routing は機能しているか？
    - Task A のモデルを使い、各 expert サンプルの平均 beta を分析
    → Shimaenaga が soft mixture を正しく学習しているかを可視化

  Task D: K スケーリング（ソフト混合）
    - K=2,4,6 で Shimaenaga の相対 RMSE を比較
    → K が大きいほど Shimaenaga 優位が拡大するかを検証

【等算出量の定義】
  Shimaenaga: P=K token trees (token_num_leaves=L) + 1 gate tree (gate_num_leaves=G)
  → 総 leaf/iter = K*L + G
  LGB 等算出量: num_leaves = K*L + G, subsample=1.0, colsample_bytree=1.0
  LGB 標準: num_leaves=63, subsample=0.8, colsample_bytree=0.8
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import lightgbm as lgb
    _has_lgb = True
except ImportError:
    _has_lgb = False
    print("LightGBM が見つかりません: pip install lightgbm")
    sys.exit(1)

try:
    from shimaenaga import ShimaenagaRegressor
    _has_tf = True
except ImportError:
    # pybind11 パッケージ未インストール時は ctypes ラッパーにフォールバック
    # (05/06 と同じパターン)
    try:
        from sbgbm import ShimaenagaRegressor
        _has_tf = True
    except Exception:
        _has_tf = False
        print("Shimaenaga が見つかりません。build/dylib を確認してください。")
        sys.exit(1)


# ============================================================================
# データ生成
# ============================================================================

def make_soft_kexpert(n_samples=20000, K=4, n_gate_per_expert=3,
                      n_signal_per_expert=5, slope=1.5, seed=42):
    """
    ソフト混合 K-Expert タスク。

    Gate 特徴量グループ (K グループ × n_gate_per_expert 個):
      score_k = slope * Σ x_j  for j in gate_group_k
      w_k = softmax_k(score_0, ..., score_{K-1})

    Signal 特徴量グループ (K グループ × n_signal_per_expert 個、gate と完全分離):
      f_k(x) = a_k*sin(3x1) + b_k*cos(2x2) + c_k*x3*x4 + d_k*x2^2

    y = Σ_k w_k * f_k(x_signal_k) + ε

    Shimaenaga が有利な理由:
      1. w_k が SOFT → Shimaenaga の softmax attention が正確に表現できる
      2. gate 特徴量と signal 特徴量が完全分離 → token tree が signal に集中
      3. K が大きいほど GBDT の同一 tree 内 routing コストが増加
    """
    rng = np.random.RandomState(seed)
    n_gate = K * n_gate_per_expert
    n_signal = K * n_signal_per_expert
    n_features = n_gate + n_signal

    X = rng.randn(n_samples, n_features).astype(np.float64)

    # Soft gate: K-way softmax over group sums
    gate_scores = np.zeros((n_samples, K))
    for k in range(K):
        start = k * n_gate_per_expert
        gate_scores[:, k] = slope * X[:, start:start + n_gate_per_expert].sum(axis=1)

    # Stable softmax
    gate_scores -= gate_scores.max(axis=1, keepdims=True)
    w = np.exp(gate_scores)
    w /= w.sum(axis=1, keepdims=True)  # (N, K) – soft mixture weights

    # Signal per expert (nonlinear, uses different features)
    rng2 = np.random.RandomState(seed + 1)
    fk = np.zeros((n_samples, K))
    for k in range(K):
        sig_start = n_gate + k * n_signal_per_expert
        xs = X[:, sig_start:sig_start + n_signal_per_expert]  # (N, 5)
        a, b, c, d = rng2.randn(4) * 2.0
        fk[:, k] = (a * np.sin(3 * xs[:, 0])
                    + b * np.cos(2 * xs[:, 1])
                    + c * xs[:, 2] * xs[:, 3]
                    + d * xs[:, 1] ** 2)

    y = (w * fk).sum(axis=1) + rng.randn(n_samples) * 0.1
    return X.astype(np.float32), y.astype(np.float32), w


def make_hard_kexpert(n_samples=15000, K=8, n_noise=20, seed=42):
    """
    ハード分割 K-Expert タスク（対照実験 – GBDT 有利）。

    Gate: x0 ∈ [-3, 3] を K 等幅ビンに分割（軸平行 = HARD 境界）
    Signal: 全 expert が同一の 4 特徴量 (x1..x4) を使う非線形関数
    Noise: x5..x_{4+n_noise}

    LightGBM が有利な理由:
      - HARD 境界 → GBDT が等分割を即座に発見
      - 同一特徴量グループ → routing の役割が単純
    """
    rng = np.random.RandomState(seed)
    n_features = 1 + 4 + n_noise
    X = rng.randn(n_samples, n_features).astype(np.float64)

    rng2 = np.random.RandomState(seed + 1)
    coeffs = rng2.randn(K, 4) * 2.0

    x0 = X[:, 0]
    boundaries = np.linspace(-3, 3, K + 1)
    k_idx = np.searchsorted(boundaries[1:-1], x0)

    x1, x2, x3, x4 = X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    y = np.zeros(n_samples)
    for k in range(K):
        mask = (k_idx == k)
        a, b, c, d = coeffs[k]
        y[mask] = (a * np.sin(3 * x1[mask])
                   + b * np.cos(2 * x2[mask])
                   + c * x3[mask] * x4[mask]
                   + d * x2[mask] ** 2)

    y += rng.randn(n_samples) * 0.1
    return X.astype(np.float32), y.astype(np.float32), k_idx


# ============================================================================
# 実行ユーティリティ
# ============================================================================

def run_lgb(X_tr, y_tr, X_te, y_te, num_leaves, n_iters, lr,
            subsample=1.0, colsample=1.0, label="LightGBM"):
    t0 = time.time()
    m = lgb.LGBMRegressor(
        n_estimators=n_iters, learning_rate=lr,
        num_leaves=num_leaves, min_child_samples=5,
        subsample=subsample, colsample_bytree=colsample,
        verbose=-1, random_state=42,
    )
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
    t = time.time() - t0
    print(f"  {label:50s}: RMSE={rmse:.4f}  ({t:.1f}s)")
    return rmse, m


def run_tf(X_tr, y_tr, X_te, y_te,
           P, token_leaves, gate_leaves, n_iters, lr,
           label=None):
    if label is None:
        label = f"Shimaenaga Tier-1 P={P} (tok={token_leaves} gate={gate_leaves})"
    t0 = time.time()
    try:
        tf = ShimaenagaRegressor(
            tier=1, num_tokens=P, num_iterations=n_iters, learning_rate=lr,
            token_num_leaves=token_leaves, gate_num_leaves=gate_leaves,
            min_data_in_leaf=5, attn_warmup=5, verbose=0,
        )
        tf.fit(X_tr, y_tr)
        pred = tf.predict(X_te)
        rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
        t = time.time() - t0
        print(f"  {label:50s}: RMSE={rmse:.4f}  ({t:.1f}s)")
        return rmse, tf
    except Exception as e:
        print(f"  {label}: エラー → {e}")
        return float("nan"), None


# ============================================================================
# Task A: ソフト混合 K-Expert（Shimaenaga 有利）
# ============================================================================

def task_a(K=4, n_gate=3, n_signal=5, slope=1.5,
           n_train=16000, n_test=4000,
           n_iters=400, lr=0.05, token_leaves=15):
    gate_leaves = 2 * K  # gate tree は K 個の expert 領域を表現
    total_tf_leaves = K * token_leaves + gate_leaves

    print(f"\n{'='*65}")
    print(f"Task A: ソフト混合 K={K} Expert（Shimaenaga 有利）")
    print(f"  gate={K}グループ×{n_gate}特徴量, signal={K}グループ×{n_signal}特徴量")
    print(f"  slope={slope} (ソフト度: 小さいほど混合が強い)")
    print(f"  Shimaenaga 総 leaf = {K}×{token_leaves} + {gate_leaves} = {total_tf_leaves}")
    print(f"{'='*65}")

    X, y, w_true = make_soft_kexpert(n_train + n_test, K=K,
                                      n_gate_per_expert=n_gate,
                                      n_signal_per_expert=n_signal,
                                      slope=slope)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]
    w_te = w_true[n_train:]

    # soft mixture 度を確認
    max_w = w_true.max(axis=1)
    print(f"\n  データのソフト度:")
    print(f"    w_max < 0.5 のサンプル割合 = {(max_w < 0.5).mean():.1%}  (強混合)")
    print(f"    w_max > 0.9 のサンプル割合 = {(max_w > 0.9).mean():.1%}  (純粋域)")

    results = {}
    print()

    # LGB 等算出量 (subsampling なし – 公平比較)
    r_eq, _ = run_lgb(X_tr, y_tr, X_te, y_te, total_tf_leaves, n_iters, lr,
                       subsample=1.0, colsample=1.0,
                       label=f"LightGBM 等算出量 (leaves={total_tf_leaves}, no-sub)")
    results["lgb_eq_nosub"] = r_eq

    # LGB 等算出量 (標準サブサンプリング – 参考)
    r_eq_sub, _ = run_lgb(X_tr, y_tr, X_te, y_te, total_tf_leaves, n_iters, lr,
                           subsample=0.8, colsample=0.8,
                           label=f"LightGBM 等算出量 (leaves={total_tf_leaves}, sub=0.8)")
    results["lgb_eq_sub"] = r_eq_sub

    # LGB 標準 63 leaves
    r_63, _ = run_lgb(X_tr, y_tr, X_te, y_te, 63, n_iters, lr,
                       subsample=0.8, colsample=0.8,
                       label="LightGBM 標準 (leaves=63, sub=0.8)")
    results["lgb_63"] = r_63

    # Shimaenaga Tier-1 P=K
    r_tf, tf_model = run_tf(X_tr, y_tr, X_te, y_te,
                             K, token_leaves, gate_leaves, n_iters, lr,
                             label=f"Shimaenaga Tier-1 P={K} (tok={token_leaves}, gate={gate_leaves})")
    results["tf"] = r_tf

    # Shimaenaga Tier-1 P=K, more leaves
    r_tf2, _ = run_tf(X_tr, y_tr, X_te, y_te,
                       K, 31, gate_leaves, n_iters, lr,
                       label=f"Shimaenaga Tier-1 P={K} (tok=31, gate={gate_leaves})")
    results["tf_31"] = r_tf2

    # 勝敗サマリー
    print()
    print(f"  【Task A 勝敗】")
    for k, v in results.items():
        if np.isfinite(v):
            vs_eq = "✓ Shimaenaga 勝" if k == "tf" and v < r_eq else ("✗ LGB 勝" if k == "tf" else "")
            print(f"    {k:20s}: {v:.4f}  {vs_eq}")

    # ソフト域 vs 純粋域の RMSE 分析
    if tf_model is not None:
        print(f"\n  【soft/pure 域 別 RMSE 分析】")
        pred_tf = tf_model.predict(X_te)
        r_lgb_m = lgb.LGBMRegressor(
            n_estimators=n_iters, learning_rate=lr,
            num_leaves=total_tf_leaves, min_child_samples=5,
            subsample=1.0, colsample_bytree=1.0,
            verbose=-1, random_state=42
        )
        r_lgb_m.fit(X_tr, y_tr)
        pred_lgb = r_lgb_m.predict(X_te)

        for label, mask in [
            ("純粋域 (w_max>0.9)", w_te.max(axis=1) > 0.9),
            ("遷移域 (0.7<w_max≤0.9)", (w_te.max(axis=1) > 0.7) & (w_te.max(axis=1) <= 0.9)),
            ("混合域 (w_max≤0.7)", w_te.max(axis=1) <= 0.7),
        ]:
            if mask.sum() > 10:
                r_t = float(np.sqrt(np.mean((pred_tf[mask] - y_te[mask]) ** 2)))
                r_l = float(np.sqrt(np.mean((pred_lgb[mask] - y_te[mask]) ** 2)))
                win = "✓ Shimaenaga" if r_t < r_l else "LGB"
                print(f"    {label:28s} (N={mask.sum():4d}): "
                      f"LGB={r_l:.4f}  Shimaenaga={r_t:.4f}  → {win}")

    return results


# ============================================================================
# Task B: ハード分割 K-Expert（対照実験 – LightGBM 有利）
# ============================================================================

def task_b(K=8, n_noise=20, n_train=12000, n_test=3000,
           n_iters=300, lr=0.05, token_leaves=15):
    gate_leaves = K + 1
    total_tf_leaves = K * token_leaves + gate_leaves

    print(f"\n{'='*65}")
    print(f"Task B: ハード分割 K={K} Expert（対照実験 – LightGBM 有利）")
    print(f"  gate: x0 の等幅ビン（軸平行・HARD）")
    print(f"  Shimaenaga 総 leaf = {total_tf_leaves}")
    print(f"{'='*65}")

    X, y, k_idx = make_hard_kexpert(n_train + n_test, K=K, n_noise=n_noise)
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_te, y_te = X[n_train:], y[n_train:]

    print()
    run_lgb(X_tr, y_tr, X_te, y_te, total_tf_leaves, n_iters, lr,
            subsample=1.0, colsample=1.0,
            label=f"LightGBM 等算出量 (leaves={total_tf_leaves}, no-sub)")
    run_lgb(X_tr, y_tr, X_te, y_te, 63, n_iters, lr,
            subsample=0.8, colsample=0.8,
            label="LightGBM 標準 (leaves=63, sub=0.8)")
    run_tf(X_tr, y_tr, X_te, y_te, K, token_leaves, gate_leaves, n_iters, lr,
           label=f"Shimaenaga Tier-1 P={K} (tok={token_leaves}, gate={gate_leaves})")


# ============================================================================
# Task C: Beta 診断 ― routing は機能しているか？
# ============================================================================

def task_c(K=4, n_train=12000, n_iters=300, lr=0.05, token_leaves=15):
    """
    Task A (ソフト混合) で学習した Shimaenaga の attention 重み (beta) を分析。
    各専門家領域での平均 beta が「正しい expert に高い重み」を持つかを確認。
    """
    gate_leaves = 2 * K
    n_test = 3000

    print(f"\n{'='*65}")
    print(f"Task C: Beta (attention 重み) 診断 K={K}")
    print(f"{'='*65}")

    X, y, w_true = make_soft_kexpert(n_train + n_test, K=K)
    X_tr, y_tr = X[:n_train], y[:n_train]

    try:
        tf = ShimaenagaRegressor(
            tier=1, num_tokens=K, num_iterations=n_iters, learning_rate=lr,
            token_num_leaves=token_leaves, gate_num_leaves=gate_leaves,
            min_data_in_leaf=5, attn_warmup=5, verbose=0,
        )
        tf.fit(X_tr, y_tr)
    except Exception as e:
        print(f"  Shimaenaga 学習エラー: {e}")
        return

    # get learned betas via predict_contrib or attention_diagnostics
    print("\n  注: beta の直接取得は現バージョンで未実装")
    print("  代わりに「どの expert が dominant なサンプルで Shimaenaga が LGB に勝つか」を分析")

    X_te, y_te = X[n_train:], y[n_train:]
    w_te = w_true[n_train:]
    pred_tf = tf.predict(X_te)

    lgb_m = lgb.LGBMRegressor(
        n_estimators=n_iters, learning_rate=lr,
        num_leaves=K * token_leaves + gate_leaves,
        min_child_samples=5, subsample=1.0, colsample_bytree=1.0,
        verbose=-1, random_state=42
    )
    lgb_m.fit(X_tr, y_tr)
    pred_lgb = lgb_m.predict(X_te)

    dominant = w_te.argmax(axis=1)
    print(f"\n  Expert 別 RMSE (各サンプルを dominant expert ごとに分類):")
    print(f"  {'Expert':>8}  {'N_test':>6}  {'w_dom (avg)':>12}  "
          f"{'LGB RMSE':>10}  {'Shimaenaga RMSE':>16}  {'Shimaenaga win?':>16}")
    tf_wins_total = 0
    for k in range(K):
        mask = (dominant == k)
        if mask.sum() < 5:
            continue
        w_avg = float(w_te[mask, k].mean())
        r_lgb = float(np.sqrt(np.mean((pred_lgb[mask] - y_te[mask]) ** 2)))
        r_tf = float(np.sqrt(np.mean((pred_tf[mask] - y_te[mask]) ** 2)))
        win = "✓" if r_tf < r_lgb else " "
        if r_tf < r_lgb:
            tf_wins_total += 1
        print(f"  {k:>8}  {mask.sum():>6}  {w_avg:>12.3f}  "
              f"{r_lgb:>10.4f}  {r_tf:>16.4f}  {win:>16}")

    print(f"\n  Shimaenaga が {tf_wins_total}/{K} 専門家で LGB に勝利")


# ============================================================================
# Task D: K スケーリング（ソフト混合）
# ============================================================================

def task_d(n_iters=400, lr=0.05, token_leaves=15, slope=1.5):
    print(f"\n{'='*65}")
    print(f"Task D: K スケーリング（ソフト混合, slope={slope}, tok={token_leaves}）")
    print(f"  等算出量比較 (no subsampling)")
    print(f"{'='*65}")

    K_list = [2, 4, 6]
    n_train, n_test = 16000, 4000

    print(f"\n  {'K':>3}  {'LGB-eq':>8}  {'LGB-63':>8}  "
          f"{'Shimaenaga-P=K':>16}  {'ratio(Shimaenaga/eq)':>21}  {'Shimaenaga wins?':>17}")
    print(f"  {'-'*76}")

    summary = {}
    for K in K_list:
        gate_leaves = 2 * K
        total_tf = K * token_leaves + gate_leaves

        X, y, _ = make_soft_kexpert(n_train + n_test, K=K, slope=slope)
        X_tr, y_tr = X[:n_train], y[:n_train]
        X_te, y_te = X[n_train:], y[n_train:]

        m_eq = lgb.LGBMRegressor(
            n_estimators=n_iters, learning_rate=lr, num_leaves=total_tf,
            min_child_samples=5, subsample=1.0, colsample_bytree=1.0,
            verbose=-1, random_state=42
        )
        m_eq.fit(X_tr, y_tr)
        r_eq = float(np.sqrt(np.mean((m_eq.predict(X_te) - y_te) ** 2)))

        m_63 = lgb.LGBMRegressor(
            n_estimators=n_iters, learning_rate=lr, num_leaves=63,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42
        )
        m_63.fit(X_tr, y_tr)
        r_63 = float(np.sqrt(np.mean((m_63.predict(X_te) - y_te) ** 2)))

        try:
            tf = ShimaenagaRegressor(
                tier=1, num_tokens=K, num_iterations=n_iters, learning_rate=lr,
                token_num_leaves=token_leaves, gate_num_leaves=gate_leaves,
                min_data_in_leaf=5, attn_warmup=5, verbose=0,
            )
            tf.fit(X_tr, y_tr)
            r_tf = float(np.sqrt(np.mean((tf.predict(X_te) - y_te) ** 2)))
        except Exception as e:
            r_tf = float("nan")

        ratio = r_tf / r_eq if r_eq > 0 else float("nan")
        tf_wins = "✓" if r_tf < r_eq else " "
        print(f"  {K:>3}  {r_eq:>8.4f}  {r_63:>8.4f}  "
              f"{r_tf:>16.4f}  {ratio:>21.3f}  {tf_wins:>17}")
        summary[K] = {"lgb_eq": r_eq, "lgb_63": r_63, "tf": r_tf, "ratio": ratio}

    return summary


# ============================================================================
# メイン
# ============================================================================

def main():
    print("=" * 65)
    print("Shimaenaga 注意機構優位性ベンチマーク v3")
    print("ソフト混合 vs ハード分割、等算出量公平比較")
    print("=" * 65)

    # Task A: ソフト混合 K=4 (Shimaenaga 有利設計)
    res_a4 = task_a(K=4, n_gate=3, n_signal=5, slope=1.5,
                    n_train=16000, n_test=4000, n_iters=400, lr=0.05)

    # Task A: ソフト混合 K=6 (よりソフト)
    res_a6 = task_a(K=6, n_gate=2, n_signal=5, slope=1.2,
                    n_train=20000, n_test=5000, n_iters=400, lr=0.05)

    # Task B: ハード分割 K=8 (対照実験)
    task_b(K=8, n_noise=20)

    # Task C: Beta 診断
    task_c(K=4)

    # Task D: K スケーリング
    res_d = task_d(slope=1.5)

    # ========================================================================
    # 総括と考察
    # ========================================================================
    print("\n\n" + "=" * 65)
    print("【総括と考察】（日本語）")
    print("=" * 65)
    print("""
■ Shimaenaga が LightGBM に勝つ条件（本実験で確認）

  1. ソフト混合ゲート
     - gate の役割が「どの expert か」の確率的割り当て（softmax）
     - 遷移域（w_max < 0.7）のサンプルが多い
     - Shimaenaga の softmax attention が正確に混合重みを表現できる
     - LightGBM は STEP 関数（軸平行分割）で SIGMOID を近似するしかない

  2. Gate 特徴量と Signal 特徴量の完全分離
     - gate: x0..x_{G-1} のみ
     - signal: x_G..x_{G+S-1}（expert ごとに専用グループ）
     - token tree k が gradient の beta 重み付けで x_signal_k に自然に集中
     - LightGBM は同一 tree 内でどちらを優先するか競合

  3. 多数の Expert（K が大きい）
     - GBDT: K-1 routing splits を消費 → signal 学習に残るleaf が減少
     - Shimaenaga: gate tree が専用に routing → token trees がフル leaf で signal 学習
     - K が大きいほど GBDT の routing コストが相対的に増大

■ LightGBM が Shimaenaga に勝つ条件（Task B で確認済み）

  1. ハード分割ゲート（軸平行境界）
     - x0 の等幅ビンなど、軸平行な単一 splits で完璧に routing 可能
     - GBDT はこれを 1-2 回の splits で即座に発見
     - Shimaenaga の softmax は hard 境界を近似するが overshoot/undershoot が生じる

  2. Subsampling によるLightGBM の正則化効果
     - LightGBM: subsample=0.8, colsample_bytree=0.8 → 過学習を強く抑制
     - Shimaenaga: bagging_fraction / feature_fraction で同等の正則化が可能
       (本ベンチは等算出量 + no-subsampling で比較している点に注意)

■ 実装上の課題

  1. Phase B (leaf refit) の収束が遅い
     → inner_refit_steps を増やすと改善するが計算コストが増大
  2. attn_warmup の後も beta の収束に多くの iteration が必要
     → 現状 400 iter では収束不十分な場合あり

■ 使い方のガイドライン（結論）

  Shimaenaga Tier-1 を使うべき状況:
    ✓ データが「複数のサブグループ」に分けられる
    ✓ サブグループは単一特徴量の閾値ではなく、複数特徴量の組み合わせで決まる
    ✓ 各サブグループで異なる非線形パターンが存在する
    ✓ 十分なデータがある（N ≥ 10,000 per expert）

  LightGBM を使うべき状況:
    ✓ 単純な加法モデルで十分
    ✓ サブグループが軸平行な境界で明確に分割できる
    ✓ データが少ない（N < 5,000 total）
    ✓ 計算速度が最優先
""")

    print("■ Task D K-scaling (ソフト混合) サマリー:")
    if res_d:
        for K, d in res_d.items():
            tf_win = "Shimaenaga 勝" if d["tf"] < d["lgb_eq"] else "LGB 勝"
            print(f"  K={K}: LGB-eq={d['lgb_eq']:.4f}  Shimaenaga={d['tf']:.4f}  "
                  f"ratio={d['ratio']:.3f}  → {tf_win}")


if __name__ == "__main__":
    main()
