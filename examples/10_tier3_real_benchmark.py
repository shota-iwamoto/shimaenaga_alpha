"""
Example 10: Tier-3 実データ精度ベンチマーク (Tier-3 vs Tier-2/0 vs LightGBM)
============================================================================
Tier-3 (Deep Attentive Tree Block) の実データでの精度を、Tier-0/2 および
LightGBM と同条件で比較します。合成データ (Friedman/Hastie) は例 06/07 が
担当するため、本スクリプトは実データのみを扱います。

比較の設計
----------
  * 複数シードで train/test 分割を変えて実行し、平均±標準偏差で報告
    (attention collapse 修正時と同じ「データ抽選」頑健性の確認手順)。
  * Tier-3 は Tier-2 と同一の Phase A 構造 + anchored 初期化のため、
    設計上 Tier-2 以下の block 目的値が保証される (数学設計書 Tier-3 §8.6
    定理 D1)。本ベンチはそれが**テスト精度**でも成立するかの実測。

対象タスク (すべて実データ)
---------------------------
  A. 回帰       : California Housing (20,640 件,  8 特徴, RMSE↓)
  B. 二値分類   : Breast Cancer      (   569 件, 30 特徴, AUC↑)
  C. 多クラス   : Digits 0-9         ( 1,797 件, 64 特徴, Accuracy↑)
  D. 多クラス   : Wine               (   178 件, 13 特徴, Accuracy↑)

実行:  python3 examples/10_tier3_real_benchmark.py [--fast] [--seeds N]
                [--tasks ABCD] [--rows N]
       --fast     反復数を減らして短時間で確認
       --seeds N  シード数 (既定 3)
       --tasks    実行タスクの選択 (例: --tasks AB)
       --rows N   1 タスクあたりの最大学習行数 (Tier-3 は活性を全行保持する
                  ため、メモリ/時間を抑えたい場合に使用。既定は無制限)
"""
import os
import sys
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from sbgbm import ShimaenagaRegressor, ShimaenagaClassifier

from sklearn.datasets import (fetch_california_housing, load_breast_cancer,
                              load_digits, load_wine)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (root_mean_squared_error, roc_auc_score,
                             accuracy_score)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true", help="反復数を減らして高速実行")
parser.add_argument("--seeds", type=int, default=3, help="シード数 (既定 3)")
parser.add_argument("--tasks", type=str, default="ABCD", help="実行タスク (例: AB)")
parser.add_argument("--rows", type=int, default=0,
                    help="学習データの最大行数 (0=無制限)")
args = parser.parse_args()

N_ITER = 40 if args.fast else 150
SEEDS = list(range(42, 42 + args.seeds))


def hr(title):
    print("\n" + "=" * 76 + f"\n  {title}\n" + "=" * 76)


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def subsample(X, y, seed):
    """--rows 指定時に学習データを層化なしで先頭から間引く (シード依存)。"""
    if args.rows <= 0 or len(X) <= args.rows:
        return X, y
    idx = np.random.RandomState(seed).permutation(len(X))[:args.rows]
    return X[idx], y[idx]


# ─────────────────────────────────────────────────────────────
# タスク定義
#   各タスク: データ読込 + モデルファクトリ (seed を受けて fit 可能な
#   推定器を返す) + 評価関数。metric の向きは lower_is_better で指定。
# ─────────────────────────────────────────────────────────────

def shim_kwargs(tier, base):
    """Tier ごとの Shimaenaga パラメータ (例 08 と同条件 + Tier-3 既定値)。"""
    kw = dict(base, verbose=0)
    if tier >= 1:
        kw.update(num_tokens=8, num_heads=2, d_attn=8)
    if tier >= 3:
        # Tier-3 基本設計書 §7 の既定値
        kw.update(attn_layers=2, d_hidden=8, d_ffn=16,
                  eta_u=0.5, eta_ffn=0.5, eta_cls=0.1)
    return kw


def task_A(seed):
    data = fetch_california_housing()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    Xtr, ytr = subsample(Xtr, ytr, seed)
    base = dict(num_iterations=N_ITER, learning_rate=0.1, token_num_leaves=31,
                gate_num_leaves=8, min_data_in_leaf=20, seed=seed)
    models = {}
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=N_ITER, learning_rate=0.1, num_leaves=31,
            min_child_samples=20, verbosity=-1, random_state=seed)
    for tier in (0, 2, 3):
        models[f"Shimaenaga Tier-{tier}"] = ShimaenagaRegressor(
            tier=tier, **shim_kwargs(tier, base))

    def evaluate(m):
        return root_mean_squared_error(yte, m.predict(Xte))
    return Xtr, ytr, models, evaluate


def task_B(seed):
    bc = load_breast_cancer()
    X, y = bc.data.astype(np.float32), bc.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=seed, stratify=y)
    base = dict(num_class=1, num_iterations=N_ITER, learning_rate=0.05,
                token_num_leaves=16, gate_num_leaves=8, min_data_in_leaf=5,
                seed=seed)
    models = {}
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=N_ITER, learning_rate=0.05, num_leaves=31,
            min_child_samples=5, verbosity=-1, random_state=seed)
    for tier in (0, 2, 3):
        models[f"Shimaenaga Tier-{tier}"] = ShimaenagaClassifier(
            tier=tier, **shim_kwargs(tier, base))

    def evaluate(m):
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    return Xtr, ytr, models, evaluate


def task_C(seed):
    dg = load_digits()
    X, y = dg.data.astype(np.float32), dg.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=seed, stratify=y)
    base = dict(num_class=10, num_iterations=N_ITER, learning_rate=0.1,
                token_num_leaves=8, gate_num_leaves=8, min_data_in_leaf=5,
                seed=seed)
    models = {}
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=N_ITER, learning_rate=0.1, num_leaves=31,
            min_child_samples=5, verbosity=-1, random_state=seed)
    # 多クラス×64特徴の tier-2/3 は重いため d_attn を絞る (例 08 と同方針)
    for tier in (0, 2, 3):
        kw = shim_kwargs(tier, base)
        if tier >= 1:
            kw.update(d_attn=4)
        models[f"Shimaenaga Tier-{tier}"] = ShimaenagaClassifier(tier=tier, **kw)

    def evaluate(m):
        return accuracy_score(yte, m.predict(Xte))
    return Xtr, ytr, models, evaluate


def task_D(seed):
    wn = load_wine()
    X, y = wn.data.astype(np.float32), wn.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=seed, stratify=y)
    base = dict(num_class=3, num_iterations=N_ITER, learning_rate=0.1,
                token_num_leaves=8, gate_num_leaves=8, min_data_in_leaf=3,
                seed=seed)
    models = {}
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=N_ITER, learning_rate=0.1, num_leaves=15,
            min_child_samples=3, verbosity=-1, random_state=seed)
    for tier in (0, 2, 3):
        models[f"Shimaenaga Tier-{tier}"] = ShimaenagaClassifier(
            tier=tier, **shim_kwargs(tier, base))

    def evaluate(m):
        return accuracy_score(yte, m.predict(Xte))
    return Xtr, ytr, models, evaluate


TASKS = {
    "A": ("回帰 — California Housing", "RMSE", True,  task_A),
    "B": ("二値分類 — Breast Cancer",  "AUC",  False, task_B),
    "C": ("多クラス — Digits 0-9",     "Acc",  False, task_C),
    "D": ("多クラス — Wine",           "Acc",  False, task_D),
}


# ─────────────────────────────────────────────────────────────
# 実行ループ: タスク × シード × モデル → 平均±SD で集計
# ─────────────────────────────────────────────────────────────

summary = {}  # task_key -> {model: (mean, sd, mean_time)}

for key in args.tasks:
    if key not in TASKS:
        print(f"  [skip] 未知のタスク '{key}'")
        continue
    title, metric_name, lower_better, make = TASKS[key]
    arrow = "↓" if lower_better else "↑"
    hr(f"Task {key}: {title} ({metric_name}{arrow}, seeds={SEEDS})")

    scores = {}   # model -> [per-seed metric]
    times = {}    # model -> [per-seed sec]
    for seed in SEEDS:
        Xtr, ytr, models, evaluate = make(seed)
        if seed == SEEDS[0]:
            print(f"  train={Xtr.shape}")
        for name, model in models.items():
            _, t = timed(lambda m=model: m.fit(Xtr, ytr))
            val = evaluate(model)
            scores.setdefault(name, []).append(val)
            times.setdefault(name, []).append(t)
            print(f"    seed={seed}  {name:<22}{metric_name}={val:.4f}"
                  f"  ({t:.1f}s)")

    print(f"\n  {'Model':<24}{metric_name + ' (mean±sd)':<22}{'Time':>9}")
    print("  " + "-" * 55)
    summary[key] = {}
    for name in scores:
        mean, sd = float(np.mean(scores[name])), float(np.std(scores[name]))
        mt = float(np.mean(times[name]))
        summary[key][name] = (mean, sd, mt)
        print(f"  {name:<24}{mean:.4f} ± {sd:.4f}      {mt:>7.1f}s")


# ─────────────────────────────────────────────────────────────
# まとめ: Tier-3 が Tier-2 / LightGBM に勝ったか
# ─────────────────────────────────────────────────────────────
hr("まとめ — Tier-3 の相対性能")
print(f"  {'Task':<6}{'Metric':<8}{'Tier-2':>12}{'Tier-3':>12}"
      f"{'Δ(3-2)':>12}{'LightGBM':>12}")
print("  " + "-" * 62)
for key in args.tasks:
    if key not in summary:
        continue
    _, metric_name, lower_better, _ = TASKS[key]
    s = summary[key]
    t2 = s.get("Shimaenaga Tier-2", (float("nan"),) * 3)[0]
    t3 = s.get("Shimaenaga Tier-3", (float("nan"),) * 3)[0]
    lg = s.get("LightGBM", (float("nan"),) * 3)[0]
    delta = t3 - t2
    better = (delta < 0) if lower_better else (delta > 0)
    mark = "✓" if better else ("=" if delta == 0 else "✗")
    print(f"  {key:<6}{metric_name:<8}{t2:>12.4f}{t3:>12.4f}"
          f"{delta:>+11.4f}{mark}{lg:>12.4f}")
print("""
  ✓ = Tier-3 が Tier-2 を上回った / ✗ = 下回った (テスト精度ベース)。
  注: 定理 D1 が保証するのは学習時 block 目的値の非悪化であり、テスト
  精度の非悪化ではない。シード間の ±sd と併せて判断すること。
""")
print("Done.")
