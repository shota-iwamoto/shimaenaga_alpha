"""Prediction-accuracy regression guards for features added in 1.2.0.

Covers what test_st_uat.py does not: robust objectives (huber/quantile/mae),
early stopping via eval_set, the tree-regularization knobs, and the tier-2
save/load path of the v2 serializer. Synthetic data (Friedman #1) keeps these
fast and deterministic; thresholds are loose regression guards.
"""
import numpy as np
import pytest

import shimaenaga as sb


def _friedman(n, noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, 10)).astype(np.float32)
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4] + noise * rng.standard_normal(n))
    return X, y.astype(np.float32)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@pytest.fixture(scope="module")
def friedman_split():
    Xtr, ytr = _friedman(2000, 1.0, seed=1)
    Xte, yte = _friedman(2000, 0.0, seed=2)  # noiseless truth
    return Xtr, ytr, Xte, yte


# ── Robust objectives ────────────────────────────────────────────────────

def test_huber_and_mae_beat_l2_under_label_outliers(friedman_split):
    Xtr, ytr, Xte, yte = friedman_split
    rng = np.random.default_rng(7)
    y_out = ytr.copy()
    idx = rng.choice(len(y_out), size=len(y_out) // 20, replace=False)  # 5%
    y_out[idx] += rng.choice([-1.0, 1.0], size=len(idx)).astype(np.float32) * 80.0

    def fit_rmse(objective):
        m = sb.ShimaenagaRegressor(objective=objective, tier=0,
                                   num_iterations=200, learning_rate=0.1,
                                   seed=0, verbose=0)
        m.fit(Xtr, y_out)
        return _rmse(m.predict(Xte), yte)

    e_l2 = fit_rmse("regression")
    e_huber = fit_rmse("huber")
    e_mae = fit_rmse("mae")
    assert np.isfinite([e_l2, e_huber, e_mae]).all()
    assert e_huber < e_l2, f"huber ({e_huber:.3f}) should beat L2 ({e_l2:.3f})"
    assert e_mae < e_l2, f"mae ({e_mae:.3f}) should beat L2 ({e_l2:.3f})"
    # huber must also stay decent in absolute terms on this easy target
    assert e_huber < 1.5, f"huber RMSE too high: {e_huber:.3f}"


def test_quantile_objective_hits_requested_coverage(friedman_split):
    Xtr, ytr, Xte, _ = friedman_split
    _, yte_noisy = _friedman(2000, 1.0, seed=2)
    m = sb.ShimaenagaRegressor(objective="quantile", quantile_alpha=0.9,
                               tier=0, num_iterations=200, learning_rate=0.1,
                               seed=0, verbose=0)
    m.fit(Xtr, ytr)
    cov = float(np.mean(yte_noisy <= m.predict(Xte)))
    assert 0.80 < cov < 0.99, f"q90 coverage {cov:.3f} outside (0.80, 0.99)"


# ── Early stopping ───────────────────────────────────────────────────────

def test_early_stopping_truncates_and_does_not_hurt(friedman_split):
    Xtr, ytr, Xte, yte = friedman_split
    Xva, yva = _friedman(500, 1.0, seed=3)
    # deliberately overfit-prone configuration
    overfit = dict(tier=0, num_iterations=600, learning_rate=0.2,
                   min_data_in_leaf=2, token_num_leaves=63, lambda_v=0.01,
                   seed=0, verbose=0)

    m_no = sb.ShimaenagaRegressor(**overfit)
    m_no.fit(Xtr, ytr)
    e_no = _rmse(m_no.predict(Xte), yte)

    m_es = sb.ShimaenagaRegressor(early_stopping_rounds=30, **overfit)
    m_es.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    e_es = _rmse(m_es.predict(Xte), yte)

    assert 0 < m_es.best_iteration_ < 600, \
        f"early stopping did not trigger (best_iteration={m_es.best_iteration_})"
    # truncating at the validation optimum must not be materially worse
    assert e_es <= e_no * 1.05, f"ES rmse {e_es:.3f} vs no-ES {e_no:.3f}"


# ── Tree regularization knobs (wired end-to-end through the JSON config) ──

def test_regularization_knobs_change_predictions(friedman_split):
    Xtr, ytr, Xte, yte = friedman_split
    base = dict(tier=0, num_iterations=100, learning_rate=0.1, seed=0, verbose=0)

    m0 = sb.ShimaenagaRegressor(**base)
    m0.fit(Xtr, ytr)
    p0 = m0.predict(Xte)
    e0 = _rmse(p0, yte)
    assert e0 < 1.5, f"baseline RMSE unexpectedly high: {e0:.3f}"

    for knob in (dict(max_depth=2), dict(lambda_l1=20.0),
                 dict(min_sum_hessian_in_leaf=50.0)):
        m = sb.ShimaenagaRegressor(**base, **knob)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        assert np.isfinite(p).all(), f"non-finite predictions with {knob}"
        assert np.max(np.abs(p - p0)) > 1e-6, f"{knob} had no effect on the model"


def test_max_bin_over_uint8_capacity_rejected(friedman_split):
    Xtr, ytr, _, _ = friedman_split
    m = sb.ShimaenagaRegressor(max_bin=1000, num_iterations=5, verbose=0)
    with pytest.raises(Exception, match="max_bin"):
        m.fit(Xtr, ytr)


# ── Attention-collapse regression guard ─────────────────────────────────
# On this exact data draw (Friedman #1, data seeds 5/6) the attention tiers
# used to die: ρ saturated to exact 0/1, starved tokens degenerated to 1-leaf
# stumps, training froze, and — after the optimizer guards — the token plan
# could still make the x0·x1 interaction unrepresentable (test RMSE ≈ 1.50 vs
# tier-0 ≈ 1.19). Guards: attn_step_clip, beta_uniform_mix, and the
# full-feature interaction token in TokenPlanner.

def test_tier2_no_collapse_on_adversarial_draw():
    Xtr, ytr = _friedman(5000, 1.0, seed=5)
    Xte, yte = _friedman(5000, 1.0, seed=6)
    e = {}
    for tier in (0, 2):
        m = sb.ShimaenagaRegressor(tier=tier, num_tokens=4, num_heads=2, d_attn=4,
                                   num_iterations=300, learning_rate=0.05,
                                   seed=0, verbose=0)
        m.fit(Xtr, ytr)
        e[tier] = _rmse(m.predict(Xte), yte)
    # Bayes RMSE = 1.0 (noise sigma); the collapsed model scored ~1.50.
    assert e[2] < 1.30, f"tier-2 collapsed again on the adversarial draw: {e[2]:.4f}"
    # attention must not be materially worse than plain GBDT on the same draw
    assert e[2] < e[0] * 1.05, f"tier-2 ({e[2]:.4f}) much worse than tier-0 ({e[0]:.4f})"


# ── Tier-2 accuracy + serializer v2 roundtrip ────────────────────────────

def test_tier2_learns_and_save_load_matches(tmp_path, friedman_split):
    Xtr, ytr, Xte, yte = friedman_split
    m = sb.ShimaenagaRegressor(tier=2, num_tokens=4, num_heads=2, d_attn=4,
                               num_iterations=80, learning_rate=0.1,
                               attn_warmup=5, seed=0, verbose=0)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    e = _rmse(pred, yte)
    assert e < 2.0, f"tier-2 failed to learn Friedman#1 (RMSE {e:.3f})"

    path = str(tmp_path / "tier2.sbb")
    m.save_model(path)
    m2 = sb.ShimaenagaRegressor(tier=2, num_tokens=4, num_heads=2, d_attn=4,
                                num_iterations=5, seed=0, verbose=0)
    m2.fit(Xtr[:200], ytr[:200])  # initialise booster, then overwrite via load
    m2.load_model(path)
    pred2 = m2.predict(Xte)
    assert np.max(np.abs(pred - pred2)) < 1e-9, \
        "tier-2 save/load roundtrip changed predictions (serializer v2 qA/kA/bA path)"
