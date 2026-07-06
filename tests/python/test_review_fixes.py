"""Regression tests for the 2026-07 review fixes:
attention diagnostics, ranking eval_group, feature_local mask, config
validation, predict dimension checks, num_threads, multiclass logloss metric.
"""
import numpy as np
import pytest

from shimaenaga import ShimaenagaRegressor, ShimaenagaClassifier, ShimaenagaRanker


@pytest.fixture(scope="module")
def small_reg():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, 2] + 0.1 * rng.normal(size=400)).astype(np.float32)
    return X, y


def test_attention_diagnostics_returns_real_beta(small_reg):
    X, y = small_reg
    m = ShimaenagaRegressor(tier=2, num_tokens=4, num_iterations=15, attn_warmup=3)
    m.fit(X, y)
    beta = m.attention_diagnostics(X[:20])["beta"]
    assert beta.shape == (20, 4)
    # rows are convex weights over tokens (E.14): sum to 1, non-negative
    assert np.allclose(beta.sum(axis=1), 1.0, atol=1e-4)
    assert (beta >= -1e-6).all()
    # and actually carry signal (the old stub returned all zeros)
    assert beta.std() > 0


def test_ranker_eval_set_requires_eval_group():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    r = ShimaenagaRanker(num_iterations=5, tier=1, num_tokens=2)
    with pytest.raises(ValueError, match="eval_group"):
        r.fit(X, y, group=[10] * 20, eval_set=[(X[:50], y[:50])])


def test_ranker_eval_group_gives_finite_valid_metric():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    s = 1.5 * X[:, 0] + X[:, 1]
    y = np.clip((1 / (1 + np.exp(-s)) * 5).astype(int), 0, 4).astype(np.float32)
    r = ShimaenagaRanker(num_iterations=20, tier=1, num_tokens=2)
    r.fit(X[:200], y[:200], group=[10] * 20,
          eval_set=[(X[200:], y[200:])], eval_group=[[10] * 10],
          early_stopping_rounds=5)
    assert r.best_iteration_ >= 1  # early stopping tracked a finite valid NDCG


def test_feature_local_mask_trains_and_roundtrips(small_reg, tmp_path):
    X, y = small_reg
    m = ShimaenagaRegressor(tier=2, num_tokens=4, num_iterations=10,
                            attn_warmup=2, attn_mask="feature_local")
    m.fit(X, y)
    before = m.predict(X[:30])
    path = str(tmp_path / "flocal.sbb")
    m.save_model(path)
    m2 = ShimaenagaRegressor(tier=2, num_tokens=4, num_iterations=2)
    m2.fit(X[:60], y[:60])
    m2.load_model(path)
    after = m2.predict(X[:30])
    # mask must be serialized (format v3) — otherwise reload changes predictions
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_manual_token_plan_rejected(small_reg):
    X, y = small_reg
    with pytest.raises(RuntimeError, match="token_plan"):
        ShimaenagaRegressor(token_plan="manual", num_iterations=2).fit(X, y)


def test_predict_feature_count_mismatch_raises(small_reg):
    X, y = small_reg
    m = ShimaenagaRegressor(num_iterations=3, tier=1, num_tokens=2)
    m.fit(X, y)
    with pytest.raises(RuntimeError, match="num_features"):
        m.predict(X[:, :5])


def test_eval_set_feature_count_mismatch_raises(small_reg):
    X, y = small_reg
    m = ShimaenagaRegressor(num_iterations=3, tier=1, num_tokens=2)
    with pytest.raises(ValueError, match="features"):
        m.fit(X, y, eval_set=[(X[:50, :5], y[:50])])


def test_num_threads_accepted_and_deterministic(small_reg):
    X, y = small_reg
    preds = []
    for nt in (1, 4):
        m = ShimaenagaRegressor(num_iterations=10, tier=2, num_tokens=4,
                                attn_warmup=2, seed=7, num_threads=nt)
        m.fit(X, y)
        preds.append(m.predict(X[:50]))
    # T10: thread count must not change the model
    np.testing.assert_array_equal(preds[0], preds[1])


def test_multiclass_early_stopping_on_logloss(multiclass_data):
    Xtr, Xte, ytr, yte = multiclass_data
    m = ShimaenagaClassifier(num_class=3, tier=1, num_tokens=2,
                             num_iterations=200, early_stopping_rounds=20)
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)])
    assert m.best_iteration_ >= 1
    proba = m.predict_proba(Xte)
    assert np.isfinite(proba).all()
