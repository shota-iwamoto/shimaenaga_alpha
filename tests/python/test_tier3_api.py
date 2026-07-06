"""Tier-3 (tier=3) Python API tests: fit/predict, sklearn clone with the new
parameters, save/load roundtrip, config validation (Tier-3 基本設計書 §5.1)."""
import os
import tempfile

import numpy as np
import pytest
from sklearn.base import clone

import shimaenaga as sb


def _t3_kwargs(**over):
    kw = dict(
        tier=3,
        num_tokens=4,
        num_heads=2,
        d_attn=4,
        attn_layers=2,
        d_hidden=6,
        d_ffn=6,
        eta_cls=0.1,
        num_iterations=25,
        learning_rate=0.1,
        token_num_leaves=8,
        gate_num_leaves=8,
        min_data_in_leaf=10,
        attn_warmup=3,
        verbose=0,
        seed=7,
    )
    kw.update(over)
    return kw


def test_tier3_regressor_fits(regression_data):
    Xtr, Xte, ytr, yte = regression_data
    m = sb.ShimaenagaRegressor(**_t3_kwargs())
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert pred.shape == (Xte.shape[0],)
    assert np.all(np.isfinite(pred))
    rmse = np.sqrt(np.mean((pred - yte) ** 2))
    base = np.sqrt(np.mean((yte - ytr.mean()) ** 2))
    assert rmse < base


def test_tier3_classifier_fits(binary_data):
    Xtr, Xte, ytr, yte = binary_data
    m = sb.ShimaenagaClassifier(**_t3_kwargs(num_iterations=30))
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    assert proba.shape == (Xte.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    assert (m.predict(Xte) == yte).mean() > 0.85


def test_tier3_sklearn_clone_keeps_params():
    m = sb.ShimaenagaRegressor(**_t3_kwargs(attn_layers=3, d_hidden=12,
                                            eta_cls=0.2, tau_learnable=False))
    c = clone(m)
    p = c.get_params()
    assert p["tier"] == 3
    assert p["attn_layers"] == 3
    assert p["d_hidden"] == 12
    assert p["eta_cls"] == 0.2
    assert p["tau_learnable"] is False


def test_tier3_save_load_roundtrip(regression_data):
    Xtr, Xte, ytr, _ = regression_data
    m = sb.ShimaenagaRegressor(**_t3_kwargs())
    m.fit(Xtr, ytr)
    before = m.predict(Xte)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t3.sbb")
        m.save_model(path)
        m2 = sb.ShimaenagaRegressor(**_t3_kwargs())
        m2.fit(Xtr[:40], ytr[:40])   # init booster, then overwrite
        m2.load_model(path)
        after = m2.predict(Xte)
    np.testing.assert_array_equal(before, after)


def test_tier3_rejects_score_tree(regression_data):
    Xtr, _, ytr, _ = regression_data
    m = sb.ShimaenagaRegressor(**_t3_kwargs(attention_mode="score_tree"))
    with pytest.raises(Exception):
        m.fit(Xtr, ytr)


def test_tier3_attention_diagnostics(regression_data):
    Xtr, Xte, ytr, _ = regression_data
    m = sb.ShimaenagaRegressor(**_t3_kwargs())
    m.fit(Xtr, ytr)
    diag = m.attention_diagnostics(Xte[:16])
    beta = diag["beta"]
    assert beta.shape == (16, 4)
    assert np.allclose(beta.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(beta >= -1e-6)
