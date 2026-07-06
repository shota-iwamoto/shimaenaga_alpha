"""Unit tests for the basic estimator API: shapes, probabilities, save/load."""
import os
import tempfile

import numpy as np
import pytest

import shimaenaga as sb


def test_regressor_predict_shape_and_finite(regression_data):
    Xtr, Xte, ytr, yte = regression_data
    m = sb.ShimaenagaRegressor(num_iterations=50, tier=0, verbose=0)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    assert pred.shape == (Xte.shape[0],)
    assert np.all(np.isfinite(pred))
    # should beat the mean baseline on this easy synthetic problem
    rmse = np.sqrt(np.mean((pred - yte) ** 2))
    base = np.sqrt(np.mean((yte - ytr.mean()) ** 2))
    assert rmse < base


def test_binary_proba_shape_and_normalised(binary_data):
    Xtr, Xte, ytr, yte = binary_data
    m = sb.ShimaenagaClassifier(num_iterations=60, tier=0, verbose=0)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    assert proba.shape == (Xte.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    assert np.all((proba >= 0) & (proba <= 1))
    assert (m.predict(Xte) == yte).mean() > 0.9


def test_multiclass_proba_shape(multiclass_data):
    Xtr, Xte, ytr, yte = multiclass_data
    m = sb.ShimaenagaClassifier(num_class=3, num_iterations=50, tier=0, verbose=0)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    assert proba.shape == (Xte.shape[0], 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    np.testing.assert_array_equal(m.classes_, [0, 1, 2])


def test_classes_preserves_original_labels():
    X = np.random.RandomState(0).rand(40, 4).astype(np.float32)
    y = np.array([10, 20] * 20)  # non-0/1 labels
    m = sb.ShimaenagaClassifier(num_iterations=20, tier=0, verbose=0)
    m.fit(X, y)
    assert set(m.predict(X)).issubset({10, 20})


def test_tier1_attention_runs(binary_data):
    Xtr, Xte, ytr, yte = binary_data
    m = sb.ShimaenagaClassifier(num_iterations=30, tier=1, num_tokens=4,
                           num_heads=2, verbose=0)
    m.fit(Xtr, ytr)
    assert (m.predict(Xte) == yte).mean() > 0.85


def test_save_load_roundtrip(regression_data):
    Xtr, Xte, ytr, yte = regression_data
    m = sb.ShimaenagaRegressor(num_iterations=40, tier=1, num_tokens=4, verbose=0)
    m.fit(Xtr, ytr)
    pred_before = m.predict(Xte)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.sbb")
        m.save_model(path)
        m2 = sb.ShimaenagaRegressor(num_iterations=40, tier=1, num_tokens=4, verbose=0)
        m2.fit(Xtr[:50], ytr[:50])   # init booster, then overwrite from file
        m2.load_model(path)
        pred_after = m2.predict(Xte)

    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-4, atol=1e-4)
