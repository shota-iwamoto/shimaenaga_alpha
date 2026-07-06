"""Unit tests for error paths and input validation."""
import numpy as np
import pytest

import shimaenaga as sb


def test_predict_before_fit_raises():
    m = sb.ShimaenagaRegressor(verbose=0)
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((3, 4), dtype=np.float32))


def test_proba_before_fit_raises():
    m = sb.ShimaenagaClassifier(verbose=0)
    with pytest.raises(RuntimeError):
        m.predict_proba(np.zeros((3, 4), dtype=np.float32))


def test_save_before_fit_raises():
    m = sb.ShimaenagaRegressor(verbose=0)
    with pytest.raises(RuntimeError):
        m.save_model("/tmp/should_not_exist.sbb")


def test_ranker_requires_group():
    X = np.random.RandomState(0).rand(20, 5).astype(np.float32)
    y = np.random.randint(0, 3, 20)
    m = sb.ShimaenagaRanker(num_iterations=10, tier=0, verbose=0)
    with pytest.raises(ValueError):
        m.fit(X, y)  # group missing


def test_binary_rejects_multiclass_labels():
    X = np.random.RandomState(0).rand(30, 4).astype(np.float32)
    y = np.array([0, 1, 2] * 10)
    m = sb.ShimaenagaClassifier(num_class=1, num_iterations=10, tier=0, verbose=0)
    with pytest.raises(ValueError):
        m.fit(X, y)


def test_ranker_fits_with_group():
    X = np.random.RandomState(0).rand(20, 5).astype(np.float32)
    y = np.random.randint(0, 3, 20)
    m = sb.ShimaenagaRanker(num_iterations=10, tier=0, verbose=0)
    m.fit(X, y, group=[10, 10])
    scores = m.predict(X)
    assert scores.shape == (20,)
    assert np.all(np.isfinite(scores))
