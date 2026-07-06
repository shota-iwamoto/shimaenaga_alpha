"""Unit tests for the sklearn parameter protocol (get_params/set_params/clone).

Regression guard for the bug where the leaf estimators' ``**kwargs`` __init__
made ``sklearn.clone`` silently drop every hyperparameter and refit with the
defaults (tier=2, num_iterations=1000), breaking Bagging/Stacking/GridSearchCV.
"""
import pytest
from sklearn.base import clone

import shimaenaga as sb


def test_get_params_returns_constructor_args():
    est = sb.ShimaenagaRegressor(num_iterations=37, learning_rate=0.123, tier=1,
                            num_tokens=4)
    params = est.get_params()
    assert params["num_iterations"] == 37
    assert params["learning_rate"] == 0.123
    assert params["tier"] == 1
    assert params["num_tokens"] == 4
    # objective is defaulted by the leaf class
    assert params["objective"] == "regression"


def test_clone_preserves_all_params_regressor():
    est = sb.ShimaenagaRegressor(num_iterations=99, learning_rate=0.07, tier=0,
                            min_data_in_leaf=11, seed=7)
    cloned = clone(est)
    # clone() must reproduce every hyperparameter, not fall back to defaults.
    for k, v in est.get_params().items():
        assert cloned.get_params()[k] == v, f"param {k} lost on clone"


def test_clone_preserves_num_class_classifier():
    est = sb.ShimaenagaClassifier(num_class=3, num_iterations=42, tier=1)
    cloned = clone(est)
    assert cloned.get_params()["num_class"] == 3
    assert cloned.get_params()["num_iterations"] == 42
    assert cloned.get_params()["tier"] == 1
    assert cloned.get_params()["objective"] == "multiclass"


def test_clone_preserves_params_ranker():
    est = sb.ShimaenagaRanker(num_iterations=15, tier=2, num_heads=4)
    cloned = clone(est)
    assert cloned.get_params()["num_iterations"] == 15
    assert cloned.get_params()["num_heads"] == 4
    assert cloned.get_params()["objective"] == "lambdarank"


def test_set_params_known_and_extra():
    est = sb.ShimaenagaRegressor()
    est.set_params(num_iterations=5, tier=1)
    assert est.num_iterations == 5 and est.tier == 1
    # unknown keys are routed to the C++ passthrough kwargs, not lost
    est.set_params(custom_flag=123)
    assert est._extra_kwargs.get("custom_flag") == 123


def test_set_params_returns_self():
    est = sb.ShimaenagaClassifier()
    assert est.set_params(num_iterations=3) is est


def test_get_set_params_roundtrip():
    est = sb.ShimaenagaRegressor(num_iterations=12, tier=1, num_tokens=6)
    params = est.get_params()
    fresh = sb.ShimaenagaRegressor().set_params(**{k: v for k, v in params.items()})
    assert fresh.get_params() == params
