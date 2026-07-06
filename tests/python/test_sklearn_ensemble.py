"""Integration tests: Shimaenaga estimators inside sklearn meta-estimators.

These exercise the clone() path end to end and assert that user hyperparameters
actually propagate (not just that the pipeline runs without error).
"""
import warnings

import numpy as np
import pytest
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score

import shimaenaga as sb

warnings.filterwarnings("ignore")


def test_bagging_classifier(binary_data):
    Xtr, Xte, ytr, yte = binary_data
    bag = BaggingClassifier(
        sb.ShimaenagaClassifier(num_iterations=40, tier=0, verbose=0),
        n_estimators=3, random_state=0)
    bag.fit(Xtr, ytr)
    acc = (bag.predict(Xte) == yte).mean()
    assert acc > 0.85
    # the cloned sub-estimators must keep the user hyperparameters
    for est in bag.estimators_:
        assert est.get_params()["num_iterations"] == 40
        assert est.get_params()["tier"] == 0


def test_bagging_regressor(regression_data):
    Xtr, Xte, ytr, yte = regression_data
    bag = BaggingRegressor(
        sb.ShimaenagaRegressor(num_iterations=40, tier=0, verbose=0),
        n_estimators=3, random_state=0)
    bag.fit(Xtr, ytr)
    pred = bag.predict(Xte)
    assert pred.shape == yte.shape
    assert np.all(np.isfinite(pred))


def test_stacking_classifier(binary_data):
    Xtr, Xte, ytr, yte = binary_data
    stack = StackingClassifier(
        estimators=[("tf", sb.ShimaenagaClassifier(num_iterations=40, tier=0, verbose=0))],
        final_estimator=LogisticRegression(max_iter=1000), cv=3)
    stack.fit(Xtr, ytr)
    acc = (stack.predict(Xte) == yte).mean()
    assert acc > 0.85


def test_stacking_regressor(regression_data):
    Xtr, Xte, ytr, yte = regression_data
    stack = StackingRegressor(
        estimators=[("tf", sb.ShimaenagaRegressor(num_iterations=40, tier=0, verbose=0))],
        final_estimator=LinearRegression(), cv=3)
    stack.fit(Xtr, ytr)
    pred = stack.predict(Xte)
    assert pred.shape == yte.shape
    assert np.all(np.isfinite(pred))


def test_cross_val_score(binary_data):
    Xtr, _, ytr, _ = binary_data
    scores = cross_val_score(
        sb.ShimaenagaClassifier(num_iterations=40, tier=0, verbose=0), Xtr, ytr, cv=3)
    assert len(scores) == 3
    assert scores.mean() > 0.85


def test_grid_search_propagates_params(binary_data):
    Xtr, _, ytr, _ = binary_data
    grid = GridSearchCV(
        sb.ShimaenagaClassifier(tier=0, verbose=0),
        param_grid={"num_iterations": [10, 60]}, cv=3)
    grid.fit(Xtr, ytr)
    # The selected hyperparameter must survive the clone() inside GridSearchCV.
    assert grid.best_params_["num_iterations"] in (10, 60)
    assert grid.best_estimator_.get_params()["num_iterations"] == \
        grid.best_params_["num_iterations"]
