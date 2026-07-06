"""System / User-Acceptance tests with quality thresholds.

These automate the acceptance scenarios that examples/01-03 demonstrate
manually, turning them into pass/fail regression guards on real datasets.
Thresholds are deliberately loose (regression-detection, not SOTA chasing).
"""
import numpy as np
import pytest
from sklearn.datasets import (fetch_california_housing, load_breast_cancer,
                              load_wine)
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import shimaenaga as sb

pytestmark = pytest.mark.uat


def test_uat_california_regression():
    X, y = fetch_california_housing(return_X_y=True)
    X = X.astype(np.float32); y = y.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    m = sb.ShimaenagaRegressor(tier=0, num_iterations=200, learning_rate=0.1,
                          token_num_leaves=31, min_data_in_leaf=20, seed=42,
                          verbose=0)
    m.fit(Xtr, ytr)
    rmse = root_mean_squared_error(yte, m.predict(Xte))
    # plain GBDT on CalifHousing comfortably beats 0.55 RMSE
    assert rmse < 0.55, f"CalifHousing RMSE regressed: {rmse:.4f}"


def test_uat_breast_cancer_classification():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X.astype(np.float32), y,
                                          test_size=0.2, random_state=42)
    m = sb.ShimaenagaClassifier(tier=1, num_tokens=4, num_iterations=150,
                           learning_rate=0.1, seed=42, verbose=0)
    m.fit(Xtr, ytr)
    acc = accuracy_score(yte, m.predict(Xte))
    assert acc > 0.92, f"BreastCancer accuracy regressed: {acc:.4f}"


def test_uat_wine_cross_validation():
    X, y = load_wine(return_X_y=True)
    X = X.astype(np.float32)
    clf = sb.ShimaenagaClassifier(num_class=3, tier=1, num_tokens=4, num_heads=2,
                             num_iterations=100, learning_rate=0.1, seed=42,
                             verbose=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf)
    assert scores.mean() > 0.90, f"Wine CV mean regressed: {scores.mean():.4f}"
