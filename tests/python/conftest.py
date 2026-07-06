"""pytest configuration: make the in-tree `shimaenaga` package importable
without installing, and provide small shared fixtures."""
import os
import sys

import numpy as np
import pytest

# Add <repo>/python to the import path so `import shimaenaga` resolves to the
# source tree (the C++ core is loaded via the ctypes backend from build/).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "python"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "uat: system/user-acceptance tests on real datasets (slower)")


@pytest.fixture(scope="session")
def binary_data():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X.astype(np.float32), y, test_size=0.2, random_state=0)


@pytest.fixture(scope="session")
def multiclass_data():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    return train_test_split(X.astype(np.float32), y, test_size=0.3, random_state=0)


@pytest.fixture(scope="session")
def regression_data():
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=400, n_features=8, noise=0.1, random_state=0)
    return train_test_split(X.astype(np.float32), y.astype(np.float32),
                            test_size=0.25, random_state=0)
