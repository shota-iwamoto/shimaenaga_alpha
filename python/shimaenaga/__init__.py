"""
Shimaenaga: Attentive Histogram GBDT with sample-level token attention.
sklearn-compatible API.
"""
from .sklearn_api import (
    ShimaenagaRegressor,
    ShimaenagaClassifier,
    ShimaenagaRanker,
)

# Try pybind11 extension first, fall back to ctypes backend.
# _shimaenaga is a top-level sibling module (built by CMake into python/,
# alongside this package), not a submodule of shimaenaga -- so it must be
# imported absolutely, not via a relative "..".
try:
    from _shimaenaga import Dataset, Booster, Config
    _backend = "pybind11"
except ImportError:
    try:
        from ._ctypes_backend import Dataset, Booster
        Config = None
        _backend = "ctypes"
    except Exception:
        Dataset = Booster = Config = None
        _backend = "none"

__version__ = "1.3.0"
__all__ = [
    "ShimaenagaRegressor",
    "ShimaenagaClassifier",
    "ShimaenagaRanker",
    "Dataset",
    "Booster",
    "Config",
]
