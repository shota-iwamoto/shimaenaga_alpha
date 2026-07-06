"""ctypes-based backend for Shimaenaga (no pybind11 required)."""
import ctypes
import json
import os
import platform
import numpy as np


def _find_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    # Search order: next to this file, project build/, project root
    pkg_root = os.path.dirname(os.path.dirname(here))   # …/Shimaenaga
    candidates = [
        os.path.join(here, f"libshimaenaga_core{ext}"),
        os.path.join(pkg_root, "build", f"libshimaenaga_core{ext}"),
        os.path.join(pkg_root, f"libshimaenaga_core{ext}"),
        f"libshimaenaga_core{ext}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"libshimaenaga_core{ext} not found.\n"
        f"Build it with:\n"
        f"  cd {pkg_root}/build && cmake .. && make shimaenaga_core"
    )


_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_find_lib())
        _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib):
    lib.SHIMAENAGA_GetLastError.restype  = ctypes.c_char_p
    lib.SHIMAENAGA_GetLastError.argtypes = []

    lib.SHIMAENAGA_DatasetCreate.restype  = ctypes.c_int
    lib.SHIMAENAGA_DatasetCreate.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.SHIMAENAGA_DatasetCreateLike.restype  = ctypes.c_int
    lib.SHIMAENAGA_DatasetCreateLike.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.SHIMAENAGA_DatasetFree.restype  = ctypes.c_int
    lib.SHIMAENAGA_DatasetFree.argtypes = [ctypes.c_void_p]

    lib.SHIMAENAGA_BoosterCreate.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterCreate.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.SHIMAENAGA_BoosterAddValid.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterAddValid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.SHIMAENAGA_BoosterTrain.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterTrain.argtypes = [ctypes.c_void_p]
    lib.SHIMAENAGA_BoosterFree.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterFree.argtypes = [ctypes.c_void_p]
    lib.SHIMAENAGA_BoosterPredict.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterPredict.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.SHIMAENAGA_BoosterPredictContrib.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterPredictContrib.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
    ]
    lib.SHIMAENAGA_BoosterGetNumTokens.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterGetNumTokens.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.SHIMAENAGA_DatasetGetNumFeatures.restype  = ctypes.c_int
    lib.SHIMAENAGA_DatasetGetNumFeatures.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.SHIMAENAGA_BoosterSave.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterSave.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.SHIMAENAGA_BoosterLoad.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterLoad.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.SHIMAENAGA_BoosterGetBestIteration.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterGetBestIteration.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.SHIMAENAGA_BoosterGetNumClasses.restype  = ctypes.c_int
    lib.SHIMAENAGA_BoosterGetNumClasses.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]


def _check(ret, lib):
    if ret != 0:
        msg = lib.SHIMAENAGA_GetLastError()
        raise RuntimeError(f"Shimaenaga error {ret}: {msg.decode() if msg else 'unknown'}")


class Dataset:
    def __init__(self, handle, lib):
        self._h = handle
        self._lib = lib

    def __del__(self):
        if self._h:
            self._lib.SHIMAENAGA_DatasetFree(self._h)
            self._h = None

    @staticmethod
    def build(X, y, weights=None, groups=None, params: dict = None):
        lib = _get_lib()
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        n, nf = X.shape

        g_ptr, ng = None, 0
        if groups is not None:
            g_arr = np.ascontiguousarray(groups, np.int32)
            g_ptr = g_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            ng = len(g_arr)

        w_arr = (np.ascontiguousarray(weights, np.float32) if weights is not None else None)
        w_ptr = (w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                 if w_arr is not None else None)

        params_json = json.dumps(params or {}).encode()
        h = ctypes.c_void_p()
        _check(lib.SHIMAENAGA_DatasetCreate(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, nf,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w_ptr, g_ptr, ng,
            params_json, ctypes.byref(h),
        ), lib)
        return Dataset(h, lib)

    def num_features(self) -> int:
        v = ctypes.c_int(0)
        self._lib.SHIMAENAGA_DatasetGetNumFeatures(self._h, ctypes.byref(v))
        return v.value

    def build_like(self, X, y, weights=None, groups=None):
        lib = self._lib
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        n = X.shape[0]
        nf_train = self.num_features()
        if X.ndim != 2 or X.shape[1] != nf_train:
            raise ValueError(
                f"build_like: X has {X.shape[1] if X.ndim == 2 else 'invalid'} "
                f"features but the training dataset has {nf_train}")

        g_ptr, ng = None, 0
        if groups is not None:
            g_arr = np.ascontiguousarray(groups, np.int32)
            g_ptr = g_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            ng = len(g_arr)

        w_arr = (np.ascontiguousarray(weights, np.float32) if weights is not None else None)
        w_ptr = (w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                 if w_arr is not None else None)

        h = ctypes.c_void_p()
        _check(lib.SHIMAENAGA_DatasetCreateLike(
            self._h,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w_ptr, g_ptr, ng,
            ctypes.byref(h),
        ), lib)
        return Dataset(h, lib)


class Booster:
    def __init__(self, train_ds: "Dataset", params: dict = None):
        lib = _get_lib()
        self._lib = lib
        self._h = ctypes.c_void_p()
        params_json = json.dumps(params or {}).encode()
        _check(lib.SHIMAENAGA_BoosterCreate(train_ds._h, params_json, ctypes.byref(self._h)), lib)

    def __del__(self):
        if self._h:
            self._lib.SHIMAENAGA_BoosterFree(self._h)
            self._h = None

    def add_valid(self, valid_ds: "Dataset"):
        _check(self._lib.SHIMAENAGA_BoosterAddValid(self._h, valid_ds._h), self._lib)

    def train(self):
        _check(self._lib.SHIMAENAGA_BoosterTrain(self._h), self._lib)

    def num_classes(self) -> int:
        v = ctypes.c_int(0)
        self._lib.SHIMAENAGA_BoosterGetNumClasses(self._h, ctypes.byref(v))
        return max(1, v.value)

    def predict(self, X) -> np.ndarray:
        """Raw score prediction — shape (n,) for regression/binary, (n, C) for multiclass."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, nf = X.shape
        C = self.num_classes()
        out = np.zeros(n * C, dtype=np.float64)
        _check(self._lib.SHIMAENAGA_BoosterPredict(
            self._h,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, nf,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ), self._lib)
        return out.reshape(n, C) if C > 1 else out

    def predict_proba(self, X) -> np.ndarray:
        """Probability output — applies sigmoid (binary) or softmax (multiclass)."""
        raw = self.predict(X)
        C = self.num_classes()
        if C == 1:
            p = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1 - p, p])
        # softmax row-wise
        e = np.exp(raw - raw.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def num_tokens(self) -> int:
        v = ctypes.c_int(0)
        self._lib.SHIMAENAGA_BoosterGetNumTokens(self._h, ctypes.byref(v))
        return v.value

    def predict_contrib(self, X):
        """Scores + attention diagnostics.

        Returns (scores, beta) where beta[i, p] is sample i's attention weight
        on token p, averaged over boosting blocks (rows sum to 1).
        """
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, nf = X.shape
        C = self.num_classes()
        P = self.num_tokens()
        out = np.zeros(n * C, dtype=np.float64)
        beta = np.zeros(n * max(P, 1), dtype=np.float32)
        _check(self._lib.SHIMAENAGA_BoosterPredictContrib(
            self._h,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, nf,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ), self._lib)
        scores = out.reshape(n, C) if C > 1 else out
        return scores, beta.reshape(n, max(P, 1))

    def best_iteration(self) -> int:
        v = ctypes.c_int(0)
        self._lib.SHIMAENAGA_BoosterGetBestIteration(self._h, ctypes.byref(v))
        return v.value

    def save_model(self, path: str):
        _check(self._lib.SHIMAENAGA_BoosterSave(self._h, path.encode()), self._lib)

    def load_model(self, path: str):
        _check(self._lib.SHIMAENAGA_BoosterLoad(self._h, path.encode()), self._lib)
