"""
Shimaenaga Python wrapper via ctypes.

pybind11 ビルドが不要な, ctypes ベースのスタンドアロン実装.
sklearn の BaseEstimator と同じインタフェースを提供する.
"""
import ctypes
import json
import os
import platform
import numpy as np

# ─── shared library loading ───────────────────────────────────────────────────

def _find_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ext = ".dylib" if platform.system() == "Darwin" else ".so"
    candidates = [
        os.path.join(root, "build", f"libshimaenaga_core{ext}"),
        os.path.join(root, f"libshimaenaga_core{ext}"),
        f"libshimaenaga_core{ext}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"libshimaenaga_core not found. Build with:\n"
        f"  cd {root}/build && cmake .. && make"
    )


_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        path = _find_lib()
        _lib = ctypes.CDLL(path)
        _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib):
    lib.SHIMAENAGA_GetLastError.restype  = ctypes.c_char_p
    lib.SHIMAENAGA_GetLastError.argtypes = []

    lib.SHIMAENAGA_DatasetCreate.restype  = ctypes.c_int
    lib.SHIMAENAGA_DatasetCreate.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # X
        ctypes.c_int64,                   # n
        ctypes.c_int32,                   # num_features
        ctypes.POINTER(ctypes.c_float),  # y
        ctypes.POINTER(ctypes.c_float),  # weights
        ctypes.POINTER(ctypes.c_int32),  # group_sizes
        ctypes.c_int32,                   # num_groups
        ctypes.c_char_p,                  # params_json
        ctypes.POINTER(ctypes.c_void_p), # out handle
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


def _np_ptr(arr, dtype):
    a = np.ascontiguousarray(arr, dtype=dtype)
    return a.ctypes.data_as(ctypes.POINTER(
        ctypes.c_float if dtype == np.float32 else
        ctypes.c_double if dtype == np.float64 else
        ctypes.c_int32
    )), a


# ─── Low-level Dataset/Booster handles ───────────────────────────────────────

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

        w_ptr = (y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if weights is None
                 else np.ascontiguousarray(weights, np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        g_ptr = None
        ng = 0
        if groups is not None:
            g_arr = np.ascontiguousarray(groups, np.int32)
            g_ptr = g_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            ng = len(g_arr)

        params_json = json.dumps(params or {}).encode()
        h = ctypes.c_void_p()
        _check(lib.SHIMAENAGA_DatasetCreate(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, nf,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            None if weights is None else np.ascontiguousarray(weights, np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            g_ptr, ng,
            params_json,
            ctypes.byref(h),
        ), lib)
        return Dataset(h, lib)

    def build_like(self, X, y, weights=None, groups=None):
        lib = self._lib
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        n = X.shape[0]
        g_ptr = None
        ng = 0
        if groups is not None:
            g_arr = np.ascontiguousarray(groups, np.int32)
            g_ptr = g_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            ng = len(g_arr)
        h = ctypes.c_void_p()
        _check(lib.SHIMAENAGA_DatasetCreateLike(
            self._h,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            None if weights is None else np.ascontiguousarray(weights, np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            g_ptr, ng,
            ctypes.byref(h),
        ), lib)
        return Dataset(h, lib)


class Booster:
    def __init__(self, train_ds: Dataset, params: dict = None):
        lib = _get_lib()
        self._lib = lib
        self._h = ctypes.c_void_p()
        params_json = json.dumps(params or {}).encode()
        _check(lib.SHIMAENAGA_BoosterCreate(train_ds._h, params_json, ctypes.byref(self._h)), lib)

    def __del__(self):
        if self._h:
            self._lib.SHIMAENAGA_BoosterFree(self._h)
            self._h = None

    def add_valid(self, valid_ds: Dataset):
        _check(self._lib.SHIMAENAGA_BoosterAddValid(self._h, valid_ds._h), self._lib)

    def train(self):
        _check(self._lib.SHIMAENAGA_BoosterTrain(self._h), self._lib)

    def predict(self, X) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        n, nf = X.shape
        num_cls = ctypes.c_int(0)
        self._lib.SHIMAENAGA_BoosterGetNumClasses(self._h, ctypes.byref(num_cls))
        C = max(1, num_cls.value)
        out = np.zeros(n * C, dtype=np.float64)
        _check(self._lib.SHIMAENAGA_BoosterPredict(
            self._h,
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n, nf,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ), self._lib)
        return out.reshape(n, C) if C > 1 else out

    def best_iteration(self) -> int:
        v = ctypes.c_int(0)
        self._lib.SHIMAENAGA_BoosterGetBestIteration(self._h, ctypes.byref(v))
        return v.value

    def save_model(self, path: str):
        _check(self._lib.SHIMAENAGA_BoosterSave(self._h, path.encode()), self._lib)

    def load_model(self, path: str):
        _check(self._lib.SHIMAENAGA_BoosterLoad(self._h, path.encode()), self._lib)


# ─── sklearn-style Estimators ────────────────────────────────────────────────

class _shimaenagaBase:
    """Base estimator for Shimaenaga sklearn-style API."""

    def __init__(
        self,
        objective:          str   = "regression",
        num_iterations:     int   = 300,
        learning_rate:      float = 0.05,
        tier:               int   = 1,
        num_tokens:         int   = 8,
        num_heads:          int   = 2,
        attention_mode:     str   = "qk_leaf",
        d_attn:             int   = 4,
        eta_attn:           float = 0.5,
        token_num_leaves:   int   = 31,
        gate_num_leaves:    int   = 31,
        inner_refit_steps:  int   = 2,
        min_data_in_leaf:   int   = 20,
        max_depth:          int   = -1,
        min_sum_hessian_in_leaf: float = 1e-3,
        lambda_l1:          float = 0.0,
        lambda_v:           float = 1.0,
        lambda_q:           float = 0.1,
        lambda_k:           float = 0.1,
        lambda_z:           float = 0.1,
        bagging_fraction:   float = 1.0,
        bagging_freq:       int   = 1,
        feature_fraction:   float = 1.0,
        huber_alpha:        float = 0.9,
        quantile_alpha:     float = 0.5,
        max_bin:            int   = 255,
        early_stopping_rounds: int = 0,
        seed:               int   = 0,
        **kwargs,
    ):
        self.objective           = objective
        self.num_iterations      = num_iterations
        self.learning_rate       = learning_rate
        self.tier                = tier
        self.num_tokens          = num_tokens
        self.num_heads           = num_heads
        self.attention_mode      = attention_mode
        self.d_attn              = d_attn
        self.eta_attn            = eta_attn
        self.token_num_leaves    = token_num_leaves
        self.gate_num_leaves     = gate_num_leaves
        self.inner_refit_steps   = inner_refit_steps
        self.min_data_in_leaf    = min_data_in_leaf
        self.max_depth           = max_depth
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.lambda_l1           = lambda_l1
        self.lambda_v            = lambda_v
        self.lambda_q            = lambda_q
        self.lambda_k            = lambda_k
        self.lambda_z            = lambda_z
        self.bagging_fraction    = bagging_fraction
        self.bagging_freq        = bagging_freq
        self.feature_fraction    = feature_fraction
        self.huber_alpha         = huber_alpha
        self.quantile_alpha      = quantile_alpha
        self.max_bin             = max_bin
        self.early_stopping_rounds = early_stopping_rounds
        self.seed                = seed
        self._extra              = kwargs
        self.booster_            = None
        self.train_ds_           = None

    def _params(self) -> dict:
        p = {k: v for k, v in self.__dict__.items()
             if not k.startswith("_") and k not in ("booster_", "train_ds_")}
        p.update(self._extra)
        # convert all values to string for C API
        return {k: str(v) for k, v in p.items()}

    def fit(self, X, y, sample_weight=None, eval_set=None, group=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        # group (query sizes) must reach Dataset.build for ranking, otherwise the
        # whole dataset is treated as one query and LambdaRank is meaningless.
        groups = np.asarray(group, np.int32) if group is not None else None
        self.train_ds_ = Dataset.build(X, y, weights=sample_weight,
                                       groups=groups, params=self._params())
        self.booster_  = Booster(self.train_ds_, self._params())

        if eval_set:
            for Xv, yv in eval_set:
                vds = self.train_ds_.build_like(
                    np.asarray(Xv, np.float32),
                    np.asarray(yv, np.float32).ravel(),
                )
                self.booster_.add_valid(vds)

        self.booster_.train()
        self.best_iteration_ = self.booster_.best_iteration()
        return self

    def save_model(self, path: str):
        if self.booster_ is None:
            raise RuntimeError("Not fitted yet")
        self.booster_.save_model(path)

    def load_model(self, path: str):
        if self.booster_ is None:
            raise RuntimeError("Fit first to initialise the booster, then load")
        self.booster_.load_model(path)

    def get_params(self, deep=True):
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and not k.endswith("_")}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class ShimaenagaRegressor(_shimaenagaBase):
    """Shimaenaga Regressor (L2 regression)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("objective", "regression")
        super().__init__(**kwargs)

    def predict(self, X) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("Not fitted yet")
        return self.booster_.predict(np.asarray(X, np.float32)).ravel()


class ShimaenagaClassifier(_shimaenagaBase):
    """Shimaenaga Classifier (binary or multiclass).

    Parameters
    ----------
    num_class : int
        1 for binary (default), K for K-class softmax.
    """

    def __init__(self, num_class: int = 1, **kwargs):
        self.num_class = num_class
        if num_class > 1:
            kwargs.setdefault("objective", "multiclass")
        else:
            kwargs.setdefault("objective", "binary")
        super().__init__(**kwargs)

    def _params(self) -> dict:
        p = super()._params()
        p["num_class"] = str(self.num_class)
        return p

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.num_class == 1:
            # binary: encode to 0/1
            y = (y == self.classes_[1]).astype(np.float32)
        else:
            # multiclass: encode to 0..K-1
            lut = {c: i for i, c in enumerate(self.classes_)}
            y = np.array([lut[yi] for yi in y], dtype=np.float32)
        return super().fit(X, y, **kwargs)

    def predict_proba(self, X) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("Not fitted yet")
        raw = self.booster_.predict(np.asarray(X, np.float32))
        if self.num_class == 1:
            # binary: sigmoid
            p = 1.0 / (1.0 + np.exp(-raw.ravel()))
            return np.column_stack([1 - p, p])
        else:
            # multiclass: softmax per row
            e = np.exp(raw - raw.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class ShimaenagaRanker(_shimaenagaBase):
    """Shimaenaga LambdaMART Ranker."""

    def __init__(self, **kwargs):
        kwargs.setdefault("objective", "lambdarank")
        super().__init__(**kwargs)

    def fit(self, X, y, group=None, **kwargs):
        if group is None:
            raise ValueError("group (query sizes) is required")
        return super().fit(X, y, group=group, **kwargs)

    def predict(self, X) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("Not fitted yet")
        return self.booster_.predict(np.asarray(X, np.float32)).ravel()
