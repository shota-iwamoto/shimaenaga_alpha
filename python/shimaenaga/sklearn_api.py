"""sklearn-compatible wrappers for Shimaenaga (基本設計書 §5.2)."""
import inspect
import numpy as np
from typing import Optional, List, Union

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    class BaseEstimator: pass
    class ClassifierMixin: pass
    class RegressorMixin: pass

try:
    from _shimaenaga import Dataset, Booster, Config
    _has_cpp = True
except ImportError:
    try:
        from ._ctypes_backend import Dataset, Booster
        _has_cpp = True
    except Exception:
        _has_cpp = False


def _make_params(kwargs: dict) -> dict:
    """Convert kwargs to string->string map for C++ Config."""
    def _s(v):
        # C++ Config::FromMap parses bools as "true"/"1" — str(True) == "True"
        # would silently read as false.
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)
    return {k: _s(v) for k, v in kwargs.items() if v is not None}


class _shimaenagaBase(BaseEstimator):
    """Base class for all Shimaenaga estimators."""

    def __init__(
        self,
        objective: str = "regression",
        num_iterations: int = 1000,
        learning_rate: float = 0.05,
        tier: int = 2,
        num_tokens: int = 8,
        num_heads: int = 2,
        attention_mode: str = "qk_leaf",
        d_attn: int = 4,
        eta_attn: float = 0.5,
        attn_mask: str = "full",
        token_num_leaves: int = 31,
        gate_num_leaves: int = 31,
        inner_refit_steps: int = 2,
        min_data_in_leaf: int = 20,
        lambda_v: float = 1.0,
        lambda_q: float = 0.1,
        lambda_k: float = 0.1,
        lambda_z: float = 0.1,
        lambda_ent: float = 1e-3,
        lambda_div: float = 1e-3,
        max_bin: int = 255,
        early_stopping_rounds: int = 0,
        num_threads: int = 0,
        seed: int = 0,
        verbose: int = 1,
        # ── Tier-3 (tier=3 のみ有効; Tier-3 基本設計書 §7) ──
        attn_layers: int = 2,
        d_hidden: int = 8,
        d_ffn: int = 16,
        eta_u: float = 0.5,
        eta_ffn: float = 0.5,
        eta_cls: float = 0.1,
        use_cls_token: bool = True,
        tau_learnable: bool = True,
        lambda_e: float = 0.1,
        lambda_W: float = 0.1,
        lambda_c: float = 0.1,
        lambda_cls: float = 1.0,
        lambda_tau: float = 1e-2,
        spectral_max: float = 2.0,
        ctx_warmup: int = -1,
        **kwargs,
    ):
        self.objective = objective
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.tier = tier
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.attention_mode = attention_mode
        self.d_attn = d_attn
        self.eta_attn = eta_attn
        self.attn_mask = attn_mask
        self.token_num_leaves = token_num_leaves
        self.gate_num_leaves = gate_num_leaves
        self.inner_refit_steps = inner_refit_steps
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_v = lambda_v
        self.lambda_q = lambda_q
        self.lambda_k = lambda_k
        self.lambda_z = lambda_z
        self.lambda_ent = lambda_ent
        self.lambda_div = lambda_div
        self.max_bin = max_bin
        self.early_stopping_rounds = early_stopping_rounds
        self.num_threads = num_threads
        self.seed = seed
        self.verbose = verbose
        self.attn_layers = attn_layers
        self.d_hidden = d_hidden
        self.d_ffn = d_ffn
        self.eta_u = eta_u
        self.eta_ffn = eta_ffn
        self.eta_cls = eta_cls
        self.use_cls_token = use_cls_token
        self.tau_learnable = tau_learnable
        self.lambda_e = lambda_e
        self.lambda_W = lambda_W
        self.lambda_c = lambda_c
        self.lambda_cls = lambda_cls
        self.lambda_tau = lambda_tau
        self.spectral_max = spectral_max
        self.ctx_warmup = ctx_warmup
        self._extra_kwargs = kwargs
        self.booster_ = None
        self.train_dataset_ = None

    # ── sklearn parameter protocol ────────────────────────────────────────
    # The leaf estimators (ShimaenagaRegressor/Classifier/Ranker) accept **kwargs,
    # which sklearn's default _get_param_names() cannot introspect. Without the
    # overrides below, sklearn.clone() — used by Bagging/Stacking/GridSearchCV/
    # cross_val_score — would silently drop every hyperparameter and refit with
    # defaults. We therefore expose the canonical _shimaenagaBase signature explicitly.
    @classmethod
    def _param_names(cls):
        sig = inspect.signature(_shimaenagaBase.__init__)
        return [name for name, p in sig.parameters.items()
                if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
                and name != "self"]

    def get_params(self, deep=True):
        out = {name: getattr(self, name) for name in self._param_names()}
        if hasattr(self, "num_class"):
            out["num_class"] = self.num_class
        # passthrough kwargs forwarded verbatim to the C++ Config
        out.update(getattr(self, "_extra_kwargs", {}) or {})
        return out

    def set_params(self, **params):
        known = set(self._param_names())
        if hasattr(self, "num_class"):
            known.add("num_class")
        for key, value in params.items():
            if key in known:
                setattr(self, key, value)
            else:
                self._extra_kwargs[key] = value
        return self

    def _get_params_map(self) -> dict:
        params = {
            "objective": self.objective,
            "num_iterations": self.num_iterations,
            "learning_rate": self.learning_rate,
            "tier": self.tier,
            "num_tokens": self.num_tokens,
            "num_heads": self.num_heads,
            "attention_mode": self.attention_mode,
            "d_attn": self.d_attn,
            "eta_attn": self.eta_attn,
            "attn_mask": self.attn_mask,
            "token_num_leaves": self.token_num_leaves,
            "gate_num_leaves": self.gate_num_leaves,
            "inner_refit_steps": self.inner_refit_steps,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_v": self.lambda_v,
            "lambda_q": self.lambda_q,
            "lambda_k": self.lambda_k,
            "lambda_z": self.lambda_z,
            "lambda_ent": self.lambda_ent,
            "lambda_div": self.lambda_div,
            "max_bin": self.max_bin,
            "early_stopping_rounds": self.early_stopping_rounds,
            "num_threads": self.num_threads,
            "seed": self.seed,
            "attn_layers": self.attn_layers,
            "d_hidden": self.d_hidden,
            "d_ffn": self.d_ffn,
            "eta_u": self.eta_u,
            "eta_ffn": self.eta_ffn,
            "eta_cls": self.eta_cls,
            "use_cls_token": self.use_cls_token,
            "tau_learnable": self.tau_learnable,
            "lambda_e": self.lambda_e,
            "lambda_W": self.lambda_W,
            "lambda_c": self.lambda_c,
            "lambda_cls": self.lambda_cls,
            "lambda_tau": self.lambda_tau,
            "spectral_max": self.spectral_max,
            "ctx_warmup": self.ctx_warmup,
        }
        params.update(self._extra_kwargs)
        return _make_params(params)

    def fit(self, X, y, sample_weight=None, eval_set=None,
            early_stopping_rounds=None, group=None, eval_group=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), optional
        eval_set : list of (X, y) tuples for validation
        early_stopping_rounds : int, optional (overrides constructor param)
        group : array-like of group sizes for ranking
        eval_group : list of group-size arrays, one per eval_set entry
            (required for ranking with eval_set — otherwise the validation
            NDCG would treat the whole eval set as a single query)
        """
        if not _has_cpp:
            raise RuntimeError("Shimaenaga C++ extension not built. "
                               "Run: cmake && make")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if early_stopping_rounds is not None:
            self.early_stopping_rounds = early_stopping_rounds

        params = self._get_params_map()

        # Build train dataset
        group_arr = None
        if group is not None:
            group_arr = np.asarray(group, dtype=np.int32)

        is_ranking = str(params.get("objective", "")) in ("lambdarank", "rank")
        if is_ranking and eval_set is not None and eval_group is None:
            raise ValueError(
                "eval_group is required when using eval_set with a ranking "
                "objective (one array of query sizes per eval_set entry)")
        if eval_set is not None and eval_group is not None and \
                len(eval_group) != len(eval_set):
            raise ValueError("eval_group must have one entry per eval_set entry")

        self.train_dataset_ = Dataset.build(
            X, y,
            weights=sample_weight,
            groups=group_arr,
            params=params,
        )

        self.booster_ = Booster(self.train_dataset_, params)

        # Validation set
        if eval_set is not None:
            for si, (Xv, yv) in enumerate(eval_set):
                Xv = np.asarray(Xv, dtype=np.float32)
                yv = np.asarray(yv, dtype=np.float32)
                gv = None
                if eval_group is not None:
                    gv = np.asarray(eval_group[si], dtype=np.int32)
                valid_ds = self.train_dataset_.build_like(Xv, yv, groups=gv)
                self.booster_.add_valid(valid_ds)

        self.booster_.train()
        self.best_iteration_ = self.booster_.best_iteration()
        return self

    def save_model(self, path: str):
        """Save model to binary file."""
        if self.booster_ is None:
            raise RuntimeError("Model not trained yet")
        self.booster_.save_model(path)

    def load_model(self, path: str):
        """Load model from binary file."""
        if self.booster_ is None:
            # Create a dummy booster
            raise RuntimeError("Create a dataset and booster first before loading")
        self.booster_.load_model(path)

    def attention_diagnostics(self, X):
        """Return attention weights β_{ip} for each sample and token.

        Returns
        -------
        dict with keys:
          'beta' : ndarray of shape (n_samples, P) — sample i's attention
                   weight on token p, averaged over boosting blocks.
                   Rows sum to 1.
        """
        if self.booster_ is None:
            raise RuntimeError("Model not trained yet")
        X = np.asarray(X, dtype=np.float32)
        _, beta = self.booster_.predict_contrib(X)
        return {"beta": beta}


class ShimaenagaRegressor(RegressorMixin, _shimaenagaBase):
    """Shimaenaga Regressor (sklearn-compatible).

    Examples
    --------
    >>> from shimaenaga import ShimaenagaRegressor
    >>> model = ShimaenagaRegressor(num_iterations=100, tier=1)
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("objective", "regression")
        super().__init__(**kwargs)

    def predict(self, X) -> np.ndarray:
        """Predict regression values."""
        if self.booster_ is None:
            raise RuntimeError("Model not trained yet")
        X = np.asarray(X, dtype=np.float32)
        return np.array(self.booster_.predict(X))


class ShimaenagaClassifier(ClassifierMixin, _shimaenagaBase):
    """Shimaenaga Classifier (sklearn-compatible).

    Parameters
    ----------
    num_class : int
        Number of classes. 1 = binary (default), >1 = multiclass.

    Examples
    --------
    >>> from shimaenaga import ShimaenagaClassifier
    >>> model = ShimaenagaClassifier(tier=2, num_tokens=8)
    >>> model.fit(X_train, y_train)
    >>> proba = model.predict_proba(X_test)
    """

    def __init__(self, num_class: int = 1, **kwargs):
        self.num_class = num_class
        if num_class > 1:
            kwargs.setdefault("objective", "multiclass")
        else:
            kwargs.setdefault("objective", "binary")
        super().__init__(**kwargs)

    def _get_params_map(self) -> dict:
        params = super()._get_params_map()
        params["num_class"] = str(self.num_class)
        return params

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        if y.dtype.kind not in ('i', 'u'):
            y = y.astype(np.int32)
        if self.num_class == 1 and len(np.unique(y)) > 2:
            raise ValueError("num_class=1 (binary) but y has >2 unique values. "
                             "Set num_class=K for multiclass.")
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            y = (y == self.classes_[1]).astype(np.float32)
        else:
            # Encode to 0..K-1
            class_map = {c: i for i, c in enumerate(self.classes_)}
            y = np.array([class_map[yi] for yi in y], dtype=np.float32)
        return super().fit(X, y, **kwargs)

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if self.booster_ is None:
            raise RuntimeError("Model not trained yet")
        X = np.asarray(X, dtype=np.float32)
        return np.array(self.booster_.predict_proba(X))


class ShimaenagaRanker(_shimaenagaBase):
    """Shimaenaga LambdaMART Ranker (sklearn-compatible).

    The ``group`` parameter specifies the number of samples per query.

    Examples
    --------
    >>> from shimaenaga import ShimaenagaRanker
    >>> model = ShimaenagaRanker(tier=2)
    >>> model.fit(X_train, y_train, group=group_train)
    >>> scores = model.predict(X_test)
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("objective", "lambdarank")
        super().__init__(**kwargs)

    def fit(self, X, y, group=None, **kwargs):
        """Fit ranker.

        Parameters
        ----------
        group : array-like of int
            Number of samples per query group (required for ranking).
        """
        if group is None:
            raise ValueError("group (query sizes) is required for ranking")
        return super().fit(X, y, group=group, **kwargs)

    def predict(self, X) -> np.ndarray:
        """Predict relevance scores."""
        if self.booster_ is None:
            raise RuntimeError("Model not trained yet")
        X = np.asarray(X, dtype=np.float32)
        return np.array(self.booster_.predict(X))
