"""Shared utilities for the CardI-HACK HCM progression models."""

import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split


# =============================================================================
# Data loading and shared configuration
# =============================================================================

# Reweighting for the two outcomes, used in both training and evaluation.
severity_weights = {0: 1.5, 1: 1.0}
mace_weights = {0: 1, 1: 3, 2: 4}


def load_data(data_dir: str = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(data_dir, "cardihack_final_train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "cardihack_final_test.csv"))
    return train, test


def get_snp_cols(train: pd.DataFrame) -> list[str]:
    return [c for c in train.columns if c.startswith("SNP")]


# =============================================================================
# SHAP aggregation
# =============================================================================

def _weighted_mean_abs_shap(shap_vals: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """
    Accepts:
      - 2D: (n_samples, n_features)
      - 3D: (n_samples, n_classes, n_features)  [multiclass]
    Returns:
      - (n_features,)
    """
    abs_vals = np.abs(shap_vals)

    if weights is None:
        # unweighted mean over samples (and classes if present)
        if abs_vals.ndim == 2:
            return abs_vals.mean(axis=0)
        elif abs_vals.ndim == 3:
            return abs_vals.mean(axis=(0, 1))
        else:
            raise ValueError(f"Unexpected shap_vals ndim={abs_vals.ndim}")

    w = np.asarray(weights, dtype=float).reshape(-1)
    denom = float(np.sum(w))
    if denom <= 0:
        if abs_vals.ndim == 2:
            return abs_vals.mean(axis=0)
        elif abs_vals.ndim == 3:
            return abs_vals.mean(axis=(0, 1))
        else:
            raise ValueError(f"Unexpected shap_vals ndim={abs_vals.ndim}")

    if abs_vals.ndim == 2:
        w2 = w.reshape(-1, 1)
        return (abs_vals * w2).sum(axis=0) / denom

    if abs_vals.ndim == 3:
    
        w3 = w.reshape(-1, 1, 1)
        weighted = (abs_vals * w3).sum(axis=0) / denom 
        return weighted.mean(axis=0)  

    raise ValueError(f"Unexpected shap_vals ndim={abs_vals.ndim}")


# =============================================================================
# OUTCOME SEVERITY metric: weighted log-loss, rescaled against baselines
# =============================================================================

def weighted_logloss_binary(y_true: np.ndarray, p1: np.ndarray, label_weights: dict[int, float]) -> float:
    # manual weighted logloss to avoid sklearn dependency
    eps = 1e-15
    p1 = np.clip(np.asarray(p1, dtype=float), eps, 1 - eps)
    y_true = np.asarray(y_true, dtype=int)
    w = np.array([label_weights[int(y)] for y in y_true], dtype=float)

    ll = -(y_true * np.log(p1) + (1 - y_true) * np.log(1 - p1))
    return float(np.sum(w * ll) / np.sum(w))


def weighted_dummy_logloss_binary(y_true: np.ndarray, label_weights: dict[int, float]) -> float:
    p = float(np.mean(np.asarray(y_true, dtype=int)))
    p1 = np.full(len(y_true), p, dtype=float)
    return weighted_logloss_binary(y_true, p1, label_weights)


def weighted_worst_logloss_binary(y_true: np.ndarray, label_weights: dict[int, float], eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=int)
    # worst predicts 1 when y=0, and 0 when y=1
    p1 = np.where(y_true == 1, eps, 1.0 - eps).astype(float)
    return weighted_logloss_binary(y_true, p1, label_weights)


def rescaled_weighted_logloss(weighted_ll: float, dummy_ll: float, worst_ll: float) -> float:
    if dummy_ll <= 0:
        return 0.0
    if weighted_ll <= dummy_ll:
        return float(1.0 - (weighted_ll / dummy_ll))
    denom = (worst_ll - dummy_ll)
    if denom <= 0:
        return float(-(weighted_ll - dummy_ll))
    return float(-(weighted_ll - dummy_ll) / denom)


# =============================================================================
# OUTCOME MACE metric: weighted quadratic kappa with ordinal thresholds
# =============================================================================

def _qwk_weights_matrix(n_classes: int = 3) -> np.ndarray:
    W = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    return W


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    n_classes = 3
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    W = _qwk_weights_matrix(n_classes)

    O = np.zeros((n_classes, n_classes), dtype=float)
    for yt, yp, w in zip(y_true, y_pred, sample_weight):
        O[yt, yp] += w

    hist_true = np.zeros(n_classes, dtype=float)
    hist_pred = np.zeros(n_classes, dtype=float)
    for yt, yp, w in zip(y_true, y_pred, sample_weight):
        hist_true[yt] += w
        hist_pred[yp] += w

    total = float(np.sum(sample_weight))
    if total <= 0:
        return 0.0

    E = np.outer(hist_true, hist_pred) / total
    num = float((W * O).sum())
    den = float((W * E).sum())
    if den == 0:
        return 0.0
    return float(1.0 - num / den)

def apply_thresholds_ordinal(score: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """
    Map a 1D ordinal score into classes 0/1/2 using thresholds.
    """
    score = np.asarray(score, dtype=float)
    yhat = np.zeros_like(score, dtype=int)
    yhat[score >= t1] = 1
    yhat[score >= t2] = 2
    return yhat

def optimize_thresholds_for_weighted_qwk(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray,
    n_grid: int = 40
) -> tuple[float, float, float]:
    """
    Threshold search on quantile grid:
    returns (best_t1, best_t2, best_qwk)
    """
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    qs = np.linspace(0.05, 0.95, n_grid)
    grid = np.quantile(score, qs)

    best = (-np.inf, None, None)  # (qwk, t1, t2)
    for i in range(len(grid)):
        for j in range(i + 1, len(grid)):
            t1, t2 = float(grid[i]), float(grid[j])
            yhat = apply_thresholds_ordinal(score, t1, t2)
            qwk = quadratic_weighted_kappa(y_true, yhat, sample_weight=sample_weight)
            if qwk > best[0]:
                best = (qwk, t1, t2)

    return best[1], best[2], float(best[0])


# =============================================================================
# LightGBM cross-validation
# =============================================================================

@dataclass(frozen=True)
class CVFold:
    """A single CV fold split (train/valid indices into the *training split* arrays)."""
    train_idx: np.ndarray
    valid_idx: np.ndarray

def make_stratified_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[CVFold]:
    """Return StratifiedKFold splits as a list."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [CVFold(tr_i, va_i) for tr_i, va_i in skf.split(np.zeros(len(y)), y)]

def train_one_fold_lgbm(
    *,
    params: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    feature_cols: Sequence[str],
    num_boost_round: int,
    early_stopping_rounds: int,
    verbose: bool,
    w_tr: Optional[np.ndarray] = None,
    w_va: Optional[np.ndarray] = None,
) -> tuple[lgb.Booster, int, np.ndarray]:
    """Train one LightGBM model on a fold and return (booster, best_it, pred_valid)."""
    dtrain = lgb.Dataset(
        X_tr, label=y_tr, weight=w_tr, feature_name=list(feature_cols), free_raw_data=True
    )
    dvalid = lgb.Dataset(
        X_va, label=y_va, weight=w_va, feature_name=list(feature_cols), reference=dtrain, free_raw_data=True
    )

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=verbose),
            lgb.log_evaluation(period=0 if not verbose else 50),
        ],
    )
    best_it = int(booster.best_iteration or num_boost_round)
    pred_va = booster.predict(X_va, num_iteration=best_it)
    return booster, best_it, pred_va

def sample_params_binary(rng: np.random.Generator) -> dict:
    """Sample parameters set for binary classification."""
    max_depth = int(rng.integers(3, 11))
    max_leaves = max(8, min(2 ** max_depth, 256))
    num_leaves = int(rng.integers(8, max_leaves + 1))
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": float(rng.uniform(0.01, 0.15)),
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_data_in_leaf": int(rng.integers(10, 120)),
        "min_sum_hessian_in_leaf": float(rng.uniform(1e-3, 5.0)),
        "feature_fraction": float(rng.uniform(0.6, 1.0)),
        "bagging_fraction": float(rng.uniform(0.6, 1.0)),
        "bagging_freq": int(rng.integers(1, 7)),
        "lambda_l1": float(rng.uniform(0.0, 10.0)),
        "lambda_l2": float(rng.uniform(0.0, 10.0)),
        "verbosity": -1,
    }

def sample_params_multiclass(rng: np.random.Generator, num_class: int = 3) -> dict:
    """Sample parameters set for multiclass classification."""
    max_depth = int(rng.integers(3, 11))
    max_leaves = max(8, min(2 ** max_depth, 256))
    num_leaves = int(rng.integers(8, max_leaves + 1))
    return {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": int(num_class),
        "boosting_type": "gbdt",
        "learning_rate": float(rng.uniform(0.01, 0.15)),
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_data_in_leaf": int(rng.integers(10, 120)),
        "min_sum_hessian_in_leaf": float(rng.uniform(1e-3, 5.0)),
        "feature_fraction": float(rng.uniform(0.6, 1.0)),
        "bagging_fraction": float(rng.uniform(0.6, 1.0)),
        "bagging_freq": int(rng.integers(1, 7)),
        "lambda_l1": float(rng.uniform(0.0, 10.0)),
        "lambda_l2": float(rng.uniform(0.0, 10.0)),
        "verbosity": -1,
    }


# =============================================================================
# SHAP-based SNP ranking
# =============================================================================

def _fit_lgbm_for_shap_binary(
    *,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    feature_cols: list[str],
    label_weights: dict[int, float],
    use_weights_for_training: bool,
    split_seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    """Train binary LightGBM with rescaled weighted logloss."""
    # Baselines for rescaling
    dummy_ll_tr = weighted_dummy_logloss_binary(y_tr, label_weights)
    worst_ll_tr = weighted_worst_logloss_binary(y_tr, label_weights)

    # Weights
    w_tr_eval = np.array([label_weights[int(y)] for y in y_tr], dtype=float)
    w_te_eval = np.array([label_weights[int(y)] for y in y_te], dtype=float)
    w_tr_fit = w_tr_eval if use_weights_for_training else None
    w_te_fit = w_te_eval if use_weights_for_training else None

    def feval_rescaled_wll(preds: np.ndarray, dataset: lgb.Dataset):
        """Metric used ONLY for early stopping / model selection."""
        y_true = dataset.get_label().astype(int)
        wll = weighted_logloss_binary(y_true, preds, label_weights)
        rll = rescaled_weighted_logloss(wll, dummy_ll_tr, worst_ll_tr)
        return ("rescaled_wll", float(rll), True)

    params = {
        "objective": "binary",
        "metric": "None",  # use custom only
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "lambda_l1": 0.0,
        "verbosity": -1,
        "seed": split_seed,
        "feature_pre_filter": False,
        "force_row_wise": True,
    }

    dtrain = lgb.Dataset(X_tr.values, label=y_tr, weight=w_tr_fit, feature_name=feature_cols, free_raw_data=True)
    dvalid = lgb.Dataset(X_te.values, label=y_te, weight=w_te_fit, feature_name=feature_cols, reference=dtrain, free_raw_data=True)

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=feval_rescaled_wll,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster

def _compute_shap_importance(
    booster: lgb.Booster,
    X: pd.DataFrame,
    feature_cols: list[str],
    *,
    use_weights_for_shap: bool,
    sample_weight: np.ndarray | None = None,
) -> pd.Series:
    """
    Binary and Multiclass
    """
    n_feat = len(feature_cols)

    contrib = booster.predict(X.values, pred_contrib=True)

    # binary
    if contrib.ndim == 2 and contrib.shape[1] == n_feat + 1:
        shap_vals = contrib[:, :-1]  # drop expected value column
        if use_weights_for_shap and sample_weight is not None:
            scores = _weighted_mean_abs_shap(shap_vals, sample_weight)
        else:
            scores = np.mean(np.abs(shap_vals), axis=0)
        return pd.Series(scores, index=feature_cols)

    # multiclass
    if contrib.ndim != 2 or contrib.shape[1] % (n_feat + 1) != 0:
        raise ValueError(
            f"Unexpected pred_contrib shape {contrib.shape} for n_feat={n_feat}. "
            "Expected (n, p+1) or (n, K*(p+1))."
        )

    n_classes = contrib.shape[1] // (n_feat + 1)
    contrib_3d = contrib.reshape(contrib.shape[0], n_classes, n_feat + 1)
    shap_3d = contrib_3d[:, :, :-1]

    if use_weights_for_shap and sample_weight is not None:
        scores = _weighted_mean_abs_shap(shap_3d, sample_weight) 
    else:
        scores = np.mean(np.abs(shap_3d), axis=(0, 1))  # mean over samples AND classes

    return pd.Series(scores, index=feature_cols)

def rank_snps_by_shap_severity_lgbm(
    train: pd.DataFrame,
    target_col: str,
    base_cols: list[str],
    snp_cols: list[str],
    severity_label_weights: dict[int, float] = {0: 1.5, 1: 1.0},
    use_weights_for_training: bool = True,
    use_weights_for_shap: bool = True,
    test_size: float = 0.2,
    split_seed: int = 42,
    num_boost_round: int = 4000,
    early_stopping_rounds: int = 120,
) -> pd.DataFrame:
    
    snp_cols = [c for c in snp_cols if c in train.columns]
    if not snp_cols:
        raise ValueError("No SNP columns found in train.")

    feature_cols = base_cols + snp_cols
    y_all = train[target_col].astype(int)

    tr_idx, te_idx = train_test_split(
        train.index, test_size=test_size, stratify=y_all, random_state=split_seed
    )

    X_tr = train.loc[tr_idx, feature_cols].copy()
    X_te = train.loc[te_idx, feature_cols].copy()
    y_tr = y_all.loc[tr_idx].values
    y_te = y_all.loc[te_idx].values

    booster = _fit_lgbm_for_shap_binary(
        X_tr=X_tr,
        y_tr=y_tr,
        X_te=X_te,
        y_te=y_te,
        feature_cols=feature_cols,
        label_weights=severity_label_weights,
        use_weights_for_training=use_weights_for_training,
        split_seed=split_seed,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )
    # Compute SHAP on the TRAIN split
    w_tr_eval = np.array([severity_label_weights[int(y)] for y in y_tr], dtype=float)
    shap_scores = _compute_shap_importance(
        booster,
        X_tr,
        feature_cols,
        use_weights_for_shap=use_weights_for_shap,
        sample_weight=w_tr_eval,
    )
    snp_scores = shap_scores.loc[snp_cols]
    rank_df = (
        snp_scores
        .rename("mean_abs_shap")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "snp"})
    )
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)

    return rank_df


def _fit_lgbm_for_shap_multiclass(
    *,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    y_te: np.ndarray,
    feature_cols: list[str],
    label_weights: dict[int, float],
    use_weights_for_training: bool,
    split_seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    """Train a multiclass LightGBM (0/1/2) with weighted QWK (thresholded)."""
    w_tr_eval = np.array([label_weights[int(y)] for y in y_tr], dtype=float)
    w_te_eval = np.array([label_weights[int(y)] for y in y_te], dtype=float)
    w_tr_fit = w_tr_eval if use_weights_for_training else None
    w_te_fit = w_te_eval if use_weights_for_training else None

    class_vals = np.array([0.0, 1.0, 2.0], dtype=float)

    def feval_weighted_qwk(preds: np.ndarray, dataset: lgb.Dataset):
        y_true = dataset.get_label().astype(int)
        proba = preds.reshape(len(y_true), 3, order="F")
        score = proba @ class_vals

        # Optimize thresholds on this eval set (quick grid) to approximate best QWK
        t1, t2, qwk = optimize_thresholds_for_weighted_qwk(
            y_true=y_true,
            score=score,
            sample_weight=dataset.get_weight(),
            n_grid=40
        )
        return ("weighted_qwk", float(qwk), True)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "None",  # custom only
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "lambda_l1": 0.0,
        "verbosity": -1,
        "seed": split_seed,
        "feature_pre_filter": False,
        "force_row_wise": True,
    }

    dtrain = lgb.Dataset(X_tr.values, label=y_tr, weight=w_tr_fit, feature_name=feature_cols, free_raw_data=True)
    dvalid = lgb.Dataset(X_te.values, label=y_te, weight=w_te_fit, feature_name=feature_cols, reference=dtrain, free_raw_data=True)

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        feval=feval_weighted_qwk,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return booster

def rank_snps_by_shap_mace_lgbm(
    train: pd.DataFrame,
    target_col: str,
    base_cols: list[str],
    snp_cols: list[str],
    mace_label_weights: dict[int, float] = {0: 1.0, 1: 3.0, 2: 4.0},
    use_weights_for_training: bool = True,
    use_weights_for_shap: bool = True,
    test_size: float = 0.2,
    split_seed: int = 42,
    num_boost_round: int = 4000,
    early_stopping_rounds: int = 120,
) -> pd.DataFrame:

    snp_cols = [c for c in snp_cols if c in train.columns]
    if not snp_cols:
        raise ValueError("No SNP columns found in train.")

    feature_cols = base_cols + snp_cols
    y_all = train[target_col].astype(int)

    tr_idx, te_idx = train_test_split(
        train.index, test_size=test_size, stratify=y_all, random_state=split_seed
    )

    X_tr = train.loc[tr_idx, feature_cols].copy()
    X_te = train.loc[te_idx, feature_cols].copy()
    y_tr = y_all.loc[tr_idx].values
    y_te = y_all.loc[te_idx].values

    booster = _fit_lgbm_for_shap_multiclass(
        X_tr=X_tr,
        y_tr=y_tr,
        X_te=X_te,
        y_te=y_te,
        feature_cols=feature_cols,
        label_weights=mace_label_weights,
        use_weights_for_training=use_weights_for_training,
        split_seed=split_seed,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    w_tr_eval = np.array([mace_label_weights[int(y)] for y in y_tr], dtype=float)
    shap_scores = _compute_shap_importance(
        booster,
        X_tr,
        feature_cols,
        use_weights_for_shap=use_weights_for_shap,
        sample_weight=w_tr_eval,
    )

    snp_scores = shap_scores.loc[snp_cols]
    rank_df = (
        snp_scores
        .rename("mean_abs_shap")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "snp"})
    )
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)

    return rank_df
