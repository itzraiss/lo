from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .db import get_all_contests
from .features import FeatureConfig, build_presence_matrix, compute_basic_features
from .utils import generate_model_version

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None


@dataclass
class TrainConfig:
    test_size: float = 0.15
    random_state: int = 42
    save_dir: str = "/workspace/models"


@dataclass
class TrainedModels:
    scaler: Optional[StandardScaler]
    baseline_weights: Dict[str, float]
    xgb_model: Optional[object]
    lgb_model: Optional[object]
    meta_model: Optional[LogisticRegression]
    feature_columns: List[str]
    version: str


BASELINE_FEATURES = [
    "freq_total",
    "freq_last_k",
    "last_seen_gap",
    "avg_gap",
    "freq_decay",
    "trend_slope",
]


def build_training_table(contests: List[Dict], feat_cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.Series]:
    presence, _ = build_presence_matrix(contests)
    features_df = compute_basic_features(presence, feat_cfg)

    # Target proxy: Use last draw presence as target for next draw probability learning surrogate.
    # For a simple tabular classifier, we simulate samples per number across rolling time windows is heavy.
    # Here, we treat each number as a single row with aggregated features and target as recent presence ratio.
    # For meta-learning, we aim to calibrate probabilities, not per-draw classification.

    last_k = min(feat_cfg.last_k_freq, presence.shape[0])
    recent_presence_ratio = presence[-last_k:].mean(axis=0)
    y = pd.Series(recent_presence_ratio, name="target_ratio")

    X = features_df[BASELINE_FEATURES].copy()
    return X, y


def train_models() -> TrainedModels:
    contests = get_all_contests()
    if len(contests) < 50:
        raise RuntimeError("Poucos concursos no banco. Faça a ingestão primeiro.")

    feat_cfg = FeatureConfig()
    X, y = build_training_table(contests, feat_cfg)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TrainConfig.test_size, random_state=TrainConfig.random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Baseline model: linear regression-like via logistic regression on ratios clipped
    y_clipped = y_train.clip(1e-4, 1 - 1e-4)
    meta_model = LogisticRegression(max_iter=200)
    meta_model.fit(X_train_s, (y_clipped > y_clipped.median()).astype(int))

    auc = roc_auc_score((y_val > y_val.median()).astype(int), meta_model.predict_proba(X_val_s)[:, 1])
    logger.info("Meta baseline AUC (median split): {:.3f}", auc)

    # Optional gradient boosted models trained to regress recent ratio
    xgb_model = None
    if xgb is not None:
        xgb_model = xgb.XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=TrainConfig.random_state,
            tree_method="hist",
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    lgb_model = None
    if lgb is not None:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=TrainConfig.random_state,
        )
        lgb_model.fit(X_train, y_train)

    version = generate_model_version()
    models = TrainedModels(
        scaler=scaler,
        baseline_weights={"auc_median": float(auc)},
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        meta_model=meta_model,
        feature_columns=list(X.columns),
        version=version,
    )

    os.makedirs(TrainConfig.save_dir, exist_ok=True)
    joblib.dump(models, os.path.join(TrainConfig.save_dir, f"{version}.joblib"))
    logger.info("Modelos salvos em {}/{}.joblib", TrainConfig.save_dir, version)

    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    if args.retrain:
        train_models()
        logger.info("Treino concluído.")
    else:
        logger.info("Use --retrain para treinar os modelos.")