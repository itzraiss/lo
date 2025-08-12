from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from .config import CONFIG
from .db import get_all_contests, insert_prediction
from .features import FeatureConfig, build_presence_matrix, compute_basic_features
from .optimizer import optimize_tickets_ilp, greedy_tickets
from .simulate import monte_carlo_simulation


MODELS_DIR = "/workspace/models"


def load_latest_model_path() -> str:
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    if not files:
        raise RuntimeError("Nenhum modelo encontrado. Treine primeiro.")
    return os.path.join(MODELS_DIR, sorted(files)[-1])


def score_numbers(models_path: str) -> Tuple[np.ndarray, str, List[int]]:
    contests = get_all_contests()
    presence, _ = build_presence_matrix(contests)
    feat_cfg = FeatureConfig()
    features = compute_basic_features(presence, feat_cfg)

    models = joblib.load(models_path)
    X = features[models.feature_columns]
    Xs = models.scaler.transform(X) if models.scaler else X.values

    # Base proba from meta model
    meta_proba = models.meta_model.predict_proba(Xs)[:, 1]

    # Optional boosted models
    preds = [meta_proba]
    if models.xgb_model is not None:
        preds.append(models.xgb_model.predict(X[models.feature_columns]))
    if models.lgb_model is not None:
        preds.append(models.lgb_model.predict(X[models.feature_columns]))

    p = np.clip(np.mean(np.vstack(preds), axis=0), 1e-4, 0.5)
    # Normalize to be comparable, but we will use as weights
    p = p / p.sum()
    return p, models.version, features["number"].tolist()


def generate_tickets(p: np.ndarray, candidate_pool_size: int, overlap_max: int) -> List[List[int]]:
    # Candidate pool: top N by p
    idx_sorted = np.argsort(p)[::-1]
    pool_idx = idx_sorted[:candidate_pool_size]
    pool_numbers = [int(i + 1) for i in pool_idx]

    try:
        tickets = optimize_tickets_ilp(pool_numbers, p[pool_idx], overlap_max=overlap_max)
    except Exception as e:
        logger.warning("ILP falhou/foi lento, usando greedy: {}", e)
        tickets = greedy_tickets(pool_numbers, p[pool_idx], overlap_max=overlap_max)

    return tickets


def run_prediction(generate: bool, simulate: bool, save: bool) -> Dict:
    model_path = load_latest_model_path()
    p, model_version, numbers = score_numbers(model_path)

    tickets = []
    sim_results = {}

    if generate:
        tickets = generate_tickets(
            p,
            candidate_pool_size=CONFIG.candidate_pool_size,
            overlap_max=CONFIG.overlap_max,
        )

    if simulate and tickets:
        sim_results = monte_carlo_simulation(
            p_vector=p,
            tickets=tickets,
            n_sim=CONFIG.n_simulations,
            seed=CONFIG.random_seed,
        )

    criteria = {
        "thresholds": {
            "18": CONFIG.threshold_p_ge_18,
            "17": CONFIG.threshold_p_ge_17,
            "16": CONFIG.threshold_p_ge_16,
        }
    }

    confidence_passed = False
    if sim_results:
        confidence_passed = (
            sim_results.get("p_at_least_18", 0) >= CONFIG.threshold_p_ge_18
            or sim_results.get("p_at_least_17", 0) >= CONFIG.threshold_p_ge_17
            or sim_results.get("p_at_least_16", 0) >= CONFIG.threshold_p_ge_16
        )

    result_doc = {
        "generated_at": datetime.utcnow(),
        "model_version": model_version,
        "p_i": {str(i + 1): float(pi) for i, pi in enumerate(p)},
        "candidate_pool": list(map(int, np.argsort(p)[::-1][: CONFIG.candidate_pool_size] + 1)),
        "tickets": tickets,
        "simulations": sim_results,
        "confidence_passed": confidence_passed,
        "criteria": criteria,
    }

    if save:
        insert_prediction(result_doc)
        logger.info("Previsão salva em MongoDB.")

    return result_doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    out = run_prediction(args.generate, args.simulate, args.save)
    print(json.dumps(out, default=str)[:1000])