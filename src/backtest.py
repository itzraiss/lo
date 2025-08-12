from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from .config import CONFIG
from .db import get_all_contests
from .features import FeatureConfig, build_presence_matrix, compute_basic_features
from .optimizer import greedy_tickets, optimize_tickets_ilp
from .simulate import monte_carlo_simulation


@dataclass
class BacktestConfig:
    start_window: int = 300
    step: int = 1
    sims: int = 50000


def run_backtest(start_window: int, step: int, sims: int) -> Dict:
    contests = get_all_contests()
    if len(contests) < start_window + 1:
        raise RuntimeError("Poucos concursos para backtest.")

    presence, contest_ids = build_presence_matrix(contests)

    results: List[Dict] = []

    for t in range(start_window, presence.shape[0] - 1, step):
        pres_train = presence[:t]
        feat = compute_basic_features(pres_train, FeatureConfig())
        # Simple probability proxy: normalized decayed frequency
        p = feat["freq_decay"].values
        p = np.clip(p, 1e-6, None)
        p = p / p.sum()

        # Candidate pool and tickets
        idx_sorted = np.argsort(p)[::-1]
        pool_idx = idx_sorted[: CONFIG.candidate_pool_size]
        pool_numbers = [int(i + 1) for i in pool_idx]
        try:
            tickets = optimize_tickets_ilp(pool_numbers, p[pool_idx], overlap_max=CONFIG.overlap_max)
        except Exception:
            tickets = greedy_tickets(pool_numbers, p[pool_idx], overlap_max=CONFIG.overlap_max)

        # Evaluate against the true next draw
        true_draw = np.where(presence[t + 1] == 1)[0] + 1
        max_hits = 0
        for tick in tickets:
            hits = len(set(tick).intersection(set(true_draw.tolist())))
            max_hits = max(max_hits, hits)

        # Monte Carlo estimate for that step (optional; can be heavy)
        sim = monte_carlo_simulation(p, tickets, n_sim=sims, seed=42)

        results.append({
            "contest_eval": int(contest_ids[t + 1]),
            "max_hits": max_hits,
            "sim": sim,
        })

    df = pd.DataFrame(results)
    df["ge_16"] = (df["max_hits"] >= 16).astype(int)
    df["ge_17"] = (df["max_hits"] >= 17).astype(int)
    df["ge_18"] = (df["max_hits"] >= 18).astype(int)

    summary = {
        "n_steps": int(len(df)),
        "freq_ge_16": float(df["ge_16"].mean()),
        "freq_ge_17": float(df["ge_17"].mean()),
        "freq_ge_18": float(df["ge_18"].mean()),
        "avg_p_ge_16": float(np.mean([r["sim"]["p_at_least_16"] for r in results])),
        "avg_p_ge_17": float(np.mean([r["sim"]["p_at_least_17"] for r in results])),
        "avg_p_ge_18": float(np.mean([r["sim"]["p_at_least_18"] for r in results])),
    }

    logger.info("Backtest summary: {}", summary)
    return {"summary": summary, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-window", type=int, default=BacktestConfig.start_window)
    parser.add_argument("--step", type=int, default=BacktestConfig.step)
    parser.add_argument("--sims", type=int, default=BacktestConfig.sims)
    args = parser.parse_args()
    out = run_backtest(args.start_window, args.step, args.sims)
    print(str(out)[0:1200])