from __future__ import annotations

from typing import Dict, List

import numpy as np


def weighted_sample_without_replacement(weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    # Use numpy choice with replace=False and normalized weights as approximation
    w = weights.astype(float)
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    return rng.choice(np.arange(1, 101), size=k, replace=False, p=w)


def evaluate_tickets(true_draw: np.ndarray, tickets: List[List[int]]) -> Dict[str, int]:
    res = {"max_hits": 0, ">=16": 0, ">=17": 0, ">=18": 0}
    true_set = set(int(x) for x in true_draw)
    for t in tickets:
        hits = len(true_set.intersection(set(t)))
        res["max_hits"] = max(res["max_hits"], hits)
        if hits >= 16:
            res[">=16"] += 1
        if hits >= 17:
            res[">=17"] += 1
        if hits >= 18:
            res[">=18"] += 1
    return res


def monte_carlo_simulation(p_vector: np.ndarray, tickets: List[List[int]], n_sim: int = 100000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    counts = {"at_least_16": 0, "at_least_17": 0, "at_least_18": 0}

    for _ in range(n_sim):
        draw = weighted_sample_without_replacement(p_vector, k=20, rng=rng)
        true_set = set(int(x) for x in draw)
        max_hits = 0
        for t in tickets:
            hits = len(true_set.intersection(set(t)))
            if hits > max_hits:
                max_hits = hits
        if max_hits >= 16:
            counts["at_least_16"] += 1
        if max_hits >= 17:
            counts["at_least_17"] += 1
        if max_hits >= 18:
            counts["at_least_18"] += 1

    return {
        "n": int(n_sim),
        "p_at_least_16": counts["at_least_16"] / n_sim,
        "p_at_least_17": counts["at_least_17"] / n_sim,
        "p_at_least_18": counts["at_least_18"] / n_sim,
    }