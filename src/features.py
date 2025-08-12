from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FeatureConfig:
    recency_half_life: int = 200
    last_k_freq: int = 100
    trend_window: int = 200


def build_presence_matrix(contests: List[Dict]) -> Tuple[np.ndarray, List[int]]:
    # Rows = contests ordered by contest number; Cols = numbers 1..100
    df = pd.DataFrame(contests)
    df = df.sort_values("contest")
    numbers_series = df["numbers"]
    num_draws = len(df)
    presence = np.zeros((num_draws, 100), dtype=np.int8)
    for idx, nums in enumerate(numbers_series):
        for n in nums:
            if 1 <= n <= 100:
                presence[idx, n - 1] = 1
    return presence, df["contest"].tolist()


def compute_basic_features(presence: np.ndarray, cfg: FeatureConfig) -> pd.DataFrame:
    num_draws, num_numbers = presence.shape
    assert num_numbers == 100

    # Total frequency
    freq_total = presence.sum(axis=0)

    # Last K frequency
    k = min(cfg.last_k_freq, num_draws)
    freq_last_k = presence[-k:].sum(axis=0)

    # Gaps
    last_seen_gap = np.zeros(num_numbers, dtype=np.int32)
    avg_gap = np.zeros(num_numbers, dtype=np.float32)

    for i in range(num_numbers):
        idxs = np.where(presence[:, i] == 1)[0]
        if idxs.size == 0:
            last_seen_gap[i] = num_draws
            avg_gap[i] = float(num_draws)
        else:
            last_seen_gap[i] = num_draws - 1 - idxs.max()
            gaps = np.diff(np.concatenate([[-1], idxs, [num_draws - 1]]))
            avg_gap[i] = float(gaps.mean())

    # Exponential decay frequency (recency weighted)
    hl = max(1, cfg.recency_half_life)
    decay = 0.5 ** (np.arange(num_draws)[::-1] / hl)
    decay = decay / decay.sum()
    freq_decay = (presence * decay[:, None]).sum(axis=0)

    # Trend slope via linear regression on rolling sum
    w = min(cfg.trend_window, num_draws)
    rolling = pd.DataFrame(presence).rolling(window=w, min_periods=max(5, w // 5)).sum()
    x = np.arange(num_draws)
    x = (x - x.mean()) / (x.std() + 1e-9)
    x = x.reshape(-1, 1)
    slopes = np.zeros(num_numbers, dtype=np.float32)
    for i in range(num_numbers):
        y = rolling[i].fillna(rolling[i].mean()).values
        y = (y - y.mean()) / (y.std() + 1e-9)
        # Simple slope = cov(x,y)/var(x)= corr(x,y) since x is standardized
        slopes[i] = float(np.mean(x.squeeze() * y))

    # Assemble DataFrame
    data = {
        "number": np.arange(1, 101),
        "freq_total": freq_total,
        "freq_last_k": freq_last_k,
        "last_seen_gap": last_seen_gap,
        "avg_gap": avg_gap,
        "freq_decay": freq_decay,
        "trend_slope": slopes,
    }
    features_df = pd.DataFrame(data)
    return features_df


def compute_pairwise_cooccurrence(presence: np.ndarray) -> np.ndarray:
    # 100x100 co-occurrence counts (symmetric, diag = freq)
    co = presence.T @ presence
    np.fill_diagonal(co, 0)
    return co