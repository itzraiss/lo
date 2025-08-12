import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def getenv_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return float(default)


def getenv_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return int(default)


@dataclass(frozen=True)
class AppConfig:
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name: str = os.getenv("DB_NAME", "lotomania")

    threshold_p_ge_18: float = getenv_float("THRESHOLD_P_GE_18", 0.02)
    threshold_p_ge_17: float = getenv_float("THRESHOLD_P_GE_17", 0.10)
    threshold_p_ge_16: float = getenv_float("THRESHOLD_P_GE_16", 0.50)

    candidate_pool_size: int = getenv_int("CANDIDATE_POOL_SIZE", 80)
    overlap_max: int = getenv_int("OVERLAP_MAX", 40)

    n_simulations: int = getenv_int("N_SIMULATIONS", 100000)
    random_seed: int = getenv_int("RANDOM_SEED", 42)


CONFIG = AppConfig()