from datetime import datetime
import hashlib
import json
from typing import Any, Dict


def generate_model_version(prefix: str = "v") -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d-%H%M%S")
    return f"{prefix}{ts}"


def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat()