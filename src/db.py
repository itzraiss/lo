from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

from .config import CONFIG


_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        logger.info("Connecting to MongoDB: {}", CONFIG.mongo_uri)
        _client = MongoClient(CONFIG.mongo_uri)
    return _client


def get_db():
    return get_client()[CONFIG.db_name]


def get_collection(name: str) -> Collection:
    return get_db()[name]


def ensure_indexes() -> None:
    contests = get_collection("contests")
    contests.create_index([("contest", ASCENDING)], unique=True)
    contests.create_index([("date", ASCENDING)], unique=False)

    predictions = get_collection("predictions")
    predictions.create_index([("generated_at", ASCENDING)], unique=False)
    predictions.create_index([("model_version", ASCENDING)], unique=False)

    for col in ["model_runs", "logs", "metrics"]:
        get_collection(col).create_index([("created_at", ASCENDING)], unique=False)

    logger.info("MongoDB indexes ensured")


def upsert_contest(doc: Dict[str, Any]) -> None:
    contests = get_collection("contests")
    doc = {**doc, "created_at": doc.get("created_at", datetime.utcnow())}
    contests.update_one({"contest": doc["contest"]}, {"$set": doc}, upsert=True)


def insert_prediction(doc: Dict[str, Any]) -> str:
    predictions = get_collection("predictions")
    doc = {**doc, "generated_at": doc.get("generated_at", datetime.utcnow())}
    res = predictions.insert_one(doc)
    return str(res.inserted_id)


def get_latest_prediction() -> Optional[Dict[str, Any]]:
    predictions = get_collection("predictions")
    return predictions.find_one(sort=[("generated_at", -1)])


def get_all_contests() -> List[Dict[str, Any]]:
    contests = get_collection("contests")
    return list(contests.find().sort("contest", ASCENDING))


def get_latest_contest_no() -> Optional[int]:
    contests = get_collection("contests")
    latest = contests.find_one(sort=[("contest", -1)])
    return latest["contest"] if latest else None