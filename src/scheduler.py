from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from .update_from_api import main as update_from_api_main
from .train import train_models
from .predict import run_prediction


def _parse_cron(expr: str) -> dict:
    # Very small parser for 5-field cron: min hour dom mon dow
    # APScheduler uses fields: minute, hour, day, month, day_of_week
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError("Cron inválida. Use 'm h dom mon dow'.")
    return {
        "minute": parts[0],
        "hour": parts[1],
        "day": parts[2],
        "month": parts[3],
        "day_of_week": parts[4],
    }


@dataclass
class ScheduleConfig:
    ingest_cron: str = os.getenv("SCHEDULE_INGEST_CRON", "0 10 * * *")
    train_cron: str = os.getenv("SCHEDULE_TRAIN_CRON", "30 10 * * *")
    predict_cron: str = os.getenv("SCHEDULE_PREDICT_CRON", "0 11 * * *")


def create_scheduler() -> BackgroundScheduler:
    cfg = ScheduleConfig()
    scheduler = BackgroundScheduler(timezone="UTC")

    def job_ingest():
        try:
            logger.info("[Scheduler] Ingest update_from_api start")
            update_from_api_main()
            logger.info("[Scheduler] Ingest done")
        except Exception as e:
            logger.exception("[Scheduler] Ingest failed: {}", e)

    def job_train():
        try:
            logger.info("[Scheduler] Train start")
            train_models()
            logger.info("[Scheduler] Train done")
        except Exception as e:
            logger.exception("[Scheduler] Train failed: {}", e)

    def job_predict():
        try:
            logger.info("[Scheduler] Predict start")
            run_prediction(generate=True, simulate=True, save=True)
            logger.info("[Scheduler] Predict done")
        except Exception as e:
            logger.exception("[Scheduler] Predict failed: {}", e)

    scheduler.add_job(job_ingest, "cron", **_parse_cron(cfg.ingest_cron), id="ingest")
    scheduler.add_job(job_train, "cron", **_parse_cron(cfg.train_cron), id="train")
    scheduler.add_job(job_predict, "cron", **_parse_cron(cfg.predict_cron), id="predict")

    return scheduler