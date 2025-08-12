import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from .db import ensure_indexes, get_latest_contest_no, upsert_contest

API_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/home/ultimos-resultados"


def fetch_latest_results() -> Dict[str, Any]:
    headers = {"accept": "application/json"}
    resp = requests.get(API_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_lotomania_entries(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # The payload includes multiple games. We try to find 'lotomania' key or entries.
    results: List[Dict[str, Any]] = []
    try:
        # Common shape: payload["lotomania"] or inside a list
        if "lotomania" in payload:
            data = payload["lotomania"]
            if isinstance(data, dict):
                results.append(data)
            elif isinstance(data, list):
                results.extend(data)
        elif "loterias" in payload:
            for item in payload.get("loterias", []):
                if item.get("tipoJogo", "").lower() == "lotomania":
                    results.append(item)
    except Exception as e:
        logger.warning("Estrutura inesperada da API: {}", e)

    return results


def parse_api_entry(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Try known fields based on CAIXA structure
    try:
        contest = int(entry.get("numero", entry.get("concurso")))
        date_str = entry.get("dataApuracao", entry.get("data"))
        numbers_raw = entry.get("dezenas", entry.get("listaDezenas"))
        if isinstance(numbers_raw, str):
            numbers = [int(x) for x in numbers_raw.split(",") if x.strip()]
        else:
            numbers = [int(x) for x in numbers_raw]
        numbers = sorted(numbers)
        date_iso = datetime.strptime(date_str[:10], "%d/%m/%Y").date().isoformat() if "/" in date_str else date_str
        doc = {
            "contest": contest,
            "date": date_iso,
            "numbers": numbers,
            "raw": entry,
            "created_at": datetime.utcnow(),
        }
        return doc
    except Exception as e:
        logger.error("Falha ao parsear entrada da API: {} | entry={}", e, json.dumps(entry)[:300])
        return None


def main() -> None:
    ensure_indexes()
    latest_no = get_latest_contest_no()
    payload = fetch_latest_results()
    entries = extract_lotomania_entries(payload)
    inserted = 0
    for e in entries:
        doc = parse_api_entry(e)
        if doc is None:
            continue
        if latest_no is None or doc["contest"] > latest_no:
            upsert_contest(doc)
            inserted += 1
    logger.info("Atualização concluída. Novos concursos inseridos: {}", inserted)


if __name__ == "__main__":
    main()