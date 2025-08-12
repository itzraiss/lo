import argparse
from datetime import datetime
from typing import List

import pandas as pd
from loguru import logger

from .db import ensure_indexes, upsert_contest


def parse_results_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Expect columns: contest, date, and 20 numbers (or a list column)
    lower_cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_cols)

    if "concurso" in df.columns and "contest" not in df.columns:
        df = df.rename(columns={"concurso": "contest"})
    if "data" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"data": "date"})

    # Try to locate numbers
    number_cols: List[str] = []
    for c in df.columns:
        if c.startswith("dezena") or c.startswith("d") or c in [str(i) for i in range(1, 21)]:
            number_cols.append(c)
    number_cols = sorted(number_cols, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

    if "numbers" in df.columns and not number_cols:
        # Already a list column
        pass
    elif number_cols:
        df["numbers"] = df[number_cols].values.tolist()
    else:
        raise ValueError("Não foi possível identificar as colunas das dezenas.")

    # Normalize types
    df["contest"] = df["contest"].astype(int)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["numbers"] = df["numbers"].apply(lambda xs: sorted(int(x) for x in xs))

    return df[["contest", "date", "numbers"]].dropna()


def main(xlsx_path: str) -> None:
    ensure_indexes()
    df = parse_results_xlsx(xlsx_path)
    logger.info("Ingesting {} contests from {}", len(df), xlsx_path)
    for _, row in df.sort_values("contest").iterrows():
        upsert_contest({
            "contest": int(row["contest"]),
            "date": row["date"],
            "numbers": list(map(int, row["numbers"])),
            "raw": {},
            "created_at": datetime.utcnow(),
        })
    logger.info("Ingestão concluída.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True, help="Caminho para results.xlsx")
    args = parser.parse_args()
    main(args.xlsx)