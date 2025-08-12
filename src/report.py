from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict

from loguru import logger

from .backtest import run_backtest
from .db import get_latest_prediction


DEFAULT_REPORT_DIR = "/workspace/logs"


def write_report(report_path: str, latest: Dict, backtest: Dict | None) -> None:
    lines = []
    lines.append(f"# Relatório Lotomania — {datetime.utcnow().isoformat()}\n")

    lines.append("## Última Previsão\n")
    if latest:
        sim = latest.get("simulations", {})
        lines.append(f"- Model version: `{latest.get('model_version','-')}`\n")
        lines.append(f"- p(≥16): {sim.get('p_at_least_16', 0):.3%}\n")
        lines.append(f"- p(≥17): {sim.get('p_at_least_17', 0):.3%}\n")
        lines.append(f"- p(≥18): {sim.get('p_at_least_18', 0):.3%}\n")
        lines.append(f"- confidence_passed: `{latest.get('confidence_passed', False)}`\n")
        lines.append("\n### Tickets\n")
        for i, t in enumerate(latest.get("tickets", []), start=1):
            lines.append(f"- Ticket {i}: {sorted(t)}\n")
    else:
        lines.append("Nenhuma previsão encontrada.\n")

    if backtest:
        lines.append("\n## Backtest (roll-forward)\n")
        s = backtest.get("summary", {})
        lines.append(f"- Steps: {s.get('n_steps', 0)}\n")
        lines.append(f"- freq max_hits ≥16: {s.get('freq_ge_16', 0):.3%}\n")
        lines.append(f"- freq max_hits ≥17: {s.get('freq_ge_17', 0):.3%}\n")
        lines.append(f"- freq max_hits ≥18: {s.get('freq_ge_18', 0):.3%}\n")
        lines.append(f"- avg p̂(≥16): {s.get('avg_p_ge_16', 0):.3%}\n")
        lines.append(f"- avg p̂(≥17): {s.get('avg_p_ge_17', 0):.3%}\n")
        lines.append(f"- avg p̂(≥18): {s.get('avg_p_ge_18', 0):.3%}\n")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Relatório gravado em {}", report_path)


def main(do_backtest: bool, start_window: int, step: int, sims: int) -> str:
    latest = get_latest_prediction() or {}
    backtest_res = None
    if do_backtest:
        backtest_res = run_backtest(start_window=start_window, step=step, sims=sims)
    report_path = os.path.join(DEFAULT_REPORT_DIR, f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md")
    write_report(report_path, latest, backtest_res)
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--start-window", type=int, default=300)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--sims", type=int, default=20000)
    args = parser.parse_args()
    main(args.backtest, args.start_window, args.step, args.sims)