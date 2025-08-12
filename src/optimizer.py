from __future__ import annotations

from typing import List, Sequence

import numpy as np
from loguru import logger

try:
    import pulp  # type: ignore
except Exception as e:  # pragma: no cover
    pulp = None


def optimize_tickets_ilp(pool_numbers: Sequence[int], pool_probs: Sequence[float], overlap_max: int) -> List[List[int]]:
    if pulp is None:
        raise RuntimeError("pulp não está disponível")

    pool_numbers = list(pool_numbers)
    pool_probs = list(pool_probs)
    n = len(pool_numbers)
    T = 3

    problem = pulp.LpProblem("LotomaniaTickets", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", ((t, i) for t in range(T) for i in range(n)), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Objective: maximize sum p_i * x_{t,i}
    problem += pulp.lpSum(pool_probs[i] * x[(t, i)] for t in range(T) for i in range(n))

    # Constraints: each ticket has exactly 50 numbers
    for t in range(T):
        problem += pulp.lpSum(x[(t, i)] for i in range(n)) == 50

    # Overlap constraints
    for a in range(T):
        for b in range(a + 1, T):
            # sum over i of x_{a,i} * x_{b,i} <= overlap_max
            # Linearize with y_{i}^{ab} <= x_{a,i}, y <= x_{b,i}, y >= x_{a,i}+x_{b,i} - 1
            y = pulp.LpVariable.dicts(f"y_{a}_{b}", (i for i in range(n)), lowBound=0, upBound=1, cat=pulp.LpBinary)
            for i in range(n):
                problem += y[i] <= x[(a, i)]
                problem += y[i] <= x[(b, i)]
                problem += y[i] >= x[(a, i)] + x[(b, i)] - 1
            problem += pulp.lpSum(y[i] for i in range(n)) <= overlap_max

    # Solve
    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
    status = pulp.LpStatus[problem.status]
    logger.info("ILP status: {}", status)
    if status not in ("Optimal", "Not Solved", "Infeasible", "Undefined", "Unbounded"):
        raise RuntimeError(f"Solver status: {status}")

    # Extract solution (fallback if not optimum reached)
    tickets: List[List[int]] = []
    for t in range(T):
        chosen = [pool_numbers[i] for i in range(n) if x[(t, i)].value() and x[(t, i)].value() > 0.5]
        if len(chosen) < 50:
            # fill greedily
            remaining = [pool_numbers[i] for i in range(n) if pool_numbers[i] not in chosen]
            probs_map = {pool_numbers[i]: pool_probs[i] for i in range(n)}
            remaining_sorted = sorted(remaining, key=lambda z: probs_map[z], reverse=True)
            chosen.extend(remaining_sorted[: 50 - len(chosen)])
        tickets.append(sorted(chosen)[:50])

    return tickets


def greedy_tickets(pool_numbers: Sequence[int], pool_probs: Sequence[float], overlap_max: int) -> List[List[int]]:
    pool_numbers = list(pool_numbers)
    pool_probs = list(pool_probs)
    prob_map = {n: p for n, p in zip(pool_numbers, pool_probs)}

    sorted_nums = sorted(pool_numbers, key=lambda x: prob_map[x], reverse=True)

    ticket1 = sorted(sorted_nums[:50])

    def build_ticket(existing: List[List[int]]) -> List[int]:
        chosen = set()
        for n in sorted_nums:
            # Penalize if n appears in many existing tickets
            overlap_penalty = sum(1 for t in existing if n in t)
            score = prob_map[n] - 0.01 * overlap_penalty
            if len(chosen) < 50:
                # Greedy pick if not violating pairwise overlap too much
                tentative = chosen | {n}
                ok = True
                for t in existing:
                    if len(tentative.intersection(t)) > overlap_max:
                        ok = False
                        break
                if ok:
                    chosen.add(n)
            if len(chosen) == 50:
                break
        if len(chosen) < 50:
            for n in sorted_nums:
                if n not in chosen:
                    chosen.add(n)
                if len(chosen) == 50:
                    break
        return sorted(list(chosen))

    ticket2 = build_ticket([ticket1])
    ticket3 = build_ticket([ticket1, ticket2])

    return [ticket1, ticket2, ticket3]