from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .config import ModelConfig
from .dynamic_stop import should_continue


@dataclass(frozen=True)
class SimulationBundle:
    costs: dict[str, np.ndarray]
    draws: dict[str, np.ndarray]
    outlays: dict[str, np.ndarray] = field(default_factory=dict)


def _resale_income(counts: np.ndarray, cfg: ModelConfig, rng: np.random.Generator) -> float:
    duplicates = np.maximum(counts - 1, 0)
    sold = rng.binomial(duplicates, cfg.resale_success_prob)
    return float(sold @ np.asarray(cfg.resale_values))


def simulate_fixed(k: int, times: int, cfg: ModelConfig, seed: int,
                   outlays: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    costs = np.empty(times)
    p = np.asarray(cfg.probabilities)
    b = np.asarray(cfg.second_hand_prices)
    for m in range(times):
        draw = rng.choice(cfg.n, size=k, p=p)
        counts = np.bincount(draw, minlength=cfg.n)
        if outlays is not None:
            outlays[m] = k * cfg.blind_price + b[counts == 0].sum()
        costs[m] = k * cfg.blind_price + b[counts == 0].sum() - _resale_income(counts, cfg, rng)
    return costs, np.full(times, k, dtype=int)


def simulate_collect(times: int, cfg: ModelConfig, seed: int,
                     outlays: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    costs, draws = np.empty(times), np.empty(times, dtype=int)
    p = np.asarray(cfg.probabilities)
    for m in range(times):
        counts = np.zeros(cfg.n, dtype=int)
        t = 0
        while np.any(counts == 0):
            counts[rng.choice(cfg.n, p=p)] += 1
            t += 1
        draws[m] = t
        if outlays is not None:
            outlays[m] = t * cfg.blind_price
        costs[m] = t * cfg.blind_price - _resale_income(counts, cfg, rng)
    return costs, draws


def simulate_dynamic(times: int, cfg: ModelConfig, seed: int,
                     outlays: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    costs, draws = np.empty(times), np.empty(times, dtype=int)
    p = np.asarray(cfg.probabilities)
    b = np.asarray(cfg.second_hand_prices)
    g = np.asarray(cfg.resale_values)
    for m in range(times):
        missing = set(range(cfg.n))
        cost, t = 0.0, 0
        while should_continue(tuple(sorted(missing)), cfg):
            cost += cfg.blind_price
            t += 1
            i = int(rng.choice(cfg.n, p=p))
            if i in missing:
                missing.remove(i)
            elif rng.random() < cfg.resale_success_prob:
                cost -= g[i]
        costs[m] = cost + b[list(missing)].sum() if missing else cost
        if outlays is not None:
            outlays[m] = t * cfg.blind_price + b[list(missing)].sum()
        draws[m] = t
    return costs, draws


def run_all_simulations(times: int, cfg: ModelConfig) -> SimulationBundle:
    a2, a3, ad, ac = (np.empty(times) for _ in range(4))
    k2, d2 = simulate_fixed(2, times, cfg, cfg.base_seed + 2, a2)
    k3, d3 = simulate_fixed(3, times, cfg, cfg.base_seed + 3, a3)
    dynamic, dd = simulate_dynamic(times, cfg, cfg.base_seed + 4, ad)
    collect, dc = simulate_collect(times, cfg, cfg.base_seed + 5, ac)
    direct = np.full(times, cfg.direct_cost)
    return SimulationBundle(
        costs={"完全二手": direct, "固定 k=2": k2, "固定 k=3": k3,
               "动态停止": dynamic, "一直抽到齐": collect},
        draws={"固定 k=2": d2, "固定 k=3": d3, "动态停止": dd, "一直抽到齐": dc},
        outlays={"完全二手": direct.copy(), "固定 k=2": a2, "固定 k=3": a3,
                 "动态停止": ad, "一直抽到齐": ac},
    )
