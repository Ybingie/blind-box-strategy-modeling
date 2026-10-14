from dataclasses import dataclass
import numpy as np

from .config import ModelConfig


@dataclass(frozen=True)
class CollectTheory:
    harmonic: float
    expected_draws: float
    expected_cost: float


def solve_collect_theory(cfg: ModelConfig) -> CollectTheory:
    p = np.asarray(cfg.probabilities)
    if not np.allclose(p, np.full(cfg.n, 1 / cfg.n)):
        raise ValueError("Coupon Collector 闭式结果只适用于等概率配置。")
    harmonic = float(sum(1 / j for j in range(1, cfg.n + 1)))
    expected_draws = cfg.n * harmonic
    expected_cost = cfg.blind_price * expected_draws - (harmonic - 1) * sum(cfg.expected_resale_values)
    return CollectTheory(harmonic, expected_draws, float(expected_cost))
