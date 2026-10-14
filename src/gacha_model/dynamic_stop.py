from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import numpy as np
from .config import ModelConfig

@dataclass(frozen=True)
class DynamicResult:
    exact_cost: float
    exact_draws: float
    state_threshold: float
    pair_records: tuple[dict, ...]
    state_records: tuple[dict, ...]

def buy_now_cost(state: tuple[int, ...], cfg: ModelConfig) -> float:
    return float(sum((cfg.second_hand_prices[i] for i in state)))

def next_draw_value(state: tuple[int, ...], cfg: ModelConfig) -> float:
    """A new design avoids its purchase price; a duplicate yields expected recovery."""
    missing = set(state)
    return float(
        sum(
            (
                cfg.probabilities[i] * (
                    cfg.second_hand_prices[i] if i in missing else cfg.expected_resale_values[i]
                ) for i in range(cfg.n)
            )
        )
    )

def should_continue(state: tuple[int, ...], cfg: ModelConfig) -> bool:
    return bool(state) and next_draw_value(state, cfg) > cfg.blind_price

def solve_dynamic(cfg: ModelConfig) -> DynamicResult:

    @lru_cache(None)
    def value(state: tuple[int, ...]) -> float:
        if not should_continue(state, cfg):
            return buy_now_cost(state, cfg)
        missing = set(state)
        p_new = sum((cfg.probabilities[i] for i in missing))
        child = sum((cfg.probabilities[i] * value(tuple((j for j in state if j != i))) for i in state))
        resale = sum(
            (cfg.probabilities[i] * cfg.expected_resale_values[i] for i in range(cfg.n) if i not in missing)
        )
        # A duplicate leaves the state unchanged. Divide out that self-loop.
        return (cfg.blind_price + child - resale) / p_new

    @lru_cache(None)
    def draws(state: tuple[int, ...]) -> float:
        if not should_continue(state, cfg):
            return 0.0
        missing = set(state)
        p_new = sum((cfg.probabilities[i] for i in missing))
        child = sum((cfg.probabilities[i] * draws(tuple((j for j in state if j != i))) for i in state))
        return (1.0 + child) / p_new
    pair_records = []
    delta = np.asarray(cfg.second_hand_prices) - np.asarray(cfg.expected_resale_values)
    for (i, j) in combinations(range(cfg.n), 2):
        state = tuple((k for k in range(cfg.n) if k not in (i, j)))
        pair_records.append(
            {
                'i': i,
                'j': j,
                'pair': f'{i + 1}+{j + 1}',
                'role_1': cfg.roles[i],
                'role_2': cfg.roles[j],
                'delta_sum': float(delta[i] + delta[j]),
                'G': next_draw_value(state, cfg),
                'action': 'Continue' if should_continue(state, cfg) else 'Stop'
            }
        )
    state_records = []
    for mask in range(1 << cfg.n):
        state = tuple((i for i in range(cfg.n) if mask & 1 << i))
        state_records.append(
            {
                'mask': mask,
                'missing_count': len(state),
                'owned_count': cfg.n - len(state),
                'G': next_draw_value(state, cfg),
                'action': 'Continue' if should_continue(state, cfg) else 'Stop'
            }
        )
    initial = tuple(range(cfg.n))
    return DynamicResult(
        value(initial),
        draws(initial),
        cfg.direct_cost - cfg.n * cfg.blind_price,
        tuple(pair_records),
        tuple(state_records)
    )
