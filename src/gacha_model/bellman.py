from dataclasses import dataclass
from functools import lru_cache
from .config import ModelConfig
from .dynamic_stop import buy_now_cost, should_continue

@dataclass(frozen=True)
class BellmanResult:
    initial_cost: float
    conflicts: tuple[dict, ...]
    state_records: tuple[dict, ...]

def solve_bellman(cfg: ModelConfig) -> BellmanResult:

    @lru_cache(None)
    def value(state: tuple[int, ...]) -> float:
        if not state:
            return 0.0
        missing = set(state)
        p_new = sum((cfg.probabilities[i] for i in missing))
        child = sum((cfg.probabilities[i] * value(tuple((j for j in state if j != i))) for i in state))
        resale = sum(
            (cfg.probabilities[i] * cfg.expected_resale_values[i] for i in range(cfg.n) if i not in missing)
        )
        continue_cost = (cfg.blind_price + child - resale) / p_new
        return min(buy_now_cost(state, cfg), continue_cost)
    (records, conflicts) = ([], [])
    for mask in range(1 << cfg.n):
        state = tuple((i for i in range(cfg.n) if mask & 1 << i))
        stop = buy_now_cost(state, cfg)
        if state:
            missing = set(state)
            p_new = sum((cfg.probabilities[i] for i in missing))
            child = sum((cfg.probabilities[i] * value(tuple((j for j in state if j != i))) for i in state))
            resale = sum(
                (
                    cfg.probabilities[i] * cfg.expected_resale_values[i] for i in range(cfg.n) if i not in missing
                )
            )
            cont = (cfg.blind_price + child - resale) / p_new
        else:
            cont = float('inf')
        bellman_action = 'Continue' if cont < stop else 'Stop'
        simple_action = 'Continue' if should_continue(state, cfg) else 'Stop'
        rec = {
            'mask': mask,
            'missing_count': len(state),
            'simple_action': simple_action,
            'bellman_action': bellman_action,
            'value': value(state)
        }
        records.append(rec)
        if bellman_action != simple_action:
            conflicts.append(rec)
    return BellmanResult(value(tuple(range(cfg.n))), tuple(conflicts), tuple(records))
