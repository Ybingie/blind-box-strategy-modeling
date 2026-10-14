from dataclasses import dataclass, replace
import csv
from pathlib import Path
import numpy as np
from .config import ModelConfig
from .dynamic_stop import solve_dynamic
from .fixed_stop import solve_fixed

@dataclass(frozen=True)
class PriceRobustnessResult:
    bootstrap_samples: int
    fixed_k_values: np.ndarray
    fixed_k_probabilities: np.ndarray
    pair_records: tuple[dict, ...]
    median_ci: tuple[tuple[float, float], ...]
    fixed_cost_median: np.ndarray
    fixed_cost_ci_low: np.ndarray
    fixed_cost_ci_high: np.ndarray

def load_price_samples(cfg: ModelConfig) -> tuple[np.ndarray, ...]:
    path = Path(__file__).resolve().parents[2] / 'data' / 'market_prices_cleaned.csv'
    grouped = [[] for _ in range(cfg.n)]
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            grouped[int(row['role_index']) - 1].append(float(row['unit_cost']))
    if any((len(x) < 2 for x in grouped)):
        raise ValueError('Each design requires at least two valid price observations for bootstrap.')
    return tuple((np.asarray(x, dtype=float) for x in grouped))

def solve_price_robustness(cfg: ModelConfig, bootstrap_samples: int=2000) -> PriceRobustnessResult:
    """Resample within each design to assess the fixed optimum and dynamic boundary."""
    samples = load_price_samples(cfg)
    rng = np.random.default_rng(cfg.base_seed + 80)
    base_pairs = solve_dynamic(cfg).pair_records
    pair_continue = np.zeros((bootstrap_samples, len(base_pairs)), dtype=float)
    best_k = np.zeros(bootstrap_samples, dtype=int)
    boot_medians = np.zeros((bootstrap_samples, cfg.n), dtype=float)
    boot_fixed_costs = np.zeros((bootstrap_samples, cfg.fixed_k_max + 1), dtype=float)
    for b in range(bootstrap_samples):
        medians = tuple((float(np.median(rng.choice(x, size=x.size, replace=True))) for x in samples))
        boot_medians[b] = medians
        local = replace(cfg, second_hand_prices=medians)
        fixed = solve_fixed(local)
        dynamic = solve_dynamic(local)
        best_k[b] = fixed.best_k
        boot_fixed_costs[b] = fixed.total_cost
        pair_continue[b] = [r['action'] == 'Continue' for r in dynamic.pair_records]
    (k_values, counts) = np.unique(best_k, return_counts=True)
    probabilities = counts / bootstrap_samples
    records = []
    for (j, base) in enumerate(base_pairs):
        p = float(pair_continue[:, j].mean())
        if p >= 0.9:
            classification = 'StableContinue'
        elif p <= 0.1:
            classification = 'StableStop'
        else:
            classification = 'Boundary-sensitive'
        records.append(
            {
                'pair': base['pair'],
                'role_1': base['role_1'],
                'role_2': base['role_2'],
                'baseline_action': base['action'],
                'continue_probability': p,
                'classification': classification
            }
        )
    ci = tuple((tuple(map(float, np.quantile(boot_medians[:, i], [0.025, 0.975]))) for i in range(cfg.n)))
    (low, median, high) = np.quantile(boot_fixed_costs, [0.025, 0.5, 0.975], axis=0)
    return PriceRobustnessResult(
        bootstrap_samples,
        k_values,
        probabilities,
        tuple(records),
        ci,
        median,
        low,
        high
    )
