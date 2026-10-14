"""Constructed probability perturbations; these are not measured product probabilities."""
from dataclasses import dataclass, replace
import numpy as np
from .bellman import solve_bellman
from .dynamic_stop import solve_dynamic
from .fixed_stop import solve_fixed
from .validation import validate_config

@dataclass(frozen=True)
class ProbabilitySensitivityResult:
    records: tuple[dict, ...]

def solve_probability_sensitivity(cfg) -> ProbabilitySensitivityResult:
    if cfg.n != 8:
        raise ValueError('This grouped probability analysis requires eight designs.')
    records = []
    for delta in np.linspace(-0.025, 0.025, 51):
        probabilities = tuple((p + (delta if i < 4 else -delta) for (i, p) in enumerate(cfg.probabilities)))
        local = replace(cfg, probabilities=probabilities)
        validate_config(local)
        fixed = solve_fixed(local)
        dynamic = solve_dynamic(local)
        bellman = solve_bellman(local)
        if bellman.conflicts or abs(dynamic.exact_cost - bellman.initial_cost) > 1e-08:
            raise AssertionError('Threshold and Bellman disagree in a probability scenario.')
        records.append(
            {
                'delta': float(delta),
                'p_1': probabilities[0],
                'p_5': probabilities[4],
                'fixed_best_k': fixed.best_k,
                'fixed_min_cost': float(fixed.total_cost[fixed.best_k]),
                'dynamic_cost': dynamic.exact_cost,
                'dynamic_draws': dynamic.exact_draws,
                'pair_continue': sum((r['action'] == 'Continue' for r in dynamic.pair_records)),
                'all_continue': sum((r['action'] == 'Continue' for r in dynamic.state_records)),
                'bellman_conflicts': len(bellman.conflicts)
            }
        )
    return ProbabilitySensitivityResult(tuple(records))
