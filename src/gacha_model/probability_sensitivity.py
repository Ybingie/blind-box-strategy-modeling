"""保持总概率为 1 的有限概率扰动实验，不把情景值当作产品概率估计。"""
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
        raise ValueError("本案例的概率扰动采用前四款与后四款分组，需要八款配置。")
    records = []
    for delta in np.linspace(-0.025, 0.025, 51):
        probabilities = tuple(p + (delta if i < 4 else -delta)
                              for i, p in enumerate(cfg.probabilities))
        local = replace(cfg, probabilities=probabilities)
        validate_config(local)
        fixed = solve_fixed(local)
        dynamic = solve_dynamic(local)
        bellman = solve_bellman(local)
        if bellman.conflicts or abs(dynamic.exact_cost - bellman.initial_cost) > 1e-8:
            raise AssertionError("概率扰动情景的一步规则与 Bellman 验证不一致。")
        records.append({
            "delta": float(delta),
            "p_1": probabilities[0], "p_5": probabilities[4],
            "fixed_best_k": fixed.best_k,
            "fixed_min_cost": float(fixed.total_cost[fixed.best_k]),
            "dynamic_cost": dynamic.exact_cost,
            "dynamic_draws": dynamic.exact_draws,
            "pair_continue": sum(r["action"] == "继续" for r in dynamic.pair_records),
            "all_continue": sum(r["action"] == "继续" for r in dynamic.state_records),
            "bellman_conflicts": len(bellman.conflicts),
        })
    return ProbabilitySensitivityResult(tuple(records))
