from dataclasses import dataclass, replace
import numpy as np
from .config import ModelConfig
from .dynamic_stop import solve_dynamic
from .fixed_stop import solve_fixed

@dataclass(frozen=True)
class OneWaySensitivity:
    key: str
    label: str
    values: np.ndarray
    baseline: float
    fixed_best_k: np.ndarray
    fixed_min_cost: np.ndarray
    dynamic_cost: np.ndarray

@dataclass(frozen=True)
class SensitivityResult:
    c_grid: np.ndarray
    eta_grid: np.ndarray
    fixed_best_k: np.ndarray
    fixed_min_cost: np.ndarray
    dynamic_pair_continue: np.ndarray
    dynamic_cost: np.ndarray
    one_way: tuple[OneWaySensitivity, ...]
    tornado_labels: tuple[str, ...]
    tornado_change: np.ndarray

def _solve_one_way(
    cfg: ModelConfig,
    key: str,
    label: str,
    values: np.ndarray,
    baseline: float
) -> OneWaySensitivity:
    best_k = np.zeros(values.size, dtype=int)
    fixed_cost = np.zeros(values.size)
    dynamic_cost = np.zeros(values.size)
    for (j, value) in enumerate(values):
        if key == 'lambda':
            local = replace(cfg, second_hand_prices=tuple((float(value) * x for x in cfg.second_hand_prices)))
        else:
            field = {
                'c': 'blind_price',
                'eta': 'resale_success_prob',
                'gamma': 'platform_fee_rate',
                'd': 'buyer_shipping',
                'h': 'seller_cost'
            }[key]
            local = replace(cfg, **{field: float(value)})
        fixed = solve_fixed(local)
        dynamic = solve_dynamic(local)
        best_k[j] = fixed.best_k
        fixed_cost[j] = fixed.total_cost[fixed.best_k]
        dynamic_cost[j] = dynamic.exact_cost
    return OneWaySensitivity(key, label, values, baseline, best_k, fixed_cost, dynamic_cost)

def solve_sensitivity(cfg: ModelConfig) -> SensitivityResult:
    c_grid = np.linspace(35.0, 45.0, 21)
    eta_grid = np.linspace(0.35, 0.8, 19)
    shape = (eta_grid.size, c_grid.size)
    best_k = np.zeros(shape)
    fixed_cost = np.zeros(shape)
    pair_count = np.zeros(shape)
    dynamic_cost = np.zeros(shape)
    for (i, eta) in enumerate(eta_grid):
        for (j, c) in enumerate(c_grid):
            local = replace(cfg, blind_price=float(c), resale_success_prob=float(eta))
            fixed = solve_fixed(local)
            dynamic = solve_dynamic(local)
            best_k[i, j] = fixed.best_k
            fixed_cost[i, j] = fixed.total_cost[fixed.best_k]
            pair_count[i, j] = sum((r['action'] == 'Continue' for r in dynamic.pair_records))
            dynamic_cost[i, j] = dynamic.exact_cost
    one_way = (
        _solve_one_way(cfg, 'c', 'Blind-draw price $c$ (CNY)', np.linspace(35.0, 45.0, 21), cfg.blind_price),
        _solve_one_way(
            cfg,
            'eta',
            'Resale success probability $\\eta$',
            np.linspace(0.35, 0.8, 19),
            cfg.resale_success_prob
        ),
        _solve_one_way(
            cfg,
            'lambda',
            'Secondary-market price multiplier $\\lambda$',
            np.linspace(0.75, 1.25, 21),
            1.0
        ),
        _solve_one_way(
            cfg,
            'gamma',
            'Platform fee rate $\\gamma$',
            np.linspace(0.0, 0.05, 21),
            cfg.platform_fee_rate
        ),
        _solve_one_way(
            cfg,
            'd',
            'Buyer shipping cost $d$ (CNY)',
            np.linspace(0.0, 12.0, 25),
            cfg.buyer_shipping
        ),
        _solve_one_way(
            cfg,
            'h',
            'Seller transaction cost $h$ (CNY)',
            np.linspace(0.0, 10.0, 21),
            cfg.seller_cost
        )
    )
    base = solve_fixed(cfg).total_cost[solve_fixed(cfg).best_k]
    cases = [
        (
            'Secondary-market price +10%',
            replace(cfg, second_hand_prices=tuple((x * 1.1 for x in cfg.second_hand_prices)))
        ),
        (
            'Secondary-market price -10%',
            replace(cfg, second_hand_prices=tuple((x * 0.9 for x in cfg.second_hand_prices)))
        ),
        ('Blind-draw price +10%', replace(cfg, blind_price=cfg.blind_price * 1.1)),
        ('Blind-draw price -10%', replace(cfg, blind_price=cfg.blind_price * 0.9)),
        (
            'Resale success probability +10%',
            replace(cfg, resale_success_prob=min(1.0, cfg.resale_success_prob * 1.1))
        ),
        ('Resale success probability -10%', replace(cfg, resale_success_prob=cfg.resale_success_prob * 0.9))
    ]
    changes = []
    for (_, local) in cases:
        f = solve_fixed(local)
        changes.append(float(f.total_cost[f.best_k] - base))
    return SensitivityResult(
        c_grid,
        eta_grid,
        best_k,
        fixed_cost,
        pair_count,
        dynamic_cost,
        one_way,
        tuple((x[0] for x in cases)),
        np.asarray(changes)
    )
