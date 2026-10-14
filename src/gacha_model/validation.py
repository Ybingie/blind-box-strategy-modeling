import numpy as np
from .config import ModelConfig

def validate_config(cfg: ModelConfig) -> None:
    if not len(cfg.roles) == len(cfg.short_roles) == len(cfg.second_hand_prices) == len(cfg.probabilities):
        raise ValueError('Names, short names, prices and probabilities must have matching lengths.')
    p = np.asarray(cfg.probabilities)
    if np.any(p <= 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError('Probabilities must be positive and sum to one.')
    if min(cfg.second_hand_prices) < 0 or cfg.blind_price < 0 or cfg.buyer_shipping < 0 or (cfg.seller_cost < 0):
        raise ValueError('Cost parameters cannot be negative.')
    if not 0 <= cfg.resale_success_prob <= 1 or not 0 <= cfg.platform_fee_rate < 1:
        raise ValueError('eta or gamma is outside its valid range.')

def validate_results(cfg, fixed, collect, dynamic, bellman, simulations, summaries, tolerance: float) -> list[str]:
    checks = []

    def require(condition: bool, message: str):
        if not condition:
            raise AssertionError(message)
        checks.append('PASS: ' + message)
    require(np.isclose(fixed.total_cost[0], cfg.direct_cost), 'k=0 equals the direct-purchase cost')
    require(len(dynamic.state_records) == 2 ** cfg.n, 'There are 256 missing-set states')
    require(len(dynamic.pair_records) == 28, 'There are 28 two-design states')
    require(len(bellman.conflicts) == 0, 'Threshold and Bellman actions have no conflicts')
    require(fixed.best_k == int(fixed.k[np.argmin(fixed.total_cost)]),
            'Selected fixed count minimizes the computed cost curve')
    require(np.isfinite(dynamic.exact_cost) and dynamic.exact_draws >= 0,
            'Dynamic cost is finite and expected draws are nonnegative')
    require(
        abs(bellman.initial_cost - dynamic.exact_cost) < 1e-08,
        'Bellman and threshold initial values agree'
    )
    theory = {
        'Secondary market only': cfg.direct_cost,
        'Fixed k=2': fixed.total_cost[2],
        'Fixed k=3': fixed.total_cost[3],
        'Dynamic stopping': dynamic.exact_cost,
        'Draw until complete': collect.expected_cost
    }
    for (name, expected) in theory.items():
        require(
            abs(summaries[name].mean - expected) <= tolerance,
            f'{name} simulation mean differs from theory by at most {tolerance:.2f} CNY'
        )
    require(
        all((len(v) == len(next(iter(simulations.costs.values()))) for v in simulations.costs.values())),
        'All strategies have the same trial count'
    )
    # These known values check the report baseline, not every possible input.
    if is_report_baseline(cfg):
        require(fixed.best_k == 3, 'Report baseline fixed optimum is k=3')
        require(abs(dynamic.exact_draws - 2.619047619) < 1e-06,
                'Report baseline expected draws are approximately 2.619')
        require(abs(dynamic.exact_cost - 356.715581) < 0.002,
                'Report baseline expected cost is approximately CNY 356.716')
    else:
        checks.append('INFO: changed parameters; report-specific numerical targets skipped')
    return checks


def is_report_baseline(cfg):
    """Compare with the report inputs even if the editable defaults change."""
    return (
        cfg.second_hand_prices == (60.0, 58.0, 60.0, 58.5, 37.75, 32.0, 20.33, 40.0)
        and cfg.probabilities == (0.125,) * 8
        and cfg.blind_price == 39.8
        and cfg.resale_success_prob == 0.60
        and cfg.platform_fee_rate == 0.006
        and cfg.buyer_shipping == 6.0
        and cfg.seller_cost == 4.0
    )
