"""Exact enumeration of purchase outlay and final net cost."""
from itertools import product
from math import comb, prod
import numpy as np
from .fixed_stop import solve_fixed

def enumerate_fixed_cash_risk(k: int, cfg) -> dict:
    if k not in range(4):
        raise ValueError('Exact enumeration supports k=0,1,2,3 only.')
    (b, g) = (np.asarray(cfg.second_hand_prices), np.asarray(cfg.resale_values))
    total_mass = mean_net = mean_outlay = net_exceed = outlay_exceed = 0.0
    for sequence in product(range(cfg.n), repeat=k):
        sequence_prob = prod((cfg.probabilities[i] for i in sequence))
        counts = np.bincount(sequence, minlength=cfg.n)
        duplicates = np.maximum(counts - 1, 0)
        outlay = float(k * cfg.blind_price + b[counts == 0].sum())
        mean_outlay += sequence_prob * outlay
        outlay_exceed += sequence_prob * (outlay > cfg.budget)
        for sales in product(*(range(int(d) + 1) for d in duplicates)):
            sale_prob = prod(
                (
                    comb(int(d), s) * cfg.resale_success_prob ** s * (1 - cfg.resale_success_prob) ** (
                        int(d) - s
                    ) for (d, s) in zip(duplicates, sales)
                )
            )
            mass = sequence_prob * sale_prob
            net_cost = outlay - float(np.asarray(sales) @ g)
            total_mass += mass
            mean_net += mass * net_cost
            net_exceed += mass * (net_cost > cfg.budget)
    theory = solve_fixed(cfg).total_cost[k]
    if not np.isclose(total_mass, 1.0) or abs(mean_net - theory) > 1e-08:
        raise AssertionError('Exact enumeration failed probability-mass or expectation checks.')
    return {
        'k': k,
        'mean_net': mean_net,
        'mean_outlay': mean_outlay,
        'net_exceed': net_exceed,
        'outlay_exceed': outlay_exceed,
        'probability_mass': total_mass
    }
