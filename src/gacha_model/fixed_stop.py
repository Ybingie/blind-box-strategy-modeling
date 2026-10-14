from dataclasses import dataclass
import numpy as np

from .config import ModelConfig


@dataclass(frozen=True)
class FixedResult:
    k: np.ndarray
    draw_cost: np.ndarray
    missing_cost: np.ndarray
    resale_income: np.ndarray
    total_cost: np.ndarray
    expected_missing: np.ndarray
    expected_duplicates: np.ndarray
    marginal: np.ndarray
    best_k: int


def expected_cost(k: int, cfg: ModelConfig) -> float:
    p = np.asarray(cfg.probabilities)
    b = np.asarray(cfg.second_hand_prices)
    r = np.asarray(cfg.expected_resale_values)
    missing = (1 - p) ** k
    duplicates = k * p - 1 + missing
    return float(k * cfg.blind_price + b @ missing - r @ duplicates)


def solve_fixed(cfg: ModelConfig) -> FixedResult:
    ks = np.arange(cfg.fixed_k_max + 1)
    p = np.asarray(cfg.probabilities)
    b = np.asarray(cfg.second_hand_prices)
    r = np.asarray(cfg.expected_resale_values)
    missing_prob = np.array([(1 - p) ** int(k) for k in ks])
    duplicates = np.array([k * p - 1 + missing_prob[k] for k in ks])
    draw = ks * cfg.blind_price
    missing_cost = missing_prob @ b
    resale = duplicates @ r
    total = draw + missing_cost - resale
    marginal = np.diff(total)
    return FixedResult(
        k=ks, draw_cost=draw, missing_cost=missing_cost, resale_income=resale,
        total_cost=total, expected_missing=missing_prob.sum(axis=1),
        expected_duplicates=duplicates.sum(axis=1), marginal=marginal,
        best_k=int(ks[np.argmin(total)]),
    )
