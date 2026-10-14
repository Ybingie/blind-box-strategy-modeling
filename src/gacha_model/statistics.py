from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class RiskSummary:
    mean: float
    sd: float
    median: float
    p75: float
    p90: float
    p95: float
    p99: float
    exceed_budget: float
    mean_ci_low: float
    mean_ci_high: float
    p95_ci_low: float
    p95_ci_high: float
    exceed_ci_low: float
    exceed_ci_high: float

    def as_dict(self) -> dict:
        return asdict(self)


def summarize(values: np.ndarray, budget: float) -> RiskSummary:
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("风险统计至少需要两个样本。")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    half = 1.96 * sd / np.sqrt(x.size)
    qs = np.quantile(x, [0.50, 0.75, 0.90, 0.95, 0.99], method="linear")
    exceed = float(np.mean(x > budget))
    # P95 区间采用经验分位数次序统计量的正态近似；超支概率采用 Wilson 区间。
    z = 1.96
    n = x.size
    rank_half = z * np.sqrt(n * 0.95 * 0.05)
    ordered = np.sort(x)
    lo_idx = int(np.clip(np.floor(0.95 * n - rank_half), 0, n - 1))
    hi_idx = int(np.clip(np.ceil(0.95 * n + rank_half), 0, n - 1))
    denom = 1 + z * z / n
    center = (exceed + z * z / (2 * n)) / denom
    wilson_half = z * np.sqrt(exceed * (1 - exceed) / n + z * z / (4 * n * n)) / denom
    return RiskSummary(
        mean, sd, *map(float, qs), exceed, mean - half, mean + half,
        float(ordered[lo_idx]), float(ordered[hi_idx]),
        max(0.0, center - wilson_half), min(1.0, center + wilson_half),
    )
