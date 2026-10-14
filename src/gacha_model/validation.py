import numpy as np

from .config import ModelConfig


def validate_config(cfg: ModelConfig) -> None:
    if not (len(cfg.roles) == len(cfg.short_roles) == len(cfg.second_hand_prices) == len(cfg.probabilities)):
        raise ValueError("名称、短名称、价格和概率长度必须一致。")
    p = np.asarray(cfg.probabilities)
    if np.any(p <= 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("概率必须全部为正且总和为 1。")
    if min(cfg.second_hand_prices) < 0 or cfg.blind_price < 0 or cfg.buyer_shipping < 0 or cfg.seller_cost < 0:
        raise ValueError("成本参数不得为负。")
    if not 0 <= cfg.resale_success_prob <= 1 or not 0 <= cfg.platform_fee_rate < 1:
        raise ValueError("eta 或 gamma 超出合法范围。")


def validate_results(cfg, fixed, collect, dynamic, bellman, simulations, summaries, tolerance: float) -> list[str]:
    checks = []
    def require(condition: bool, message: str):
        if not condition:
            raise AssertionError(message)
        checks.append("PASS: " + message)
    require(np.isclose(fixed.total_cost[0], cfg.direct_cost), "k=0 等于完全二手成本")
    require(fixed.best_k == 3, "当前基准固定最优 k=3")
    require(len(dynamic.state_records) == 2 ** cfg.n, "动态状态数为 256")
    require(len(dynamic.pair_records) == 28, "两款状态数为 28")
    require(len(bellman.conflicts) == 0, "一步前瞻与 Bellman 无动作冲突")
    require(abs(dynamic.exact_draws - 2.619047619) < 1e-6, "动态解析平均抽数约为 2.619")
    require(abs(dynamic.exact_cost - 356.715581) < 2e-3, "动态精确期望成本约为 356.716 元")
    require(abs(bellman.initial_cost - dynamic.exact_cost) < 1e-8, "Bellman 与动态规则初始成本一致")
    theory = {"完全二手": cfg.direct_cost, "固定 k=2": fixed.total_cost[2],
              "固定 k=3": fixed.total_cost[3], "动态停止": dynamic.exact_cost,
              "一直抽到齐": collect.expected_cost}
    for name, expected in theory.items():
        require(abs(summaries[name].mean - expected) <= tolerance,
                f"{name}模拟均值与理论值偏差不超过 {tolerance:.2f} 元")
    require(all(len(v) == len(next(iter(simulations.costs.values()))) for v in simulations.costs.values()),
            "各策略模拟样本量一致")
    return checks
