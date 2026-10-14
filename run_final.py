#!/usr/bin/env python
"""一键运行随机谷子收集策略最终模型。"""
from argparse import ArgumentParser
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from gacha_model.bellman import solve_bellman
from gacha_model.collect_all import solve_collect_theory
from gacha_model.config import ModelConfig, RunConfig
from gacha_model.cash_risk import enumerate_fixed_cash_risk
from gacha_model.dynamic_stop import solve_dynamic
from gacha_model.export import export_all
from gacha_model.fixed_stop import solve_fixed
from gacha_model.plotting import make_data_figures, make_model_figures, make_risk_figures, make_sensitivity_figures, make_uncertainty_figure, make_probability_figure
from gacha_model.probability_sensitivity import solve_probability_sensitivity
from gacha_model.plotting.style import setup_style
from gacha_model.plotting.i18n import set_language
from gacha_model.sensitivity import solve_sensitivity
from gacha_model.simulation import run_all_simulations
from gacha_model.statistics import summarize
from gacha_model.uncertainty import solve_price_robustness
from gacha_model.validation import validate_config, validate_results


def main() -> int:
    parser = ArgumentParser(description="随机谷子数学建模最终求解与绘图")
    parser.add_argument("--quick", action="store_true", help="使用 10,000 次模拟；默认 100,000 次")
    parser.add_argument("--language", choices=("zh", "en", "both"), default="both",
                        help="图形语言：zh、en 或 both；默认同时生成中英文")
    args = parser.parse_args()
    cfg = ModelConfig(); run = RunConfig.build(SCRIPT_DIR, args.quick)
    print(f"运行模式: {'quick' if run.quick else 'final'}，每种随机策略模拟 {run.simulations:,} 次")
    validate_config(cfg)
    fixed = solve_fixed(cfg); collect = solve_collect_theory(cfg); dynamic = solve_dynamic(cfg); bellman = solve_bellman(cfg)
    simulations = run_all_simulations(run.simulations, cfg)
    summaries = {name: summarize(values, cfg.budget) for name, values in simulations.costs.items()}
    outlay_summaries = {name: summarize(values, cfg.budget) for name, values in simulations.outlays.items()}
    cash_exact = tuple(enumerate_fixed_cash_risk(k, cfg) for k in (2, 3))
    probability_sensitivity = solve_probability_sensitivity(cfg)
    sensitivity = solve_sensitivity(cfg)
    robustness = solve_price_robustness(cfg, 500 if run.quick else 2_000)
    checks = validate_results(cfg, fixed, collect, dynamic, bellman, simulations, summaries, 5.0 if run.quick else 2.0)
    for name in simulations.costs:
        if not all(simulations.outlays[name] >= simulations.costs[name] - 1e-8):
            raise AssertionError(f"{name}的转售前支出不应低于净成本。")
    checks += ["PASS: 51 个概率扰动情景与 Bellman 逐状态一致",
               "PASS: 固定两次、三次的预算风险精确枚举与理论期望一致",
               "PASS: 所有样本转售前支出不低于最终净成本"]
    figures = []
    languages = ("zh", "en") if args.language == "both" else (args.language,)
    for language in languages:
        set_language(language)
        font = setup_style(language)
        print(f"绘图语言: {language}，字体: {font}")
        current = []
        current += make_data_figures(cfg, collect, simulations, robustness, run)
        current += make_model_figures(cfg, fixed, collect, dynamic, simulations, robustness, run)
        current += make_risk_figures(cfg, fixed, collect, dynamic, simulations, summaries, run)
        current += make_sensitivity_figures(cfg, sensitivity, run)
        current += make_uncertainty_figure(cfg, robustness, run)
        current += make_probability_figure(cfg, probability_sensitivity, run)
        if len([x for x in current if x["group"] == "main"]) != 15 or len([x for x in current if x["group"] == "appendix"]) != 11:
            raise AssertionError(f"{language} 图形数量不符合 15 张正文图 + 11 张附录图的设计。")
        figures += current
    export_all(cfg, run, fixed, collect, dynamic, bellman, summaries, sensitivity, robustness, checks, figures)
    from gacha_model.export import export_audit_diagnostics, export_record_audit
    export_audit_diagnostics(run, fixed, probability_sensitivity, summaries, outlay_summaries, cash_exact)
    export_record_audit(cfg, run)
    print("\n关键结果")
    print(f"固定最优 k={fixed.best_k}，理论成本 {fixed.total_cost[fixed.best_k]:.4f} 元")
    print(f"动态精确成本 {dynamic.exact_cost:.4f} 元，解析平均抽数 {dynamic.exact_draws:.4f}")
    print(f"Bellman 冲突状态数 {len(bellman.conflicts)}")
    for name, s in summaries.items():
        print(f"{name:<10} Mean={s.mean:8.2f}  P95={s.p95:8.2f}  P(C>{cfg.budget:.0f})={s.exceed_budget:6.2%}")
    print("\n输出目录:", run.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
