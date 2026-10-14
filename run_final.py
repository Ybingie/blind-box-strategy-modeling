"""Run the blind-draw collection model and figures."""
from argparse import ArgumentParser
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / 'src'))
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
from gacha_model.sensitivity import solve_sensitivity
from gacha_model.simulation import run_all_simulations
from gacha_model.statistics import summarize
from gacha_model.uncertainty import solve_price_robustness
from gacha_model.validation import validate_config, validate_results
from prepare_data import prepare_data

def main() -> int:
    parser = ArgumentParser(description='Blind-draw collection modeling and figures')
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use 10,000 trials instead of the default 100,000'
    )
    args = parser.parse_args()
    audit = prepare_data()
    print(f"Loaded {audit['usable_records']} usable price records from the workbook.")
    cfg = ModelConfig()
    run = RunConfig.build(SCRIPT_DIR, args.quick)
    print(
        f"Run mode: {('quick' if run.quick else 'final')}; trials per randomized strategy: {run.simulations:,}"
    )
    validate_config(cfg)
    fixed = solve_fixed(cfg)
    collect = solve_collect_theory(cfg)
    dynamic = solve_dynamic(cfg)
    bellman = solve_bellman(cfg)
    simulations = run_all_simulations(run.simulations, cfg)
    summaries = {name: summarize(values, cfg.budget) for (name, values) in simulations.costs.items()}
    outlay_summaries = {name: summarize(values, cfg.budget) for (name, values) in simulations.outlays.items()}
    cash_exact = tuple((enumerate_fixed_cash_risk(k, cfg) for k in (2, 3)))
    probability_sensitivity = solve_probability_sensitivity(cfg)
    sensitivity = solve_sensitivity(cfg)
    robustness = solve_price_robustness(cfg, 500 if run.quick else 2000)
    checks = validate_results(
        cfg,
        fixed,
        collect,
        dynamic,
        bellman,
        simulations,
        summaries,
        5.0 if run.quick else 2.0
    )
    for name in simulations.costs:
        if not all(simulations.outlays[name] >= simulations.costs[name] - 1e-08):
            raise AssertionError(f'{name}: purchase outlay must not be below final net cost.')
    checks += [
        'PASS: all 51 probability scenarios agree with Bellman actions',
        'PASS: exact two- and three-draw cash enumeration agrees with theory',
        'PASS: purchase outlay is never below final net cost'
    ]
    figures = []
    font = setup_style()
    print(f'English figures; font: {font}')
    figures += make_data_figures(cfg, collect, simulations, robustness, run)
    figures += make_model_figures(cfg, fixed, collect, dynamic, simulations, robustness, run)
    figures += make_risk_figures(cfg, fixed, collect, dynamic, simulations, summaries, run)
    figures += make_sensitivity_figures(cfg, sensitivity, run)
    figures += make_uncertainty_figure(cfg, robustness, run)
    figures += make_probability_figure(cfg, probability_sensitivity, run)
    if len([x for x in figures if x['group'] == 'main']) != 15 or len([x for x in figures if x['group'] == 'appendix']) != 11:
        raise AssertionError('Expected 15 main and 11 appendix figures.')
    export_all(
        cfg,
        run,
        fixed,
        collect,
        dynamic,
        bellman,
        summaries,
        sensitivity,
        robustness,
        checks,
        figures
    )
    from gacha_model.export import export_audit_diagnostics, export_record_audit
    export_audit_diagnostics(run, fixed, probability_sensitivity, summaries, outlay_summaries, cash_exact)
    export_record_audit(cfg, run)
    print('\nKey results')
    print(f'Optimal fixed k={fixed.best_k}; theoretical cost {fixed.total_cost[fixed.best_k]:.4f} CNY')
    print(f'Exact dynamic cost {dynamic.exact_cost:.4f} CNY; exact expected draws {dynamic.exact_draws:.4f}')
    print(f'Bellman action conflicts {len(bellman.conflicts)}')
    for (name, s) in summaries.items():
        print(
            f'{name:<10} Mean={s.mean:8.2f}  P95={s.p95:8.2f}  P(C>{cfg.budget:.0f})={s.exceed_budget:6.2%}'
        )
    print('\nOutput directory: outputs/')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
