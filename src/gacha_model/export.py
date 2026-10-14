import csv
import json
from pathlib import Path

def _write_csv(path: Path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def export_all(
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
    figure_index
):
    tables = run.output_root / 'tables'
    summary_dir = run.output_root / 'summary'
    tables.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        tables / 'fixed_stop_results.csv',
        [
            'k',
            'draw_cost',
            'missing_cost',
            'resale_income',
            'total_cost',
            'expected_missing',
            'expected_duplicates'
        ],
        zip(
            fixed.k,
            fixed.draw_cost,
            fixed.missing_cost,
            fixed.resale_income,
            fixed.total_cost,
            fixed.expected_missing,
            fixed.expected_duplicates
        )
    )
    _write_csv(
        tables / 'strategy_risk_summary.csv',
        [
            'strategy',
            'mean',
            'sd',
            'median',
            'p75',
            'p90',
            'p95',
            'p99',
            'exceed_budget',
            'mean_ci_low',
            'mean_ci_high',
            'p95_ci_low',
            'p95_ci_high',
            'exceed_ci_low',
            'exceed_ci_high'
        ],
        ([name, *s.as_dict().values()] for (name, s) in summaries.items())
    )
    _write_csv(
        tables / 'price_bootstrap_state_robustness.csv',
        ['pair', 'role_1', 'role_2', 'baseline_action', 'continue_probability', 'classification'],
        (
            [
                r[k] for k in [
                    'pair',
                    'role_1',
                    'role_2',
                    'baseline_action',
                    'continue_probability',
                    'classification'
                ]
            ] for r in robustness.pair_records
        )
    )
    _write_csv(
        tables / 'price_bootstrap_fixed_k.csv',
        ['k', 'probability'],
        zip(robustness.fixed_k_values, robustness.fixed_k_probabilities)
    )
    _write_csv(
        tables / 'dynamic_pair_states.csv',
        ['pair', 'role_1', 'role_2', 'delta_sum', 'G', 'action'],
        (
            [
                r['pair'],
                r['role_1'],
                r['role_2'],
                r['delta_sum'],
                r['G'],
                r['action']
            ] for r in dynamic.pair_records
        )
    )
    _write_csv(
        tables / 'bellman_state_summary.csv',
        ['mask', 'missing_count', 'simple_action', 'bellman_action', 'value'],
        (
            [
                r['mask'],
                r['missing_count'],
                r['simple_action'],
                r['bellman_action'],
                r['value']
            ] for r in bellman.state_records
        )
    )
    _write_csv(
        tables / 'one_way_sensitivity.csv',
        ['parameter', 'value', 'fixed_best_k', 'fixed_min_cost', 'dynamic_cost'],
        (
            [item.key, value, best_k, fixed_cost, dynamic_cost]
            for item in sensitivity.one_way
            for value, best_k, fixed_cost, dynamic_cost in zip(
                item.values, item.fixed_best_k, item.fixed_min_cost, item.dynamic_cost
            )
        )
    )
    _write_csv(
        tables / 'figure_index.csv',
        ['figure_id', 'language', 'group', 'chapter', 'purpose', 'png', 'svg', 'pdf'],
        (
            [
                r[k] for k in ['figure_id', 'language', 'group', 'chapter', 'purpose', 'png', 'svg', 'pdf']
            ] for r in figure_index
        )
    )
    params = {
        'roles': cfg.roles,
        'second_hand_prices': cfg.second_hand_prices,
        'probabilities': cfg.probabilities,
        'blind_price': cfg.blind_price,
        'eta': cfg.resale_success_prob,
        'gamma': cfg.platform_fee_rate,
        'buyer_shipping': cfg.buyer_shipping,
        'seller_cost': cfg.seller_cost,
        'budget': cfg.budget,
        'simulations': run.simulations,
        'base_seed': cfg.base_seed,
        'percentile_method': 'numpy linear',
        'price_bootstrap_samples': robustness.bootstrap_samples,
        'parameter_provenance': {
            'second_hand_prices': 'Within-design medians of 125 usable listing records',
            'blind_price': 'Scenario draw cost based on the official JPY 550 price',
            'resale_success_prob': 'Scenario assumption without transaction-level resale tracking',
            'platform_fee_rate': 'Approximate fee scenario for the collection period',
            'buyer_shipping': 'Scenario shipping adjustment',
            'seller_cost': 'Scenario value for packaging and shipping',
            'budget': 'Integer reference above the direct-purchase cost'
        }
    }
    (summary_dir / 'parameters.json').write_text(
        json.dumps(params, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    (summary_dir / 'validation_report.txt').write_text('\n'.join(checks) + '\n', encoding='utf-8')
    lines = [
        'Blind-draw collection results',
        '=' * 48,
        f"Simulation mode: {('quick' if run.quick else 'final')}",
        f'Trial count: {run.simulations}',
        f'Direct-purchase cost: {cfg.direct_cost:.4f}',
        f'Optimal fixed k: {fixed.best_k}',
        f'Minimum fixed expected cost: {fixed.total_cost[fixed.best_k]:.6f}',
        f'Collect-all expected draws: {collect.expected_draws:.6f}',
        f'Collect-all expected net cost: {collect.expected_cost:.6f}',
        f'Exact dynamic expected draws: {dynamic.exact_draws:.6f}',
        f'Exact dynamic expected cost: {dynamic.exact_cost:.6f}',
        f'Bellman initial value: {bellman.initial_cost:.6f}',
        f'Bellman action conflicts: {len(bellman.conflicts)}',
        '',
        'Risk measures:'
    ]
    for (name, s) in summaries.items():
        lines.append(
            f'{name}: Mean={s.mean:.4f}, SD={s.sd:.4f}, P95={s.p95:.4f}, P(C>{cfg.budget:.0f})={s.exceed_budget:.4%}'
        )
    (summary_dir / 'model_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def export_audit_diagnostics(run, fixed, probabilities, net_summaries, outlay_summaries, cash_exact):
    """Export diagnostics and table rows using unrounded values."""
    tables = run.output_root / 'tables'
    keys = tuple(probabilities.records[0])
    _write_csv(
        tables / 'probability_sensitivity.csv',
        keys,
        ([r[k] for k in keys] for r in probabilities.records)
    )
    _write_csv(
        tables / 'cash_outlay_risk_summary.csv',
        ['strategy', 'mean_outlay', 'p95_outlay', 'outlay_exceed', 'net_exceed'],
        (
            [
                name,
                s.mean,
                s.p95,
                s.exceed_budget,
                net_summaries[name].exceed_budget
            ] for (name, s) in outlay_summaries.items()
        )
    )
    _write_csv(
        tables / 'fixed_cash_exact.csv',
        tuple(cash_exact[0]),
        ([r[k] for k in cash_exact[0]] for r in cash_exact)
    )
    fixed_rows = [
        f'{k} & {fixed.draw_cost[k]:.2f} & {fixed.missing_cost[k]:.2f} & {fixed.resale_income[k]:.2f} & {fixed.total_cost[k]:.2f} \\\\' for k in range(
            6
        )
    ]
    selected = [probabilities.records[j] for j in (0, 25, 50)]
    probability_rows = [
        f"{r['p_1']:.3f} & {r['p_5']:.3f} & {r['fixed_best_k']} & {r['fixed_min_cost']:.2f} & {r['dynamic_cost']:.2f} \\\\" for r in selected
    ]
    names = {
        'Secondary market only': 'Secondary market only',
        'Fixed k=2': 'Fixed $k=2$',
        'Fixed k=3': 'Fixed $k=3$',
        'Dynamic stopping': 'Dynamic stopping',
        'Draw until complete': 'Draw until complete'
    }
    cash_rows = [
        f'{names[name]} & {s.mean:.2f} & {s.p95:.2f} & {100 * s.exceed_budget:.2f}\\% & {100 * net_summaries[name].exceed_budget:.2f}\\% \\\\' for (
            name,
            s
        ) in outlay_summaries.items()
    ]
    for (filename, rows) in (
        ('fixed_selected_rows_en.tex', fixed_rows),
        ('probability_selected_rows_en.tex', probability_rows),
        ('cash_risk_rows_en.tex', cash_rows)
    ):
        (tables / filename).write_text('\n'.join(rows) + '\n', encoding='utf-8')
    export_latex_tables(tables, include_record_audit=False)

def export_record_audit(cfg, run):
    from dataclasses import replace
    from .dynamic_stop import solve_dynamic
    from .fixed_stop import solve_fixed
    path = Path(__file__).resolve().parents[2] / 'data' / 'market_trace_audit.json'
    audit = json.loads(path.read_text(encoding='utf-8'))
    records = []
    for (label, count, medians) in (
        ('Original usable records', audit['usable_records'], audit['baseline_medians']),
        (
            'Conservative record check',
            audit['conservative_check_records'],
            audit['conservative_check_medians']
        )
    ):
        local = replace(cfg, second_hand_prices=tuple(medians))
        fixed = solve_fixed(local)
        dynamic = solve_dynamic(local)
        records.append(
            {
                'scenario': label,
                'records': count,
                'direct_cost': local.direct_cost,
                'fixed_best_k': fixed.best_k,
                'fixed_min_cost': float(fixed.total_cost[fixed.best_k]),
                'dynamic_cost': dynamic.exact_cost
            }
        )
    tables = run.output_root / 'tables'
    _write_csv(
        tables / 'market_record_sensitivity.csv',
        tuple(records[0]),
        ([r[k] for k in records[0]] for r in records)
    )
    rows = [
        f"{r['records']} & {r['direct_cost']:.2f} & {r['fixed_best_k']} & {r['fixed_min_cost']:.2f} & {r['dynamic_cost']:.2f} \\\\" for r in records
    ]
    (tables / 'record_audit_rows_en.tex').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    export_latex_tables(tables)

def export_latex_tables(tables, include_record_audit=True):
    """Export complete LaTeX tabular environments."""
    definitions = [
        (
            'fixed_selected',
            'rrrrr',
            '$k$ & Draw spending & Missing-item cost & Resale offset & Expected net cost \\\\'
        ),
        (
            'probability_selected',
            'rrrrr',
            '$p_H$ & $p_L$ & Fixed $k^*$ & Fixed cost (CNY) & Dynamic cost (CNY) \\\\'
        ),
        ('cash_risk', 'lrrrr', 'Strategy & Mean $A$ & P95 of $A$ & $\\Prob(A>400)$ & $\\Prob(C>400)$ \\\\')
    ]
    if include_record_audit:
        definitions.append(
            ('record_audit', 'rrrrr', 'Records & Direct cost & Fixed $k^*$ & Fixed cost & Dynamic cost \\\\')
        )
    for (stem, columns, header) in definitions:
        rows = (tables / f'{stem}_rows_en.tex').read_text(encoding='utf-8')
        body = '\\begin{tabular}{' + columns + '}\n' + '\\toprule' + '\n' + header + '\n' + '\\midrule' + '\n' + rows + '\\bottomrule' + '\n' + '\\end{tabular}' + '\n'
        (tables / f'{stem}_table_en.tex').write_text(body, encoding='utf-8')
