"""只读检查默认八款案例；不绘图、不请求网络、不改写数据。"""
from dataclasses import replace
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from gacha_model.config import ModelConfig
from gacha_model.fixed_stop import solve_fixed
from gacha_model.dynamic_stop import solve_dynamic
from gacha_model.bellman import solve_bellman
from gacha_model.cash_risk import enumerate_fixed_cash_risk
from gacha_model.probability_sensitivity import solve_probability_sensitivity
from gacha_model.uncertainty import load_price_samples
from gacha_model.validation import validate_config


def main():
    cfg = ModelConfig()
    validate_config(cfg)
    fixed = solve_fixed(cfg)
    dynamic = solve_dynamic(cfg)
    bellman = solve_bellman(cfg)
    assert fixed.best_k == 3
    assert abs(fixed.total_cost[3] - 357.307254734375) < 1e-8
    assert abs(dynamic.exact_cost - 356.7155814285714) < 1e-8
    assert abs(dynamic.exact_draws - 2.619047619047619) < 1e-8
    assert len(bellman.state_records) == 256 and not bellman.conflicts
    assert abs(bellman.initial_cost - dynamic.exact_cost) < 1e-8
    scenarios = solve_probability_sensitivity(cfg)
    assert len(scenarios.records) == 51
    assert all(record['bellman_conflicts'] == 0 for record in scenarios.records)
    assert tuple(float(np.median(x)) for x in load_price_samples(cfg)) == cfg.second_hand_prices
    for k, net, outlay in ((2, .034375, .0625), (3, .072109375, .1328125)):
        result = enumerate_fixed_cash_risk(k, cfg)
        assert abs(result['net_exceed'] - net) < 1e-10
        assert abs(result['outlay_exceed'] - outlay) < 1e-10
    with (ROOT / 'data/market_prices_cleaned.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    audit = json.loads((ROOT / 'data/market_trace_audit.json').read_text(encoding='utf-8'))
    assert len(records) == audit['usable_records'] == 125
    retained = [r for r in records if r['evidence_id'] not in audit['conservative_check_excluded_ids']]
    assert len(retained) == audit['conservative_check_records'] == 122
    medians = tuple(float(np.median([float(r['unit_cost']) for r in retained
                                    if int(r['role_index']) == i])) for i in range(1, 9))
    assert medians == tuple(audit['conservative_check_medians'])
    local = replace(cfg, second_hand_prices=medians)
    assert solve_fixed(local).best_k == 2
    assert abs(solve_dynamic(local).exact_cost - 354.32297666666665) < 1e-8
    print('PASS: baseline, 256 Bellman states, 51 probability scenarios, exact cash risk, '
          '125 price records and 122-record conservative check')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
