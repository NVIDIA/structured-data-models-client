# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Current source policy behind the historical real-data refusal counts.

T4.4 implements ClientRules; it is no longer an intentionally unreachable
temporary refusal whose count can be inferred from SQL provider selection.
SQL providers retain ValueByValue, and the benchmark still asks relational
questions, so its model-refusal count retains that narrower explanation.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_PROVIDERS = ('duckdb', 'sqlite', 'postgres')


def test_client_rules_are_implemented_and_sql_keeps_its_declared_reading():
    source = (ROOT / 'rust/crates/sdm-core/src/materialize.rs').read_text()
    assert 'not_yet("text_time_rules")' not in source
    probes = json.loads((ROOT / 'contract/refusal-census.json').read_text())
    assert all(
        probe['site'] != 'rule:text_time_rules' for probe in probes['probes']
    )
    for name in SQL_PROVIDERS:
        source = (ROOT / f'rust/crates/sdm-data/src/{name}.rs').read_text()
        assert 'text_times: Some(TextTimes::ValueByValue)' in source, name


def test_the_real_data_benchmark_asks_no_tabular_model():
    source = (ROOT / 'scripts/live/real_data_benchmark.py').read_text()
    assert 'KumoTabular' not in source
    assert 'model=KumoRelational()' in source
    predict = (ROOT / 'rust/crates/sdm-request/src/predict.rs').read_text()
    assert 'return Err(not_yet("model"));' in predict
