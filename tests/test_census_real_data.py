# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Historical census assumptions that still apply after Phase 4.

Built-in SQL providers retain their value-by-value date reading and the
benchmark still asks Relational questions.
ClientRules is now supported; its former global source-count restriction
must not prohibit frame or custom-provider support.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_PROVIDERS = ('duckdb', 'sqlite', 'postgres', 'databricks', 'snowflake')


def test_builtin_sql_providers_keep_value_by_value_dates():
    for name in SQL_PROVIDERS:
        source = (ROOT / f'rust/crates/sdm-data/src/{name}.rs').read_text()
        assert 'text_times: Some(TextTimes::ValueByValue)' in source, name


def test_the_real_data_benchmark_asks_no_tabular_model():
    source = (ROOT / 'scripts/live/real_data_benchmark.py').read_text()
    assert 'KumoTabular' not in source
    assert 'model=KumoRelational()' in source
    predict = (ROOT / 'rust/crates/sdm-request/src/predict.rs').read_text()
    assert 'return Err(not_yet("model"));' in predict
