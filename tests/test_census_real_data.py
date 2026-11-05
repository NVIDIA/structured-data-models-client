# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SQL providers retain value-by-value date interpretation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_PROVIDERS = ('duckdb', 'sqlite', 'postgres', 'databricks', 'snowflake')


def test_builtin_sql_providers_keep_value_by_value_dates():
    for name in SQL_PROVIDERS:
        source = (ROOT / f'rust/crates/sdm-data/src/{name}.rs').read_text()
        assert 'text_times: Some(TextTimes::ValueByValue)' in source, name
