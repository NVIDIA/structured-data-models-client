# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: a month-day-nano interval column tagged as built from a driver's
``timedelta``s (``sdm:interval = timedelta``, the Postgres provider's
reading of psycopg's) has the ``timedelta``'s own canonical cell, read on
the Python runner and in ``sdm-testkit``'s ``values`` alike, one holding
months unencodable; an untagged interval keeps its own cell (so
DuckDB's committed results are read as before), and another interval
still differs.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
VALUES = ROOT / 'rust/crates/sdm-testkit/src/values.rs'
TAG = {b'sdm:interval': b'timedelta'}


def _run_python() -> ModuleType:
    name = 't21e_run_python_interval'
    if name not in sys.modules:
        path = ROOT / 'contract/runners/run_python.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _arrow(values: list[pa.MonthDayNano], tagged: bool) -> dict:
    field = pa.field(
        'f', pa.month_day_nano_interval(), metadata=TAG if tagged else None
    )
    table = pa.Table.from_arrays(
        [pa.array(values, type=pa.month_day_nano_interval())],
        schema=pa.schema([field]),
    )
    return _run_python().arrow_values(table)


def test_a_tagged_interval_reads_as_the_timedelta_it_was_built_from() -> None:
    rp = _run_python()
    delta = dt.timedelta(days=31, microseconds=5)
    found = re.search(
        r'const DELTA_ROW: &str = r#"(.*?)"#;',
        VALUES.read_text(encoding='utf-8'),
    )
    assert found is not None
    assert rp._canonical_text([rp.canonical_cell(delta, [])]) == found.group(1)
    as_rows = rp.result_values(['f'], None, [[delta]], duckdb_types=False)
    one = pa.MonthDayNano([0, 31, 5_000])
    assert _arrow([one], tagged=True)['ordered'] == as_rows['ordered']
    # A timedelta holds no months (psycopg folds a month into 30 days and a
    # year into 365, which months cannot tell apart): one that does is
    # unencodable, never equal.
    month = pa.MonthDayNano([1, 1, 5_000])
    assert _arrow([month], tagged=True)['unencodable'] == [
        'arrow:interval months'
    ]
    # Untagged, the interval keeps its own cell; another interval differs.
    assert _arrow([one], tagged=False)['ordered'] != as_rows['ordered']
    other = pa.MonthDayNano([0, 31, 6_000])
    assert _arrow([other], tagged=True)['ordered'] != as_rows['ordered']
