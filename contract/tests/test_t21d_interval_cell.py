# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1d: an interval has one canonical cell, read as a row's value
(pyarrow's ``MonthDayNano``, as the Postgres provider hands it over) or
read from Arrow (``month_day_nano_interval``), so a row read and an Arrow
read of the same interval compare equal; a plain tuple stays a list.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')


def test_an_interval_row_value_is_its_arrow_cell() -> None:
    value = pa.MonthDayNano([14, -3, 5_000_000_001])
    odd: list[str] = []
    row = run_python.canonical_cell(value, odd)
    table = pa.table(
        {'f': pa.array([value], type=pa.month_day_nano_interval())}
    )
    arrow = run_python.arrow_values(table)
    assert row == ['interval', 14, -3, '5000000001']
    rows = run_python.result_values(['f'], None, [[value]], duckdb_types=False)
    assert rows['ordered'] == arrow['ordered']
    assert rows['unencodable'] == arrow['unencodable'] == []
    assert odd == []
    assert run_python.canonical_cell((14, -3, 5), []) == [
        'list',
        [['i', '14'], ['i', '-3'], ['i', '5']],
    ]
