# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: a JSON value has one canonical cell, read as a driver's row
value (``json.loads``'s dict or list, as psycopg reads ``json`` and
``jsonb``) or from Arrow text marked ``arrow.json`` (the Postgres
provider's), on the Python runner and in ``sdm-testkit``'s ``values``
alike, so a row read and an Arrow read of one JSON value compare equal;
equality stays exact on the value, and another value still differs.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
VALUES = ROOT / 'rust/crates/sdm-testkit/src/values.rs'


def _run_python() -> ModuleType:
    name = 't21e_run_python_json'
    if name not in sys.modules:
        path = ROOT / 'contract/runners/run_python.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _rust_literal(name: str) -> str:
    r"""The raw string ``name`` in ``values.rs``'s test, which holds the
    Rust reading to the same text and row."""
    found = re.search(
        rf'const {name}: &str = r#"(.*?)"#;', VALUES.read_text(encoding='utf-8')
    )
    assert found is not None, name
    return found.group(1)


def test_a_json_row_value_is_its_arrow_json_cell() -> None:
    rp = _run_python()
    text, row = _rust_literal('TEXT'), _rust_literal('ROW')
    odd: list[str] = []
    assert rp._canonical_text([rp.canonical_cell(json.loads(text), odd)]) == row
    assert odd == []
    as_rows = rp.result_values(
        ['f'], None, [[json.loads(text)]], duckdb_types=False
    )
    as_arrow = rp.arrow_values(
        pa.table({'f': pa.array([text], type=pa.json_())})
    )
    assert as_rows['ordered'] == as_arrow['ordered']
    assert as_arrow['unencodable'] == []
    # Another JSON value differs, read either way.
    other = rp.arrow_values(
        pa.table({'f': pa.array(['{"a": 2}'], type=pa.json_())})
    )
    assert other['ordered'] != as_rows['ordered']
    # Unmarked text stays text; text that is no JSON is unencodable.
    plain = rp.arrow_values(pa.table({'f': pa.array([text])}))
    assert plain['ordered'] != as_rows['ordered']
    broken = rp.arrow_values(pa.table({'f': pa.array(['{'], type=pa.json_())}))
    assert broken['unencodable'] == ['arrow:json']
