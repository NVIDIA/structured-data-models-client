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


#: The field metadata Arrow's ``arrow.json`` extension sets.
JSON_TAG = {
    b'ARROW:extension:name': b'arrow.json',
    b'ARROW:extension:metadata': b'',
}


def _ipc(table: pa.Table) -> pa.Table:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return pa.ipc.open_stream(sink.getvalue()).read_all()


def _json_tables(texts: list[str]) -> dict[str, pa.Table]:
    r"""``texts`` in each way an Arrow reading marks them as JSON: a string
    field carrying the extension's metadata, as built and after an IPC round
    trip (which a newer pyarrow reads back as the extension type, and
    pyarrow 14, the floor, as the metadata), and, where the pyarrow has it,
    the extension type itself."""
    field = pa.field('f', pa.string(), metadata=JSON_TAG)
    tagged = pa.Table.from_arrays([pa.array(texts)], schema=pa.schema([field]))
    tables = {'metadata': tagged, 'metadata after IPC': _ipc(tagged)}
    if hasattr(pa, 'json_'):
        tables['extension'] = pa.table({'f': pa.array(texts, type=pa.json_())})
        tables['extension after IPC'] = _ipc(tables['extension'])
    return tables


def test_a_json_row_value_is_its_arrow_json_cell() -> None:
    rp = _run_python()
    text, row = _rust_literal('TEXT'), _rust_literal('ROW')
    odd: list[str] = []
    assert rp._canonical_text([rp.canonical_cell(json.loads(text), odd)]) == row
    assert odd == []
    as_rows = rp.result_values(
        ['f'], None, [[json.loads(text)]], duckdb_types=False
    )
    for way, table in _json_tables([text]).items():
        as_arrow = rp.arrow_values(table)
        assert as_arrow['ordered'] == as_rows['ordered'], way
        assert as_arrow['unencodable'] == [], way
    for way, table in _json_tables(['{"a": 2}']).items():
        # Another JSON value differs, read either way.
        assert rp.arrow_values(table)['ordered'] != as_rows['ordered'], way
    for way, table in _json_tables(['{']).items():
        # Text that is no JSON is unencodable.
        assert rp.arrow_values(table)['unencodable'] == ['arrow:json'], way
    # Unmarked text stays text, as built and after IPC.
    plain = pa.table({'f': pa.array([text])})
    for table in (plain, _ipc(plain)):
        assert (
            rp.arrow_values(table)['ordered']
            == rp.result_values(['f'], None, [[text]], duckdb_types=False)[
                'ordered'
            ]
        )
