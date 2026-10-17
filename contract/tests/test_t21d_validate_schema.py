# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1d's comparator change for ``validate_query`` off DuckDB (T2.2's
test author's): the inputs rule holds the schema ``validate_query`` is
handed to the case's with each column's dtype the one the contract
loader's type map gives it on the database (SQLite keeps the shop's
timestamp as ISO text, so ``ts`` is a string there), every dtype still
compared.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASE = 's0.accept.binary-classification'


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't21d_compare')
SOURCE = ROOT / f'contract/cases/s0/{CASE}.json'


def _observation(database: str, schema: dict[str, Any]) -> dict[str, Any]:
    r"""python-ref's line of the case on ``database`` made python-core's,
    its core started on ``schema``."""
    key = CASE if database == 'duckdb' else f'{CASE}@{database}'
    [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    [line] = [
        json.loads(text)
        for text in path.read_text(encoding='utf-8').splitlines()
        if json.loads(text)['case'] == key
    ]
    case = json.loads(SOURCE.read_text(encoding='utf-8'))
    spec = case['core']['validate_query']
    observation = copy.deepcopy(line)
    observation['identity'].update(
        runner='python-core',
        reference=None,
        native={'sha256': '0' * 64},
        native_loaded=True,
        operations=['validate_query'],
    )
    observation['core'] = {
        'inputs': [
            {
                'op': 'validate_query',
                'query': case['input']['pql'],
                'profile': spec['profile'],
                'schema': schema,
            }
        ],
        'fetches': [],
        'accepted': None,
        'described': None,
        'materialized': None,
        'predicted': None,
        'error': None,
    }
    return observation


def _input_failures(
    tmp_path: Path, database: str, schema: dict[str, Any]
) -> list[str]:
    where = tmp_path / f'{database}-{len(list(tmp_path.iterdir()))}'
    cases = where / 'cases'
    cases.mkdir(parents=True)
    (cases / SOURCE.name).write_bytes(SOURCE.read_bytes())
    observed = where / 'python-core.jsonl'
    observed.write_text(json.dumps(_observation(database, schema)) + '\n')
    report = COMPARE.compare([cases], None, [observed], database=database)
    assert not [t for _, t in report.failures if 'not valid' in t]
    return [
        text
        for rule, text in report.failures
        if rule == 'inputs' and 'core.inputs' in text
    ]


def _retyped(schema: dict[str, Any], column: str, dtype: str) -> dict[str, Any]:
    out = copy.deepcopy(schema)
    for table in out['tables']:
        for each in table['columns']:
            if each['name'] == column:
                each['dtype'] = dtype
    return out


def test_validate_query_s_schema_off_duckdb_is_the_type_map_s(
    tmp_path: Path,
) -> None:
    r"""On SQLite the schema with ``ts`` a string (the type map stores a
    timestamp as ISO text there) passes the inputs rule; the case's DuckDB
    ``time``, and ``age`` (an INTEGER there, kept as the case's ``int``) as
    a float, each fail it. On DuckDB the case's dtypes pass and ``ts`` as a
    string fails."""
    case = json.loads(SOURCE.read_text(encoding='utf-8'))
    pinned = case['core']['validate_query']['schema']
    on_sqlite = _retyped(pinned, 'ts', 'string')
    assert on_sqlite != pinned
    assert _input_failures(tmp_path, 'sqlite', on_sqlite) == []
    assert _input_failures(tmp_path, 'sqlite', pinned) != []
    assert (
        _input_failures(tmp_path, 'sqlite', _retyped(on_sqlite, 'age', 'float'))
        != []
    )
    assert _input_failures(tmp_path, 'duckdb', pinned) == []
    assert _input_failures(tmp_path, 'duckdb', on_sqlite) != []
