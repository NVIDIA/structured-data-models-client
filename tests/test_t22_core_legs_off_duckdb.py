# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The runner and comparator change for the core legs off DuckDB, one
test each (T2.2's test author's).

A case names its DuckDB routes (``.../duckdb``) and DuckDB's SQL provider
in its ``core`` inputs. Off DuckDB the core legs run the same case on that
database: ``run_python.py --mode python-core --database sqlite`` forces the
same paths of SQLite's backend (``run_python.on_database``), and
``compare.py --database sqlite`` holds the core's recorded inputs to the
case's with SQLite's dialect in DuckDB's place. Without either, no case's
core legs could run on SQLite: the DuckDB routes leave a SQLite source on
the old route, and the inputs rule fails every case on its dialect.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASE = 'base.regression'


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _module(ROOT / 'contract/runners/run_python.py', 't22_run_python')
COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't22_compare')
ROUTING = _module(
    ROOT / 'packages/structured-data-models-client/src/sdm_client/_routing.py',
    't22_routing',
)


def _case() -> dict[str, Any]:
    path = ROOT / f'contract/cases/s1/{CASE}.json'
    found: dict[str, Any] = json.loads(path.read_text(encoding='utf-8'))
    return found


def test_a_case_s_routes_on_sqlite_are_its_duckdb_paths_of_sqlite() -> None:
    r"""python-core on SQLite forces each DuckDB path of a case as SQLite's
    (``.../duckdb`` to ``.../sqlite``), leaves a path of no backend as it
    is, and every path so forced, over every case, is one the client
    routes (``_routing.PATHS``)."""
    assert [
        RUNNER.on_database(path, 'sqlite')
        for path in (
            'query.validate/accept/any',
            'schema.from-source/all-tables/duckdb',
            'client.predict/regression/duckdb',
        )
    ] == [
        'query.validate/accept/any',
        'schema.from-source/all-tables/sqlite',
        'client.predict/regression/sqlite',
    ]
    assert RUNNER.on_database('x/duckdb', 'duckdb') == 'x/duckdb'
    # Not vacuous: the S1 cases name DuckDB routes, and each is a path of
    # SQLite's backend too.
    forced = [
        path
        for case, _ in RUNNER.load_cases([ROOT / 'contract/cases/s1'])
        for path in (RUNNER.on_database(r, 'sqlite') for r in case['routes'])
    ]
    assert any(path.endswith('/sqlite') for path in forced)
    assert not [p for p in forced if p not in ROUTING.PATHS]
    assert not [p for p in forced if p.endswith('/duckdb')]


def _on(core: dict[str, Any], dialect: str) -> dict[str, Any]:
    r"""A case's core inputs with ``dialect`` in DuckDB's SQL provider's
    place."""
    out = copy.deepcopy(core)
    for spec in out.values():
        if spec.get('provider') == {'provider': 'sql', 'dialect': 'duckdb'}:
            spec['provider'] = {'provider': 'sql', 'dialect': dialect}
    return out


def _observation(dialect: str) -> dict[str, Any]:
    r"""python-ref's ``base.regression@sqlite`` line made python-core's, its
    core started with the case's inputs naming ``dialect``."""
    case = _case()
    [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    [line] = [
        json.loads(text)
        for text in path.read_text(encoding='utf-8').splitlines()
        if json.loads(text)['case'] == f'{CASE}@sqlite'
    ]
    observation = copy.deepcopy(line)
    observation['identity']['runner'] = 'python-core'
    observation['identity']['reference'] = None
    observation['identity']['native'] = {'sha256': '0' * 64}
    observation['identity']['native_loaded'] = True
    observation['identity']['operations'] = sorted(case['core'])
    core = _on(case['core'], dialect)
    observation['core'] = {
        'inputs': [
            {'op': 'describe_source', **core['describe_source']},
            {
                'op': 'materialize',
                'query': case['input']['pql'],
                'schema': {},
                **core['materialize'],
            },
            {
                'op': 'predict',
                **core['predict'],
                'state': {'version': 2, 'query': '', 'payload': {}},
            },
        ],
        'fetches': [],
        'accepted': None,
        'described': {},
        'materialized': None,
        'predicted': None,
        'error': None,
    }
    return observation


def _input_failures(tmp_path: Path, dialect: str) -> list[str]:
    cases = tmp_path / dialect / 'cases'
    cases.mkdir(parents=True)
    source = ROOT / f'contract/cases/s1/{CASE}.json'
    (cases / source.name).write_bytes(source.read_bytes())
    observed = tmp_path / dialect / 'python-core.jsonl'
    observed.write_text(json.dumps(_observation(dialect)) + '\n')
    report = COMPARE.compare([cases], None, [observed], database='sqlite')
    # The seeded observation is valid, so the rules ran.
    assert not [t for _, t in report.failures if 'not valid' in t]
    return [
        text
        for rule, text in report.failures
        if rule == 'inputs' and 'core.inputs' in text
    ]


def test_the_inputs_rule_off_duckdb_takes_the_database_s_dialect(
    tmp_path: Path,
) -> None:
    r"""On SQLite the inputs rule holds python-core's recorded inputs to the
    case's with SQLite's dialect: they pass, and the case's DuckDB dialect
    as recorded fails."""
    assert _case()['core']['describe_source']['provider'] == {
        'provider': 'sql',
        'dialect': 'duckdb',
    }
    assert _input_failures(tmp_path, 'sqlite') == []
    assert _input_failures(tmp_path, 'duckdb') != []
