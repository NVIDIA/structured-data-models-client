# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.11's ``unseeded`` exit on ``python-core`` (the package's test
author's): an unseeded sample judged by its properties (the owner,
2026-10-20, KO2-76), not row by row.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t211_unseeded.py -q

``random_seed=None`` draws anew on every call, on every database, so no
request of the reference can be compared row by row. The cases,
``contract/tests/t211-unseeded/*.json`` (out of ``contract/cases/``, whose
every request the gate compares row by row), run on DuckDB, SQLite and
Postgres; ``contract/t211-unseeded-vectors.json``, recorded from the
reference by ``contract/tools/t211_unseeded_record.py``, holds what does not
change from draw to draw. python-core must:

- end as the reference does (a success through predict, one request);
- draw the right number of rows: as many context examples, the same
  entities asked about at the same anchor with as many rows of each table
  around them, the same request schema and the same result;
- draw anew on each call: a second run's context examples differ;
- draw only rows the source holds: every context example's entity is a
  user, its label is that user's at its anchor time (the orders in the 30
  days after it, the rule the recorder checked against the reference's
  draws), and every sampled row of every table is a row of that table;
- and give the same rows when a seed is given: ``sample.seed-7``'s
  request is the reference's row by row (``test_t211_sampling.py``).

The core refuses an unseeded sample today by name (``UNSUPPORTED_REQUEST``,
rule ``unseeded``). Gating: ``t211-pending.json``'s ``pending`` names, as
``unseeded-py:<case key>``, the keys whose test failed when written; each
must still fail (strict) until it is taken out.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
RECORDER = ROOT / 'contract/tools/t211_unseeded_record.py'
CASES = ROOT / 'contract/tests/t211-unseeded'
VECTORS = ROOT / 'contract/t211-unseeded-vectors.json'
PENDING = Path(__file__).with_name('t211-pending.json')
DATABASES = ('duckdb', 'sqlite', 'postgres')
CASE_IDS = tuple(sorted(path.stem for path in CASES.glob('*.json')))


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t211u_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('t211u_record', RECORDER)


KEYS_ALL = tuple(runner().case_key(c, d) for d in DATABASES for c in CASE_IDS)


@functools.cache
def vectors() -> dict[str, Any]:
    return json.loads(VECTORS.read_text(encoding='utf-8'))


@functools.cache
def cases() -> dict[str, dict[str, Any]]:
    return {
        case['id']: case
        for case, _ in runner().load_cases(
            [CASES / f'{c}.json' for c in CASE_IDS]
        )
    }


def held(name: str, check: Any) -> None:
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    if f'unseeded-py:{name}' not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take unseeded-py:{name} out of '
        'contract/tests/t211-pending.json pending'
    )


def test_the_unseeded_keys_are_authored() -> None:
    r"""Control: ``authored`` names every unseeded case's key on each
    database on python-core, and on DuckDB and SQLite on rust-host."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    authored = {k for k in gating['authored'] if k.startswith('unseeded')}
    assert authored == {f'unseeded-py:{k}' for k in KEYS_ALL} | {
        f'unseeded-rust:{k}' for k in KEYS_ALL if not k.endswith('@postgres')
    }


@pytest.mark.parametrize('case_id', CASE_IDS)
def test_each_case_asks_for_an_unseeded_sample(case_id: str) -> None:
    r"""Exit "``unseeded``": the case passes ``random_seed=None`` through
    predict on all three databases, and the reference drew different
    context rows on two runs of it on each, so it exercises an unseeded
    draw; its seeded twin is ``sample.seed-7``."""
    case = cases()[case_id]
    assert case['input']['materialize']['random_seed'] is None
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    assert case['core']['materialize']['seed'] is None
    for database in DATABASES:
        line = vectors()['keys'][runner().case_key(case_id, database)]
        assert line['outcome']['result'] == 'success'
        assert line['draws_differ'] is True
        assert line['context_rows'] > 0
    assert (ROOT / 'contract/cases/s1/sample.seed-7.json').exists()


def _server() -> bool:
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


@functools.cache
def core_run(database: str, run: int = 0) -> dict[str, Any]:
    r"""python-core over the unseeded cases on ``database``: each
    observation by its key, or under ``''`` why the run failed whole."""
    with tempfile.TemporaryDirectory(prefix='t211u-') as tmp:
        out = Path(tmp) / 'python-core.jsonl'
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(RUNNER),
                '--mode',
                'python-core',
                '--database',
                database,
                '--out',
                str(out),
                str(CASES),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=1800,
        )
        if done.returncode != 0:
            return {'': done.stdout + done.stderr}
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


def problems(
    case_id: str, line: dict[str, Any], want: dict[str, Any]
) -> list[str]:
    r"""Each property of an unseeded draw ``line`` breaks."""
    found = []
    got = recorder().properties(line)
    if got['outcome'] != want['outcome']:
        return [f'{got["outcome"]} where the reference has {want["outcome"]}']
    for name in (
        'schema',
        'predict',
        'context_rows',
        'context_tables',
        'result',
    ):
        if got.get(name) != want.get(name):
            found.append(f"its {name} is not the reference draws'")
    case = cases()[case_id]
    given = vectors()['cases'][case_id]
    entities = set(given['entities'])
    for entity, anchor, target in recorder().examples(line):
        if entity not in entities:
            found.append(f'a context example of no user: {entity}')
        elif target != recorder().label(case, entity, anchor):
            found.append(f'user {entity} at {anchor} labelled {target}')
    request = recorder().sent(line)
    for section in ('context', 'predict'):
        for table, sampled in request[section]['related_tables'].items():
            columns = recorder().FEATURES.get(table)
            if columns is None:
                found.append(f'{section} samples {table}, which it cannot')
                continue
            rows = {json.dumps(r) for r in given['source'][table]}
            at = [sampled['columns'].index(c) for c in columns]
            for row in sampled['rows']:
                values = [recorder().written(row[i]) for i in at]
                if json.dumps(values) not in rows:
                    found.append(f'{section} {table} row {values} is not in it')
    return found[:8]


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_core_draws_an_unseeded_sample_with_the_reference_s_properties(
    case_key: str,
) -> None:
    r"""Exit "``unseeded``", judged by properties (KO2-76): python-core
    answers the case on the database on the core route as the reference
    does, drawing the right number of rows, each from the source."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        assert line['identity']['native_loaded'] and line['core'] is not None
        assert 'materialize' in line['identity']['operations']
        found = problems(
            case_key.split('@')[0], line, vectors()['keys'][case_key]
        )
        # A new sample each call: a second run draws other context
        # examples, as the reference's two runs did (draws_differ).
        again = core_run(database, 1).get(case_key)
        if not found and again is not None:
            if recorder().examples(line) == recorder().examples(again):
                found.append('two runs drew the same context')
        assert not found, f'{case_key}: ' + '; '.join(found)

    held(case_key, check)
