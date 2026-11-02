# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Special column names preserve current-client schema, payloads and prediction."""

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

DATABASES = ('duckdb', 'sqlite', 'postgres')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'column_names_record', ROOT / 'contract/tools/column_names_record.py'
)

RECORDED: dict[str, Any] = {'vectors': R.vectors()}

BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}


def reachable(database: str) -> bool:
    if database != 'postgres' or os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')


def _cases() -> dict[str, dict[str, Any]]:
    runner = _module('t218_run_python', RUNNER)
    return {
        case['id']: case
        for case, _ in runner.load_cases([ROOT / 'contract/cases/s1'])
        if case['id'].startswith('name.')
    }


def as_named(value: Any, twin: dict[str, Any], vector: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {
            as_named(key, twin, vector): as_named(item, twin, vector)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [as_named(item, twin, vector) for item in value]
    if value == twin['pql']:
        return vector['pql']
    if value == twin['name']:
        return vector['name']
    return value


def _vectors(database: str) -> list[dict[str, Any]]:
    return [
        vector
        for vector in RECORDED['vectors']
        if vector['database'] == database and vector['character'] is not None
    ]


def test_the_name_reading_reads_only_the_name() -> None:
    twin = {'name': 'a_b', 'pql': 'PREDICT users.a_b FOR EACH users.user_id'}
    vector = {
        'name': 'a.b',
        'pql': 'PREDICT users.`a.b` FOR EACH users.user_id',
    }
    given = {
        'a_b': ['a_b', 'xa_b', 1, None],
        'q': twin['pql'],
        'other': {'name': 'a_bc'},
    }
    assert as_named(given, twin, vector) == {
        'a.b': ['a.b', 'xa_b', 1, None],
        'q': vector['pql'],
        'other': {'name': 'a_bc'},
    }


@functools.cache
def _core_answer(vector_id: str) -> dict[str, Any]:
    vector = BY_ID[vector_id]
    answer = R.run(vector)
    if vector['character'] is None:
        assert answer['outcome'] == {
            'result': 'success',
            'stage': 'materialize',
        }
        assert answer['schema']['tables']
        assert answer['payloads']
        assert answer['sdm_warnings'] == {
            'from_source': 0,
            'materialize': int(vector['role'] == 'target'),
        }
    return answer


@pytest.mark.parametrize('database', DATABASES)
def test_the_core_keeps_every_column_whatever_its_name(database: str) -> None:
    if not reachable(database):
        return

    def check() -> None:
        problems = []
        for vector in _vectors(database):
            twin = BY_ID[vector['twin']]
            got = _core_answer(vector['id'])
            wanted = as_named(_core_answer(twin['id'])['schema'], twin, vector)
            if got['schema'] != wanted:
                problems.append(f'{vector["id"]}: {got["outcome"]}')
            elif (
                got['sdm_warnings'].get('from_source')
                != (_core_answer(twin['id'])['sdm_warnings']['from_source'])
            ):
                problems.append(f'{vector["id"]}: warned at from_source')
        assert not problems, '\n'.join(problems)

    check()


@pytest.mark.parametrize('database', DATABASES)
def test_a_query_names_any_column_in_every_role(database: str) -> None:
    if not reachable(database):
        return

    def check() -> None:
        problems = []
        for vector in _vectors(database):
            twin = BY_ID[vector['twin']]
            got = _core_answer(vector['id'])
            if got['outcome'] != _core_answer(twin['id'])['outcome']:
                problems.append(f'{vector["id"]}: {got["outcome"]}')
                continue
            if got['payloads'] != as_named(
                _core_answer(twin['id'])['payloads'], twin, vector
            ):
                problems.append(f"{vector['id']}: not the twin's request")
            if got['sdm_warnings'] != _core_answer(twin['id'])['sdm_warnings']:
                problems.append(f'{vector["id"]}: {got["sdm_warnings"]}')
        assert not problems, '\n'.join(problems)

    check()


@functools.cache
def _cases_on(database: str) -> dict[str, dict[str, Any]]:
    paths = sorted(
        path for path in (ROOT / 'contract/cases/s1').glob('name.*.json')
    )
    with tempfile.TemporaryDirectory(prefix='t218-cases-') as tmp:
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
                *map(str, paths),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


@pytest.mark.parametrize('database', DATABASES)
def test_each_case_predicts_through_native_operations(
    database: str,
) -> None:
    if not reachable(database):
        return

    def check() -> None:
        runner = _module('t218_run_python', RUNNER)
        observed = _cases_on(database)
        problems = []
        for case_id, case in sorted(_cases().items()):
            key = runner.case_key(case_id, database)
            line = observed[key]
            outcome = line['outcome']
            if (outcome['result'], outcome['stage']) != ('success', 'predict'):
                problems.append(f'{key}: {outcome}')
                continue
            identity = line['identity']
            if not identity['native_loaded'] or not set(case['core']) <= set(
                identity['operations']
            ):
                problems.append(f'{key}: not on the core route')
        assert not problems, '\n'.join(problems)

    check()
