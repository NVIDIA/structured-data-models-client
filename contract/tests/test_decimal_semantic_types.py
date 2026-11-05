# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact decimal values retain their declared semantic types.

Current-client checks use retained expected values and scripted endpoint responses.
No archived client runtime is executed.
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
CASES = ROOT / 'contract/cases/s1'

PREFIX = 'stype:'
CASE_ID = 'decimal.stype-past-float-precision'
DATABASES = ('duckdb', 'sqlite', 'postgres')


def runner() -> ModuleType:
    name = 'test_decimal_semantic_types_run_python'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, RUNNER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


KEYS = tuple(runner().case_key(CASE_ID, d) for d in DATABASES)


@functools.cache
def case() -> tuple[dict[str, Any], str]:
    [(loaded, digest)] = runner().load_cases([CASES / f'{CASE_ID}.json'])
    return loaded, digest


@functools.cache
def expected_cases() -> dict[str, dict[str, Any]]:
    """Independent expected facts, indexed by case key."""
    path = ROOT / 'contract/fixtures/maintained/request-expectations.jsonl'
    return {
        json.loads(line)['case']: json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    }


def column_d(line: dict[str, Any]) -> dict[str, Any]:
    r"""``orders.d`` in a line's public schema."""
    [orders] = [
        t for t in line['public']['schema']['tables'] if t['name'] == 'orders'
    ]
    [d] = [c for c in orders['columns'] if c['name'] == 'd']
    return d


def test_the_case_is_a_scaled_decimal_past_float64() -> None:
    r"""The case is what it names: one setup statement adding ``orders.d``,
    a ``DECIMAL(38, 18)`` stepping by 1e-18 over the shop base's 80 orders,
    run as far as ``from_source`` on all three databases."""
    loaded, _ = case()
    assert loaded['status'] == 'acceptance' and loaded['slice'] == 'S1'
    assert runner().case_databases(loaded) == DATABASES
    assert loaded['until'] == 'from_source'
    [setup] = loaded['input']['setup']
    assert '0.000000000000000001' in setup and 'DECIMAL(38, 18) AS d' in setup


@pytest.mark.parametrize('case_key', KEYS)
def test_python_ref_s_line_is_at_the_case_file(case_key: str) -> None:
    """Independent expected facts agree with this case's stated behavior."""
    _, digest = case()
    line = expected_cases().get(case_key)
    assert line is not None, f'{case_key}: no expected facts'
    assert line['outcome']['result'] == 'success'
    assert line['outcome']['stage'] == 'from_source'
    want = (
        {'dtype': 'string', 'stype': 'text'}
        if case_key.endswith('@sqlite')
        else {'dtype': 'float64', 'stype': 'categorical'}
    )
    assert {k: column_d(line)[k] for k in want} == want


def _server() -> bool:
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    r"""python-core over the case on ``database``: its observation by key,
    or under ``''`` why the run failed whole."""
    with tempfile.TemporaryDirectory(prefix='t214b-stype-') as tmp:
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
                str(CASES / f'{CASE_ID}.json'),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=900,
        )
        if done.returncode != 0:
            return {'': done.stdout + done.stderr}
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


@pytest.mark.parametrize('case_key', KEYS)
def test_python_core_reads_the_column_as_the_reference_does(
    case_key: str,
) -> None:
    r"""The rides-along exit: python-core, on the core route
    (``describe_source`` run), gives ``orders.d`` the reference's dtype and
    stype on each database, and the whole schema is the reference's."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        assert line['identity']['native_loaded']
        assert 'describe_source' in line['identity']['operations']
        want = expected_cases()[case_key]
        assert column_d(line) == column_d(want), (
            f'{case_key}: orders.d is {column_d(line)} where python-ref '
            f'has {column_d(want)}'
        )
        assert line['public']['schema'] == want['public']['schema']

    check()
