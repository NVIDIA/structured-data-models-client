# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact decimal values retain their declared semantic types.

Assertions run the current client on each local connector.
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
    with tempfile.TemporaryDirectory(prefix='decimal_semantic_types-') as tmp:
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
def test_decimal_column_has_connector_specific_semantic_type(
    case_key: str,
) -> None:
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test DSN is not configured')
    run = core_run(database)
    assert '' not in run, run.get('')
    line = run[case_key]
    assert line['identity']['native_loaded']
    assert 'describe_source' in line['identity']['operations']
    assert line['outcome']['result'] == 'success'
    assert line['outcome']['stage'] == 'from_source'
    sqlite = database == 'sqlite'

    def column(name: str, dtype: str, stype: str) -> dict[str, str]:
        return {'dtype': dtype, 'name': name, 'stype': stype}

    assert line['public']['schema'] == {
        'links': [['orders', 'user_id', 'users', 'user_id']],
        'tables': [
            {
                'columns': [
                    column('order_id', 'int64', 'ID'),
                    column('user_id', 'int64', 'ID'),
                    column('price', 'float64', 'numerical'),
                    column(
                        'ts',
                        'string' if sqlite else 'timestamp[us]',
                        'timestamp',
                    ),
                    column('d', 'string', 'text')
                    if sqlite
                    else column('d', 'float64', 'categorical'),
                ],
                'duplicate_keys': 0,
                'name': 'orders',
                'primary_key': 'order_id',
                'time_column': 'ts',
            },
            {
                'columns': [
                    column('user_id', 'int64', 'ID'),
                    column('age', 'int64', 'numerical'),
                    column('country', 'string', 'categorical'),
                ],
                'duplicate_keys': 0,
                'name': 'users',
                'primary_key': 'user_id',
                'time_column': None,
            },
        ],
    }
