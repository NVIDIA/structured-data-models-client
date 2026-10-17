# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.2's round 1 SQLite regressions on ``python-core`` (the package's
test author's), through the public calls with SQLite's core routes forced,
each held to the reference's answer in
``contract/sqlite-regression-vectors.json`` (recorded by
``contract/tools/sqlite_regressions_record.py``):

- a caller's own temporary table ``__sdm_seeds__`` survives
  ``materialize`` unchanged (the provider's scratch relation never
  replaces a relation it does not own);
- an ``INT PRIMARY KEY`` holding 101.5 to 107.5 gets the reference's
  seeded context keys, 102.5, 106.5 and 107.5 (the hash is chosen by the
  column's type, not each value's storage class);
- an ``INT PRIMARY KEY`` holding a NULL materializes as the reference does,
  the null filtered (a SQLite primary key that is not a rowid alias keeps
  its catalog nullability).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'sqlite_regressions_record',
    ROOT / 'contract/tools/sqlite_regressions_record.py',
)
RECORDED: dict[str, Any] = json.loads(
    (ROOT / 'contract/sqlite-regression-vectors.json').read_text('utf-8')
)


def _on_the_core(monkeypatch: pytest.MonkeyPatch, name: str) -> dict[str, Any]:
    r"""The scenario through the public calls on SQLite's core routes:
    ``from_source`` with tables named, ``materialize`` and validation."""
    from sdm_client import _routing

    routes = (
        _routing.FROM_SOURCE_SQLITE_TABLES
        + _routing.MATERIALIZE_SQLITE
        + _routing.S0
    )
    # Not vacuous: the build routes each forced path on the core.
    assert set(routes) <= _routing.CORE_ROUTED
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(routes))
    answer: dict[str, Any] = R.run(RECORDED['scenarios'][name])
    return answer


def _context_keys(answer: dict[str, Any]) -> list[Any]:
    [payload] = answer['payloads']
    rows = payload['context']['instance_table']['rows']
    return [row[1] for row in rows]


def test_the_vectors_are_the_reference_s() -> None:
    r"""Control: recorded from the reference this tree names, every
    scenario of the recorder, each a success there."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert set(RECORDED['scenarios']) == set(R.SCENARIOS)
    for name, scenario in RECORDED['scenarios'].items():
        assert scenario['setup'] == R.SCENARIOS[name]['setup']
        assert scenario['reference']['outcome']['result'] == 'success'
    seeds = RECORDED['scenarios']['seeds-table']['reference']['temp_after']
    assert seeds == {'columns': ['important'], 'rows': [['keep me']]}
    floats = RECORDED['scenarios']['float-keys']['reference']
    assert _context_keys(floats) == [102.5, 106.5, 107.5]


def test_a_caller_s_temporary_table_survives_materialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The caller's ``temp.__sdm_seeds__`` holds its one column and
    ``'keep me'`` after ``materialize``, as on the reference, and the
    request is the reference's."""
    wanted = RECORDED['scenarios']['seeds-table']['reference']
    got = _on_the_core(monkeypatch, 'seeds-table')
    assert got['temp_after'] == wanted['temp_after']
    assert got['outcome'] == wanted['outcome']
    assert got['payloads'] == wanted['payloads']


def test_float_keys_of_an_int_column_draw_the_reference_s_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""An ``INT PRIMARY KEY`` holding 101.5 to 107.5: the seeded context
    keys are the reference's (102.5, 106.5, 107.5), and the request is the
    reference's."""
    wanted = RECORDED['scenarios']['float-keys']['reference']
    got = _on_the_core(monkeypatch, 'float-keys')
    assert got['outcome'] == wanted['outcome']
    assert _context_keys(got) == _context_keys(wanted)
    assert got['payloads'] == wanted['payloads']


def test_a_null_primary_key_is_filtered_as_the_reference_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""An ``INT PRIMARY KEY`` holding a NULL: ``materialize`` succeeds, the
    null filtered, and the request is the reference's."""
    wanted = RECORDED['scenarios']['null-key']['reference']
    got = _on_the_core(monkeypatch, 'null-key')
    assert got['outcome'] == wanted['outcome']
    assert got['payloads'] == wanted['payloads']
