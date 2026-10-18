# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: a SQLite or Postgres line is held to the calls it makes.

A case's core block is DuckDB's run. Off DuckDB the reference's line may
stop at an earlier call: SQLite stores a decimal as its canonical text,
which the reference types as text and refuses as a number at materialize
(decimal.sum-10-2@sqlite), so python-core, refused there as the
reference is, never runs the core's materialize. The route, inputs and
core-consistency rules then hold the line's core operations up to that
call, and any later one the runner ran; a call before the stop that did
not run still fails, and DuckDB's lines are held to the whole block.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'contract/cases/s1/decimal.sum-10-2.json'


def _compare() -> ModuleType:
    name = 't21e_compare_stop'
    if name not in sys.modules:
        path = ROOT / 'scripts/rust_core/compare.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _observation(operations: list[str]) -> dict[str, Any]:
    return {
        'identity': {
            'native_loaded': True,
            'native': {'sha256': '0' * 64},
            'operations': operations,
        },
        'core': {},
    }


def _route(case: dict[str, Any], operations: list[str]) -> list[str]:
    c = _compare()
    report = c.Report()
    c._route(report, case, 'python-core', _observation(operations))
    return [text for _, text in report.failures]


def test_a_database_line_is_held_to_the_calls_up_to_its_stop() -> None:
    c = _compare()
    case = {
        **json.loads(CASE.read_text(encoding='utf-8')),
        'id': 'decimal.sum-10-2@sqlite',
    }
    reference = {
        'outcome': {
            'result': 'refusal',
            'stage': 'materialize',
            'code': 'INVALID_REQUEST',
            'error': 'StructuredDataError',
        }
    }
    stop = c.line_stop(None, reference)
    assert stop == 'materialize'
    # Refused at materialize by the client's query check, as the reference
    # is, python-core ran no core materialize: the line is held to the
    # calls before its stop, which it ran.
    ran = ['describe_source']
    held = c.core_held(case, _observation(ran), stop)
    assert sorted(held['core']) == ['describe_source']
    assert _route(held, ran) == []
    # A call before the stop that did not run still fails.
    assert _route(c.core_held(case, _observation([]), stop), [])
    # A call at or after the stop the runner ran is still held.
    both = ['describe_source', 'materialize']
    held = c.core_held(case, _observation(both), stop)
    assert sorted(held['core']) == ['describe_source', 'materialize']
    # An answering line, or an approved move to an answer, is held to the
    # whole block, as every DuckDB line is.
    assert c.line_stop({'result': 'success'}, reference) is None
    assert c.core_held(case, _observation(ran), None) is case
    assert _route(case, ran)


def test_rust_host_s_refused_edit_is_one_error_by_its_operations() -> None:
    r"""rust-host records no edits: a line held to its from_source call,
    whose last operation is a refused edit, has a description and the
    edit's error, as python-core's recorded edits say for it."""
    c = _compare()
    case = {
        'id': 'decimal.target-regression-collide@sqlite',
        'core': {'describe_source': {}},
    }
    observation = {
        'identity': {'operations': ['describe_source', 'edit_schema']},
        'core': {
            'described': {'tables': []},
            'error': {'kind': 'invalid_request', 'code': 'INVALID_REQUEST'},
            'fetches': [{}],
        },
        'outcome': {'result': 'refusal', 'code': 'INVALID_REQUEST'},
    }
    report = c.Report()
    c._core_consistent(report, case, 'rust-host', observation, 'sqlite')
    assert report.failures == []
    observation['identity']['operations'] = ['describe_source']
    report = c.Report()
    c._core_consistent(report, case, 'rust-host', observation, 'sqlite')
    assert report.failures
