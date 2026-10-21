# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1r (added scope 2): ``more_exact`` names a path in one of several
calls.

Where predict makes several calls, rust-host's bodies are under
``core.predicted.calls/<n>/`` (T2.1g); T2.9's lag.condition-per-call
(register P2F-1) asks one entity a call, and the core keeps each lag value
on its own row where the reference moves them up, so its more-exact cells
are at ``core.predicted.calls/0/context/instance_table/rows/0/4``. An
owner-approved ``more_exact`` entry names such a path as it names
``core.predicted.body/...``, and the payload rule leaves out only that
path, for that case and runner.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't21r_more_calls')
CASE = {'id': 'lag.condition-per-call', 'until': 'predict'}
#: The lag cells of every call's context rows, as P2F-1 would name them.
NAMED = 'core.predicted.calls/*/context/instance_table/rows/*/4'


def _call(n: int) -> dict[str, Any]:
    r"""A call's body asking one entity: its context rows, the fifth
    column a lag value."""
    return {
        'model': 'kumo-relational',
        'context': {
            'instance_table': {
                'rows': [
                    [
                        {'number': str(n)},
                        'a',
                        'b',
                        {'number': '0.5'},
                        {'number': '1.25'},
                    ]
                ]
            }
        },
        'metadata': {'request_id': f'sdm-{n}'},
    }


def _failures(tmp: Path, mine: list[dict[str, Any]], named: bool) -> list[str]:
    r"""The payload rule's failures for rust-host's calls ``mine`` against
    the reference's (two calls of :func:`_call`), under an owner-approved
    entry naming :data:`NAMED` (``named``) or no entry."""
    entries = []
    if named:
        entries.append(
            {
                'id': 'P2F-1',
                'status': 'approved',
                'approved_by': 'owner',
                # A row of this harness's decisions the owner approved.
                'decision_ref': {
                    'file': 'docs/design/rust-core/phase-2/decisions.md',
                    'row': 'KO2-35',
                },
                'more_exact': [
                    {
                        'case': CASE['id'],
                        'runners': ['rust-host'],
                        'fields': [NAMED],
                    }
                ],
            }
        )
    register = tmp / 'register.yaml'
    register.write_text(yaml.safe_dump({'entries': entries}), encoding='utf-8')
    report = COMPARE.Report({'payload': 'gate'})
    report.more_exact = COMPARE.more_exact(register)
    sent = [_call(0), _call(1)]
    observation = {'core': {'predicted': {'body': mine[-1], 'calls': mine}}}
    reference = {'public': {'payloads': [{'parsed': b} for b in sent]}}
    COMPARE._core_payload(report, CASE, 'rust-host', observation, reference)
    return [text for _, text in report.failures]


def _lag(calls: list[dict[str, Any]], n: int, column: int) -> None:
    row = calls[n]['context']['instance_table']['rows'][0]
    row[column] = {'number': '9.75'}


def test_a_named_per_call_path_is_left_out(tmp_path: Path) -> None:
    r"""Exit: the entry's path under core.predicted.calls/<n>/ passes."""
    mine = [_call(0), _call(1)]
    _lag(mine, 0, 4)
    _lag(mine, 1, 4)
    # Not vacuous: with no entry the lag cell fails, by its call's path.
    [found] = _failures(tmp_path, mine, named=False)
    assert 'core.predicted.calls/0/context/instance_table/rows/0/4' in found
    assert _failures(tmp_path, mine, named=True) == []
    # Beside it, a difference outside the named paths is still the one
    # reported.
    _lag(mine, 1, 3)
    [found] = _failures(tmp_path, mine, named=True)
    assert 'core.predicted.calls/1/context/instance_table/rows/0/3' in found


def test_an_unnamed_per_call_path_still_fails(tmp_path: Path) -> None:
    r"""Exit: nothing outside the named paths loosens."""
    mine = [_call(0), _call(1)]
    _lag(mine, 1, 3)
    [found] = _failures(tmp_path, mine, named=True)
    assert 'core.predicted.calls/1/context/instance_table/rows/0/3' in found
