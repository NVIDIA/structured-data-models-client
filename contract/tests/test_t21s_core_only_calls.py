# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1s (added scope, 2026-10-20 overnight, part a): a core-only column
in one of several calls' bodies.

T2.10's rank.odd-column (register P2E-1, KO2-36) keeps ``items.i.d`` and
``buys.i.d``, which the reference leaves out. Where predict makes several
calls, rust-host's bodies are under ``core.predicted.calls/<n>/`` (T2.1g),
and the core_only_columns rule must take the named columns out of each
of them as it does out of a single call's ``core.predicted.body``; a
column the entry does not name still fails, in any call.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASE = {'id': 'rank.odd-column', 'until': 'predict'}


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't21s_core_only')


def _call(n: int) -> dict[str, Any]:
    r"""A call's body as the reference sends it: ``items`` and ``buys``
    without their ``i.d`` key, which the reference leaves out."""
    return {
        'model': 'kumo-relational',
        'context': {
            'related_tables': {
                'items': {'columns': ['kind'], 'rows': [['a'], ['b']]},
                'buys': {
                    'columns': ['buy_id', 'user_id', 'ts'],
                    'rows': [[{'number': str(n)}, {'number': '1'}, 't0']],
                },
            }
        },
        'metadata': {'request_id': f'sdm-{n}'},
    }


def _kept(body: dict[str, Any], table: str, column: str) -> None:
    r"""``body`` with ``table``'s column ``column`` kept, as the core does,
    first among its columns."""
    section = body['context']['related_tables'][table]
    section['columns'].insert(0, column)
    for at, row in enumerate(section['rows']):
        row.insert(0, {'number': str(at)})


def _failures(
    tmp: Path, bodies: list[dict[str, Any]], named: bool
) -> list[str]:
    r"""The payload rule's failures for rust-host's ``bodies`` (one call,
    or several) against the reference's calls of :func:`_call`, the
    register's owner entry naming ``items.i.d`` and ``buys.i.d`` under
    ``core_only_columns`` (``named``) or none."""
    entries = []
    if named:
        entries.append(
            {
                'id': 'P2E-1',
                'status': 'approved',
                'approved_by': 'owner',
                'cases': [CASE['id']],
                # A row of this harness's decisions the owner approved.
                'decision_ref': {
                    'file': 'docs/design/rust-core/phase-2/decisions.md',
                    'row': 'KO2-36',
                },
                'core_only_columns': ['items.i.d', 'buys.i.d'],
            }
        )
    register = tmp / 'register.yaml'
    register.write_text(json.dumps({'entries': entries}), encoding='utf-8')
    report = COMPARE.Report({'payload': 'gate'})
    report.core_only = COMPARE.core_only_columns(register)
    predicted: dict[str, Any] = {'body': bodies[-1]}
    if len(bodies) > 1:
        predicted['calls'] = bodies
    seen = COMPARE.without_core_only(
        {'core': {'predicted': predicted}},
        report.core_only.get(CASE['id'], {}),
    )
    sent = [{'parsed': _call(n)} for n in range(len(bodies))]
    reference = {'public': {'payloads': sent}}
    COMPARE._core_payload(report, CASE, 'rust-host', seen, reference)
    return [text for _, text in report.failures]


def _core_bodies() -> list[dict[str, Any]]:
    r"""rust-host's two calls, each keeping both named columns."""
    bodies = [_call(0), _call(1)]
    for body in bodies:
        _kept(body, 'items', 'i.d')
        _kept(body, 'buys', 'i.d')
    return bodies


def test_a_named_core_only_column_is_left_out_of_every_call(
    tmp_path: Path,
) -> None:
    r"""Exit (a): the rule applies to every per-call body as to the single
    call."""
    bodies = _core_bodies()
    # Not vacuous: with no entry the kept columns fail, by the call's path.
    found = _failures(tmp_path, bodies, named=False)
    assert found and 'core.predicted.calls/0/' in found[0], found
    assert _failures(tmp_path, bodies, named=True) == []
    # As in a single call's body, which the rule already reads.
    assert _failures(tmp_path, bodies[:1], named=True) == []
    # Beside them, a column the entry does not name is the one reported.
    _kept(bodies[1], 'items', 'other')
    [found] = _failures(tmp_path, bodies, named=True)
    assert 'core.predicted.calls/1/' in found, found


def test_an_unnamed_column_in_a_call_still_fails(tmp_path: Path) -> None:
    r"""Exit (a): nothing else loosens: a column the entry does not name,
    kept in the second call alone, still fails by that call's path."""
    bodies = [_call(0), _call(1)]
    _kept(bodies[1], 'items', 'other')
    [found] = _failures(tmp_path, bodies, named=True)
    assert 'core.predicted.calls/1/' in found, found
