# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: on Postgres the reference's read whose result it never uses, each
table's ``SELECT COUNT(*) FROM "schema"."table"``, is not expected of a
core runner (KO2-58); every other statement is, count by count and
statement by statement, ``SELECT current_schema()`` included, which the
core makes too since T3.7, and DuckDB and SQLite lines are held to every
one.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / 'contract/reference/43e55f4f58924384c3cb52f67c8ed2e73ba45fa1.jsonl'
)
LINE = 's0.accept.sum-window@postgres'


def _compare() -> ModuleType:
    name = 't21e_compare_unused'
    if name not in sys.modules:
        path = ROOT / 'scripts/rust_core/compare.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _reference() -> dict[str, Any]:
    for line in REFERENCE.read_text(encoding='utf-8').splitlines():
        observation = json.loads(line)
        if observation['case'] == LINE:
            return dict(observation)
    raise AssertionError(f'no reference line {LINE}')


def _failures(
    ours: dict[str, Any], theirs: dict[str, Any], database: str
) -> list[tuple[str, str]]:
    c = _compare()
    report = c.Report()
    c._sql(
        report, {'id': LINE}, 'python-core', ours, theirs, {}, database=database
    )
    return [(rule, text) for rule, text in report.failures]


def _without(observation: dict[str, Any], drop: Any) -> dict[str, Any]:
    changed = copy.deepcopy(observation)
    capture = changed['driver']['capture']
    capture['statements'] = [s for s in capture['statements'] if not drop(s)]
    return changed


def test_only_the_unused_row_counts_are_not_expected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = _compare()
    # The record's completeness is its own rule; here only what is expected.
    monkeypatch.setattr(c, '_complete', lambda *_: True)
    theirs = _reference()
    unused = [
        s['sql']
        for s in theirs['driver']['capture']['statements']
        if s['sql'].startswith('SELECT COUNT(*) FROM "public".')
    ]
    assert len(unused) == 2, unused
    ours = _without(theirs, lambda s: s['sql'] in unused)
    # The core's run without them is every other statement, as expected.
    assert _failures(ours, theirs, 'postgres') == []
    # On another database every statement is expected.
    assert any(
        rule == 'sql.count' for rule, _ in _failures(ours, theirs, 'sqlite')
    )
    # Any other statement missing still fails: the session's schema, which
    # the core reads with the reference's statement (T3.7), and the seed.
    for missing in ('SELECT current_schema()', 'SELECT setseed'):
        without = _without(ours, lambda s, m=missing: s['sql'].startswith(m))
        assert len(without['driver']['capture']['statements']) < len(
            ours['driver']['capture']['statements']
        ), missing
        assert any(
            rule == 'sql.count'
            for rule, _ in _failures(without, theirs, 'postgres')
        ), missing
    # And so does one more: an unused read run where none is expected.
    extra = copy.deepcopy(ours)
    statements = extra['driver']['capture']['statements']
    statements.append({**statements[-1], 'sql': 'SELECT 1'})
    assert any(
        rule == 'sql.count' for rule, _ in _failures(extra, theirs, 'postgres')
    )
    # A count with a condition is no unused read.
    counted = copy.deepcopy(theirs)
    for each in counted['driver']['capture']['statements']:
        if each['sql'].startswith('SELECT COUNT(*) FROM "public".'):
            each['sql'] += ' WHERE true'
    assert any(
        rule == 'sql.count' for rule, _ in _failures(ours, counted, 'postgres')
    )
