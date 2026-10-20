# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1s (KO2-83): fewer repeats of a DeprecationWarning, for a named line.

On SQLite and Postgres the reference records five DeprecationWarnings for
s0.boundary.escapes (``invalid escape sequence '\/'``, from
predictive_query's constant.py:134, each the same: checked by hand, the
evidence in the register entry's text), where python-core records two.
An owner-approved register entry naming the line (``<case>@<database>``)
and the runner under ``fewer_deprecations`` lets the ``warnings`` rule
take fewer DeprecationWarnings than the reference's there, as HOST-5
narrows the rule: never more, every other class the same in count and in
relative order, and nothing for a line, runner or database the entry does
not name, or an entry the owner did not approve.

The record holds classes only (no format change, no reference recorded
again), so the rule counts by class.
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

ROOT = Path(__file__).resolve().parents[2]
CASE = 's0.boundary.escapes'
REFERENCE = (
    ROOT / 'contract/reference/6298913c3b66da9459bb605d0127f8493adaa1fc.jsonl'
)
DEPRECATION = 'DeprecationWarning'


def _module(path: Path, name: str) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


COMPARE = _module(ROOT / 'scripts/rust_core/compare.py', 't21s_compare')


def _reference(key: str) -> dict[str, Any]:
    for line in REFERENCE.read_text(encoding='utf-8').splitlines():
        observation: dict[str, Any] = json.loads(line)
        if observation['case'] == key:
            return observation
    raise AssertionError(key)


def _fewer(warnings: list[str], keep: int) -> list[str]:
    r"""``warnings`` with only the first ``keep`` DeprecationWarnings, the
    rest in their order."""
    out, seen = [], 0
    for name in warnings:
        if name == DEPRECATION:
            seen += 1
            if seen > keep:
                continue
        out.append(name)
    return out


def _entry(
    lines: list[str],
    approved_by: str = 'owner',
    runners: list[str] | None = None,
) -> dict[str, Any]:
    return {
        'id': 'X-1',
        'status': 'approved',
        'approved_by': approved_by,
        # A row of this harness's decisions the owner approved.
        'decision_ref': {
            'file': 'docs/design/rust-core/phase-2/decisions.md',
            'row': 'KO2-35',
        },
        'fewer_deprecations': [
            {'case': line, 'runners': runners or ['python-core']}
            for line in lines
        ],
    }


def _warnings_failures(
    tmp: Path,
    database: str,
    shown: list[str] | None,
    entries: list[dict[str, Any]],
) -> list[str]:
    r"""``compare()``'s ``warnings`` failures on python-core's line of
    ``CASE`` on ``database``, the reference's own line but for its
    warnings, ``shown`` (two DeprecationWarnings of the reference's five
    when None), under a register of ``entries``."""
    key = CASE if database == 'duckdb' else f'{CASE}@{database}'
    reference = _reference(key)
    mine = copy.deepcopy(reference)
    mine['identity']['runner'] = 'python-core'
    mine['identity']['reference'] = None
    if shown is None:
        shown = _fewer(reference['host']['warnings'], 2)
    mine['host'] = {'warnings': shown}
    (tmp / 'ref.jsonl').write_text(json.dumps(reference) + '\n')
    (tmp / 'core.jsonl').write_text(json.dumps(mine) + '\n')
    (tmp / 'register.yaml').write_text(
        json.dumps({'format': 'sdm-register/2', 'entries': entries})
    )
    report = COMPARE.compare(
        [ROOT / 'contract/cases'],
        tmp / 'ref.jsonl',
        [tmp / 'core.jsonl'],
        register_file=tmp / 'register.yaml',
        database=database,
    )
    return [
        text
        for rule, text in report.failures
        if rule == 'warnings' and text.startswith(f'{key}: ')
    ]


BOTH = [f'{CASE}@sqlite', f'{CASE}@postgres']


def test_the_reference_shows_five_deprecations_there() -> None:
    r"""Not vacuous: the reference's lines hold the five repeats."""
    for key in BOTH:
        assert _reference(key)['host']['warnings'].count(DEPRECATION) == 5


@pytest.mark.parametrize('database', ['sqlite', 'postgres'])
def test_an_owner_entry_takes_fewer_deprecations(
    tmp_path: Path, database: str
) -> None:
    r"""Exit: with the owner's entry naming the line, two pass."""
    assert _warnings_failures(tmp_path, database, None, [_entry(BOTH)]) == []


@pytest.mark.parametrize('database', ['sqlite', 'postgres'])
def test_fewer_deprecations_fail_without_an_owner_entry(
    tmp_path: Path, database: str
) -> None:
    r"""Exit: no entry, or a reviewer's, leaves the rule as it was."""
    assert _warnings_failures(tmp_path, database, None, [])
    reviewer = _entry(BOTH, approved_by='reviewer')
    assert _warnings_failures(tmp_path, database, None, [reviewer])


def test_more_deprecations_than_the_reference_fail(tmp_path: Path) -> None:
    r"""Exit: never more than the reference's."""
    warnings = _reference(f'{CASE}@sqlite')['host']['warnings']
    more = [DEPRECATION, *warnings]
    assert _warnings_failures(tmp_path, 'sqlite', more, [_entry(BOTH)])


def test_another_class_is_held_exactly(tmp_path: Path) -> None:
    r"""Exit: every other class the same in count and relative order."""
    fewer = _fewer(_reference(f'{CASE}@sqlite')['host']['warnings'], 2)
    assert 'UserWarning' in fewer and 'SDMWarning' in fewer
    owner = [_entry(BOTH)]
    # A UserWarning too many, or one missing.
    extra = [*fewer[:-1], 'UserWarning', fewer[-1]]
    assert _warnings_failures(tmp_path, 'sqlite', extra, owner)
    missing = [name for name in fewer if name != 'UserWarning']
    assert _warnings_failures(tmp_path, 'sqlite', missing, owner)
    # The same classes, the UserWarning and SDMWarning swapped.
    swapped = [n for n in fewer if n not in ('UserWarning', 'SDMWarning')]
    swapped += ['SDMWarning', 'UserWarning']
    assert _warnings_failures(tmp_path, 'sqlite', swapped, owner)


def test_an_unnamed_line_or_runner_is_held_as_before(tmp_path: Path) -> None:
    r"""Exit: nothing changes for a case, database or runner the entry
    does not name."""
    # The SQLite line alone named: Postgres's is held as before.
    assert _warnings_failures(
        tmp_path, 'postgres', None, [_entry([f'{CASE}@sqlite'])]
    )
    # The case's DuckDB key names that line, not SQLite's.
    assert _warnings_failures(tmp_path, 'sqlite', None, [_entry([CASE])])
    # Another case named.
    other = [_entry(['s0.boundary.date-packed@sqlite'])]
    assert _warnings_failures(tmp_path, 'sqlite', None, other)
    # Another runner named.
    rust = [_entry(BOTH, runners=['rust-host'])]
    assert _warnings_failures(tmp_path, 'sqlite', None, rust)
