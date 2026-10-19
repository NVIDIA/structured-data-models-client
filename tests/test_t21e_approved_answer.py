# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: a call an approved move turns from the reference's refusal into
an answer, by the entry's opt-in.

Where an owner-approved entry moves a runner from the reference's refusal
at a call to a success and its ``case_outcomes`` row names that call under
``statements_differ`` (the owner's KO2-61 on
time.feature-time-of-day@sqlite), the reference's statements up to its
refusal are no expectation of that call's: its ``sql.count`` and
``sql.result`` differences against the reference are reported as
diagnostic. Without the opt-in, every call gates as before (Phase 1's
``test_t132g_later_stage.py`` guard); register.py takes the opt-in only on
the call the reference refused, on a move to a success.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KEY = 'time.feature-time-of-day@sqlite'


def _compare() -> ModuleType:
    name = 't21e_compare'
    if name not in sys.modules:
        path = ROOT / 'scripts/rust_core/compare.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _reference() -> dict[str, Any]:
    [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    for text in path.read_text(encoding='utf-8').splitlines():
        line = json.loads(text)
        if line['case'] == KEY:
            return line
    raise AssertionError(f'{KEY}: no reference line')


def _failures(answered: bool) -> tuple[list[str], list[str]]:
    r"""The sql rules' failures and notes for a runner whose last
    materialize statement and first from_source statement read other rows
    than the reference's, judged past the reference's materialize refusal."""
    c = _compare()
    reference = _reference()
    observation = copy.deepcopy(reference)
    statements = observation['driver']['capture']['statements']
    for statement in (statements[0], statements[-1]):
        values = statement['results'][0]['values']
        values['unordered'] = values['ordered'] = '0' * 64
    report = c.Report()
    c._sql(
        report,
        {'id': KEY},
        'python-core',
        observation,
        reference,
        {},
        stop='materialize',
        database='sqlite',
        answered=answered,
    )
    return [text for _, text in report.failures], report.notes


def test_an_approved_answer_reports_its_call_s_statements_and_no_other() -> (
    None
):
    failures, notes = _failures(answered=True)
    assert failures
    assert all(': from_source statement' in text for text in failures)
    assert any(
        ': materialize statement' in text and 'sql.result' in text
        for text in notes
    )
    held, _ = _failures(answered=False)
    assert any(': materialize statement' in text for text in held)
    assert any(': from_source statement' in text for text in held)


def _register(tmp_path: Path, approved_by: str, calls: str) -> Path:
    path = tmp_path / 'register.yaml'
    path.write_text(
        textwrap.dedent(
            f"""\
            entries:
            - id: X-1
              status: approved
              approved_by: {approved_by}
              decision_ref: {{file: docs/design/rust-core/phase-2/decisions.md, row: KO2-61}}
              case_outcomes:
              - case: {KEY}
                cell: ref-pandas3
                runner: python-core
                old: 'refusal INVALID_REQUEST at materialize (StructuredDataError)'
                new: 'success at predict'
                summary: '-'
            {calls}
            """
        ),
        encoding='utf-8',
    )
    return path


def test_the_opt_in_is_the_owner_approved_row_s_and_named_calls_only(
    tmp_path: Path,
) -> None:
    c = _compare()
    opted = '    statements_differ: [materialize]'
    assert c.statements_differ(_register(tmp_path, 'owner', opted)) == {
        (KEY, 'python-core'): ('materialize',)
    }
    # The default: no opt-in, or one not the owner's, opts in nothing.
    assert c.statements_differ(_register(tmp_path, 'owner', '')) == {}
    assert c.statements_differ(_register(tmp_path, 'reviewer', opted)) == {}


def test_register_takes_the_opt_in_only_on_the_refused_call_moved_to_an_answer() -> (
    None
):
    r = importlib.util.spec_from_file_location(
        't21e_register', ROOT / 'scripts/rust_core/register.py'
    )
    assert r is not None and r.loader is not None
    register = importlib.util.module_from_spec(r)
    r.loader.exec_module(register)
    entry = {'id': 'X-1', 'status': 'approved', 'approved_by': 'owner'}
    row = {
        'case': KEY,
        'runner': 'python-core',
        'old': 'refusal INVALID_REQUEST at materialize (StructuredDataError)',
        'new': 'success at predict',
    }
    corpus = {'time.feature-time-of-day': {}}
    ok = register._outcome_problems(
        entry, {**row, 'statements_differ': ['materialize']}, corpus, {}
    )
    assert ok == []
    for wrong in (
        {**row, 'statements_differ': ['from_source']},
        {
            **row,
            'statements_differ': ['materialize'],
            'new': 'refusal X at predict',
        },
    ):
        assert register._outcome_problems(entry, wrong, corpus, {})


def test_a_move_to_a_later_refusal_never_opts_in() -> None:
    r"""Only a move to a success opts a call in: a move to a later refusal,
    an unmoved line and a row naming another call gate as before."""
    c = _compare()
    report = c.Report()
    report.statements_differ = {(KEY, 'python-core'): ('materialize',)}
    success = {'result': 'success', 'stage': 'predict'}
    later = {'result': 'refusal', 'stage': 'predict', 'code': 'X'}
    opted = c.statements_opted_in
    assert opted(report, KEY, 'python-core', 'materialize', success)
    assert not opted(report, KEY, 'python-core', 'materialize', later)
    assert not opted(report, KEY, 'python-core', None, success)
    assert not opted(report, KEY, 'python-core', 'from_source', success)
    assert not opted(report, KEY, 'rust-host', 'materialize', success)
