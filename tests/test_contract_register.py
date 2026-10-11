# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The register, ``sdm-register/2``, and ``register.py validate`` (T1.14;
design, "Register v2"): the committed register passes, and each rule fails
a seeded fault, missing, empty and retyped evidence included."""

from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/rust_core/register.py'
spec = importlib.util.spec_from_file_location('sdm_register', SCRIPT)
assert spec is not None and spec.loader is not None
register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(register)

COMMITTED = yaml.safe_load(
    (ROOT / 'docs/design/rust-core/register.yaml').read_text(encoding='utf-8')
)


def test_the_committed_register_validates() -> None:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), 'validate'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout
    assert done.stdout.strip() == 'register: ok'


def _problems(change: Callable[[dict[str, Any]], None]) -> list[str]:
    document = copy.deepcopy(COMMITTED)
    change(document)
    checkpoint = copy.deepcopy(COMMITTED)
    found: list[str] = register.problems(document, ROOT, checkpoint)
    return found


def _entry(document: dict[str, Any], entry_id: str) -> dict[str, Any]:
    found: dict[str, Any] = next(
        e for e in document['entries'] if e['id'] == entry_id
    )
    return found


def _outcomes(
    old: str, new: str, approved_by: str | None = 'reviewer'
) -> Callable[[dict[str, Any]], None]:
    def change(document: dict[str, Any]) -> None:
        entry = _entry(document, 'PQL-2')
        entry['status'] = 'approved' if approved_by else 'proposed'
        entry['approved_by'] = approved_by
        entry['case_outcomes'] = [
            {
                'case': 's0.accept.sum-window',
                'cell': 'ref-pandas3',
                'runner': 'python-core',
                'old': old,
                'new': new,
                'summary': 'seeded',
            }
        ]

    return change


FAULTS: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    'format 1': (
        lambda d: d.update(format='sdm-register/1'),
        "format fails 'const'",
    ),
    'no entries': (lambda d: d.update(entries=[]), "entries fails 'minItems'"),
    'entries retyped': (lambda d: d.update(entries={}), "entries fails 'type'"),
    'an ID repeated': (
        lambda d: d['entries'].append(copy.deepcopy(d['entries'][0])),
        'PK-1: the ID repeats',
    ),
    'an ID removed': (lambda d: d['entries'].pop(), 'PQL-2: removed'),
    'a case that resolves to nothing': (
        lambda d: _entry(d, 'PQL-2')['cases'].append('no.such-case'),
        'case no.such-case: not a corpus case',
    ),
    'no cases and no refs': (
        lambda d: _entry(d, 'PK-2').pop('refs'),
        "fails 'anyOf'",
    ),
    'a decision row that does not exist': (
        lambda d: _entry(d, 'PK-1')['decision_ref'].update(row=999),
        'decision_ref: 999 is not a row',
    ),
    'a decisions file that does not exist': (
        lambda d: _entry(d, 'PK-1')['decision_ref'].update(
            file='docs/design/rust-core/phase-9/decisions.md'
        ),
        'decision_ref: no file',
    ),
    'decision_ref still a number': (
        lambda d: _entry(d, 'PK-1').update(decision_ref=29),
        "decision_ref fails 'oneOf'",
    ),
    'approved by nobody': (
        lambda d: _entry(d, 'PQL-2').update(status='approved'),
        "fails 'enum'",
    ),
    'proposed and approved by someone': (
        lambda d: _entry(d, 'PQL-2').update(approved_by='reviewer'),
        "fails 'const'",
    ),
    'an outcome in no form ratchet.py writes': (
        _outcomes('fine', 'success'),
        "old fails 'pattern'",
    ),
    'a case outcome of a case that does not exist': (
        lambda d: (
            _outcomes('success', 'success at predict')(d),
            _entry(d, 'PQL-2')['case_outcomes'][0].update(case='no.such-case'),
        ),
        'no.such-case on python-core: not a corpus case',
    ),
    'a new refusal approved by the reviewer': (
        _outcomes(
            'success at materialize', 'refusal INVALID_REQUEST at materialize'
        ),
        'only the owner approves it',
    ),
    'a refusal of an input the reference accepts': (
        _outcomes('none', 'refusal INVALID_REQUEST at materialize'),
        'only the owner approves it',
    ),
    'a panic accepted': (
        _outcomes('success', 'refusal INTERNAL_ERROR', 'owner'),
        'no entry accepts INTERNAL_ERROR',
    ),
    'a case outcome with no summary': (
        lambda d: (
            _outcomes('success', 'success at predict')(d),
            _entry(d, 'PQL-2')['case_outcomes'][0].pop('summary'),
        ),
        "fails 'required'",
    ),
}


@pytest.mark.parametrize('fault', sorted(FAULTS))
def test_a_seeded_fault_fails(fault: str) -> None:
    change, wanted = FAULTS[fault]
    found = _problems(change)
    assert any(wanted in problem for problem in found), found


def test_an_owner_approved_refusal_and_a_proposed_one_pass() -> None:
    new = 'refusal INVALID_REQUEST at materialize'
    assert _problems(_outcomes('success at materialize', new, 'owner')) == []
    assert _problems(_outcomes('success at materialize', new, None)) == []


@pytest.mark.parametrize(
    ('text', 'wanted'),
    [
        ('[' * 100_000 + ']' * 100_000, 'nested past what the parser reads'),
        ('entries: [\n', 'not YAML'),
        (
            '{' + ', '.join(f'k{n}: {n}' for n in range(100_000)) + '}',
            "fails 'additionalProperties'",
        ),
    ],
    ids=['deep', 'not-yaml', 'wide'],
)
def test_a_register_it_cannot_read_is_refused_by_name(
    tmp_path: Path, text: str, wanted: str
) -> None:
    path = tmp_path / 'register.yaml'
    path.write_text(text)
    try:
        found = register.problems(register.load(path), ROOT)
    except register.Unreadable as error:
        found = [str(error)]
    assert any(wanted in problem for problem in found), found
