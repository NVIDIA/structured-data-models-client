# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The schema's edits on the core, held across the two core hosts (T1.24b).

Run where the contract tests run:

    python -m pytest contract/tests -q

The ``edit.*`` cases (``contract/cases/s1``) make edits to the schema
``Schema.from_source`` returned, each through the case format's edits:
seven the reference refuses at the assignment (a key on a column the table
lacks or on its time column, a time column that cannot hold a time, a
meaning that is no semantic type or one the values cannot carry, a link to
a column that is not the key, the key left out) and one whose four edits it
takes, then materializes. Each edit is the core's ``edit_schema`` on both
core hosts (decision E67). What holds them:

- the reference's outcome: python-core's outcome is each case's
  ``expect``, which the committed reference observation records;
- the two core hosts agree: what python-core's host started the core with
  (``core.inputs``: the description, the schema held and the edit of each
  ``edit_schema``, and ``materialize``'s inputs), the core's error code and
  the request state are the ones ``contract/pinned/edits.core.json`` pins
  as digests, which ``sdm-testkit``'s ``tests/edit_cases.rs`` holds
  rust-host to, so CI gates the two hosts' equality where neither job runs
  both.

``SDM_EDITS_PIN_WRITE=1`` rewrites the pin from python-core's run.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = sorted((ROOT / 'contract/cases/s1').glob('edit.*.json'))
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))
PINNED = ROOT / 'contract/pinned/edits.core.json'
FORMAT = 'sdm-edits-pin/1'
FIELDS = ('inputs', 'error', 'materialized')


def digest(value: Any) -> str:
    r"""The SHA-256 of ``value`` as canonical JSON (keys sorted, no
    whitespace, UTF-8), as ``refusal_map.digest`` and ``sdm-testkit``
    write it."""
    text = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def agreed(observation: dict[str, Any]) -> dict[str, Any]:
    r"""What the pin holds of one python-core observation."""
    core = observation['core']
    error = core['error']
    return {
        'inputs': digest(core['inputs']),
        'error': None if error is None else error['code'],
        'materialized': (
            None
            if core['materialized'] is None
            else digest(core['materialized'])
        ),
    }


def problems(pin: Any, seen: dict[str, dict[str, Any]]) -> list[str]:
    r"""Each way ``seen`` departs from ``pin``; missing, empty or retyped
    evidence is a problem, not a pass."""
    if not isinstance(pin, dict) or pin.get('format') != FORMAT:
        return ['the pin is not an sdm-edits-pin/1 document']
    cases = pin.get('cases')
    if not isinstance(cases, dict) or not cases:
        return ['the pin holds no case']
    found = []
    for name in sorted(set(cases) | set(seen)):
        wanted = cases.get(name)
        if not isinstance(wanted, dict) or set(wanted) != set(FIELDS):
            found.append(f'{name}: not pinned as {", ".join(FIELDS)}')
            continue
        if not isinstance(wanted['inputs'], str) or len(wanted['inputs']) != 64:
            found.append(f'{name}: inputs is not a digest')
            continue
        if name not in seen:
            found.append(f'{name}: pinned, and not run')
            continue
        for field in FIELDS:
            if seen[name][field] != wanted[field]:
                found.append(f"{name}: {field} is not the pin's")
    return found


@pytest.fixture(scope='module')
def observed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp('edits') / 'python-core.jsonl'
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNER),
            '--mode',
            'python-core',
            '--out',
            str(out),
            *map(str, CASES),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    observations = [json.loads(line) for line in out.read_text().splitlines()]
    return {each['case']: each for each in observations}


def test_there_are_refused_and_accepted_edit_cases() -> None:
    expects = [json.loads(path.read_text())['expect'] for path in CASES]
    assert sum(e['outcome'] == 'refusal' for e in expects) == 7
    assert sum(e['outcome'] == 'success' for e in expects) == 1


def test_each_edit_case_ends_as_the_reference_ends_it(
    observed: dict[str, Any],
) -> None:
    reference = {
        line['case']: line
        for line in map(json.loads, REFERENCE.read_text().splitlines())
    }
    for path in CASES:
        case = json.loads(path.read_text())
        mine = observed[case['id']]
        theirs = reference[case['id']]
        assert mine['outcome'] == theirs['outcome'], case['id']
        expect = case['expect']
        assert mine['outcome']['result'] == expect['outcome'], case['id']
        assert mine['outcome']['stage'] == expect['stage'], case['id']
        assert mine['outcome']['code'] == expect.get('code'), case['id']
        # Each edit the case makes is the core's, up to a refused one.
        ops = [each['op'] for each in mine['core']['inputs']]
        made = ops.count('edit_schema')
        if expect['outcome'] == 'refusal':
            assert made == len(case['input']['edits']), case['id']
            assert ops[-1] == 'edit_schema', case['id']
            assert mine['core']['error']['code'] == expect['code']
        else:
            assert made == len(case['input']['edits']), case['id']
            assert ops[-1] == 'materialize', case['id']


def test_the_core_hosts_agree_as_the_pin_holds_them(
    observed: dict[str, Any],
) -> None:
    seen = {name: agreed(each) for name, each in observed.items()}
    if os.environ.get('SDM_EDITS_PIN_WRITE') == '1':
        PINNED.write_text(
            json.dumps(
                {
                    'format': FORMAT,
                    'about': (
                        "python-core's run of the edit.* cases, as "
                        'contract/tests/test_edit_cases.py writes it: per '
                        "case, the digest of the core's inputs, the core's "
                        'error code, and the digest of the request state; '
                        "sdm-testkit's tests/edit_cases.rs holds rust-host "
                        'to it (T1.24b).'
                    ),
                    'cases': seen,
                },
                indent=2,
                sort_keys=True,
            )
            + '\n'
        )
    assert problems(json.loads(PINNED.read_text()), seen) == []


def test_the_pin_check_fails_closed() -> None:
    pin = json.loads(PINNED.read_text())
    seen = {name: dict(each) for name, each in pin['cases'].items()}
    assert problems(pin, seen) == []
    name = sorted(seen)[0]
    differing = {**seen, name: {**seen[name], 'error': 'INTERNAL_ERROR'}}
    assert problems(pin, differing) == [f"{name}: error is not the pin's"]
    missing = {k: v for k, v in seen.items() if k != name}
    assert problems(pin, missing) == [f'{name}: pinned, and not run']
    assert problems({**pin, 'cases': {}}, seen) == ['the pin holds no case']
    assert problems({**pin, 'format': 'other'}, seen) != []
    retyped = {**pin, 'cases': {**pin['cases'], name: ['inputs']}}
    assert problems(retyped, seen)[0].startswith(f'{name}: not pinned')
    short = {
        **pin,
        'cases': {**pin['cases'], name: {**pin['cases'][name], 'inputs': 'x'}},
    }
    assert problems(short, seen) == [f'{name}: inputs is not a digest']
