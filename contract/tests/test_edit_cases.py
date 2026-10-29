# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The schema's edits on the core, held across the two core hosts (T1.24b).

Run where the contract tests run:

    python -m pytest contract/tests -q

The ``edit.*`` cases (``contract/cases/s1``, moved there at T1.27 from
T1.24b's fixtures, once the gate's checkpoint loads the format's new
edits; diagnostic, so the corpus reports them and this test gates them)
make edits to the schema
``Schema.from_source`` returned, each through the case format's edits:
seven the reference refuses at the assignment (a key on a column the table
lacks or on its time column, a time column that cannot hold a time, a
meaning that is no semantic type or one the values cannot carry, a link to
a column that is not the key, the key left out) and two whose edits it
takes, then materializes: four edits, and from T1.27 a time column set on
a VARCHAR column of dates, so the time column's rule is reached over text
on both core hosts. Each edit is the core's ``edit_schema`` on both
core hosts (decision E67). What holds them:

- the reference's outcome: python-core's outcome is each case's
  ``expect``, which each case's line of the committed reference file
  (``contract/reference``, recorded on python-ref under T1.13's rules)
  records;
- the two core hosts agree: what python-core's host started the core with
  (``core.inputs``, and ``core.edits``: the description, the schema held
  and the edit of each ``edit_schema``, and ``materialize``'s held schema),
  the core's error code and
  the request state are the ones ``contract/pinned/edits.core.json`` pins
  as digests, which ``sdm-testkit``'s ``tests/edit_cases.rs`` holds
  rust-host to, so CI gates the two hosts' equality where neither job runs
  both.

``SDM_EDITS_PIN_WRITE=1`` rewrites the pin from python-core's run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _actual_core import checked_operations, checked_view

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = sorted((ROOT / 'contract/cases/s1').glob('edit.*.json'))
REFERENCE = ROOT / 'contract/reference'
PINNED = ROOT / 'contract/pinned/edits.core.json'
FORMAT = 'sdm-edits-pin/1'
FIELDS = ('inputs', 'edits', 'error', 'materialized')


def digest(value: Any) -> str:
    r"""The SHA-256 of ``value`` as canonical JSON (keys sorted, no
    whitespace, UTF-8), as ``refusal_map.digest`` and ``sdm-testkit``
    write it."""
    text = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def agreed(observation: dict[str, Any]) -> dict[str, Any]:
    r"""Compare actual common materialization meaning to the historical pin.

    Complete current declaration flags are asserted separately before
    hashing the older fields, which the immutable pin actually observed.
    """
    core = observation['core']
    logical = checked_view(observation)
    name = observation['case']
    error = core['error']
    return {
        'inputs': digest(
            historical_starts(name, logical['inputs'], edits=False)
        ),
        'edits': digest(historical_starts(name, core['edits'], edits=True)),
        'error': None if error is None else error['code'],
        'materialized': (
            None
            if logical['materialized'] is None
            else digest(logical['materialized'])
        ),
    }


def historical_starts(
    name: str, starts: list[Any], *, edits: bool
) -> list[Any]:
    """Check complete current flags, then compare the older independent data.

    The frozen pin's original nine raw journeys omit these members.
    Current public journeys retain true for the source's time meaning and
    for age after its accepted meaning edit.
    Every other input, edit, order and value remains in the hashed data.
    """
    assert name in {path.stem for path in CASES}
    accepted = name in (
        'edit.accepted-meaning-key-links',
        'edit.time-column-text',
    )
    expected_count = (
        4 if edits and name == 'edit.accepted-meaning-key-links' else 1
    )
    if not edits:
        expected_count = 2 if accepted else 1
    assert len(starts) == expected_count
    projected = copy.deepcopy(starts)
    for at, start in enumerate(projected):
        if not edits and at == 0:
            assert start['op'] == 'describe_source' and 'schema' not in start
            continue
        assert start['op'] == ('edit_schema' if edits else 'materialize')
        time = 'day' if name == 'edit.time-column-text' else 'ts'
        age_declared = name == 'edit.accepted-meaning-key-links' and (
            not edits or at > 0
        )
        expected = {
            ('orders', column): (False, column == time)
            for column in ('order_id', 'user_id', 'price', time)
        } | {
            ('users', column): (False, column == 'age' and age_declared)
            for column in ('user_id', 'age', 'country')
        }
        seen = set()
        for table in start['schema']['tables']:
            for column in table['columns']:
                key = table['name'], column['name']
                assert key in expected and key not in seen, (
                    'held-column flags inventory'
                )
                seen.add(key)
                flags = column.get('untyped'), column.get('stype_declared')
                assert (
                    all(type(value) is bool for value in flags)
                    and flags == expected[key]
                ), 'held-column flags differ'
                del column['untyped'], column['stype_declared']
        assert seen == set(expected), 'held-column flags incomplete'
    return projected


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
    assert sum(e['outcome'] == 'success' for e in expects) == 2


def test_each_edit_case_ends_as_the_reference_ends_it(
    observed: dict[str, Any],
) -> None:
    reference = {
        line['case']: line
        for path in sorted(REFERENCE.glob('*.jsonl'))
        for line in map(json.loads, path.read_text().splitlines())
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
        # identity.operations: every core run started, in order.
        ops = checked_operations(mine)
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
                        "case, the digests of the core's inputs and of the "
                        "edits' starts, the core's error code, and the "
                        'digest of the request state; '
                        "sdm-testkit's tests/edit_cases.rs holds rust-host "
                        'to it (T1.24b; the cases in contract/cases/s1 '
                        'from T1.27).'
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


@pytest.mark.parametrize('changed', ['missing', 'false', 'unexpected', 'type'])
def test_native_flags_are_checked_before_the_historical_pin(
    observed: dict[str, Any],
    changed: str,
) -> None:
    name = 'edit.accepted-meaning-key-links'
    original = observed[name]
    original_before = copy.deepcopy(original)
    assert agreed(original) == json.loads(PINNED.read_text())['cases'][name]
    altered = copy.deepcopy(original)
    columns = altered['core']['edits'][0]['schema']['tables'][0]['columns']
    if changed == 'missing':
        columns[3].pop('stype_declared')
    elif changed == 'false':
        columns[3]['stype_declared'] = False
    elif changed == 'unexpected':
        columns[0]['untyped'] = True
    else:
        columns[3]['stype_declared'] = 1
    before = copy.deepcopy(altered)
    with pytest.raises(AssertionError, match='held-column flags'):
        agreed(altered)
    assert altered == before
    assert original == original_before


def test_the_historical_projection_keeps_later_native_edit_flags(
    observed: dict[str, Any],
) -> None:
    name = 'edit.accepted-meaning-key-links'
    altered = copy.deepcopy(observed[name])
    # The first edit really declares age's meaning. Its subsequent starts
    # must retain true even though the older pin never observed that bit.
    age = altered['core']['edits'][1]['schema']['tables'][1]['columns'][1]
    assert age['stype_declared'] is True
    age['stype_declared'] = False
    with pytest.raises(AssertionError, match='held-column flags'):
        agreed(altered)


def test_the_historical_projection_keeps_actual_edit_operands(
    observed: dict[str, Any],
) -> None:
    name = 'edit.accepted-meaning-key-links'
    altered = copy.deepcopy(observed[name])
    altered['core']['edits'][0]['edit']['stype']['value'] = 'numerical'
    seen = {name: agreed(altered)}
    pin = {
        'format': FORMAT,
        'cases': {name: json.loads(PINNED.read_text())['cases'][name]},
    }
    assert problems(pin, seen) == [f"{name}: edits is not the pin's"]
