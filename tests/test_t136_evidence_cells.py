# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.36 (final review P2-3): the evidence pack excuses an outcome under the
gate's cell rule, cell for cell, no more.

Written by the package's independent test author, before the fix.

    python -m pytest tests/test_t136_evidence_cells.py -q

The gate's rule, which ``scripts/rust_core/evidence.py`` must apply:

- A runner's outcome is judged against that runner's effective expectation,
  keyed by case and runner, never by cell. It is the ``ref-pandas3``
  baseline (``ratchet.baselines`` reads ``contract/baselines/ref-pandas3``
  only), or an approved entry's move from it, or, on a first admission,
  from ``none`` (``ratchet.effective``, ``moved``). ``check_baselines``
  holds every runner's observations to those baselines, whatever cell they
  name. ``compare.py``'s outcome rule (``_effective``) gates the outcome in
  every cell; outside ``ref-pandas3`` only the host fields against the
  reference are reported instead of gated (H1). So python-core in
  ``dbr17-pandas2`` and ``floor-pandas2.2`` is held to what holds it in
  ``ref-pandas3``.
- An entry moves an outcome only through a ``case_outcomes`` item whose
  ``cell`` is ``ratchet.CELL`` (``ref-pandas3``; ``authorized``, ``moved``
  and ``approved_move`` all match it), for that case and runner. The entry
  is approved, its ``new`` is the observed outcome, and its ``new`` is never
  ``INTERNAL_ERROR``. Where the expectation succeeded and the runner
  refuses, the entry is the owner's. An item naming ``dbr17-pandas2`` or
  ``floor-pandas2.2`` moves nothing, in any cell.
- An ``INTERNAL_ERROR`` never passes. The exception is python-old's own
  crash, equal to the reference line (``old_client_crash``).
- rust-host has no cell and is matched under ``ref-pandas3``, as before.
- A Python runner's observation with no cell recorded (an environment that
  matches none of ``run_python.CELLS``) is gated nowhere. No CI job makes
  one, since each pins its cell. ``gate.sh --locked`` marks checks 1 and 4
  "reported, not gated" and fails them unless the Python runners observed
  in exactly ``ref-pandas3`` (gate.sh's ``runners_cover`` and the locked
  table). So no entry excuses it, which is what evidence.py's
  ``excused_by`` docstring already says.

So ``evidence.py`` (``excused_by``, through ``classify``) matches an item
under ``ratchet.CELL`` for every runner, Python ones included, never under
the observed cell. The exception is a Python observation with no cell,
which nothing excuses. Everything else stays as it is: the comparison is
with the reference's outcome, the entry is approved, its ``new`` is the
observed outcome text and it is not a panic, and the owner's entry is
needed for a refusal of a reference success.

What is held here: python-core's approved new successes, under an entry
naming ``ref-pandas3`` only, are an ``approved improvement`` in every cell,
both seeded and as the real register states them
(``key.varchar-large-buffers`` and ``nested.fixed-array``, under P1B-6's
addition to P1B-33, P1B-9 and P1B-53). ``key.hugeint``,
``key.decimal-18-0`` and ``key.decimal-38-10`` were among them until the
reference at 6298913c (main's eb6c68eb) answered them: python-core's
success there is now a correct success in every cell, and a refusal of
them no improvement. The fail-closed seeds stay ``regression`` in every cell: another
outcome than the entry states, an entry not approved, a new refusal under a
reviewer's entry, a panic, an item naming the observed cell but not
``ref-pandas3``, and a Python observation with no cell. rust-host is
unchanged. A consistency check also runs over a seeded set of
observations, with baselines held and as first admissions: evidence.py's
verdict on each runner and cell is the verdict the ratchet's own functions
reach (``effective``, ``projection``, ``old_client_crash``).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CELLS = ('ref-pandas3', 'dbr17-pandas2', 'floor-pandas2.2')
OTHER_CELLS = ('dbr17-pandas2', 'floor-pandas2.2')
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'
REFERENCE = next(iter(sorted((ROOT / 'contract/reference').glob('*.jsonl'))))
#: python-core's owner-approved new successes (final review P2-3) where
#: the reference at 6298913c still refuses.
APPROVED_SUCCESSES = (
    'key.varchar-large-buffers',
    'nested.fixed-array',
)
#: The others of P2-3's five: since main's eb6c68eb (the reference at
#: 6298913c) the reference answers them too, so python-core's success is a
#: correct one, needing no entry.
ANSWERED_BY_THE_REFERENCE = (
    'key.hugeint',
    'key.decimal-18-0',
    'key.decimal-38-10',
)


def _load(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: a dataclass reads its module from sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evidence() -> ModuleType:
    return _load(ROOT / 'scripts/rust_core/evidence.py', 't136_evidence')


def ratchet() -> ModuleType:
    r"""The gate's ratchet, loaded on its own, not through evidence.py."""
    return _load(ROOT / 'scripts/rust_core/ratchet.py', 't136_ratchet')


# --- outcomes, lines and entries ---------------------------------------------------

Outcome = tuple[str, str | None, str | None, str | None]

SUCCESS: Outcome = ('success', 'predict', None, None)
CRASH: Outcome = (
    'refusal',
    'materialize',
    'StructuredDataError',
    'INTERNAL_ERROR',
)
UNNAMED: Outcome = ('refusal', 'from_source', 'RuntimeError', None)
NAMED_LATE: Outcome = (
    'refusal',
    'materialize',
    'StructuredDataError',
    'INVALID_REQUEST',
)


def _line(
    case: str, runner: str, cell: str | None, outcome: Outcome
) -> dict[str, Any]:
    result, stage, error, code = outcome
    if runner == 'rust-host':
        stage, error, cell = None, None, None
    return {
        'case': case,
        'identity': {'runner': runner, 'cell': cell},
        'outcome': {
            'result': result,
            'stage': stage,
            'error': error,
            'code': code,
        },
        'host': None if runner == 'rust-host' else {'warnings': []},
    }


def _as(line: dict[str, Any], runner: str, cell: str | None) -> dict[str, Any]:
    r"""``line`` observed by ``runner`` in ``cell``: python-old's copy of the
    reference line is its own crash where the reference crashed."""
    found = copy.deepcopy(line)
    found['identity'] = {**found['identity'], 'runner': runner, 'cell': cell}
    return found


def _text(runner: str, outcome: Outcome) -> str:
    r = ratchet()
    projected = r.projection(
        _line('x', runner, 'ref-pandas3', outcome), list(r.OUTCOME_FIELDS)
    )
    return str(r.outcome_text(projected))


def _item(
    case: str, runner: str, old: str, new: str, cell: str = 'ref-pandas3'
) -> dict[str, Any]:
    return {
        'case': case,
        'cell': cell,
        'runner': runner,
        'old': old,
        'new': new,
        'summary': 'seeded',
    }


def _entry(
    ident: str,
    items: list[dict[str, Any]],
    status: str = 'approved',
    approver: str | None = 'owner',
) -> dict[str, Any]:
    return {
        'id': ident,
        'status': status,
        'approved_by': approver if status == 'approved' else None,
        'cases': sorted({item['case'] for item in items}),
        'case_outcomes': items,
    }


def _classify(
    case: str,
    reference: dict[str, Any],
    observed: dict[tuple[str, str | None], dict[str, Any]],
    register: list[dict[str, Any]],
) -> dict[str, Any]:
    r"""evidence.py's row for ``case``: ``observed`` by runner and cell."""
    found: dict[str, Any] = evidence().classify(
        case,
        reference,
        {key: {case: line} for key, line in observed.items()},
        register,
        Counter(),
    )
    return found


# --- the seeded world --------------------------------------------------------------


def _seed(
    reference: Outcome,
    core: Outcome,
    items: Callable[[str, str], list[dict[str, Any]]],
    status: str = 'approved',
    approver: str | None = 'owner',
) -> dict[str, Any]:
    r"""A case: the reference's outcome, python-core's and rust-host's
    observed outcome (python-old answers as the reference), and the one
    entry: its status, approver and items, made by ``items(case, seed,
    first)``, ``first`` a first admission (old ``none``) or a baseline
    held (old the reference's outcome)."""
    return {
        'reference': reference,
        'core': core,
        'items': items,
        'status': status,
        'approver': approver,
    }


def _moves(*cells: str, new: Outcome | None = None) -> Callable[..., Any]:
    r"""Items moving python-core and rust-host to the observed outcome (or
    to ``new``), each under ``cells``."""

    def items(case: str, seed: dict[str, Any], first: bool) -> list[Any]:
        found = []
        for runner in ('python-core', 'rust-host'):
            old = 'none' if first else _text(runner, seed['reference'])
            want = _text(runner, seed['core'] if new is None else new)
            found += [_item(case, runner, old, want, cell) for cell in cells]
        return found

    return items


def _panic_items(case: str, seed: dict[str, Any], first: bool) -> list[Any]:
    old = 'none' if first else _text('python-core', seed['reference'])
    return [
        _item(case, runner, old, _text(runner, CRASH))
        for runner in ('python-core', 'rust-host')
    ]


SEEDS: dict[str, dict[str, Any]] = {
    # P1B-33's shape: the reference crashes, the core succeeds (owner).
    'improves-a-crash': _seed(CRASH, SUCCESS, _moves('ref-pandas3')),
    # P1B-9's or P1B-53's shape: an unnamed failure, now a success; a
    # reviewer's entry is enough for a success.
    'improves-an-unnamed-failure': _seed(
        UNNAMED, SUCCESS, _moves('ref-pandas3'), approver='reviewer'
    ),
    # A new refusal of a reference success, the owner's.
    'new-refusal-by-the-owner': _seed(
        SUCCESS, NAMED_LATE, _moves('ref-pandas3')
    ),
    # Fail closed: the same refusal under a reviewer's entry.
    'new-refusal-by-a-reviewer': _seed(
        SUCCESS, NAMED_LATE, _moves('ref-pandas3'), approver='reviewer'
    ),
    # Fail closed: the entry states another outcome than observed.
    'another-outcome-than-stated': _seed(
        CRASH,
        SUCCESS,
        _moves('ref-pandas3', new=NAMED_LATE),
    ),
    # Fail closed: the entry is not approved.
    'entry-proposed': _seed(
        CRASH, SUCCESS, _moves('ref-pandas3'), status='proposed'
    ),
    'entry-rejected': _seed(
        CRASH, SUCCESS, _moves('ref-pandas3'), status='rejected'
    ),
    # Fail closed: a panic, which an entry stating it never excuses.
    'a-core-panic': _seed(SUCCESS, CRASH, _panic_items),
    # Fail closed: the items name the other cells, never ref-pandas3; the
    # gate matches no item outside ratchet.CELL.
    'items-outside-the-reference-cell': _seed(
        CRASH, SUCCESS, _moves(*OTHER_CELLS)
    ),
    # A control: every runner answers as the reference, no entry.
    'as-the-reference': _seed(SUCCESS, SUCCESS, lambda *_: []),
}

#: The seeds whose differences must be excused in every named cell.
EXCUSED = (
    'improves-a-crash',
    'improves-an-unnamed-failure',
    'new-refusal-by-the-owner',
)
FAIL_CLOSED = (
    'new-refusal-by-a-reviewer',
    'another-outcome-than-stated',
    'entry-proposed',
    'entry-rejected',
    'a-core-panic',
    'items-outside-the-reference-cell',
)


def _world(
    name: str, first: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    r"""``name``'s reference line and register; ``first``: the entry's old
    outcome is ``none`` (a first admission, as the real entries are), else
    the reference's (a baseline held)."""
    seed = SEEDS[name]
    items = seed['items'](name, seed, first)
    register = (
        [_entry(f'T6-{name}', items, seed['status'], seed['approver'])]
        if items
        else []
    )
    return _line(name, 'python-ref', 'ref-pandas3', seed['reference']), register


def _observed(
    name: str, runner: str, cell: str | None, reference: dict[str, Any]
) -> dict[str, Any]:
    seed = SEEDS[name]
    if runner == 'python-old':
        return _as(reference, 'python-old', cell)
    return _line(name, runner, cell, seed['core'])


# --- 1: approved new successes in every cell -----------------------------------------


@pytest.mark.parametrize('cell', OTHER_CELLS)
@pytest.mark.parametrize('name', EXCUSED)
def test_a_reference_cell_entry_excuses_the_same_outcome_in_another_cell(
    name: str, cell: str
) -> None:
    r"""python-core's outcome in ``cell``, the one an approved entry states
    under ``ref-pandas3``, is an approved improvement, as the gate holds it
    (ratchet.CELL for every runner), not a regression."""
    reference, register = _world(name)
    row = _classify(
        name,
        reference,
        {
            ('python-core', cell): _observed(
                name, 'python-core', cell, reference
            )
        },
        register,
    )
    assert row['unexcused'] == [], row['unexcused']
    assert row['class'] == 'approved improvement'
    assert row['excused'] == {f'T6-{name}'}


@pytest.mark.parametrize('name', EXCUSED)
def test_the_whole_row_is_an_improvement_with_every_cell(name: str) -> None:
    r"""Every cell's python-core, python-old and rust-host in one row."""
    reference, register = _world(name)
    observed = {
        ('python-core', cell): _observed(name, 'python-core', cell, reference)
        for cell in CELLS
    }
    observed[('python-old', 'ref-pandas3')] = _observed(
        name, 'python-old', 'ref-pandas3', reference
    )
    observed[('rust-host', None)] = _observed(
        name, 'rust-host', None, reference
    )
    row = _classify(name, reference, observed, register)
    assert row['unexcused'] == [], row['unexcused']
    assert row['class'] == 'approved improvement'


# --- 2: the real register's entries ---------------------------------------------------


def _real() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    register = yaml.safe_load(REGISTER.read_text(encoding='utf-8'))
    lines = [
        json.loads(text)
        for text in REFERENCE.read_text(encoding='utf-8').splitlines()
        if text.strip()
    ]
    return register['entries'], {line['case']: line for line in lines}


def _approved_naming(entries: list[dict[str, Any]], case: str) -> set[str]:
    return {
        str(entry['id'])
        for entry in entries
        if entry.get('status') == 'approved'
        and any(
            item.get('case') == case and item.get('cell') == 'ref-pandas3'
            for item in entry.get('case_outcomes') or []
        )
    }


def test_the_real_entries_name_only_the_reference_cell() -> None:
    r"""The premise: the entries approving these successes name python-core
    under ``ref-pandas3`` alone, so only the gate's cell rule excuses the
    other cells."""
    entries, reference = _real()
    for case in APPROVED_SUCCESSES:
        assert case in reference, f'{case}: no reference line'
        items = [
            item
            for entry in entries
            if entry.get('status') == 'approved'
            for item in entry.get('case_outcomes') or []
            if item.get('case') == case and item.get('runner') == 'python-core'
        ]
        assert items, f'{case}: no approved python-core item'
        assert {item['cell'] for item in items} == {'ref-pandas3'}
        assert {item['new'] for item in items} == {'success at predict'}
        assert reference[case]['outcome']['result'] == 'refusal'


@pytest.mark.parametrize('cell', OTHER_CELLS)
@pytest.mark.parametrize('case', APPROVED_SUCCESSES)
def test_the_real_approved_successes_are_improvements_in_every_cell(
    case: str, cell: str
) -> None:
    r"""python-core's success at predict, observed in ``cell``, under the
    real register and the committed reference line, is an approved
    improvement, named by an approved entry for the case."""
    entries, references = _real()
    reference = references[case]
    core = _as(reference, 'python-core', cell)
    core['outcome'] = {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    }
    row = _classify(case, reference, {('python-core', cell): core}, entries)
    assert row['unexcused'] == [], row['unexcused']
    assert row['class'] == 'approved improvement'
    assert row['excused'] and row['excused'] <= _approved_naming(entries, case)


@pytest.mark.parametrize('case', APPROVED_SUCCESSES)
def test_the_real_rows_are_improvements_with_every_runner_and_cell(
    case: str,
) -> None:
    r"""The real row: python-core succeeding in all three cells, python-old
    as the reference, rust-host succeeding."""
    entries, references = _real()
    reference = references[case]
    observed: dict[tuple[str, str | None], dict[str, Any]] = {}
    for cell in CELLS:
        core = _as(reference, 'python-core', cell)
        core['outcome'] = {
            'result': 'success',
            'stage': 'predict',
            'error': None,
            'code': None,
        }
        observed[('python-core', cell)] = core
    observed[('python-old', 'ref-pandas3')] = _as(
        reference, 'python-old', 'ref-pandas3'
    )
    observed[('rust-host', None)] = _line(case, 'rust-host', None, SUCCESS)
    row = _classify(case, reference, observed, entries)
    assert row['unexcused'] == [], row['unexcused']
    assert row['class'] == 'approved improvement'


@pytest.mark.parametrize('cell', CELLS)
@pytest.mark.parametrize('case', ANSWERED_BY_THE_REFERENCE)
def test_a_success_the_reference_now_gives_is_correct_in_every_cell(
    case: str, cell: str
) -> None:
    r"""The real reference line is a success at predict now; python-core's
    success observed in ``cell`` is a correct success, excused by nothing
    and needing nothing; its refusal there is no improvement."""
    entries, references = _real()
    reference = references[case]
    assert (
        reference['outcome']['result'],
        reference['outcome']['stage'],
    ) == ('success', 'predict')
    core = _as(reference, 'python-core', cell)
    row = _classify(case, reference, {('python-core', cell): core}, entries)
    assert row['unexcused'] == [], row['unexcused']
    assert row['class'] == 'correct success'
    refused = copy.deepcopy(core)
    refused['outcome'] = {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'INVALID_REQUEST',
    }
    row = _classify(case, reference, {('python-core', cell): refused}, entries)
    assert row['class'] not in ('correct success', 'approved improvement')


# --- 3: what stays a regression ------------------------------------------------------


@pytest.mark.parametrize('cell', CELLS)
@pytest.mark.parametrize('name', FAIL_CLOSED)
def test_a_fail_closed_seed_stays_a_regression_in_every_cell(
    name: str, cell: str
) -> None:
    reference, register = _world(name)
    row = _classify(
        name,
        reference,
        {
            ('python-core', cell): _observed(
                name, 'python-core', cell, reference
            )
        },
        register,
    )
    assert row['class'] == 'regression', row
    assert row['excused'] == set(), row


@pytest.mark.parametrize('name', [*EXCUSED, 'as-the-reference'])
def test_a_python_observation_with_no_cell_is_excused_by_nothing(
    name: str,
) -> None:
    r"""No cell recorded on a Python runner: gated nowhere, so no entry
    excuses it, though one under ``ref-pandas3`` states it. A control with
    no difference shows the row is not refused for the missing cell alone."""
    reference, register = _world(name)
    row = _classify(
        name,
        reference,
        {
            ('python-core', None): _observed(
                name, 'python-core', None, reference
            )
        },
        register,
    )
    if name == 'as-the-reference':
        assert row['class'] == 'correct success', row
    else:
        assert row['class'] == 'regression', row
        assert row['excused'] == set(), row


@pytest.mark.parametrize('name', [*EXCUSED, *FAIL_CLOSED])
def test_rust_host_is_matched_under_the_reference_cell_as_before(
    name: str,
) -> None:
    reference, register = _world(name)
    row = _classify(
        name,
        reference,
        {('rust-host', None): _observed(name, 'rust-host', None, reference)},
        register,
    )
    if name in EXCUSED:
        assert row['class'] == 'approved improvement', row
    else:
        assert row['class'] == 'regression', row


# --- 4: evidence.py agrees with the ratchet, runner by runner and cell by cell -----


def _gate_passes(
    name: str,
    runner: str,
    cell: str | None,
    line: dict[str, Any],
    reference: dict[str, Any],
    register: list[dict[str, Any]],
    first: bool,
) -> bool:
    r"""The gate's verdict on one observation, by the ratchet's own
    functions: the runner's effective expectation (``effective``: the
    ``ref-pandas3`` baseline, here the reference's outcome, or an approved
    entry's move from it, or on a first admission from ``none``; keyed by
    case and runner, the same in every cell), held on the fields the runner
    observes; an ``INTERNAL_ERROR`` never, but python-old's own crash
    (``old_client_crash``). A Python runner with no cell is gated nowhere
    (``gate.sh --locked``: checks 1 and 4 "reported, not gated"), so no
    entry moves it: it passes only as the reference answered."""
    r = ratchet()
    if runner != 'rust-host' and cell is None:
        register, first = [], True
    fields = evidence().fields(runner)
    now = r.projection(line, list(r.OUTCOME_FIELDS))
    if now['outcome.code'] == r.PANIC:
        return bool(
            r.old_client_crash(runner, line, reference, evidence().ordered())
        )
    baseline = r.projection(reference, list(r.OUTCOME_FIELDS))
    if runner == 'rust-host':
        baseline.update({'outcome.stage': None, 'outcome.error': None})
    held = {} if first else {runner: {name: {'projection': baseline}}}
    want = (
        r.effective(held, r.approved({'entries': register}), {name: reference})
        .get(name, {})
        .get(runner, baseline)
    )
    return all(now[field] == want[field] for field in fields)


def _runs() -> list[tuple[str, str | None]]:
    found: list[tuple[str, str | None]] = []
    for runner in ('python-core', 'python-old'):
        found += [(runner, cell) for cell in (*CELLS, None)]
    return [*found, ('rust-host', None)]


@pytest.mark.parametrize(
    'first', [True, False], ids=['first-admission', 'baselined']
)
@pytest.mark.parametrize('name', sorted(SEEDS))
def test_evidence_agrees_with_the_ratchet_in_every_cell(
    name: str, first: bool
) -> None:
    reference, register = _world(name, first)
    disagree = []
    for runner, cell in _runs():
        line = _observed(name, runner, cell, reference)
        gate = _gate_passes(
            name, runner, cell, line, reference, register, first
        )
        row = _classify(name, reference, {(runner, cell): line}, register)
        if gate != (row['unexcused'] == []):
            disagree.append(
                f'{runner} ({cell}): gate {"passes" if gate else "refuses"}, '
                f'evidence {row["class"]} {row["unexcused"]}'
            )
    assert disagree == []


def test_the_consistency_check_sees_both_verdicts() -> None:
    r"""The check is not vacuous: over the seeds, the gate both passes and
    refuses, in the reference's cell and outside it."""
    seen = set()
    for name in SEEDS:
        reference, register = _world(name)
        for runner, cell in _runs():
            line = _observed(name, runner, cell, reference)
            seen.add(
                (
                    cell == 'ref-pandas3',
                    _gate_passes(
                        name, runner, cell, line, reference, register, True
                    ),
                )
            )
    assert seen == {(True, True), (True, False), (False, True), (False, False)}


# --- 5: the pack itself --------------------------------------------------------------


def _t135() -> ModuleType:
    r"""T1.35's seeded world and pack readers (``tests/test_t135_evidence.py``)."""
    return _load(ROOT / 'tests/test_t135_evidence.py', 't136_t135_evidence')


@pytest.mark.gate_tooling
def test_the_pack_classes_a_reference_cell_entry_as_an_improvement(
    tmp_path: Path,
) -> None:
    r"""T1.35's world, its approved improvement's python-core items cut down to
    ``ref-pandas3`` alone, as the real register states them: ``cases.md``
    still classes the case ``approved improvement``, naming the entry, with
    python-core succeeding in all three cells."""
    t135 = _t135()
    state = t135.world()
    for entry in state['register']:
        if entry['id'] == 'ZZ-1':
            entry['case_outcomes'] = [
                item
                for item in entry['case_outcomes']
                if item['cell'] == 'ref-pandas3'
            ]
            assert {item['runner'] for item in entry['case_outcomes']} == {
                'python-core',
                'rust-host',
            }
    paths, _ = t135.generate(tmp_path, state)
    assert t135.class_of(paths, 'nested.improved') == 'approved improvement'
    assert 'ZZ-1' in t135.entry_of(paths, 'nested.improved')
    # The seeded regression with no entry stays one.
    assert t135.class_of(paths, 'key.regress') == 'regression'


@pytest.mark.gate_tooling
def test_the_pack_excuses_nothing_by_an_item_outside_the_reference_cell(
    tmp_path: Path,
) -> None:
    r"""T1.35's world, its approved improvement's python-core item for
    ``ref-pandas3`` taken out: the items naming ``dbr17-pandas2`` and
    ``floor-pandas2.2`` excuse nothing, in any cell, since the gate matches
    none, so the case is a regression."""
    t135 = _t135()
    state = t135.world()
    for entry in state['register']:
        if entry['id'] == 'ZZ-1':
            entry['case_outcomes'] = [
                item
                for item in entry['case_outcomes']
                if item['runner'] == 'rust-host'
                or item['cell'] != 'ref-pandas3'
            ]
    paths, _ = t135.generate(tmp_path, state)
    assert t135.class_of(paths, 'nested.improved') == 'regression'
