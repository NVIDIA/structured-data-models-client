# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's acceptance tests for the case format's database axis and its new
fields (the independent test author's).

The design (choice 2): a case runs on DuckDB, SQLite and Postgres unless its
file says otherwise, and no case file may say otherwise without an
owner-approved exclusion; the format gains the fields Phase 1's could not
carry, ``candidates`` and ``top_k`` (E153), with accept and refuse vectors in
``contract/tests/malformed-cases.json``, run by both loaders
(``run_python.py``'s and ``sdm-testkit``'s ``case::load``, whose
``tests/s0_cases.rs`` runs every vector of that file).

The vectors are listed here, as the inputs the format must take or refuse.
The implementer adds each to ``malformed-cases.json``'s ``refused`` or
``accepted`` list (same ``why``, ``path`` and ``value``), so the Rust loader
runs them too; :func:`test_the_vectors_are_in_the_shared_file` holds that.

Gating: the tests that need the feature are marked :data:`PENDING`, a
strict xfail, so CI stays green until the implementation lands and turns red
if one passes before it. The implementer deletes the ``PENDING`` marker (its
one definition below) in the commit that makes them pass. The unmarked tests
are controls that pass on the base and must keep passing.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MALFORMED = ROOT / 'contract/tests/malformed-cases.json'
DECISIONS = ROOT / 'docs/design/rust-core/phase-2/decisions.md'


# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: The databases every case runs on, in this order, unless its file says
#: otherwise.
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: The case's top-level ``databases``: the databases it runs on, a non-empty
#: list of distinct names from DATABASES; absent means all three. Allowed
#: only beside ``exclusion``, the owner's decision row that approves leaving
#: the others out (``KO2-<n>``, or a later phase's ``KO<p>-<n>``), and
#: ``exclusion`` only beside ``databases``.
#: ``input.materialize.candidates``: a list of entity keys, each an integer
#: literal of any size or a string, as ``indices`` are read (a number with a
#: fraction or an exponent, ``1.0`` included, refused).
#: ``input.materialize.top_k``: a JSON integer, written as one. A value the
#: reference refuses (``0``) is a case's input, not a malformed case.
#: ``run_python.case_databases(case)``: the databases a loaded case runs on,
#: a tuple in DATABASES order.
# ---------------------------------------------------------------------------------


def _module(path: Path, name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _module(ROOT / 'contract/runners/run_python.py', 't21a_run_python')

M = ['input', 'materialize']

#: Each a change to malformed-cases.json's case: the loaders refuse it.
REFUSED: list[dict[str, Any]] = [
    {'why': 'databases with no database', 'path': ['databases'], 'value': []},
    {
        'why': 'databases naming a database the corpus does not run',
        'path': ['databases'],
        'value': ['duckdb', 'oracle'],
    },
    {
        'why': 'databases naming one twice',
        'path': ['databases'],
        'value': ['duckdb', 'duckdb'],
    },
    {
        'why': 'databases as one name, not a list',
        'path': ['databases'],
        'value': 'duckdb',
    },
    {
        'why': 'databases naming a database in upper case',
        'path': ['databases'],
        'value': ['DuckDB'],
    },
    {
        'why': 'an exclusion that is no decision row',
        'path': ['exclusion'],
        'value': 'the owner said so',
    },
    {
        'why': 'candidates that are not a list',
        'path': [*M, 'candidates'],
        'value': 7,
    },
    {
        'why': 'candidates as one text, not a list',
        'path': [*M, 'candidates'],
        'value': '7',
    },
    {
        'why': 'a candidate written with a fraction',
        'path': [*M, 'candidates'],
        'value': [1, 1.0],
    },
    {
        'why': 'a candidate that is a boolean',
        'path': [*M, 'candidates'],
        'value': [True],
    },
    {
        'why': 'a candidate that is null',
        'path': [*M, 'candidates'],
        'value': [None],
    },
    {
        'why': 'a candidate that is a list',
        'path': [*M, 'candidates'],
        'value': [[1]],
    },
    {'why': 'top_k as text', 'path': [*M, 'top_k'], 'value': '3'},
    {
        'why': 'top_k written with a fraction',
        'path': [*M, 'top_k'],
        'value': 3.0,
    },
    {'why': 'top_k as a boolean', 'path': [*M, 'top_k'], 'value': True},
    {'why': 'top_k null', 'path': [*M, 'top_k'], 'value': None},
]

#: Each a change the loaders take.
ACCEPTED: list[dict[str, Any]] = [
    {
        'why': 'candidates of integers, one past 2**64 read without loss',
        'path': [*M, 'candidates'],
        'value': [1, 2, 123456789012345678901234567890],
    },
    {
        'why': 'candidates of text keys',
        'path': [*M, 'candidates'],
        'value': ['a', 'b'],
    },
    {'why': 'top_k of 3', 'path': [*M, 'top_k'], 'value': 3},
    {
        'why': 'top_k of 0, an input the reference refuses, carried as a case',
        'path': [*M, 'top_k'],
        'value': 0,
    },
]

#: A case with databases and its exclusion, both set together (the
#: shared file's single-path changes cannot set two keys, so this one is
#: held here alone, through both keys).
EXCLUDED = {'databases': ['duckdb', 'sqlite'], 'exclusion': 'KO2-19'}


def _changed(
    case: dict[str, Any], path: list[Any], value: Any
) -> dict[str, Any]:
    changed = copy.deepcopy(case)
    at = changed
    for key in path[:-1]:
        at = at[key]
    at[path[-1]] = copy.deepcopy(value)
    return changed


def _base() -> dict[str, Any]:
    listed = json.loads(MALFORMED.read_text())
    return json.loads((ROOT / listed['case']).read_text())


def _loads(case: dict[str, Any], tmp_path: Path) -> list[Any] | str:
    path = tmp_path / 'case.json'
    path.write_text(json.dumps(case))
    try:
        return RUNNER.load_cases([path])
    except RUNNER.RunnerFailure as failure:
        return str(failure)


def _wrap(change: dict[str, Any]) -> dict[str, Any]:
    r"""A refused ``databases`` change is shown beside a valid exclusion, and
    a refused ``exclusion`` beside valid databases, so the refusal is the
    seed's and not the missing partner's."""
    case = _base()
    if change['path'] == ['databases']:
        case['exclusion'] = EXCLUDED['exclusion']
    if change['path'] == ['exclusion']:
        case['databases'] = EXCLUDED['databases']
    return _changed(case, change['path'], change['value'])


def test_the_base_case_loads_and_takes_none_of_the_new_keys() -> None:
    r"""Control: the shared file's case loads, and today holds none of the
    keys the vectors set (so each vector is the one introducing it)."""
    case = _base()
    assert 'databases' not in case and 'exclusion' not in case
    assert not {'candidates', 'top_k'} & set(case['input']['materialize'])


@pytest.mark.parametrize('change', REFUSED, ids=[c['why'] for c in REFUSED])
def test_a_malformed_new_field_is_refused(
    change: dict[str, Any], tmp_path: Path
) -> None:
    r"""Seeded (holds today, and must keep holding once the fields exist):
    the Python loader refuses each malformed value by name."""
    found = _loads(_wrap(change), tmp_path)
    assert isinstance(found, str) and 'not a case' in found, change['why']


@pytest.mark.parametrize('change', ACCEPTED, ids=[c['why'] for c in ACCEPTED])
def test_a_well_formed_new_field_is_taken(
    change: dict[str, Any], tmp_path: Path
) -> None:
    r"""Exit: the format carries ``candidates`` and ``top_k``."""
    found = _loads(_changed(_base(), change['path'], change['value']), tmp_path)
    assert not isinstance(found, str), found


def test_databases_with_their_exclusion_are_taken(tmp_path: Path) -> None:
    r"""Exit: a case may run on fewer databases, under an exclusion."""
    found = _loads({**_base(), **EXCLUDED}, tmp_path)
    assert not isinstance(found, str), found
    [(case, _)] = found
    assert RUNNER.case_databases(case) == ('duckdb', 'sqlite')


def test_a_case_without_databases_runs_on_all_three(tmp_path: Path) -> None:
    r"""Exit: every case runs on DuckDB, SQLite and Postgres unless its file
    says otherwise."""
    found = _loads(_base(), tmp_path)
    assert not isinstance(found, str), found
    [(case, _)] = found
    assert RUNNER.case_databases(case) == DATABASES


@pytest.mark.parametrize('key', ['databases', 'exclusion'])
def test_databases_and_exclusion_come_together(
    key: str, tmp_path: Path
) -> None:
    r"""Seeded: ``databases`` with no exclusion fails (no case leaves a
    database out without the owner), and an exclusion leaving nothing out
    fails. Its control, the two together, is taken
    (:func:`test_databases_with_their_exclusion_are_taken`)."""
    alone = {**_base(), key: EXCLUDED[key]}
    found = _loads(alone, tmp_path)
    assert isinstance(found, str) and 'not a case' in found
    # And the pair is taken, so the refusal is the missing partner's.
    assert not isinstance(_loads({**_base(), **EXCLUDED}, tmp_path), str)


def test_the_vectors_are_in_the_shared_file() -> None:
    r"""Exit: the vectors are in ``malformed-cases.json``, which both loaders
    run (``sdm-testkit``'s ``tests/s0_cases.rs`` and
    ``tests/test_contract_cases.py``)."""
    listed = json.loads(MALFORMED.read_text())

    def keyed(items: list[dict[str, Any]]) -> list[str]:
        return [json.dumps([c['path'], c['value']]) for c in items]

    refused, accepted = keyed(listed['refused']), keyed(listed['accepted'])
    # A databases or exclusion change needs its partner, which a one-path
    # change cannot give: those are held here, through both keys.
    single = [
        c for c in REFUSED if c['path'][0] not in ('databases', 'exclusion')
    ]
    missing = [c['why'] for c in single if keyed([c])[0] not in refused] + [
        c['why'] for c in ACCEPTED if keyed([c])[0] not in accepted
    ]
    assert not missing, missing


def test_every_exclusion_in_the_corpus_is_an_owner_approved_row() -> None:
    r"""Exit: no case file runs on fewer databases without an
    owner-approved exclusion: each ``exclusion`` names a row of the phase's
    decisions file whose status is the owner's approval."""
    rows = {}
    for line in DECISIONS.read_text().splitlines():
        cells = [cell.strip() for cell in line.split('|')]
        if len(cells) > 2 and re.fullmatch(r'KO2-[0-9]+', cells[1]):
            rows[cells[1]] = cells[-2]
    assert rows, 'no decision rows read'
    unapproved = []
    for path in sorted((ROOT / 'contract/cases').rglob('*.json')):
        case = json.loads(path.read_text())
        if 'databases' in case or 'exclusion' in case:
            status = rows.get(case.get('exclusion', ''), '')
            if not status.startswith('approved by owner'):
                unapproved.append(case['id'])
    assert not unapproved, unapproved
