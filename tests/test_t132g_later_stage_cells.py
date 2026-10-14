# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2f: past the reference's stop in a cell other than the
reference's.

Written by the package's independent test author, before the change. In a
runtime cell other than ref-pandas3 (H1), ``compare.py``'s ``_cell_pair``
holds python-core's host fields (``schema``, ``warnings``, ``result``,
``payload``) to python-old's there, with no stop: part 2d's rule reaches
only the comparison against python-ref. On the key cases and
nested.fixed-array python-old crashes where the reference does and
python-core goes on to predict, so everything past the stop differs.

The rule (the orchestrator's decision, part 2d's): where python-core's
``past_stop`` is set (an approved move past the reference's stop) and
python-old's outcome is the reference's, ``_cell_pair`` compares the two
only up to where python-old stopped, under part 2d's stage rules (the
schema only past from_source, the result and payload only at predict,
python-old's warnings the first of python-core's). Otherwise nothing
changes.

The observations are real: ``tests/golden/t132g-later-stage``'s
python-core and rust-host lines (part 2d), and its ``python-old.jsonl``,
python-old's own lines for the three cases from a gate run, under the
fixtures' ids; both Python runners' lines are labelled the dbr17-pandas2
cell (python-old's there are the reference cell's but for the pandas
version: the reviewer's dbr17 runs of part 2a showed its warnings and
outcome the same). Expectations come from the real ``ratchet.check
--effective-out`` on ``tests/test_contract_ratchet.py``'s temporary
repository; the comparison is the real ``compare.compare``.

Gating (``contract/tests/t132g_gate.py``): a test named in
``t132g-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CELL = 'dbr17-pandas2'
PAIR = 'python-core against python-old'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132g_gate', ROOT / 'contract/tests/t132g_gate.py')
later = _module(
    't132g_later_stage_tests', ROOT / 'tests/test_t132g_later_stage.py'
)
helpers = later.helpers
compare, ratchet, run_python = later.compare, later.ratchet, later.run_python

origin, repo = helpers.origin, helpers.repo
commit, land, append_entry = helpers.commit, helpers.land, helpers.append_entry
HUGEINT, FIXED, CORES = later.HUGEINT, later.FIXED, later.CORES
item, entry, of, has = later.item, later.entry, later.of, later.has
FIXTURES = later.FIXTURES
RUNNERS = ('python-core', 'python-old', 'rust-host')

Line = dict[str, Any]
Change = Callable[[dict[str, dict[str, Line]]], None]


def lines() -> dict[str, dict[str, Line]]:
    found: dict[str, dict[str, Line]] = later.lines()
    found['python-old'] = {
        line['case']: line
        for line in map(
            json.loads, (FIXTURES / 'python-old.jsonl').read_text().splitlines()
        )
    }
    for who in ('python-core', 'python-old'):
        for line in found[who].values():
            line['identity']['cell'] = CELL
    return found


def compared(
    r: Path,
    tmp: Path,
    seed: dict[str, Any] | None,
    change: Change | None = None,
) -> Any:
    landed = land(r, 'T1.32f', later.with_fixtures)
    if seed is not None:
        commit(r, 'T1.32g: an entry', lambda tree: append_entry(tree, seed))
    expect = tmp / 'expect.json'
    ratchet.check(r, landed, [], '', effective_out=expect)
    found, cases = lines(), later.case_files()
    if change is not None:
        change(found)
    directory = tmp / 'cases'
    directory.mkdir(exist_ok=True)
    digests = {}
    for name, spec in cases.items():
        path = directory / f'{name}.json'
        path.write_text(json.dumps(spec, indent=2))
        digests[name] = run_python.digest(path.read_bytes())
    bases = run_python.digest(run_python.BASES.read_bytes())
    files = {}
    for who, by_case in found.items():
        for name, line in by_case.items():
            line['identity']['inputs']['case'] = digests[name]
            line['identity']['inputs']['bases'] = bases
        path = tmp / f'{who}.jsonl'
        path.write_text(''.join(json.dumps(x) + '\n' for x in by_case.values()))
        files[who] = path
    return compare.compare(
        [directory],
        files['python-ref'],
        [files[who] for who in RUNNERS],
        expect_file=expect,
    )


def pair(report: Any, case: str) -> list[tuple[str | None, str]]:
    r"""The gating failures of the cell's pair, python-core against
    python-old, for ``case``."""
    return [(rule, text) for rule, text in of(report, case) if PAIR in text]


def held_past(found: list[tuple[str | None, str]]) -> bool:
    r"""python-core still held to python-old past python-old's stop."""
    return has(found, 'payload', PAIR) and has(found, 'result', PAIR)


#: The approved first admission past the reference's stop, both cases.
APPROVED = entry(*(item(c, w) for c in (HUGEINT, FIXED) for w in CORES))


def test_the_fixtures_are_what_they_say() -> None:
    found = lines()
    for case in (HUGEINT, FIXED):
        old, ref = found['python-old'][case], found['python-ref'][case]
        assert old['identity']['runner'] == 'python-old'
        assert old['identity']['native_loaded'] is False
        assert old['outcome'] == ref['outcome'], case
        assert old['host']['warnings'] == ref['host']['warnings'], case
        assert found['python-core'][case]['outcome']['stage'] == 'predict'
        assert {
            found[w][case]['identity']['cell']
            for w in (*CORES[:1], 'python-old')
        } == {CELL}


# --- passes ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', [HUGEINT, FIXED])
def test_the_cell_s_pair_compares_only_what_python_old_reached(
    repo: Path, tmp_path: Path, case: str, request: pytest.FixtureRequest
) -> None:
    report = compared(repo, tmp_path, APPROVED)

    def clean() -> None:
        assert pair(report, case) == [], pair(report, case)
        # Nothing else gates the case in this cell either.
        assert of(report, case) == [], of(report, case)

    G.held('exits', f'py:{request.node.name}', clean)


# --- still fails --------------------------------------------------------------------


def _old(case: str, **outcome: Any) -> Change:
    def change(found: dict[str, dict[str, Line]]) -> None:
        found['python-old'][case]['outcome'].update(outcome)

    return change


OLD_OTHERWISE = {
    'another stage': _old(HUGEINT, stage='from_source'),
    'another error': _old(HUGEINT, error='ValueError'),
}


@pytest.mark.parametrize('seed', sorted(OLD_OTHERWISE))
def test_python_old_otherwise_than_the_reference_relaxes_nothing(
    repo: Path, tmp_path: Path, seed: str
) -> None:
    found = pair(
        compared(repo, tmp_path, APPROVED, OLD_OTHERWISE[seed]), HUGEINT
    )
    assert held_past(found), found


def _core(case: str, edit: Callable[[Line], None]) -> Change:
    def change(found: dict[str, dict[str, Line]]) -> None:
        edit(found['python-core'][case])

    return change


BEFORE_THE_STOP = {
    # The public schema: python-old got past from_source on hugeint.
    'the public schema': (
        _core(
            HUGEINT,
            lambda line: line['public']['schema']['tables'][0].update(
                duplicate_keys=7
            ),
        ),
        'schema',
    ),
    # python-old's warnings, raised before it stopped.
    'python-old s warnings missing': (
        _core(
            HUGEINT, lambda line: line['host'].update(warnings=[], origins=[])
        ),
        'warnings',
    ),
}


@pytest.mark.parametrize('seed', sorted(BEFORE_THE_STOP))
def test_evidence_before_python_old_s_stop_still_fails_when_it_differs(
    repo: Path, tmp_path: Path, seed: str
) -> None:
    change, rule = BEFORE_THE_STOP[seed]
    found = pair(compared(repo, tmp_path, APPROVED, change), HUGEINT)
    assert has(found, rule, PAIR), found


NOT_APPROVED: dict[str, dict[str, Any] | None] = {
    'no entry': None,
    'a proposed entry': entry(
        *(item(HUGEINT, w) for w in CORES),
        status='proposed',
        approved_by=None,
    ),
}


@pytest.mark.parametrize('seed', sorted(NOT_APPROVED))
def test_without_an_approved_move_the_pair_is_held_in_full(
    repo: Path, tmp_path: Path, seed: str
) -> None:
    found = pair(compared(repo, tmp_path, NOT_APPROVED[seed]), HUGEINT)
    assert held_past(found), found
