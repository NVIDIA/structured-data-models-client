# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2d: an approved entry moving a runner past the reference's
stop, and DuckDB's large Arrow buffers, in ``compare.py``.

Written by the package's independent test author, before the change. Part
2a's first admission clears ``compare.py``'s ``outcome`` difference, but
its other rules against python-ref still fail a runner an approved entry
moves to a later stage than the reference reached: key.hugeint's core
succeeds through predict where the reference crashes at materialize
(``payload``, ``result``, ``sql.count``), nested.fixed-array's where the
reference refuses at from_source (``schema``, ``warnings`` too).

The rule (the orchestrator's decision), as these tests read it: for a case
and runner whose effective expectation (``ratchet.py check
--effective-out``, handed to ``compare.py --expect``) is an approved
entry's move of the reference's outcome to a later stage, the rules
against python-ref compare only the evidence the reference produced:

- statements (``sql.*``): every stage before the reference's stop as
  before; at its stop stage, the reference's statements, each held to the
  runner's statement at its position (the runner's further statements
  there came after the reference stopped); none after;
- ``schema``: held where the reference got past from_source (its public
  schema recorded); ``result`` and ``payload`` (public and the core's
  call): held only where the reference reached predict;
- ``warnings``: the reference's warnings, in order, are the first of the
  runner's (those raised before it stopped);
- evidence past the stop stays held between the core runners
  (``python-core`` against ``rust-host``: ``core``, ``sql.*``), and by the
  baselines. Without such an approved entry nothing changes.

And the large buffers (the orchestrator's decision): a row read's DuckDB
types predict the Arrow types an Arrow read of the same statement gives
(``_arrow_types``); where the case's own setup runs ``SET
arrow_large_buffer_size = true`` (reachable from ``compare.py``: the case
file's ``input.setup``), ``VARCHAR`` is ``large_string`` and ``X[]``
``large_list<l: X>``, on that path only.

The observations are real (lesson of part 2a): ``tests/golden/
t132g-later-stage/`` holds three cases as the bundle (5a5a9804) admits
them, under ids of their own so no state of the corpus changes them, with
the committed reference's lines for them and python-core's and
rust-host's as the bundle's gate run recorded them; only the case ids are
changed, and the comparator's input digests set to the files compared.
Each claim builds the expectations through the real ``ratchet.check`` on
``tests/test_contract_ratchet.py``'s temporary repository and compares
through the real ``compare.compare``.

Gating (``contract/tests/t132g_gate.py``): a test named in
``t132g-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/golden/t132g-later-stage'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132g_gate', ROOT / 'contract/tests/t132g_gate.py')
helpers = _module('ratchet_tests', ROOT / 'tests/test_contract_ratchet.py')
compare = _module('compare_for_ratchet', ROOT / 'scripts/rust_core/compare.py')
run_python = _module('run_python', ROOT / 'contract/runners/run_python.py')
ratchet = helpers.ratchet

origin, repo = helpers.origin, helpers.repo
commit, land = helpers.commit, helpers.land
append_entry, outcome = helpers.append_entry, helpers.outcome

HUGEINT = 't132g.later-key-hugeint'
LARGE = 't132g.later-key-varchar-large-buffers'
FIXED = 't132g.later-nested-fixed-array'
CORES = ('python-core', 'rust-host')
#: What the core answers, as each runner records it.
NEW = {'python-core': 'success at predict', 'rust-host': 'success'}
LARGE_SETUP = 'SET arrow_large_buffer_size = true'

Line = dict[str, Any]
Lines = dict[str, dict[str, Line]]


def lines() -> Lines:
    r"""The fixtures' lines, by runner and case."""
    return {
        who: {
            line['case']: line
            for line in map(
                json.loads,
                (FIXTURES / f'{who}.jsonl').read_text().splitlines(),
            )
        }
        for who in ('python-ref', *CORES)
    }


def case_files() -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text())
        for path in sorted((FIXTURES / 'cases').glob('*.json'))
    }


def with_fixtures(r: Path) -> None:
    r"""The fixture cases and their reference lines in the temporary
    repository, as a landing (not judged) adds them."""
    for path in (FIXTURES / 'cases').glob('*.json'):
        shutil.copy2(path, r / 'contract/cases/s1' / path.name)
    reference = next((r / 'contract/reference').glob('*.jsonl'))
    reference.write_text(
        reference.read_text() + (FIXTURES / 'python-ref.jsonl').read_text()
    )


def item(case: str, runner: str, new: str | None = None) -> dict[str, str]:
    return outcome(case, runner, 'none', new or NEW[runner])


def entry(*items: dict[str, str], **fields: Any) -> dict[str, Any]:
    return {
        'id': 'SEED-6',
        'kind': 'change',
        'cases': sorted({each['case'] for each in items}),
        'case_outcomes': list(items),
        **fields,
    }


APPROVED = entry(*(item(c, w) for c in (HUGEINT, LARGE, FIXED) for w in CORES))


def compared(
    r: Path,
    tmp: Path,
    seed: dict[str, Any] | None = APPROVED,
    change: Callable[[Lines, dict[str, Any]], None] | None = None,
) -> Any:
    r"""The fixtures compared under the expectations ``ratchet.check``
    writes for a candidate holding ``seed`` (an entry, or none), the
    fixture cases and their reference lines landed at its checkpoint;
    ``change`` changes the lines and the case files first."""
    landed = land(r, 'T1.32f', with_fixtures)
    if seed is not None:
        commit(r, 'T1.32g: an entry', lambda tree: append_entry(tree, seed))
    expect = tmp / 'expect.json'
    ratchet.check(r, landed, [], '', effective_out=expect)
    found, cases = lines(), case_files()
    if change is not None:
        change(found, cases)
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
        [files[who] for who in CORES],
        expect_file=expect,
    )


def of(report: Any, case: str) -> list[tuple[str | None, str]]:
    return [(rule, text) for rule, text in report.failures if case in text]


def has(found: list[tuple[str | None, str]], rule: str, *words: str) -> bool:
    return any(
        r == rule and all(word in text for word in words) for r, text in found
    )


def statement(line: Line, stage: str, position: int) -> dict[str, Any]:
    found: dict[str, Any] = [
        s for s in line['driver']['capture']['statements']
        if s['stage'] == stage
    ][position]  # fmt: skip
    return found


def values(line: Line, stage: str, position: int) -> dict[str, Any]:
    found: dict[str, Any] = statement(line, stage, position)['results'][0][
        'values'
    ]
    return found


# --- the fixtures are what they say -----------------------------------------------


def test_the_fixtures_are_real_and_past_the_reference_s_stop() -> None:
    found, cases = lines(), case_files()
    assert sorted(cases) == sorted([HUGEINT, LARGE, FIXED])
    for name in cases:
        ref = found['python-ref'][name]
        assert ref['identity']['runner'] == 'python-ref'
        assert ref['outcome']['result'] == 'refusal'
        assert found['python-core'][name]['outcome'] == {
            'result': 'success',
            'stage': 'predict',
            'error': None,
            'code': None,
        }
        assert found['rust-host'][name]['outcome']['result'] == 'success'
        # Real core lines: the core's own record, never a relabelled
        # reference line.
        for who in CORES:
            assert found[who][name]['identity']['runner'] == who
            assert found[who][name]['core'] is not None
        counts = compare.counts(found['python-core'][name])
        assert counts is not None
        assert counts != compare.counts(ref)
    assert found['python-ref'][HUGEINT]['outcome']['stage'] == 'materialize'
    assert found['python-ref'][FIXED]['outcome']['stage'] == 'from_source'
    assert LARGE_SETUP in cases[LARGE]['input']['setup']
    assert not any(
        LARGE_SETUP in step for step in cases[HUGEINT]['input']['setup']
    )


# --- passes ------------------------------------------------------------------------


@pytest.mark.parametrize('case', [HUGEINT, FIXED])
def test_an_approved_move_past_the_reference_s_stop_compares_what_it_reached(
    repo: Path, tmp_path: Path, case: str, request: pytest.FixtureRequest
) -> None:
    report = compared(repo, tmp_path)

    def clean() -> None:
        assert of(report, case) == [], of(report, case)

    G.held('exits', f'py:{request.node.name}', clean)


def test_the_large_buffer_case_passes_under_an_approved_move(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    report = compared(repo, tmp_path)

    def clean() -> None:
        assert of(report, LARGE) == [], of(report, LARGE)

    G.held('exits', f'py:{request.node.name}', clean)


# --- still fails: evidence the reference produced --------------------------------


def _python(case: str, edit: Callable[[Line], None]) -> Any:
    def change(found: Lines, cases: dict[str, Any]) -> None:
        edit(found['python-core'][case])

    return change


def _drop_last_materialize(line: Line) -> None:
    r"""python-core ran three of the reference's four materialize
    statements: the last of its own six dropped, and the one before, so
    the record stays complete (its count is DuckDB's query log's)."""
    statements = line['driver']['capture']['statements']
    places = [
        i for i, s in enumerate(statements) if s['stage'] == 'materialize'
    ]
    for i in reversed(places[3:]):
        statements.pop(i)
    line['driver']['query_log']['statements'] -= len(places[3:])


REACHED: dict[str, tuple[Callable[[Line], None], str, tuple[str, ...]]] = {
    # A statement before the reference's stop.
    'a from_source statement': (
        lambda line: values(line, 'from_source', 3).update(
            unordered='0' * 64, ordered='0' * 64
        ),
        'sql.result',
        ('python-core against python-ref', 'from_source statement 3'),
    ),
    # A statement at the reference's stop, one the reference ran.
    'a materialize statement the reference ran': (
        lambda line: values(line, 'materialize', 1).update(
            unordered='0' * 64, ordered='0' * 64
        ),
        'sql.result',
        ('python-core against python-ref', 'materialize statement 1'),
    ),
    # Fewer statements at the stop than the reference ran.
    'fewer statements at the stop': (
        _drop_last_materialize,
        'sql.count',
        ('python-core against python-ref', 'materialize'),
    ),
    # The public schema: the reference got past from_source.
    'the public schema': (
        lambda line: line['public']['schema']['tables'][0].update(
            duplicate_keys=7
        ),
        'schema',
        ('python-core against python-ref',),
    ),
    # The warnings the reference raised before it stopped.
    'the reference s warnings missing': (
        lambda line: line['host'].update(warnings=[], origins=[]),
        'warnings',
        ('python-core against python-ref',),
    ),
}


@pytest.mark.parametrize('seed', sorted(REACHED))
def test_evidence_the_reference_produced_still_fails_when_it_differs(
    repo: Path, tmp_path: Path, seed: str, request: pytest.FixtureRequest
) -> None:
    r"""Each fails today, but a statement at the stop the reference ran:
    today the stage's counts differ (six against four), so its statements
    are never compared one by one."""
    edit, rule, words = REACHED[seed]
    found = of(compared(repo, tmp_path, change=_python(HUGEINT, edit)), HUGEINT)

    def fails() -> None:
        assert has(found, rule, *words), found

    G.held('exits', f'py:{request.node.name}', fails)


def test_rust_host_s_statement_the_reference_ran_still_fails(
    repo: Path, tmp_path: Path
) -> None:
    def change(found: Lines, cases: dict[str, Any]) -> None:
        values(found['rust-host'][HUGEINT], 'from_source', 3).update(
            unordered='0' * 64, ordered='0' * 64
        )

    found = of(compared(repo, tmp_path, change=change), HUGEINT)
    assert has(
        found,
        'sql.result',
        'rust-host against python-ref',
        'from_source statement 3',
    ), found


# --- still fails: no approved later-stage entry ---------------------------------


NOT_APPROVED: dict[str, dict[str, Any] | None] = {
    'no entry': None,
    'a proposed entry': entry(
        *(item(HUGEINT, w) for w in CORES),
        status='proposed',
        approved_by=None,
    ),
    'an entry for another runner': entry(item(HUGEINT, 'rust-host')),
    'an entry for another case': entry(*(item(FIXED, w) for w in CORES)),
    'an entry for another cell': entry(
        *({**item(HUGEINT, w), 'cell': 'dbr17-pandas2'} for w in CORES)
    ),
}


@pytest.mark.parametrize('seed', sorted(NOT_APPROVED))
def test_without_an_approved_entry_for_the_runner_nothing_is_relaxed(
    repo: Path, tmp_path: Path, seed: str
) -> None:
    found = of(compared(repo, tmp_path, NOT_APPROVED[seed]), HUGEINT)
    where = 'python-core against python-ref'
    assert has(found, 'sql.count', where, 'materialize'), found
    assert has(found, 'payload', where), found
    assert has(found, 'result', where), found


def test_an_entry_moving_to_an_earlier_stage_relaxes_nothing(
    repo: Path, tmp_path: Path
) -> None:
    refusal = 'refusal INVALID_REQUEST at from_source (ValueError)'
    seed = entry(item(HUGEINT, 'python-core', refusal))

    def earlier(line: Line) -> None:
        line['outcome'].update(
            result='refusal',
            stage='from_source',
            error='ValueError',
            code='INVALID_REQUEST',
        )

    found = of(
        compared(repo, tmp_path, seed, _python(HUGEINT, earlier)), HUGEINT
    )
    where = 'python-core against python-ref'
    assert has(found, 'sql.count', where, 'materialize'), found
    assert has(found, 'payload', where), found


# --- still fails: the core runners against each other past the stop -------------


def test_the_core_runners_still_agree_past_the_reference_s_stop(
    repo: Path, tmp_path: Path
) -> None:
    def change(found: Lines, cases: dict[str, Any]) -> None:
        # Statement 5 of materialize: the reference ran four.
        values(found['rust-host'][HUGEINT], 'materialize', 5).update(
            unordered='0' * 64, ordered='0' * 64
        )

    found = of(compared(repo, tmp_path, change=change), HUGEINT)
    assert has(
        found,
        'sql.result',
        'python-core against rust-host',
        'materialize statement 5',
    ), found


def test_the_core_runners_calls_still_agree_past_the_reference_s_stop(
    repo: Path, tmp_path: Path
) -> None:
    def change(found: Lines, cases: dict[str, Any]) -> None:
        predicted = found['rust-host'][HUGEINT]['core']['predicted']
        body = copy.deepcopy(predicted['body'])
        body['seeded'] = {'number': '1'}
        predicted['body'] = body

    found = of(compared(repo, tmp_path, change=change), HUGEINT)
    assert has(found, 'core', 'python-core against rust-host'), found


# --- the large buffers ----------------------------------------------------------------


def _catalog_mismatches(found: list[tuple[str | None, str]]) -> list[str]:
    return [
        text
        for rule, text in found
        if rule == 'sql.result'
        and 'Arrow schema differs from the types' in text
    ]


def test_large_buffer_rows_and_arrow_agree(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    found = of(compared(repo, tmp_path), LARGE)

    def agree() -> None:
        assert _catalog_mismatches(found) == [], found

    G.held('exits', f'py:{request.node.name}', agree)


def test_large_strings_without_the_setting_still_fail(
    repo: Path, tmp_path: Path
) -> None:
    def change(found: Lines, cases: dict[str, Any]) -> None:
        setup = cases[LARGE]['input']['setup']
        setup.remove(LARGE_SETUP)

    found = of(compared(repo, tmp_path, change=change), LARGE)
    assert _catalog_mismatches(found), found


def _small_string(line: Line) -> None:
    r"""from_source statement 0 (a catalog read, rows on the reference) as
    if read without the large buffers: ``string``."""
    for field in statement(line, 'from_source', 0)['results'][0]['schema']:
        assert field['type'] == 'large_string', field
        field['type'] = 'string'


def test_small_strings_under_the_setting_fail(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    found = of(
        compared(repo, tmp_path, change=_python(LARGE, _small_string)), LARGE
    )

    def fails() -> None:
        assert any(
            'python-core against python-ref: from_source statement 0' in text
            for text in _catalog_mismatches(found)
        ), found

    G.held('exits', f'py:{request.node.name}', fails)


def test_a_value_difference_under_the_setting_still_fails(
    repo: Path, tmp_path: Path
) -> None:
    def other(line: Line) -> None:
        values(line, 'from_source', 0).update(
            unordered='0' * 64, ordered='0' * 64
        )

    found = of(compared(repo, tmp_path, change=_python(LARGE, other)), LARGE)
    assert has(
        found,
        'sql.result',
        'python-core against python-ref',
        'from_source statement 0',
        'rows differ',
    ), found
