# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0f's acceptance tests for the ``<case>@<database>`` keys in the
observation schema and the ratchet (the independent test author's).

The design (choice 2, "Reference"): SQLite and Postgres lines are keyed
``<case>@sqlite`` and ``<case>@postgres``, DuckDB lines keep the bare case
ID, and observations and baselines use the same keys. T2.0f teaches the
observation schema's case pattern and the ratchet's removed-case and
baseline rules the suffix (``compare.py``'s known-case check is held in
``contract/tests/test_t20f_gate.py``, through the gate).

The schema: through ``contract/runners/run_python.py``'s validator, the one
``compare.py`` validates every observation with. ``X@sqlite`` and
``X@postgres`` are valid observation keys; seeded faults, refused: an
unknown database, ``@duckdb`` (DuckDB keeps the bare ID), an empty part
on either side, two suffixes, upper case. Controls: the bare ID stays
valid, and a case file's own ID never takes a suffix.

The ratchet: ``ratchet.py check`` on a real git history, built as
``tests/test_contract_ratchet.py`` builds it (its helpers, reused), with
two more landings on top, standing in for T2.1a's recordings:

- ``recorded``: the reference holds ``X@sqlite`` and ``X@postgres`` lines
  (copies of ``X``'s line in place of those T2.1a recorded, ``X`` an S0
  acceptance case baselined on every runner), and no baseline of them yet;
  a runner observes X on a database where a test says so, as a copy of
  its X line (``observe`` replays the reference's DuckDB lines only);
- ``held``: on top, every runner's baseline of ``X@sqlite`` and
  ``X@postgres``.

Removal (exit: "removing a case removes its database lines"): at ``held``,
an approved removal of ``X`` that removes its file and every line keyed
``X``, ``X@sqlite`` or ``X@postgres`` (reference and baselines) passes,
the entry for ``X`` covering its database lines. Seeded: the same removal
leaving ``X@postgres``'s reference line, or one runner's ``X@postgres``
baseline, fails, naming ``X@postgres``.

Baselines (exit, as the coordinator corrected the design's row: "an
admitted acceptance case needs a baseline on each database the core is
observed on"): at ``recorded``, a reference line for ``X@sqlite`` with no
core observation on SQLite passes with no SQLite baseline (control); with
python-core observing ``X@sqlite`` and ``X@postgres`` and the candidate
proposing both baselines it passes, and seeded, python-core's ``X@postgres``
baseline missing fails, naming ``X@postgres`` and python-core.

``ratchet.py accept`` (review round 1): with python-core and python-old
observing X on all three databases, the baselines ``accept`` writes hold
``X@sqlite`` and ``X@postgres`` too, and ``check`` then passes.

Approving a database line (review round 1; the case is the unit, its
databases are lines of it): an unapproved move of python-old's
``X@postgres`` baseline fails, and the ratchet's message suggests the
``case_outcomes`` item; that item, as suggested (its ``summary`` filled
in), is one the register accepts and approves the move. In the same form
(the runner ``python-ref``, the reference line's old and new outcome), an
entry approves a changed ``X@postgres`` reference line, which fails
without one.

After the rebase onto T2.0e (review round 1): an owner-approved
re-recording of the reference that drops an approved-removed case's
database lines passes; dropping a kept case's ``X@sqlite`` line fails,
naming it. It builds T2.0e's approval as T2.0e's tests assume it (an owner
row in ``phase-2/decisions.md`` naming the full new SHA and the word
"reference", every citation of the old SHA outside ``docs/`` moved). It
cannot pass on this branch's base, which has no T2.0e, so it is gated
like the rest: the implementer removes ``PENDING`` once rebased onto
T2.0e's landing.

Every seeded fault is shown on a candidate whose unseeded twin passes, so a
refusal is the seed's.

Gating: the tests that need the feature are marked :data:`PENDING`, a
strict xfail, so CI stays green until the implementation lands and turns
red if one passes before it (a vacuous test). The implementer deletes the
``PENDING`` marker (the one definition below) in the commit that makes them
pass. The unmarked tests are controls that pass on the base and must keep
passing.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[1]

# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: The databases a key may name after ``@``; DuckDB's lines keep the bare
#: case ID, so ``@duckdb`` is no key.
DATABASES = ('sqlite', 'postgres')
#: A removal's register entry names the bare case in each
#: ``case_outcomes`` item (``cases`` names the task, since a removed case
#: is no corpus case), each baselined runner's old outcome and
#: ``removed``, as today; it covers the case's database lines, whose
#: outcomes here equal the bare line's.
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


helpers = _module(ROOT / 'tests/test_contract_ratchet.py', 'ratchet_tests')
RUNNER = _module(ROOT / 'contract/runners/run_python.py', 't20f_run_python')
ratchet = helpers.ratchet
git, commit, land = helpers.git, helpers.commit, helpers.land
CELL = helpers.CELL

#: The case whose database lines the tests key: an S0 acceptance case,
#: baselined on every runner.
X = 's0.accept.sum-window'
SQLITE, POSTGRES = f'{X}@sqlite', f'{X}@postgres'
RUNNERS = ('python-core', 'python-old', 'rust-host')


# --- the observation schema ------------------------------------------------------


def _reference_line() -> dict[str, Any]:
    path = next((ROOT / 'contract/reference').glob('*.jsonl'))
    for line in path.read_text().splitlines():
        found: dict[str, Any] = json.loads(line)
        if found['case'] == X:
            return found
    raise AssertionError(f'no reference line of {X}')


def _invalid(case: str) -> str | None:
    line = copy.deepcopy(_reference_line())
    line['case'] = case
    return RUNNER._invalid(RUNNER.validator('observation'), line)


def test_the_bare_key_is_a_valid_observation() -> None:
    r"""Control: DuckDB's lines keep the bare case ID."""
    assert _invalid(X) is None


@pytest.mark.parametrize('database', DATABASES)
def test_a_database_key_is_a_valid_observation(database: str) -> None:
    r"""Exit: the observation schema's case pattern takes
    ``<case>@sqlite`` and ``<case>@postgres``."""
    assert _invalid(f'{X}@{database}') is None


@pytest.mark.parametrize(
    'key',
    [
        f'{X}@oracle',
        f'{X}@duckdb',
        f'{X}@',
        '@sqlite',
        f'{X}@sqlite@postgres',
        f'{X}@SQLITE',
        f'{X} @sqlite',
        f'{X}@sqlite ',
    ],
)
def test_a_malformed_database_key_is_not_an_observation(key: str) -> None:
    r"""Seeded (holds today and must keep holding): an unknown database,
    ``@duckdb``, an empty part, two suffixes, upper case, a space."""
    assert _invalid(key) is not None


def test_a_case_file_s_own_id_takes_no_database_suffix() -> None:
    r"""Control: the suffix is the observation's key, never a case's ID
    (the part before ``@`` is a case)."""
    path = ROOT / f'contract/cases/s0/{X}.json'
    case = json.loads(path.read_text())
    validate = RUNNER.validator('case')
    assert RUNNER._invalid(validate, case) is None
    case['id'] = SQLITE
    assert RUNNER._invalid(validate, case) is not None


# --- the ratchet's history ---------------------------------------------------------


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _write_jsonl(path: Path, lines: list[dict[str, Any]]) -> None:
    path.write_text(''.join(json.dumps(x) + '\n' for x in lines))


def _reference_file(tree: Path) -> Path:
    found = list((tree / 'contract/reference').glob('*.jsonl'))
    assert len(found) == 1, found
    return found[0]


def _baseline_file(tree: Path, runner: str) -> Path:
    return tree / f'contract/baselines/{CELL}/{runner}.jsonl'


def _record_database_lines(tree: Path) -> None:
    r"""The reference's ``X@sqlite`` and ``X@postgres`` lines: copies of
    ``X``'s, keyed by database, in place of the lines T2.1a recorded for
    them, so each database's baseline is a copy of ``X``'s."""
    path = _reference_file(tree)
    lines = [x for x in _jsonl(path) if x['case'] not in (SQLITE, POSTGRES)]
    bare = next(x for x in lines if x['case'] == X)
    for key in (SQLITE, POSTGRES):
        added = copy.deepcopy(bare)
        added['case'] = key
        lines.append(added)
    _write_jsonl(path, lines)


def _hold_database_baselines(tree: Path) -> None:
    r"""Every runner's ``X@sqlite`` and ``X@postgres`` baselines: copies of
    its ``X`` baseline (the digest is of the projection, so it holds)."""
    for runner in RUNNERS:
        path = _baseline_file(tree, runner)
        lines = _jsonl(path)
        bare = next(x for x in lines if x['case'] == X)
        for key in (SQLITE, POSTGRES):
            added = copy.deepcopy(bare)
            added['case'] = key
            lines.append(added)
        _write_jsonl(path, lines)


@pytest.fixture(scope='module')
def origin(owned_module_dir: Callable[[str], Path]) -> dict[str, Any]:
    repo, _ = helpers.build(owned_module_dir('origin'))
    recorded = land(
        repo, 'T2.1a', _record_database_lines, protected='contract/**'
    )
    held = land(repo, 'T2.2', _hold_database_baselines, protected='contract/**')
    return {'repo': repo, 'recorded': recorded, 'held': held}


def _worktree(origin: dict[str, Any], tmp_path: Path, at: str) -> Path:
    return helpers.repo.__wrapped__((origin['repo'], at), tmp_path)


@pytest.fixture
def recorded(origin: dict[str, Any], tmp_path: Path) -> Path:
    r"""A worktree at the landing whose reference holds X's database
    lines, with no baseline of them."""
    return _worktree(origin, tmp_path, origin['recorded'])


@pytest.fixture
def held(origin: dict[str, Any], tmp_path: Path) -> Path:
    r"""A worktree at the landing that also holds every runner's baseline
    of X's database lines."""
    return _worktree(origin, tmp_path, origin['held'])


def _fields(tree: Path) -> list[str]:
    found: list[str] = yaml.safe_load(
        (tree / 'contract/compare-policy.yaml').read_text()
    )['baseline']
    return found


def _observed(files: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for path in files:
        for line in _jsonl(path):
            found.setdefault(line['identity']['runner'], {})[line['case']] = (
                line
            )
    return found


def _propose(
    files: tuple[Path, ...], keys: dict[str, tuple[str, ...]]
) -> Callable[[Path], None]:
    r"""Appends, for each runner in ``keys``, its proposed baseline of each
    key from its observation (as ``ratchet.py accept`` writes it)."""
    observed = _observed(files)

    def edit(tree: Path) -> None:
        fields = _fields(tree)
        for runner, wanted in keys.items():
            path = _baseline_file(tree, runner)
            lines = _jsonl(path)
            for key in wanted:
                projection = ratchet.projection(observed[runner][key], fields)
                lines.append(
                    {
                        'case': key,
                        'runner': runner,
                        'cell': CELL,
                        'projection': projection,
                        'digest': ratchet.digest(projection),
                    }
                )
            _write_jsonl(path, lines)

    return edit


def _on_databases(
    keys: tuple[str, ...],
    then: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[dict[str, Any]], None]:
    r"""A runner's observations (``test_contract_ratchet.observe``, which
    replays the reference's DuckDB lines only) with X observed under each
    of ``keys`` too, as a copy of its X line (the reference's database
    lines of X being copies of X's, :func:`_record_database_lines`); then
    ``then``."""

    def change(lines: dict[str, Any]) -> None:
        for key in keys:
            line = copy.deepcopy(lines[X])
            line['case'] = key
            lines[key] = line
        if then is not None:
            then(lines)

    return change


def _observe(
    tree: Path, tmp_path: Path, keep: tuple[str, ...]
) -> tuple[Path, ...]:
    r"""python-core's and python-old's observations, made from the
    reference's (``test_contract_ratchet.observe``), keeping of X's lines
    only ``keep``."""
    on = tuple(key for key in (SQLITE, POSTGRES) if key in keep)

    def drop(lines: dict[str, Any]) -> None:
        if X not in keep:
            lines.pop(X)

    return tuple(
        helpers.observe(tree, tmp_path, runner, _on_databases(on, drop))
        for runner in ('python-core', 'python-old')
    )


def _naming(problems: list[str], key: str) -> list[str]:
    r"""The problems naming ``key`` as a key (not as part of a longer one)."""
    return [
        p
        for p in problems
        if any(
            p[i + len(key) : i + len(key) + 1] in ('', ' ', ':', ',', ')', "'")
            for i in _starts(p, key)
        )
    ]


def _starts(text: str, key: str) -> list[int]:
    found, at = [], text.find(key)
    while at != -1:
        if at == 0 or text[at - 1] in ' :,(\'"':
            found.append(at)
        at = text.find(key, at + 1)
    return found


# --- baselines on each database the core is observed on -------------------------


def test_a_database_recording_with_no_core_observation_needs_no_baseline(
    recorded: Path, tmp_path: Path
) -> None:
    r"""Control (the coordinator's correction): the reference holds
    ``X@sqlite`` and ``X@postgres``, the core is observed on DuckDB only,
    and no database baseline is proposed: it passes."""
    landed = git(recorded, 'rev-parse', 'HEAD')
    files = _observe(recorded, tmp_path, keep=(X,))
    commit(recorded, 'T2.0f: a change')
    assert helpers.check(recorded, landed, files) == []


def test_a_core_observation_on_a_database_without_its_baseline_fails_by_name(
    recorded: Path, tmp_path: Path
) -> None:
    r"""Exit: an admitted acceptance case needs a baseline on each database
    the core is observed on. Twin: python-core and python-old observe X on
    SQLite and Postgres and both baselines are proposed: it passes.
    Seeded: python-core's ``X@postgres`` baseline missing fails, naming
    ``X@postgres`` and python-core."""
    landed = git(recorded, 'rev-parse', 'HEAD')
    files = _observe(recorded, tmp_path, keep=(X, SQLITE, POSTGRES))
    both = (SQLITE, POSTGRES)
    commit(
        recorded,
        'T2.0f: the twin',
        _propose(files, {'python-core': both, 'python-old': both}),
    )
    assert helpers.check(recorded, landed, files) == []
    git(recorded, 'reset', '-q', '--hard', landed)
    commit(
        recorded,
        'T2.0f: the seed',
        _propose(files, {'python-core': (SQLITE,), 'python-old': both}),
    )
    found = helpers.check(recorded, landed, files)
    assert found, 'a core observation on Postgres with no baseline passed'
    assert _naming(found, POSTGRES) == found, found
    assert any('python-core' in p for p in found), found
    assert not _naming(found, SQLITE), found


# --- removing a case removes its database lines ---------------------------------


def _remove_x(
    leave_reference: tuple[str, ...] = (),
    leave_baseline: tuple[tuple[str, str], ...] = (),
) -> Callable[[Path], None]:
    r"""Removes X's case file and every line keyed X, ``X@sqlite`` or
    ``X@postgres`` from the reference and the baselines, but those left;
    and adds the approved entry for X's removal on every runner."""

    def edit(tree: Path) -> None:
        (tree / f'contract/cases/s0/{X}.json').unlink()
        keys = {X, SQLITE, POSTGRES}
        path = _reference_file(tree)
        _write_jsonl(
            path,
            [
                x
                for x in _jsonl(path)
                if x['case'] not in keys or x['case'] in leave_reference
            ],
        )
        items = []
        for runner in RUNNERS:
            path = _baseline_file(tree, runner)
            lines = _jsonl(path)
            bare = next(x for x in lines if x['case'] == X)
            items.append(
                helpers.outcome(
                    X,
                    runner,
                    ratchet.outcome_text(bare['projection']),
                    'removed',
                )
            )
            _write_jsonl(
                path,
                [
                    x
                    for x in lines
                    if x['case'] not in keys
                    or (runner, x['case']) in leave_baseline
                ],
            )
        helpers.append_entry(
            tree, {'id': 'SEED-1', 'cases': ['T2.0f'], 'case_outcomes': items}
        )

    return edit


@pytest.mark.parametrize(
    'leave',
    [
        pytest.param({'leave_reference': (POSTGRES,)}, id='reference-line'),
        pytest.param(
            {'leave_baseline': (('python-core', POSTGRES),)}, id='baseline'
        ),
    ],
)
def test_an_approved_removal_removes_the_case_s_database_lines(
    held: Path, tmp_path: Path, leave: dict[str, Any]
) -> None:
    r"""Exit: removing a case removes its database lines. Twin: X removed
    under an approved entry, with its reference lines and baselines on
    every database: it passes. Seeded: the same removal leaving
    ``X@postgres``'s reference line, or python-core's ``X@postgres``
    baseline, fails, naming ``X@postgres``."""
    landed = git(held, 'rev-parse', 'HEAD')
    files = _observe(held, tmp_path, keep=())
    commit(held, 'T2.0f: the twin', _remove_x())
    twin = helpers.check(held, landed, files)
    assert twin == [], '\n'.join(twin)
    git(held, 'reset', '-q', '--hard', landed)
    commit(held, 'T2.0f: the seed', _remove_x(**leave))
    found = helpers.check(held, landed, files)
    assert found, 'a removal leaving a database line passed'
    assert _naming(found, POSTGRES) == found, found


# --- ratchet.py accept writes the database baselines ----------------------------


def test_accept_writes_the_database_baselines_check_then_needs(
    recorded: Path, tmp_path: Path
) -> None:
    r"""Review round 1: python-core and python-old observe X on DuckDB,
    SQLite and Postgres, matching the reference; ``ratchet.py accept``
    (the command line) writes their ``X@sqlite`` and ``X@postgres``
    baselines beside ``X``'s, and ``check`` on that commit passes."""
    landed = git(recorded, 'rev-parse', 'HEAD')
    files = _observe(recorded, tmp_path, keep=(X, SQLITE, POSTGRES))
    commit(recorded, 'T2.0f: a change')
    given = [x for path in files for x in ('--observations', str(path))]
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / 'scripts/rust_core/ratchet.py'),
            'accept',
            '--repo',
            str(recorded),
            '--checkpoint',
            landed,
            *given,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    for runner in ('python-core', 'python-old'):
        written = {x['case'] for x in _jsonl(_baseline_file(recorded, runner))}
        assert {X, SQLITE, POSTGRES} <= written, (runner, sorted(written))
    commit(recorded, 'T2.0f: the baselines accept wrote')
    found = helpers.check(recorded, landed, files)
    assert found == [], '\n'.join(found)


# --- an entry approves a database line --------------------------------------------


def _replace_baseline(
    files: tuple[Path, ...], runner: str, key: str
) -> Callable[[Path], None]:
    r"""``runner``'s proposed baseline of ``key`` replaced by its
    observation's projection (as ``ratchet.py accept`` writes it)."""
    observation = _observed(files)[runner][key]

    def edit(tree: Path) -> None:
        projection = ratchet.projection(observation, _fields(tree))
        path = _baseline_file(tree, runner)
        lines = _jsonl(path)
        line = next(x for x in lines if x['case'] == key)
        line.update(projection=projection, digest=ratchet.digest(projection))
        _write_jsonl(path, lines)

    return edit


def _suggested(problem: str) -> dict[str, Any]:
    r"""The ``case_outcomes`` item the ratchet's message suggests, its
    ``summary`` filled in."""
    found = yaml.safe_load(problem[problem.index('{') :].replace('...', 'x'))
    assert isinstance(found, dict), problem
    found['summary'] = 'Seeded by the T2.0f tests.'
    return found


def _moved_on_postgres(
    held: Path, tmp_path: Path
) -> tuple[str, tuple[Path, ...], dict[str, Any]]:
    r"""python-old's ``X@postgres`` observation moved to predict: the
    checkpoint, the observations, and the item the ratchet suggests for
    the move, unapproved (the worktree back at the checkpoint)."""
    landed = git(held, 'rev-parse', 'HEAD')

    def moved(lines: dict[str, Any]) -> None:
        lines[POSTGRES]['outcome']['stage'] = 'predict'

    files = (
        helpers.observe(
            held,
            tmp_path,
            'python-old',
            _on_databases((SQLITE, POSTGRES), moved),
        ),
    )
    commit(held, 'T2.0f: no entry')
    found = helpers.check(held, landed, files)
    assert found, 'an unapproved move of a database baseline passed'
    assert _naming(found, POSTGRES) == found, found
    suggesting = [p for p in found if '{' in p]
    assert len(suggesting) == 1, found
    git(held, 'reset', '-q', '--hard', landed)
    return landed, files, _suggested(suggesting[0])


def test_the_suggested_entry_approves_a_database_baseline_move(
    held: Path, tmp_path: Path
) -> None:
    r"""Review round 1: the item the ratchet suggests for a moved
    ``X@postgres`` baseline is one the register accepts, and an approved
    entry holding it lets the move pass (the proposed baseline the new
    observation)."""
    landed, files, item = _moved_on_postgres(held, tmp_path)

    def approve(tree: Path) -> None:
        helpers.append_entry(
            tree, {'id': 'SEED-1', 'cases': [X], 'case_outcomes': [item]}
        )
        _replace_baseline(files, 'python-old', POSTGRES)(tree)

    commit(held, 'T2.0f: the move approved', approve)
    found = helpers.check(held, landed, files)
    assert found == [], '\n'.join(found)


def test_an_entry_in_the_same_form_approves_a_changed_database_reference_line(
    held: Path, tmp_path: Path
) -> None:
    r"""Review round 1: a changed ``X@postgres`` reference line fails with
    no entry, naming it; an entry in the suggested form, for python-ref
    with the line's old and new outcome, approves it."""
    landed, _, item = _moved_on_postgres(held, tmp_path)
    fields = _fields(held)
    before = next(
        x for x in _jsonl(_reference_file(held)) if x['case'] == POSTGRES
    )
    after = copy.deepcopy(before)
    after['outcome']['stage'] = 'predict'

    def change(tree: Path) -> None:
        path = _reference_file(tree)
        _write_jsonl(
            path,
            [after if x['case'] == POSTGRES else x for x in _jsonl(path)],
        )

    commit(held, 'T2.0f: the line changed', change)
    found = helpers.check(held, landed)
    assert found, 'a changed database reference line passed'
    assert _naming(found, POSTGRES) == found, found
    git(held, 'reset', '-q', '--hard', landed)
    reference = {
        **item,
        'runner': 'python-ref',
        'old': ratchet.outcome_text(ratchet.projection(before, fields)),
        'new': ratchet.outcome_text(ratchet.projection(after, fields)),
    }

    def approve(tree: Path) -> None:
        change(tree)
        helpers.append_entry(
            tree, {'id': 'SEED-1', 'cases': [X], 'case_outcomes': [reference]}
        )

    commit(held, 'T2.0f: the change approved', approve)
    found = helpers.check(held, landed)
    assert found == [], '\n'.join(found)


# --- after the rebase onto T2.0e: a re-recording drops removed lines --------------

#: The re-recorded reference's fake SHA, and T2.0e's approval of it as
#: T2.0e's tests assume it (tests/test_t20e_reference_change.py).
NEW = '5a' * 20
DECISIONS = 'docs/design/rust-core/phase-2/decisions.md'
HEADER = (
    '# Phase 2 decisions (seeded by the T2.0f tests)\n\n'
    '| # | Decision | Reason | Status |\n|---|---|---|---|\n'
)
ROW = (
    f'| P2H-901 | The reference moves to `{NEW}`: its recording replaces '
    'the recorded one | The owner moved the reference | approved by owner '
    '2026-10-16 |\n'
)


def _citing(root: Path, sha: str) -> list[str]:
    r"""The tracked files citing ``sha`` in full, but ``docs/`` and the
    recordings."""
    done = subprocess.run(
        ['git', '-C', str(root), 'grep', '-l', '-F', sha],
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(
        name
        for name in done.stdout.splitlines()
        if not name.startswith(('docs/', 'contract/reference/'))
    )


def _approved_new_reference(tree: Path) -> str:
    r"""A landing holding every file of this checkout that cites the
    reference SHA, and the owner's row approving ``NEW``."""
    old = _reference_file(tree).stem
    wanted = [n for n in _citing(ROOT, old) if not (tree / n).exists()]

    def edit(t: Path) -> None:
        for name in wanted:
            (t / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, t / name)
        path = t / DECISIONS
        path.parent.mkdir(parents=True, exist_ok=True)
        before = path.read_text() if path.is_file() else HEADER
        path.write_text(before + ROW)

    landed: str = land(tree, 'T2.0d', edit, protected='contract/** docs/**')
    return landed


def _rerecord(drop: tuple[str, ...] = ()) -> Callable[[Path], None]:
    r"""The reference re-recorded at ``NEW``, less the lines ``drop``, every
    citation of the old SHA outside ``docs/`` moved."""

    def edit(tree: Path) -> None:
        source = _reference_file(tree)
        old = source.stem
        lines = [
            json.loads(x.replace(old, NEW))
            for x in source.read_text().splitlines()
            if x.strip()
        ]
        source.unlink()
        _write_jsonl(
            source.with_name(f'{NEW}.jsonl'),
            [x for x in lines if x['case'] not in drop],
        )
        for name in _citing(tree, old):
            path = tree / name
            path.write_text(path.read_text().replace(old, NEW))
        env = tree / 'contract/reference-env.txt'
        assert re.search(rf'(?m)^reference: {NEW}$', env.read_text())

    return edit


def test_an_approved_rerecord_drops_a_removed_case_s_database_lines(
    held: Path,
) -> None:
    r"""Review round 1, after the rebase onto T2.0e: the owner approves
    the new reference; the candidate removes X under an approved entry
    (its file, baselines and every line keyed X, ``X@sqlite`` or
    ``X@postgres``) and re-records the reference without them: it passes.
    Seeded: X kept, the re-recording dropping its ``X@sqlite`` line fails,
    naming it."""
    checkpoint = _approved_new_reference(held)

    def both(tree: Path) -> None:
        _remove_x()(tree)
        _rerecord()(tree)

    commit(held, 'T2.0c: the reference re-recorded, X removed', both)
    found = ratchet.check(held, checkpoint, [], None).problems
    assert found == [], '\n'.join(found)
    git(held, 'reset', '-q', '--hard', checkpoint)
    commit(held, 'T2.0c: the reference re-recorded', _rerecord(drop=(SQLITE,)))
    found = ratchet.check(held, checkpoint, [], None).problems
    assert found, 'a kept case re-recorded without its SQLite line passed'
    assert _naming(found, SQLITE), found
