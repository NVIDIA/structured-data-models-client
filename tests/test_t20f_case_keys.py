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
``tests/test_t20f_gate.py``, through the gate).

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
  (copies of ``X``'s line, ``X`` an S0 acceptance case baselined on every
  runner), and no baseline of them yet;
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
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

#: The marker the implementer deletes (see the module docstring).
PENDING = pytest.mark.xfail(
    strict=True,
    reason='T2.0f: the <case>@<database> keys are not implemented yet',
)

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


@PENDING
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
    ``X``'s, keyed by database."""
    path = _reference_file(tree)
    lines = _jsonl(path)
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
def origin(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    repo, _ = helpers.build(tmp_path_factory.mktemp('origin'))
    recorded = land(
        repo, 'T2.1a', _record_database_lines, protected='contract/**'
    )
    held = land(repo, 'T2.2', _hold_database_baselines, protected='contract/**')
    return {'repo': repo, 'recorded': recorded, 'held': held}


def _worktree(origin: dict[str, Any], tmp_path: Path, at: str) -> Path:
    tree = tmp_path / 'tree'
    git(origin['repo'], 'worktree', 'add', '-q', '--detach', str(tree), at)
    return tree


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


def _observe(
    tree: Path, tmp_path: Path, keep: tuple[str, ...]
) -> tuple[Path, ...]:
    r"""python-core's and python-old's observations, made from the
    reference's (``test_contract_ratchet.observe``), keeping of X's lines
    only ``keep``."""
    mine = {X, SQLITE, POSTGRES} - set(keep)

    def drop(lines: dict[str, Any]) -> None:
        for key in mine:
            lines.pop(key, None)

    return tuple(
        helpers.observe(tree, tmp_path, runner, drop)
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


@PENDING
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


@PENDING
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
