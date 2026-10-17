# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's acceptance tests for the rule that stages the core on a database
(the independent test author's).

The design (choice 2, "CI"): one rule, defined once in T2.1a's runner and
used by both ``gate.sh`` and CI: the core legs (python-core and rust-host;
the Python provider kit once T2.5 lands) run on a database exactly when the
core under test reports a provider for it, through a private, plain-data
marker (a list of database names in the private native module for
python-core, and the test kit's provider list for rust-host), and
otherwise the reference side only (the old client over the loaded database,
checked against the committed reference lines), so a job still runs cases.
A job that runs no case on its database fails. The row's controls: no
provider runs the reference side only and still runs cases; a provider
present runs the core legs; a provider present with a core leg missing or
skipped fails by name.

The rule is held here as a pure function over plain data, the providers
each leg reports and the observations a job made, so it is judged without
building a core per database.

Gating: :data:`PENDING`, a strict xfail; the implementer deletes its one
definition. Unmarked tests are controls that pass today.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: ``run_python.check_legs(database, providers, observations) -> list[str]``:
#: ``providers`` maps each core leg (``python-core``, ``rust-host``) to the
#: databases it reports a provider for; ``observations`` are a job's
#: observations on ``database`` (python-old's and any core leg's, keyed by
#: the bare case ID on DuckDB and ``<case>@<database>`` elsewhere). The
#: problems, each naming what fails: no case observed on the database; a
#: leg reporting a provider for it with no observation there (missing), or
#: without one of the cases the reference side observed there (skipped,
#: naming the case's key). A leg reporting no provider for it is not
#: required. No problem: an empty list.
#: ``sdm_client._native.PROVIDERS``: the databases python-core reports, a
#: tuple of names.
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


RUNNER = _module(
    ROOT / 'contract/runners/run_python.py', 't21a_rule_run_python'
)

CASES = ('s0.accept.sum-window', 's0.accept.binary-classification')
ONLY_DUCKDB = {'python-core': ('duckdb',), 'rust-host': ('duckdb',)}
SQLITE_BOTH = {
    'python-core': ('duckdb', 'sqlite'),
    'rust-host': ('duckdb', 'sqlite'),
}


def _reference_line(case: str) -> dict[str, Any]:
    path = next((ROOT / 'contract/reference').glob('*.jsonl'))
    for line in path.read_text().splitlines():
        found: dict[str, Any] = json.loads(line)
        if found['case'] == case:
            return found
    raise AssertionError(f'no reference line of {case}')


def _observed(
    runner: str, database: str, cases: tuple[str, ...] = CASES
) -> list[dict[str, Any]]:
    lines = []
    for case in cases:
        line = copy.deepcopy(_reference_line(case))
        line['identity']['runner'] = runner
        line['identity']['reference'] = None
        line['case'] = case if database == 'duckdb' else f'{case}@{database}'
        lines.append(line)
    return lines


def _named(problems: list[str], *words: str) -> bool:
    return any(all(word in problem for word in words) for problem in problems)


def test_the_seeded_observations_are_valid() -> None:
    r"""Control: the observations the rule is given are ones the runner's
    validator takes, on every database's key."""
    validate = RUNNER.validator('observation')
    for database in ('duckdb', 'sqlite', 'postgres'):
        for line in _observed('python-core', database):
            assert RUNNER._invalid(validate, line) is None


def test_no_provider_runs_the_reference_side_only_and_still_runs_cases() -> (
    None
):
    r"""Control: with no core provider for SQLite, the old client's cases
    alone pass."""
    assert (
        RUNNER.check_legs(
            'sqlite', ONLY_DUCKDB, _observed('python-old', 'sqlite')
        )
        == []
    )


def test_a_database_job_that_runs_no_case_fails() -> None:
    r"""Seeded: a job on its database with no case observed fails, naming
    the database, provider or not."""
    for providers in (ONLY_DUCKDB, SQLITE_BOTH):
        problems = RUNNER.check_legs('sqlite', providers, [])
        assert problems and _named(problems, 'sqlite'), problems


def test_a_provider_present_runs_the_core_legs() -> None:
    r"""Control: a provider for SQLite on both legs, and both legs observed
    on every case there, passes."""
    observations = [
        *_observed('python-old', 'sqlite'),
        *_observed('python-core', 'sqlite'),
        *_observed('rust-host', 'sqlite'),
    ]
    assert RUNNER.check_legs('sqlite', SQLITE_BOTH, observations) == []


@pytest.mark.parametrize('leg', ['python-core', 'rust-host'])
def test_a_provider_present_with_its_leg_missing_fails_by_name(
    leg: str,
) -> None:
    r"""Seeded: a leg that reports a provider for SQLite and made no
    observation there fails, naming the leg and the database."""
    other = 'rust-host' if leg == 'python-core' else 'python-core'
    observations = [
        *_observed('python-old', 'sqlite'),
        *_observed(other, 'sqlite'),
    ]
    problems = RUNNER.check_legs('sqlite', SQLITE_BOTH, observations)
    assert _named(problems, leg, 'sqlite'), problems
    assert not _named(problems, other), problems


def test_a_provider_present_with_a_case_skipped_fails_by_name() -> None:
    r"""Seeded: python-core observed one of the two cases on SQLite; the
    skipped one is named by its key."""
    skipped = f'{CASES[1]}@sqlite'
    observations = [
        *_observed('python-old', 'sqlite'),
        *_observed('python-core', 'sqlite', CASES[:1]),
        *_observed('rust-host', 'sqlite'),
    ]
    problems = RUNNER.check_legs('sqlite', SQLITE_BOTH, observations)
    assert _named(problems, 'python-core', skipped), problems


def test_observations_on_another_database_do_not_stand_in() -> None:
    r"""Seeded: python-core's DuckDB observations (bare keys) do not count
    as its SQLite leg."""
    observations = [
        *_observed('python-old', 'sqlite'),
        *_observed('python-core', 'duckdb'),
        *_observed('rust-host', 'sqlite'),
    ]
    problems = RUNNER.check_legs('sqlite', SQLITE_BOTH, observations)
    assert _named(problems, 'python-core', 'sqlite'), problems


def test_duckdb_keeps_bare_keys_and_needs_both_legs() -> None:
    r"""Control and seed on DuckDB, where every core leg reports a provider
    today: both legs observed passes; rust-host missing fails."""
    old, core, host = (
        _observed(runner, 'duckdb')
        for runner in ('python-old', 'python-core', 'rust-host')
    )
    assert RUNNER.check_legs('duckdb', ONLY_DUCKDB, [*old, *core, *host]) == []
    problems = RUNNER.check_legs('duckdb', ONLY_DUCKDB, [*old, *core])
    assert _named(problems, 'rust-host', 'duckdb'), problems


def test_the_built_core_reports_its_providers_as_plain_data() -> None:
    r"""Exit: python-core's marker is a private list of database names in
    the native module, DuckDB among them; never public API."""
    import sdm_client
    import sdm_client._native as native

    providers = native.PROVIDERS
    assert isinstance(providers, tuple)
    assert all(isinstance(name, str) for name in providers)
    assert 'duckdb' in providers
    assert set(providers) <= {'duckdb', 'sqlite', 'postgres'}
    assert not hasattr(sdm_client, 'PROVIDERS')
