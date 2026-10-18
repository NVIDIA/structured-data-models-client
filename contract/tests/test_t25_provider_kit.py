# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.5's and T2.6's exits on the Python host's providers (the package's
test author's): the provider conformance kit and the in-memory fallback.

    python -m pytest contract/tests/test_t25_provider_kit.py -q

The kit's data is ``contract/fixtures/provider-kit/``: ``dataset.sql``,
the one fixture dataset, and ``expected.json``, the cases and their
expected answers (some hand-checked, marked ``checked: hand``), both read
by this file and by ``sdm-testkit``'s ``tests/t25_provider_kit.rs``, which
holds ``rust-host``'s providers to the same exits. ``record.py`` beside
them recomputes every answer from the dataset in plain Python.

The interface assumed, the kit's runner on this host:

    python -I contract/runners/provider_kit.py --database DATABASE \
        [--fault truncate|silent_capability] --out REPORT

It builds the dataset on DATABASE (DuckDB from ``dataset.sql``; SQLite and
Postgres loaded from it by the one contract loader, Postgres at
``SDM_POSTGRES_TEST_DSN``), runs each case of ``expected.json`` through the
core on that database's Python provider (``sdm_client._rust.providers``)
and the Python run loop, as ``expected.json``'s ``about`` says (its
``page_rows``, ``seeds``, ``capabilities`` and ``inject``), with the fault
named seeded in the provider path as ``expected.json``'s ``faults`` say,
and writes REPORT, JSON::

    {"format": "sdm-provider-kit/1", "host": "python-core",
     "database": DATABASE, "fault": null | FAULT,
     "cases": [{"id": ..., "fetched": [<request name>, ...],
                "answer": {"columns": [...], "rows": [[...], ...]}
                | "error": {"code": ..., "kind": ..., "retryable": ...,
                            "fields": {<name>: <value>, ...}}}, ...]}

``fetched`` names the request of every fetch the provider path answered or
failed for the case, one per page, in order; ``answer`` is what the core
admitted, its values as JSON (numbers, text, null); ``error`` is the
core's error the case ended with, its fields by name (each a JSON value,
or the contract's scalar as it serializes). The runner exits 0
when every case passes its own judgment, 1 when any case fails, and
otherwise when it could not run. This file judges the report itself too.

Each test's one line says the exit it proves:

- T2.5, "every provider landed by then passes": the kit passes on each
  database's Python provider: paging (each answer of more than
  ``page_rows`` rows came in that many pages or more), limits, the
  expected answers, capability failure (a declared request that fails is
  ``PROVIDER_ERROR``, with or without a fallback declared; an undeclared
  one with no fallback ``UNSUPPORTED_REQUEST`` naming it) and retryability
  (a page failing after one that answered is an error, retryable as the
  provider said, either way);
- T2.6, "kit cases with the capability undeclared give the pushdown's
  answer": each ``same_as`` case, its request withheld and the fallback
  declared, answers as the pushdown case does, and the provider was never
  asked the withheld request;
- T2.6, "a limit exceeded is a named error": past ``max_rows`` or
  ``max_bytes``, ``UNSUPPORTED_REQUEST`` naming the request, rule
  ``fallback_limit`` and the limit, where the same request within its
  limits answers;
- T2.6, round 1's regressions: a fallback scan past ``max_bytes`` on a
  page stops there with the limit's error, asking no later page, even of
  a provider that would fail it; and a float key's -0.0 and 0.0 are one
  value through the fallback as through pushdown (``key_counts``,
  ``repeated_key``);
- T2.5, "a seeded truncation ... in each host's provider path ... fails
  the kit": with ``--fault truncate`` the runner exits 1 and each case the
  fault names answered short, with no error;
- T2.5, "a seeded silent capability failure ... fails the kit": with
  ``--fault silent_capability`` the runner exits 1, each neighbors case
  answered with no rows and no error, and every other case still passes.

Controls, passing before the implementation: the committed answers are
the dataset's (``record.py``), the cases are well formed and each fault
names cases it must change, and the dataset loads into SQLite through
the contract loader.

Postgres runs where ``SDM_POSTGRES_TEST_DSN`` names a server, and fails,
never stands down, where ``SDM_REQUIRE_POSTGRES`` asks for one (H26).

Gating: ``t25-pending.json``'s ``pending`` names, as ``py:<test>`` (with
``[<database>]``), the tests that failed when written. Each must still
fail (strict); once one passes it fails, naming itself, until it is taken
out. ``pending`` stays within ``authored``. Nothing is skipped.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
KIT_DIR = ROOT / 'contract/fixtures/provider-kit'
KIT = ROOT / 'contract/runners/provider_kit.py'
PENDING = Path(__file__).with_name('t25-pending.json')
DATABASES = ('duckdb', 'sqlite', 'postgres')
FAULTS = ('truncate', 'silent_capability')
OPTIONAL = (
    'preview',
    'count_rows',
    'key_counts',
    'time_bounds',
    'column_ranges',
    'repeated_key',
    'sample_entities',
    'neighbors',
    'lookup',
)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


EXPECTED: dict[str, Any] = json.loads(
    (KIT_DIR / 'expected.json').read_text(encoding='utf-8')
)
BY_ID: dict[str, dict[str, Any]] = {c['id']: c for c in EXPECTED['cases']}
PAGE_ROWS: int = EXPECTED['page_rows']


def held(name: str, check: Callable[[], None]) -> None:
    r"""Runs ``check``: it must pass, or, while ``py:<name>`` is pending,
    fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    if f'py:{name}' not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take py:{name} out of '
        'contract/tests/t25-pending.json pending'
    )


def reachable(database: str) -> bool:
    r"""Whether ``database`` can be run here: DuckDB and SQLite always;
    Postgres where ``SDM_POSTGRES_TEST_DSN`` names a server, and a failure
    where ``SDM_REQUIRE_POSTGRES`` asks for one that is not named (H26)."""
    if database != 'postgres' or os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


# --- the cases, resolved ------------------------------------------------------


def request_case(case: dict[str, Any]) -> dict[str, Any]:
    r"""The case whose request and seeds ``case`` runs."""
    named = case.get('same_as') or case.get('request_of')
    return BY_ID[named] if named else case


def wanted_answer(case: dict[str, Any]) -> dict[str, Any] | None:
    r"""The answer ``case`` expects, its own or its ``same_as`` case's."""
    if 'same_as' in case:
        return BY_ID[case['same_as']]['answer']
    return case.get('answer')


def wanted_rows(answer: dict[str, Any]) -> int:
    return answer['count'] if 'any_of' in answer else len(answer['rows'])


def withheld(case: dict[str, Any]) -> list[str]:
    return list(case.get('capabilities', {}).get('withhold', []))


# --- the judgment -------------------------------------------------------------


def _value(value: Any) -> Any:
    r"""A value as compared: a number by its value, a zero of either sign
    one value (round 1), anything else as it is."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) + 0.0
    return value


def fetched_problem(case: dict[str, Any], fetched: list[str]) -> str | None:
    r"""Where ``case`` names its fetches exactly, how ``fetched`` differs."""
    if 'fetched' in case and fetched != case['fetched']:
        return f'fetched {fetched}, not exactly {case["fetched"]}'
    return None


def _key(row: list[Any]) -> str:
    return json.dumps([_value(v) for v in row])


def _projected(
    got: dict[str, Any], columns: list[str]
) -> list[list[Any]] | str:
    r"""``got``'s rows, the listed columns in order, or why not."""
    names = list(got.get('columns', []))
    missing = [c for c in columns if c not in names]
    if missing:
        return f'no column {missing} among {names}'
    at = [names.index(c) for c in columns]
    return [[_value(row[i]) for i in at] for row in got.get('rows', [])]


def answer_problem(got: dict[str, Any], wanted: dict[str, Any]) -> str | None:
    r"""How ``got`` differs from the answer ``wanted``, by
    ``expected.json``'s rules; ``None`` when it does not."""
    rows = _projected(got, wanted['columns'])
    if isinstance(rows, str):
        return rows
    if 'any_of' in wanted:
        pool = sorted(_key(r) for r in wanted['any_of'])
        if len(rows) != wanted['count']:
            return f'{len(rows)} rows, not {wanted["count"]}'
        for row in rows:
            if _key(row) not in pool:
                return f'{row} is not one of the rows it may be'
            pool.remove(_key(row))
        return None
    want = [[_value(v) for v in row] for row in wanted['rows']]
    if wanted['ordered']:
        if rows != want:
            return f'rows {rows}, not {want} in order'
    elif sorted(map(_key, rows)) != sorted(map(_key, want)):
        return f'rows {rows}, not the multiset {want}'
    return None


def error_problem(got: dict[str, Any], wanted: dict[str, Any]) -> str | None:
    for name in ('code', 'kind', 'retryable'):
        if got.get(name) != wanted[name]:
            return f'{name} {got.get(name)!r}, not {wanted[name]!r}'
    fields = got.get('fields') or {}
    for name, value in wanted['fields'].items():
        if _text(fields.get(name)) != str(value):
            return f'field {name} {fields.get(name)!r}, not {value!r}'
    return None


def _text(detail: Any) -> str:
    r"""An error field's value as text: the value itself, or the value of
    the contract's scalar as it serializes (``{"t": ..., "v": ...}``)."""
    if isinstance(detail, dict) and 'v' in detail:
        return str(detail['v'])
    return str(detail)


def judge(case: dict[str, Any], seen: dict[str, Any] | None) -> list[str]:
    r"""Why the report's ``seen`` does not pass ``case``; empty when it
    does."""
    if seen is None:
        return ['not reported']
    fetched = list(seen.get('fetched', []))
    wanted = wanted_answer(case)
    if wanted is not None:
        if 'error' in seen or 'answer' not in seen:
            return [f'an error, {seen.get("error")}, not an answer']
        problem = answer_problem(seen['answer'], wanted)
        if problem:
            return [problem]
        if 'fetched' in case:
            return [p for p in [fetched_problem(case, fetched)] if p]
        name = request_case(case)['request']['request']
        problems = []
        held_back = withheld(case)
        if held_back:
            # The fallback answered: the provider was asked only what it
            # declares, and something.
            if not fetched or set(fetched) & set(held_back):
                problems.append(
                    f'fetched {fetched}: the fallback asks none of '
                    f'{held_back}, and asks something'
                )
        else:
            pages = math.ceil(wanted_rows(wanted) / PAGE_ROWS)
            if fetched.count(name) < pages:
                problems.append(
                    f'fetched {fetched}: {wanted_rows(wanted)} rows come '
                    f'in {pages} pages of {PAGE_ROWS} or more'
                )
        return problems
    if 'answer' in seen or 'error' not in seen:
        return [f'an answer, not the error {case["error"]}']
    problem = error_problem(seen['error'], case['error'])
    if problem:
        return [problem]
    if 'fetched' in case:
        return [p for p in [fetched_problem(case, fetched)] if p]
    injected = case.get('inject')
    if injected and len(fetched) < injected['page']:
        return [f'fetched {fetched}: page {injected["page"]} was never asked']
    return []


# --- the kit's runs -----------------------------------------------------------

_OUT = Path(tempfile.mkdtemp(prefix='t25-provider-kit-'))


@functools.cache
def kit(database: str, fault: str | None = None) -> tuple[int, dict[str, Any]]:
    r"""The kit's runner on ``database``, with ``fault`` seeded: its exit
    status and report. Fails when there is no runner or no report."""
    assert KIT.is_file(), f'no kit runner: {KIT.relative_to(ROOT)}'
    out = _OUT / f'{database}-{fault}.json'
    command = [sys.executable, '-I', str(KIT), '--database', database]
    if fault:
        command += ['--fault', fault]
    done = subprocess.run(
        [*command, '--out', str(out)],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert out.is_file(), (
        f'the kit wrote no report (exit {done.returncode}): '
        f'{done.stdout[-2000:]}{done.stderr[-2000:]}'
    )
    report = json.loads(out.read_text(encoding='utf-8'))
    assert report.get('format') == EXPECTED['format'], report.get('format')
    assert report.get('host') == 'python-core', report.get('host')
    assert report.get('database') == database, report.get('database')
    assert report.get('fault') == fault, report.get('fault')
    return done.returncode, report


def by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {seen['id']: seen for seen in report.get('cases', [])}


def judged(report: dict[str, Any]) -> dict[str, list[str]]:
    r"""Each case's problems in ``report``, by case."""
    seen = by_id(report)
    return {c['id']: judge(c, seen.get(c['id'])) for c in EXPECTED['cases']}


# --- controls -----------------------------------------------------------------


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only tests the test author wrote."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])


def test_the_expected_answers_are_the_dataset_s() -> None:
    r"""Control: every committed answer is the one ``record.py`` computes
    from the dataset, and each says how it was checked (``hand``, with why,
    or ``recorded``), some by hand."""
    record = _module('provider_kit_record', KIT_DIR / 'record.py')
    assert record.check() == []
    answered = [c for c in EXPECTED['cases'] if 'answer' in c]
    assert {c['checked'] for c in answered} == {'hand', 'recorded'}
    assert all(c.get('why') for c in answered if c['checked'] == 'hand')


def test_the_cases_are_well_formed_and_each_fault_bites() -> None:
    r"""Control: ids are unique; every case runs a request and expects one
    answer or one error; ``same_as`` and ``request_of`` name a case with a
    request; a ``same_as`` case withholds its request and declares the
    fallback; each truncation case has more rows than a page, and each
    silent capability case is a pushdown ``neighbors`` with rows to lose."""
    ids = [c['id'] for c in EXPECTED['cases']]
    assert len(ids) == len(set(ids))
    for case in EXPECTED['cases']:
        source = request_case(case)
        assert 'request' in source, case['id']
        assert (wanted_answer(case) is None) != ('error' not in case), case
        for name in withheld(case):
            assert name in OPTIONAL, (case['id'], name)
        if 'same_as' in case:
            assert 'request' in BY_ID[case['same_as']]
            kind = source['request']['request']
            assert withheld(case) == [kind], case['id']
            assert 'fallback' in case['capabilities'], case['id']
        if 'request' in source and source['request']['request'] in (
            'neighbors',
            'lookup',
        ):
            assert source.get('seeds'), case['id']
    faults = {f['name']: f for f in EXPECTED['faults']}
    assert set(faults) == set(FAULTS)
    for name in faults['truncate']['fails']:
        case = BY_ID[name]
        assert not withheld(case) and 'inject' not in case, name
        assert wanted_rows(case['answer']) > PAGE_ROWS, name
    for name in faults['silent_capability']['fails']:
        case = BY_ID[name]
        assert case['request']['request'] == 'neighbors', name
        assert not withheld(case) and case['answer']['rows'], name
    # The fallback's limits fail where the same request within its
    # limits answers.
    for name in ('fallback.rows-exceeded', 'fallback.bytes-exceeded'):
        assert BY_ID[name]['request_of'] == 'neighbors.ordered'
    assert BY_ID['fallback.neighbors-ordered']['same_as'] == 'neighbors.ordered'


def test_the_dataset_loads_into_sqlite_through_the_contract_loader(
    tmp_path: Path,
) -> None:
    r"""Control: the one contract loader loads the fixture into SQLite with
    every row as DuckDB holds it, so the kit's SQLite runs read the same
    data."""
    import sqlite3

    import duckdb

    loader = _module('t25_databases', ROOT / 'contract/runners/databases.py')
    record = _module('provider_kit_record', KIT_DIR / 'record.py')
    duck = duckdb.connect()
    duck.execute((KIT_DIR / 'dataset.sql').read_text(encoding='utf-8'))
    target = tmp_path / 'kit.sqlite'
    loader.load(duck, 'sqlite', target)
    duck.close()
    with sqlite3.connect(target) as sqlite:
        for table, (columns, rows) in record.tables().items():
            quoted = ', '.join(f'"{c}"' for c in columns)
            got = sqlite.execute(f'SELECT {quoted} FROM "{table}"').fetchall()
            assert sorted(map(_key, map(list, got))) == sorted(
                map(_key, map(list, rows))
            ), table


# --- T2.5: the kit on each Python provider ------------------------------------


@pytest.mark.parametrize('database', DATABASES)
def test_the_kit_passes_on_each_python_provider(database: str) -> None:
    r"""T2.5: every case passes on ``database``'s Python provider: paging,
    limits, the expected answers, capability failure and retryability, the
    runner exiting 0 and every case judged here too."""
    if not reachable(database):
        return

    def check() -> None:
        status, report = kit(database)
        problems = {k: v for k, v in judged(report).items() if v}
        assert not problems, problems
        assert set(by_id(report)) == set(BY_ID), sorted(
            set(by_id(report)) ^ set(BY_ID)
        )
        assert status == 0, f'the kit exited {status}'

    held(f'test_the_kit_passes_on_each_python_provider[{database}]', check)


# --- T2.6: the in-memory fallback ---------------------------------------------


@pytest.mark.parametrize('database', DATABASES)
def test_the_fallback_gives_the_pushdown_s_answer(database: str) -> None:
    r"""T2.6: each case with its request undeclared and the fallback
    declared answers as the pushdown case does, the provider never asked
    the request withheld."""
    if not reachable(database):
        return

    def check() -> None:
        _, report = kit(database)
        problems = judged(report)
        fallback = [c['id'] for c in EXPECTED['cases'] if 'same_as' in c]
        assert fallback
        assert not {k: problems[k] for k in fallback if problems[k]}
        # The pushdown answers they are held to pass too.
        pushdown = {BY_ID[k]['same_as'] for k in fallback}
        assert not {k: problems[k] for k in pushdown if problems[k]}

    held(f'test_the_fallback_gives_the_pushdown_s_answer[{database}]', check)


@pytest.mark.parametrize('database', DATABASES)
def test_a_fallback_limit_exceeded_is_a_named_error(database: str) -> None:
    r"""T2.6: past the fallback's declared rows or bytes, the request fails
    by name (``UNSUPPORTED_REQUEST``, rule ``fallback_limit``, the limit),
    where the same request within its limits answers."""
    if not reachable(database):
        return

    def check() -> None:
        _, report = kit(database)
        problems = judged(report)
        named = (
            'fallback.rows-exceeded',
            'fallback.bytes-exceeded',
            'fallback.neighbors-ordered',
        )
        assert not {k: problems[k] for k in named if problems[k]}

    held(f'test_a_fallback_limit_exceeded_is_a_named_error[{database}]', check)


# --- round 1's regressions -----------------------------------------------------


def _cases_pass(database: str, names: tuple[str, ...]) -> None:
    _, report = kit(database)
    problems = judged(report)
    assert not {k: problems[k] for k in names if problems[k]}


@pytest.mark.parametrize('database', DATABASES)
def test_r1_the_byte_limit_stops_at_the_page_that_passes_it(
    database: str,
) -> None:
    r"""T2.6, round 1: a fallback scan past ``max_bytes`` on a page that is
    not its last fails by name at that page, asking no later one; a
    provider that would fail page 2 (retryable) is never asked it, so the
    error is the limit's."""
    if not reachable(database):
        return
    names = (
        'fallback.bytes-on-the-first-page',
        'fallback.bytes-before-a-failing-page',
    )
    held(
        f'test_r1_the_byte_limit_stops_at_the_page_that_passes_it[{database}]',
        lambda: _cases_pass(database, names),
    )


@pytest.mark.parametrize('database', DATABASES)
def test_r1_a_signed_zero_through_the_fallback_is_the_pushdown_s(
    database: str,
) -> None:
    r"""T2.6, round 1: over a float key holding -0.0 and 0.0, ``key_counts``
    and ``repeated_key`` through the fallback answer as pushdown does: one
    distinct value, and the zero repeated."""
    if not reachable(database):
        return
    names = (
        'key-counts.signed-zero',
        'repeated-key.signed-zero',
        'fallback.key-counts-signed-zero',
        'fallback.repeated-key-signed-zero',
    )
    held(
        'test_r1_a_signed_zero_through_the_fallback_is_the_pushdown_s'
        f'[{database}]',
        lambda: _cases_pass(database, names),
    )


# --- T2.5: the seeded faults --------------------------------------------------


def _fault(name: str) -> dict[str, Any]:
    return next(f for f in EXPECTED['faults'] if f['name'] == name)


@pytest.mark.parametrize('database', DATABASES)
def test_a_seeded_truncation_fails_the_kit(database: str) -> None:
    r"""T2.5: with the last page of every multi-page answer dropped in
    the Python provider path, the kit fails (exit 1), each case the fault
    names answered short and without an error."""
    if not reachable(database):
        return

    def check() -> None:
        status, report = kit(database, 'truncate')
        seen = by_id(report)
        problems = judged(report)
        for name in _fault('truncate')['fails']:
            assert problems[name], f'{name} passes truncated'
            got = seen[name]
            assert 'answer' in got and 'error' not in got, (name, got)
            wanted = wanted_rows(BY_ID[name]['answer'])
            assert len(got['answer']['rows']) < wanted, (name, got)
        assert status == 1, f'the kit exited {status}, not 1'

    held(f'test_a_seeded_truncation_fails_the_kit[{database}]', check)


@pytest.mark.parametrize('database', DATABASES)
def test_a_seeded_silent_capability_failure_fails_the_kit(
    database: str,
) -> None:
    r"""T2.5: with every neighbors fetch answered empty and no error in the
    Python provider path, the kit fails (exit 1), each neighbors case
    answered with no rows, and every case that asks no neighbors still
    passes."""
    if not reachable(database):
        return

    def check() -> None:
        status, report = kit(database, 'silent_capability')
        seen = by_id(report)
        problems = judged(report)
        fails = _fault('silent_capability')['fails']
        for name in fails:
            assert problems[name], f'{name} passes with no rows'
            got = seen[name]
            assert 'answer' in got and not got['answer']['rows'], (name, got)
            assert 'neighbors' in got['fetched'], (name, got)
        others = {k: v for k, v in problems.items() if v and k not in fails}
        assert not others, others
        assert status == 1, f'the kit exited {status}, not 1'

    held(
        f'test_a_seeded_silent_capability_failure_fails_the_kit[{database}]',
        check,
    )
