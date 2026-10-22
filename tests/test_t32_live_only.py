# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.2's ``live_only`` outcome and the warehouse legs in the comparator,
the observation schema and the ratchet (design choice 2, "The ``live_only``
state in the tools"; the package's test author's), one test per change:

- the observation schema keys a line ``<case>@<leg>`` on each warehouse
  leg, takes the runner ``java``, and carries the stand-in's verdict,
  ``standin: {live_only: <code>, flagged: [<codes>]}``;
- ``compare.py`` on a warehouse leg: a case whose stand-in marks it live
  only with a listed reason (B1, B2, B4) is ``live_only``, neither passed
  nor failed; a mark with a reason not in ``compare-policy.yaml``'s list is
  refused (the brief's "a case marked live-only with no stand-in reason is
  refused"); a case the stand-in flagged B4 passes when it matches the old
  client's answer and is ``live_only`` (B4) when it differs; a difference
  the stand-in did not flag fails as any difference does;
- H9 on stand-in legs (KO3-45): the statement count is diagnostic there,
  and still gates on DuckDB;
- the locked run (locked check 1): a ``live_only`` item with no E-1
  verdict for the final candidate fails, one with an E-1 verdict naming
  the final candidate passes, and one naming another commit fails;
- the runner ``java`` in ``ratchet.py``: its baselines are read, and a
  Java leg once baselined is held by the no-removal rule; and the Java
  legs' admission (choice 10): on a warehouse leg, the Java host runs every
  case there exactly when it reports the provider (``run_python.py legs``).

The fixture is a real reference line: ``base.binary``'s, python-ref's on
DuckDB, made a case of its own (``t32.live.binary``, no ``core``) and keyed
on the leg, beside a python-core line that copies it, as the stand-in
would answer when the core matches the old client. On DuckDB the same
pair compares with no difference (``test_t32_the_fixture_is_valid_on_duckdb``,
which passes today and keeps the others from being vacuous).

Interfaces assumed (T3.2 defines them; this file is their first reader):

- ``compare.compare(...)``'s report holds ``live_only``: the items so
  judged, by key, each with its code (a mapping or a collection of keys);
  none is in ``failures``;
- ``compare.py ... --database LEG --verdicts MANIFEST... --candidate SHA``
  (``gate.sh --locked`` passes them): exit 1 naming each ``live_only`` item
  without a verdict ``equal`` in a manifest whose ``recording_commit`` is
  SHA; the core's live capture manifest (``contract/captures/``, written by
  E-1) carries ``"verdicts": {"<case>@<leg>": "equal" | "differs"}``.

Each test listed in ``contract/tests/t32-pending.json`` failed when written,
before the implementation, and is held to failing (the ``pending``
decorator, as T2.10's ``held``) until the implementer takes it out of
``pending``; ``pending`` stays within
``authored``, the test author's list.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / 'scripts/rust_core/compare.py'
RATCHET = ROOT / 'scripts/rust_core/ratchet.py'
RUNNER = ROOT / 'contract/runners/run_python.py'
PENDING_FILE = ROOT / 'contract/tests/t32-pending.json'
SOURCE = 'base.binary'
CASE = 't32.live.binary'
WAREHOUSES = ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')
CANDIDATE = 'c' * 40


def _pending_names(key: str) -> set[str]:
    return set(json.loads(PENDING_FILE.read_text(encoding='utf-8'))[key])


def pending(test: Any) -> Any:
    """Holds ``test`` to failing while ``contract/tests/t32-pending.json``
    lists it, as T2.10's ``held`` does: a pending test passes only by
    failing, and once it passes it fails, naming itself, until the
    implementer takes it out of ``pending``."""
    assert test.__name__ in _pending_names('authored'), test.__name__

    @functools.wraps(test)
    def held(*arguments: Any, **keywords: Any) -> None:
        if test.__name__ not in _pending_names('pending'):
            test(*arguments, **keywords)
            return
        try:
            test(*arguments, **keywords)
        except Exception:  # noqa: BLE001 - pending: failing, as when written
            return
        raise AssertionError(
            f'{test.__name__} passes now: take it out of '
            'contract/tests/t32-pending.json pending'
        )

    return held


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def compare_module() -> ModuleType:
    return _module('t32_live_compare', COMPARE)


def _reference_line() -> dict[str, Any]:
    [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    for text in path.read_text(encoding='utf-8').splitlines():
        line = json.loads(text)
        if line['case'] == SOURCE:
            found: dict[str, Any] = line
            return found
    raise AssertionError(f'no reference line of {SOURCE}')


class Pair:
    r"""A case, its reference line and a python-core line, on one leg,
    written under ``tmp``."""

    def __init__(self, tmp: Path, database: str) -> None:
        self.database = database
        self.key = CASE if database == 'duckdb' else f'{CASE}@{database}'
        case = json.loads(
            (ROOT / f'contract/cases/s1/{SOURCE}.json').read_text(
                encoding='utf-8'
            )
        )
        case.pop('core')
        case['id'] = CASE
        if database != 'duckdb':
            case['slice'] = 'S2'
            case['warehouses'] = [database]
        self.cases = tmp / 'cases'
        self.cases.mkdir(parents=True)
        data = json.dumps(case, indent=2).encode()
        (self.cases / f'{CASE}.json').write_bytes(data)
        line = _reference_line()
        line['case'] = self.key
        line['identity']['inputs']['case'] = hashlib.sha256(data).hexdigest()
        self.reference = tmp / 'reference.jsonl'
        self.reference.write_text(json.dumps(line) + '\n', encoding='utf-8')
        core = copy.deepcopy(line)
        core['identity'].update(
            runner='python-core',
            reference=None,
            native={'sha256': '0' * 64},
            native_loaded=True,
            operations=[],
        )
        core['core'] = {
            'inputs': [],
            'fetches': [],
            'accepted': None,
            'described': None,
            'materialized': None,
            'predicted': None,
            'error': None,
        }
        self.core = core
        self.observed = tmp / 'python-core.jsonl'

    def write(self, line: dict[str, Any]) -> Path:
        self.observed.write_text(json.dumps(line) + '\n', encoding='utf-8')
        return self.observed

    def differing(self, standin: dict[str, Any] | None) -> dict[str, Any]:
        r"""The core's line with another answer: its result's digest."""
        line = self.answered(standin)
        line['public']['result']['digest'] = '0' * 64
        return line

    def answered(self, standin: dict[str, Any] | None) -> dict[str, Any]:
        line = copy.deepcopy(self.core)
        if standin is not None:
            line['standin'] = standin
        return line

    def judge(self, line: dict[str, Any]) -> Any:
        return compare_module().compare(
            [self.cases],
            self.reference,
            [self.write(line)],
            database=self.database,
        )


def _live_only(report: Any) -> set[str]:
    found = report.live_only
    return set(found)


def test_t32_the_fixture_is_valid_on_duckdb(tmp_path: Path) -> None:
    r"""Not vacuous: on DuckDB the pair compares with no difference, and a
    changed result or one statement fewer each fails, so the same changes
    on a warehouse leg below are judged by the rules, not refused as a
    malformed line."""
    pair = Pair(tmp_path, 'duckdb')
    assert pair.judge(pair.answered(None)).failures == []
    rules = [rule for rule, _ in pair.judge(pair.differing(None)).failures]
    assert rules == ['result']
    fewer = pair.answered(None)
    fewer['driver']['capture']['statements'].pop()
    fewer['driver']['query_log']['statements'] -= 1
    rules = [rule for rule, _ in pair.judge(fewer).failures]
    assert rules == ['sql.count']


@pending
def test_t32_the_observation_schema_takes_legs_java_and_the_stand_in_verdict() -> (
    None
):
    r"""The observation schema: ``<case>@<leg>`` on each warehouse leg, the
    runner ``java``, and ``standin`` with each listed code; a key on no leg
    is refused, and a line without ``standin`` is valid as before."""
    run = _module('t32_live_run_python', RUNNER)
    validate = run.validator('observation')
    line = _reference_line()
    assert run._invalid(validate, line) is None
    problems = []
    for leg in WAREHOUSES:
        keyed = copy.deepcopy(line)
        keyed['case'] = f'{SOURCE}@{leg}'
        if run._invalid(validate, keyed) is not None:
            problems.append(f'{leg}: refused')
    java = copy.deepcopy(line)
    java['identity']['runner'] = 'java'
    java['identity']['reference'] = None
    if run._invalid(validate, java) is not None:
        problems.append('java: refused')
    for standin in (
        {'live_only': 'B1', 'flagged': []},
        {'live_only': 'B2', 'flagged': []},
        {'live_only': 'B4', 'flagged': ['B4']},
        {'live_only': None, 'flagged': ['B4']},
        {'live_only': None, 'flagged': []},
    ):
        marked = copy.deepcopy(line)
        marked['case'] = f'{SOURCE}@databricks-sql'
        marked['standin'] = standin
        if run._invalid(validate, marked) is not None:
            problems.append(f'{standin}: refused')
    nowhere = copy.deepcopy(line)
    nowhere['case'] = f'{SOURCE}@bigquery'
    if run._invalid(validate, nowhere) is None:
        problems.append('@bigquery: taken')
    assert not problems, problems


@pending
@pytest.mark.parametrize('code', ['B1', 'B2', 'B4'])
def test_t32_a_listed_live_only_mark_is_live_only_neither_passed_nor_failed(
    tmp_path: Path, code: str
) -> None:
    r"""A case the stand-in could not answer, with a reason of the policy's
    list, is ``live_only`` with its code, whatever its answer: never a pass
    and never a failure offline."""
    pair = Pair(tmp_path, 'databricks-sql')
    report = pair.judge(pair.differing({'live_only': code, 'flagged': []}))
    assert report.failures == []
    assert pair.key in _live_only(report)


@pending
@pytest.mark.parametrize('code', ['B3', 'B9', ''])
def test_t32_a_live_only_mark_without_a_listed_reason_is_refused(
    tmp_path: Path, code: str
) -> None:
    r"""Only the stand-in's listed reasons admit ``live_only``: B3 (a blind
    spot covered another way), an unknown code or an empty one is refused
    by name, and the item is not ``live_only``."""
    pair = Pair(tmp_path, 'databricks-sql')
    report = pair.judge(pair.answered({'live_only': code, 'flagged': []}))
    assert report.failures, 'a live-only mark with no listed reason passed'
    assert pair.key not in _live_only(report)
    assert any(
        'standin' in text or 'live_only' in text or (code and code in text)
        for text in report.texts
    ), report.texts


@pending
def test_t32_a_b4_flagged_case_passes_when_it_matches(tmp_path: Path) -> None:
    r"""B4: the stand-in answered a cross-type key comparison on the local
    engine; where the outcome equals the old client's, the case passes."""
    pair = Pair(tmp_path, 'databricks-sql')
    report = pair.judge(pair.answered({'live_only': None, 'flagged': ['B4']}))
    assert report.failures == []
    assert pair.key not in _live_only(report)


@pending
def test_t32_a_b4_flagged_case_that_differs_is_live_only(
    tmp_path: Path,
) -> None:
    r"""B4: where the flagged case's outcome differs, the difference may be
    the local engine's casts, so the case is ``live_only`` (B4), not a
    failure; the same difference unflagged fails."""
    pair = Pair(tmp_path, 'databricks-sql')
    flagged = pair.judge(pair.differing({'live_only': None, 'flagged': ['B4']}))
    assert flagged.failures == []
    assert pair.key in _live_only(flagged)
    unflagged = pair.judge(pair.differing({'live_only': None, 'flagged': []}))
    assert [rule for rule, _ in unflagged.failures] == ['result']
    assert pair.key not in _live_only(unflagged)


@pending
def test_t32_statements_are_diagnostic_on_a_stand_in_leg(
    tmp_path: Path,
) -> None:
    r"""KO3-45 (H9 on stand-in legs): one statement fewer than the old
    client is reported, not gated, on a warehouse leg on the stand-in;
    the fixture test shows it gates on DuckDB."""
    pair = Pair(tmp_path, 'databricks-sql')
    fewer = pair.answered({'live_only': None, 'flagged': []})
    fewer['driver']['capture']['statements'].pop()
    fewer['driver']['query_log']['statements'] -= 1
    report = pair.judge(fewer)
    assert report.failures == [], report.texts
    assert any('sql.count' in note for note in report.notes), report.notes


def _verdicts(path: Path, commit: str, verdict: str | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'format': 'sdm-capture/1',
                'recording_commit': commit,
                'reference_commit': '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1',
                'legs': {'databricks-sql': {}},
                'cases': {},
                'verdicts': {}
                if verdict is None
                else {f'{CASE}@databricks-sql': verdict},
            }
        ),
        encoding='utf-8',
    )
    return path


@pending
def test_t32_the_locked_run_needs_an_e1_verdict_for_each_live_only_item(
    tmp_path: Path,
) -> None:
    r"""Locked check 1: a ``live_only`` item with no E-1 verdict for the
    final candidate fails the locked run; an ``equal`` verdict recorded on
    the final candidate passes; one recorded on another commit, or a
    ``differs`` verdict, fails."""
    pair = Pair(tmp_path, 'databricks-sql')
    observed = pair.write(pair.answered({'live_only': 'B1', 'flagged': []}))

    def locked(*manifests: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(COMPARE),
                '--cases',
                str(pair.cases),
                '--reference',
                str(pair.reference),
                '--database',
                'databricks-sql',
                '--candidate',
                CANDIDATE,
                '--verdicts',
                *map(str, manifests),
                '--',
                str(observed),
            ],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'},
            timeout=600,
        )

    results = {}
    for name, commit, verdict in (
        ('none', CANDIDATE, None),
        ('equal on the candidate', CANDIDATE, 'equal'),
        ('equal on another commit', 'd' * 40, 'equal'),
        ('differs on the candidate', CANDIDATE, 'differs'),
    ):
        manifest = _verdicts(
            tmp_path / name.replace(' ', '-') / 'e1.json', commit, verdict
        )
        done = locked(manifest)
        assert 'Traceback' not in done.stderr, done.stderr
        results[name] = done.returncode
        if done.returncode:
            assert pair.key in done.stdout, done.stdout
    assert results == {
        'none': 1,
        'equal on the candidate': 0,
        'equal on another commit': 1,
        'differs on the candidate': 1,
    }


@pending
def test_t32_a_java_leg_once_baselined_is_held_by_the_no_removal_rule() -> None:
    r"""The runner ``java`` in ``ratchet.py``: a tree's Java baselines are
    read (``contract/baselines/<cell>/java.jsonl``), and a Java line
    baselined and not observed fails, so a later candidate cannot drop a
    leg once admitted."""
    ratchet = _module('t32_live_ratchet', RATCHET)
    assert 'java' in ratchet.RUNNERS
    line = {
        'case': 's0.accept.binary-classification',
        'cell': ratchet.CELL,
        'runner': 'java',
        'projection': {
            'outcome.code': None,
            'outcome.error': None,
            'outcome.result': 'success',
            'outcome.stage': None,
        },
    }
    line['digest'] = ratchet.digest(line['projection'])

    class Tree:
        def read(self, path: str) -> str | None:
            if path.endswith('/java.jsonl'):
                return json.dumps(line) + '\n'
            return ''

    held = ratchet.baselines(Tree())
    assert held['java'] == {line['case']: line}
    report = ratchet.Report()
    ratchet.check_baselines(
        report,
        held,
        {},
        [],
        [],
        {'java': {}},
        {},
        [],
        set(),
        {},
        None,
    )
    texts = [str(item) for item in getattr(report, 'failures', [])] or [
        str(report)
    ]
    assert any('baselined, and not observed' in text for text in texts), texts


@pending
def test_t32_a_java_leg_runs_where_the_java_host_reports_the_provider(
    tmp_path: Path,
) -> None:
    r"""Choice 10's admission: on ``databricks-sql``, the Java host runs
    each case the reference observed there once it reports the provider
    (a skipped case fails, by its key), and is not required there before
    (``run_python.py legs``, the staging rule gate.sh and CI read)."""
    line = _reference_line()
    lines = []
    for runner in ('python-ref', 'python-core', 'java'):
        for case in ('t32.one', 't32.two'):
            if runner == 'java' and case == 't32.two':
                continue
            made = copy.deepcopy(line)
            made['case'] = f'{case}@databricks-sql'
            made['identity']['runner'] = runner
            lines.append(made)
    observations = tmp_path / 'observations.jsonl'
    observations.write_text(
        ''.join(json.dumps(item) + '\n' for item in lines), encoding='utf-8'
    )

    def legs(
        providers: dict[str, list[str]],
    ) -> subprocess.CompletedProcess[str]:
        listed = tmp_path / 'providers.json'
        listed.write_text(json.dumps(providers), encoding='utf-8')
        return subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                'legs',
                '--database',
                'databricks-sql',
                '--providers',
                str(listed),
                str(observations),
            ],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=300,
        )

    reported = legs(
        {
            'python-core': ['databricks-sql'],
            'java': ['duckdb', 'databricks-sql'],
        }
    )
    assert reported.returncode == 1, reported.stdout + reported.stderr
    assert (
        'java' in reported.stdout
        and 't32.two@databricks-sql' in reported.stdout
    )
    before = legs({'python-core': ['databricks-sql'], 'java': ['duckdb']})
    assert before.returncode == 0, before.stdout + before.stderr
