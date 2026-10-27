# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T3.2's warehouse legs in the comparator, the observation schema and
the ratchet (the package's test author's), one test per change. The
``live_only`` outcome, the stand-in's verdicts and the locked run's E-1
verdict check went with the stand-in (the owner's B-lite, 2026-10-23), and
their tests with them:

- the observation schema keys a line ``<case>@<leg>`` on each warehouse
  leg and takes the runner ``java``;
- ``compare.py`` judges a warehouse leg against its recorded reference by
  every rule: an equal answer passes, a different one fails, and the
  statement count gates (H9 in full);
- the runner ``java`` in ``ratchet.py``: its baselines are read, and a
  Java leg once baselined is held by the no-removal rule; and the Java
  legs' admission (choice 10): on a warehouse leg, the Java host runs every
  case there exactly when it reports the provider (``run_python.py legs``).

The fixture is a real reference line: ``base.binary``'s, python-ref's on
DuckDB, made a case of its own (``t32.live.binary``, no ``core``) and keyed
on the leg, beside a python-core line that copies it. On DuckDB the same
pair compares with no difference (``test_t32_the_fixture_is_valid_on_duckdb``,
which keeps the others from being vacuous).

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
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / 'scripts/rust_core/compare.py'
RATCHET = ROOT / 'scripts/rust_core/ratchet.py'
RUNNER = ROOT / 'contract/runners/run_python.py'
PENDING_FILE = ROOT / 'contract/tests/t32-pending.json'
SOURCE = 'base.binary'
CASE = 't32.live.binary'
WAREHOUSES = ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')


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
def test_t32_the_observation_schema_takes_legs_and_java() -> None:
    r"""The observation schema: ``<case>@<leg>`` on each warehouse leg and
    the runner ``java``; a key on no leg is refused."""
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
    nowhere = copy.deepcopy(line)
    nowhere['case'] = f'{SOURCE}@bigquery'
    if run._invalid(validate, nowhere) is None:
        problems.append('@bigquery: taken')
    assert not problems, problems


@pending
def test_t32_a_warehouse_leg_equal_to_its_recording_passes(
    tmp_path: Path,
) -> None:
    r"""B-lite: a warehouse leg is judged against its recorded reference
    by every rule; an answer equal to the old client's passes."""
    pair = Pair(tmp_path, 'databricks-sql')
    assert pair.judge(pair.answered(None)).failures == []


@pending
def test_t32_a_warehouse_leg_that_differs_fails(tmp_path: Path) -> None:
    r"""B-lite: a different answer on a warehouse leg fails, as on DuckDB."""
    pair = Pair(tmp_path, 'databricks-sql')
    report = pair.judge(pair.differing(None))
    assert [rule for rule, _ in report.failures] == ['result'], report.texts


@pending
def test_t32_statements_gate_on_a_warehouse_leg(tmp_path: Path) -> None:
    r"""B-lite: with no stand-in, H9 applies in full on a warehouse leg:
    one statement fewer than the old client's recording fails."""
    pair = Pair(tmp_path, 'databricks-sql')
    fewer = pair.answered(None)
    fewer['driver']['capture']['statements'].pop()
    fewer['driver']['query_log']['statements'] -= 1
    report = pair.judge(fewer)
    assert 'sql.count' in [rule for rule, _ in report.failures], report.texts


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
