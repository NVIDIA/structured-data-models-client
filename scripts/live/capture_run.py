#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The early capture (C-1) and the reference top-up (E-0), per warehouse
leg (Phase 3 design, "The live-test plan"; T3.3).

    capture_run.py plan --leg LEG [--e0]
    capture_run.py run --leg LEG --reference CHECKOUT --out DIR [--e0]

``plan`` is the dry mode, run anywhere, reaching nothing: the cases the
leg records, in order (on Snowflake a case creating a semantic view from
its YAML first, its statements shown, so a statement the engine refuses
shows before anything is recorded; C-1: every case that runs on the leg; with
``--e0``, only those pending live, with no capture on the leg or one whose
input hash is not the case's), each case whose tables the leg's engine
cannot hold listed by the loader's reason, never recorded, each case
:data:`NOT_LIVE` leaves out of the leg listed by name and reason, and the
environment the run needs, by variable name only (no value is read for
print). ``run`` records: the old client (``run_python.py --mode
python-ref``, T3.2's recording mode) over the planned cases on the leg's
live connection (``SDM_LIVE_CONFIG``), in this process, with every
statement any driver runs, the loader's and the old client's alike,
through the live write guard (KO3-51; ``write_guard.py`` beside this file,
installed on the drivers themselves, without which the run does not
start), refused outside the one test schema; case by case, a case that
fails (the runner's failure, any other error, or any statement of the
case the guard refused, even one the old client's call took for its
answer) written with its reason to ``DIR/<leg>.failed.json``, none of it
imported, while the others go on; their lines written to ``DIR/<leg>.record.jsonl``, then each written as a capture under
``DIR/<leg>/<case>/`` with the leg's manifest (``captures.py import``: the
placeholders and the private-name search applied again). Nothing is
written into the repository: the orchestrator commits the captures under
the protected-path trailer.

The two Databricks legs load the one Databricks test schema, which holds
one case's tables at a time (the loader refuses a relation it did not
load): they run one after the other, never together. The SQL-warehouse
and Snowflake legs run from the reference machine; the Spark and Lakebase
legs in one-time serverless jobs (``capture.job.template.json``, filled
and submitted by ``dbx_runs.py``, its notebook ``capture_job.py``).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import signal
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
CASES = ROOT / 'contract/cases'
CAPTURES = ROOT / 'contract/captures'
LEGS = ('databricks-sql', 'databricks-spark', 'snowflake', 'lakebase')
#: The order the legs record in: the Databricks legs one after the other.
ORDER = ('databricks-sql', 'snowflake', 'databricks-spark', 'lakebase')
#: The environment a leg's recording reads, by name: the live
#: configuration, each placeholder's private name, the write guard's test
#: schema, and Snowflake's credentials (KO3-24, loaded from the owner's
#: file, never read here).
ENVIRONMENT = {
    'databricks-sql': (
        'SDM_LIVE_CONFIG',
        'SDM_CAPTURE_CATALOG',
        'SDM_CAPTURE_SCHEMA',
        'SDM_LIVE_TEST_DATABASE',
        'SDM_LIVE_TEST_SCHEMA',
    ),
    'databricks-spark': (
        'SDM_LIVE_CONFIG',
        'SDM_CAPTURE_CATALOG',
        'SDM_CAPTURE_SCHEMA',
        'SDM_LIVE_TEST_DATABASE',
        'SDM_LIVE_TEST_SCHEMA',
    ),
    'snowflake': (
        'SDM_LIVE_CONFIG',
        'SDM_CAPTURE_CATALOG',
        'SDM_CAPTURE_SCHEMA',
        'SDM_LIVE_TEST_DATABASE',
        'SDM_LIVE_TEST_SCHEMA',
        'SNOWFLAKE_URL',
        'SNOWFLAKE_USERNAME',
        'SNOWFLAKE_ACCESS_TOKEN',
    ),
    'lakebase': (
        'SDM_LIVE_CONFIG',
        'SDM_CAPTURE_SCHEMA',
        'SDM_LIVE_TEST_SCHEMA',
    ),
}


def _module(name: str) -> ModuleType:
    key = f'capture_run_{name}'
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def plan(leg: str, e0: bool = False) -> dict[str, Any]:
    r"""What ``leg`` records, reaching nothing (the dry mode)."""
    if leg not in LEGS:
        raise SystemExit(f'no leg {leg!r}; it is one of {LEGS}')
    runner = _module('run_python')
    loader = _module('databases')
    captures = _module('captures')
    bases = runner.load_bases()
    engine = loader.ENGINE[leg]
    record: list[str] = []
    refused: list[str] = []
    current: list[str] = []
    not_live: list[str] = []
    for path in sorted(CASES.rglob('*.json')):
        ((case, _),) = runner.load_cases([path])
        if leg not in runner.case_warehouses(case):
            continue
        if leg in NOT_LIVE.get(case['id'], ((), ''))[0]:
            not_live.append(f'{case["id"]}: {NOT_LIVE[case["id"]][1]}')
            continue
        if e0:
            capture = CAPTURES / leg / case['id']
            if (
                capture / captures.DOCUMENT
            ).is_file() and captures.read_document(capture)[
                'input_hash'
            ] == runner.input_hash(case, bases):
                current.append(case['id'])
                continue
        reason = None
        if engine in loader.WAREHOUSE_ENGINES:
            duck = runner._connection(case, bases)
            try:
                for table in loader._tables(duck):
                    loader.stored(duck, table, engine)
            except loader.Unloadable as error:
                reason = str(error)
            finally:
                duck.close()
        if reason is None:
            record.append(str(path))
        else:
            refused.append(f'{case["id"]}: {reason}')
    first: dict[str, Any] | None = None
    if leg == 'snowflake':
        # A semantic view created from a case's YAML first: a statement the
        # live engine refuses shows before any case is recorded.
        viewed = [
            path
            for path in record
            if (
                json.loads((ROOT / path).read_text())
                .get('input', {})
                .get('view')
                or {}
            ).get('yaml')
        ]
        record = viewed + [path for path in record if path not in viewed]
        if viewed:
            ((case, _),) = runner.load_cases([Path(viewed[0])])
            ran: list[str] = []
            schema = loader.CaseSchema(
                'snowflake', ran.append, ('{catalog}', '{schema}')
            )
            duck = runner._connection(case, bases)
            try:
                schema.load(duck, fold=True)
            finally:
                duck.close()
            view = case['input']['view']
            schema.create_view(view['kind'], view['name'], view['yaml'])
            first = {
                'case': case['id'],
                'statements': [
                    statement[:160] + (' ...' if len(statement) > 160 else '')
                    for statement in ran
                    if not statement.startswith('INSERT')
                ],
            }
    return {
        'leg': leg,
        'campaign': 'E-0' if e0 else 'C-1',
        'first': first,
        'order': list(ORDER),
        'record': record,
        'refused_by_the_loader': refused,
        'not_live': not_live,
        'current': current,
        'environment': {name: name in os.environ for name in ENVIRONMENT[leg]},
    }


#: The cases a leg's live plan leaves out, by name, with the legs and the
#: reason: their names hold a character the write guard's one strict name
#: rule refuses (a backtick on Databricks, a line break anywhere), so every
#: load of theirs is refused by design (the owner's allowlist decision).
_BACKTICK = (
    'a name holds a backtick, which the write guard refuses on Databricks'
)
_LINE_BREAK = 'a name holds a line break, which the write guard refuses'
NOT_LIVE: dict[str, tuple[tuple[str, ...], str]] = {
    **{
        f'name.backtick-{part}': (
            ('databricks-sql', 'databricks-spark'),
            _BACKTICK,
        )
        for part in ('aggregation', 'entity', 'feature', 'filter', 'target')
    },
    **{
        f'name.line-break-{part}': (LEGS, _LINE_BREAK)
        for part in ('aggregation', 'entity', 'feature', 'filter', 'target')
    },
}


#: The live write guard, beside this file (KO3-51): a run without it does
#: not start.
GUARD = Path(__file__).resolve().with_name('write_guard.py')


def load_guard(path: Path | str | None = GUARD) -> ModuleType:
    r"""The live write guard's module (``write_guard.py``, its
    ``WriteGuard``, ``install`` and ``from_environment``); a run without it refuses to
    start."""
    if not path or not Path(path).is_file():
        raise SystemExit(
            f'no live write guard at {path}; every live write goes through '
            'it (KO3-51)'
        )
    spec = importlib.util.spec_from_file_location('sdm_live_write_guard', path)
    if spec is None or spec.loader is None:
        raise SystemExit(f'the live write guard {path} cannot be read')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ('WriteGuard', 'install', 'from_environment'):
        if not hasattr(module, name):
            raise SystemExit(f'the live write guard {path} has no {name}')
    return module


#: Each leg's dialect in the write guard.
DIALECT = {
    'databricks-sql': 'databricks',
    'databricks-spark': 'databricks',
    'snowflake': 'snowflake',
    'lakebase': 'postgres',
}


def install_guard(leg: str, guard: ModuleType, loader: Any) -> Any:
    r"""Every live statement of the run through the one write guard:
    every driver's entry points (``guard.install``, each dialect's test
    schema from ``SDM_LIVE_TEST_*``, a dialect with none refusing every
    statement) and the loader's own statements (``databases.WRITE_GUARD``).
    The leg's test schema must be set, and be the one its cases resolve
    ``{catalog}`` and ``{schema}`` to (``SDM_CAPTURE_*``), or the run does
    not start. Returns what undoes it."""
    guards = guard.from_environment()
    own = guards.get(DIALECT[leg])
    if own is None:
        raise SystemExit(
            f'the write guard names no test schema for {leg} (SDM_LIVE_TEST_*)'
        )
    for variable, held in (
        ('SDM_CAPTURE_CATALOG', own.database),
        ('SDM_CAPTURE_SCHEMA', own.schema),
    ):
        value = os.environ.get(variable)
        if held is None or value is None:
            continue
        same = (
            value.lower() == held.lower()
            if own.dialect == 'databricks'
            else value == held
        )
        if not same:
            raise SystemExit(
                f"{variable} is not the write guard's test schema "
                '(SDM_LIVE_TEST_*): the loads would be refused'
            )
    undo = guard.install(guards)
    loader.WRITE_GUARD = guard._keeping(own.check)
    loader.WRITE_GUARD_SETUP = getattr(guard, 'setup', None)

    def uninstall() -> None:
        loader.WRITE_GUARD = None
        loader.WRITE_GUARD_SETUP = None
        undo()

    return uninstall


#: How long one case may run, in seconds, before it fails alone: three
#: times the heaviest case the Spark leg recorded (549 s, a 30 MiB body;
#: the Spark recording's root-cause analysis, 2026-10-24).
CASE_TIMEOUT = 1800


class CaseTimedOut(BaseException):
    r"""A case past its time limit. Not an :class:`Exception`, so no
    handler of the runner's or a driver's takes it for the case's own
    error or answer; :func:`run` catches it and goes on."""


@contextlib.contextmanager
def _limited(seconds: float | None) -> Iterator[None]:
    r"""Raises :class:`CaseTimedOut` in the block once ``seconds`` have
    passed, by the interval timer, which interrupts a statement waiting on
    its driver's socket; no limit for ``None`` or 0. :func:`run` refuses a
    limit off the main thread, where no signal arrives."""
    if not seconds:
        yield
        return

    def expired(signum: int, frame: Any) -> None:
        raise CaseTimedOut

    before = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, before)


def run(
    leg: str,
    reference: Path,
    out: Path,
    e0: bool = False,
    guard_path: Path | str | None = GUARD,
    case_timeout: float | None = CASE_TIMEOUT,
) -> int:
    r"""Records ``leg`` live (see the module's text), every write through
    the live write guard; returns the exit status. As each case ends, a
    line for it is added to ``DIR/<leg>.progress.jsonl`` (its number,
    name, ``recorded`` or ``failed`` and its seconds) and printed, and a
    failure is written to ``DIR/<leg>.failed.json`` at once, so a run cut
    short leaves what it did; a case running past ``case_timeout``
    seconds fails alone, and the run goes on; a limit asked for off the
    main thread is refused before anything runs."""
    if (
        case_timeout
        and threading.current_thread() is not threading.main_thread()
    ):
        # The limit is the interval timer's signal, which only the main
        # thread takes (Claude r1 P2): no run goes on believing itself
        # limited.
        raise SystemExit(
            'a per-case time limit needs the main thread, where the timer '
            'signal arrives; run there, or pass case_timeout 0 for none'
        )
    guard = load_guard(guard_path)
    planned = plan(leg, e0)
    missing = [
        name for name, held in planned['environment'].items() if not held
    ]
    if missing:
        print(f'{leg}: the environment lacks {", ".join(missing)}')
        return 2
    if not planned['record']:
        print(f'{leg}: nothing to record')
        return 0
    # The runner in this process, with the loader it loads (by the names
    # its _sibling reads), so the guard holds every write it makes.
    loader = _named('databases')
    runner = _named('run_python')
    out.mkdir(parents=True, exist_ok=True)
    parts = out / f'{leg}.cases'
    parts.mkdir(exist_ok=True)
    failed: dict[str, str] = {}
    recorded: list[Path] = []
    progress = out / f'{leg}.progress.jsonl'
    # Rewritten whole after each case, never appended to: a Databricks
    # volume refuses an append to a file once written (Errno 29, the
    # second recording's Spark run).
    progressed: list[str] = []
    progress.write_text('', encoding='utf-8')
    failures = out / f'{leg}.failed.json'

    def write_failures() -> None:
        failures.write_text(
            json.dumps(failed, indent=1, sort_keys=True) + '\n',
            encoding='utf-8',
        )

    write_failures()
    uninstall = install_guard(leg, guard, loader)
    try:
        # One case at a time: a case that fails (refused, not opened, or
        # past its time limit) is written with its reason, and the others
        # go on.
        for number, path in enumerate(planned['record']):
            case = Path(path).stem
            started = time.monotonic()
            reason = _one_case(
                runner,
                guard,
                path,
                reference,
                leg,
                parts / f'{number:04d}',
                case_timeout,
            )
            if reason is None:
                recording = parts / f'{number:04d}.record.jsonl'
                if recording.is_file():
                    recorded.append(recording)
            else:
                failed[case] = reason
                write_failures()
            line = {
                'number': number,
                'case': case,
                'status': 'recorded' if reason is None else 'failed',
                'seconds': round(time.monotonic() - started, 1),
            }
            progressed.append(json.dumps(line, sort_keys=True) + '\n')
            progress.write_text(''.join(progressed), encoding='utf-8')
            print(
                f'{leg}: {number + 1}/{len(planned["record"])} {case} '
                f'{line["status"]} in {line["seconds"]} s',
                flush=True,
            )
    finally:
        uninstall()
    whole = out / f'{leg}.record.jsonl'
    whole.write_text(
        ''.join(part.read_text(encoding='utf-8') for part in recorded),
        encoding='utf-8',
    )
    for case, reason in sorted(failed.items()):
        print(f'{leg}: {case} failed: {reason}')
    for manifest in _module('captures').import_recording(whole, out):
        print(f'{leg}: wrote {manifest}')
    return 1 if failed else 0


def _one_case(
    runner: Any,
    guard: ModuleType,
    path: str,
    reference: Path,
    leg: str,
    stem: Path,
    case_timeout: float | None,
) -> str | None:
    r"""Records the case at ``path``, its lines beside ``stem``
    (``.record.jsonl``, ``.python-ref.jsonl``), within ``case_timeout``
    seconds; returns why it failed, or ``None``."""
    recording = stem.with_name(stem.name + '.record.jsonl')
    observations = stem.with_name(stem.name + '.python-ref.jsonl')
    refusals = len(guard.REFUSALS)
    try:
        with _limited(case_timeout):
            lines = runner.run(
                'python-ref',
                runner.load_cases([Path(path)]),
                reference,
                database=leg,
                record=recording,
            )
    except CaseTimedOut:
        return f'past its time limit of {case_timeout} s'
    except runner.RunnerFailure as failure:
        return str(failure)
    except Exception as error:  # noqa: BLE001 - this case alone fails
        return f'{type(error).__name__}: {error}'
    if len(guard.REFUSALS) > refusals:
        # A statement of the case refused, whoever sent it: the old
        # client's call would record the refusal as its own answer.
        return 'the write guard refused: ' + '; '.join(
            guard.REFUSALS[refusals:]
        )
    observations.write_text(
        ''.join(json.dumps(item, sort_keys=True) + '\n' for item in lines),
        encoding='utf-8',
    )
    return None


def _named(name: str) -> ModuleType:
    r"""``contract/runners/<name>.py`` loaded under its own name, as the
    runner's ``_sibling`` loads its neighbours."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The reference's client package, which the recording mode imports from
#: the reference checkout itself (``run_python.py``'s
#: ``_require_reference_client``), beside the reference's wheels.
CLIENT_SOURCE = Path('packages/structured-data-models-client/src')


def bootstrap_reference(reference: Path, site: Path) -> Path:
    r"""Makes the interpreter whose site directory is ``site`` import
    ``sdm_client`` from the reference checkout ``reference``: a ``.pth``
    file naming the checkout's client source, written after the
    reference's wheels are installed and its client wheel removed (the
    serverless jobs' layout, ``capture_job.py``), so the client the
    recording runs is the checkout's, as the runner requires; returns the
    file."""
    source = (Path(reference) / CLIENT_SOURCE).resolve()
    if not (source / 'sdm_client' / '__init__.py').is_file():
        raise SystemExit(f'{reference}: no reference client at {CLIENT_SOURCE}')
    path = Path(site) / 'sdm-reference-client.pth'
    path.write_text(f'{source}\n', encoding='utf-8')
    return path


def installable_wheels(
    folder: Path | str, tags: Iterable[Any] | None = None
) -> list[str]:
    r"""The wheels in ``folder`` this host installs, by its tags
    (``tags``, else ``packaging.tags.sys_tags()``), sorted: the folder
    holds every platform's build the jobs' hosts need (serverless runs on
    x86_64 and aarch64), and pip refuses a whole install naming a wheel
    the host does not support."""
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    supported = set(sys_tags() if tags is None else tags)
    return sorted(
        str(path)
        for path in Path(folder).glob('*.whl')
        if supported & set(parse_wheel_filename(path.name)[3])
    )


def use_reference_client(reference: Path) -> Path:
    r"""Makes this process import ``sdm_client`` from the reference
    checkout ``reference``: its client source first on ``sys.path``, any
    ``sdm_client`` already imported forgotten, so the client the recording
    runs is the checkout's, as the runner requires, beside the reference's
    other packages installed from its wheels (``capture_job.py``; a
    notebook's environment reads no ``.pth`` file). Returns the source."""
    source = (Path(reference) / CLIENT_SOURCE).resolve()
    if not (source / 'sdm_client' / '__init__.py').is_file():
        raise SystemExit(f'{reference}: no reference client at {CLIENT_SOURCE}')
    sys.path.insert(0, str(source))
    for name in [n for n in sys.modules if n.split('.')[0] == 'sdm_client']:
        del sys.modules[name]
    return source


#: Each job parameter that names part of a run's environment, and the
#: variable it sets: the live configuration's path, the placeholders'
#: private names and the write guard's test schema, filled into the job's
#: spec when it is submitted (``dbx_runs.py spec --set``), never written
#: in the repository.
JOB_ENVIRONMENT = {
    'live_config': 'SDM_LIVE_CONFIG',
    'capture_catalog': 'SDM_CAPTURE_CATALOG',
    'capture_schema': 'SDM_CAPTURE_SCHEMA',
    'test_database': 'SDM_LIVE_TEST_DATABASE',
    'test_schema': 'SDM_LIVE_TEST_SCHEMA',
}


#: The job parameter holding the other placeholders' private names, a JSON
#: object by placeholder name (``{"user": ..., "shop_metric_view": ...}``),
#: each set as ``SDM_CAPTURE_<NAME>``: an existing view's name, its
#: schema, the view owner the warehouse's describe shows.
JOB_NAMES = 'capture_names'


def job_environment(parameters: dict[str, str]) -> list[str]:
    r"""Sets the run's environment from a job's parameters
    (:data:`JOB_ENVIRONMENT`, and each placeholder's name of
    :data:`JOB_NAMES`), each given one; no value is printed. A name that
    is not a placeholder's (``[a-z][a-z0-9_]*``), or a value that is not
    text, is refused by its key. Returns the variables set."""
    found = []
    for parameter, variable in JOB_ENVIRONMENT.items():
        value = (parameters.get(parameter) or '').strip()
        if value:
            os.environ[variable] = value
            found.append(variable)
    names = json.loads((parameters.get(JOB_NAMES) or '').strip() or '{}')
    if not isinstance(names, dict):
        raise SystemExit(f'{JOB_NAMES} is not a JSON object')
    for key, value in sorted(names.items()):
        if not (
            re.fullmatch(r'[a-z][a-z0-9_]*', key)
            and isinstance(value, str)
            and value.strip()
        ):
            raise SystemExit(f'{JOB_NAMES}: {key!r} is no placeholder name')
        variable = 'SDM_CAPTURE_' + key.upper()
        os.environ[variable] = value.strip()
        found.append(variable)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    planning = commands.add_parser('plan', help='what a leg records (dry)')
    planning.add_argument('--leg', required=True, choices=LEGS)
    planning.add_argument('--e0', action='store_true')
    running = commands.add_parser('run', help='record a leg live')
    running.add_argument('--leg', required=True, choices=LEGS)
    running.add_argument('--reference', type=Path, required=True)
    running.add_argument('--out', type=Path, required=True)
    running.add_argument('--e0', action='store_true')
    running.add_argument(
        '--case-timeout',
        type=float,
        default=CASE_TIMEOUT,
        help='seconds one case may run before it fails alone',
    )
    booting = commands.add_parser(
        'bootstrap', help='import sdm_client from the reference checkout'
    )
    booting.add_argument('--reference', type=Path, required=True)
    booting.add_argument('--site', type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == 'bootstrap':
        print(
            f'wrote {bootstrap_reference(arguments.reference, arguments.site)}'
        )
        return 0
    if arguments.command == 'plan':
        print(json.dumps(plan(arguments.leg, arguments.e0), indent=1))
        return 0
    return run(
        arguments.leg,
        arguments.reference,
        arguments.out,
        arguments.e0,
        case_timeout=arguments.case_timeout,
    )


if __name__ == '__main__':
    sys.exit(main())
