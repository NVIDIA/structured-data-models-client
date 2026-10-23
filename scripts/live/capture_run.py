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
cannot hold listed by the loader's reason, never recorded, and the
environment the run needs, by variable name only (no value is read for
print). ``run`` records: the old client (``run_python.py --mode
python-ref``, T3.2's recording mode) over the planned cases on the leg's
live connection (``SDM_LIVE_CONFIG``), in this process, with every write
the loader makes, and on the SQL warehouse and Snowflake every statement on
the connection, through the live write guard (KO3-51; ``write_guard.py``
beside this file, without which the run does not start), refused outside
the one test schema; its lines
written to ``DIR/<leg>.record.jsonl``, then each written as a capture under
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
import importlib.util
import json
import os
import sys
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
    for path in sorted(CASES.rglob('*.json')):
        ((case, _),) = runner.load_cases([path])
        if leg not in runner.case_warehouses(case):
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
        'current': current,
        'environment': {name: name in os.environ for name in ENVIRONMENT[leg]},
    }


#: The live write guard, beside this file (KO3-51): a run without it does
#: not start.
GUARD = Path(__file__).resolve().with_name('write_guard.py')


def load_guard(path: Path | str | None = GUARD) -> ModuleType:
    r"""The live write guard's module (``write_guard.py``, its
    ``WriteGuard`` and ``guard_connection``); a run without it refuses to
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
    for name in ('WriteGuard', 'guard_connection'):
        if not hasattr(module, name):
            raise SystemExit(f'the live write guard {path} has no {name}')
    return module


def _postgres_check(schema: str) -> Any:
    r"""Lakebase's write check: a statement the loader writes names its
    table or type in the one test schema, first after its verb."""
    import re

    loader = _module('databases')
    allowed = re.compile(
        r'^(?:DROP TABLE IF EXISTS|CREATE TABLE|DROP TYPE IF EXISTS|CREATE TYPE|COPY) '
        + re.escape(loader.quote(schema) + '.')
    )

    def check(statement: str) -> None:
        if not allowed.match(statement):
            raise PermissionError(
                'a Lakebase write outside the test schema, or of another form, '
                'refused'
            )

    return check


def install_guard(
    leg: str, guard: ModuleType, runner: Any, loader: Any
) -> None:
    r"""Every live write of ``leg`` through ``guard``: the loader's
    statements (``databases.WRITE_GUARD``), and on the SQL warehouse and
    Snowflake every statement on the connection the runner opens
    (``guard_connection``), each refused when it writes outside the one
    test schema, ``SDM_LIVE_TEST_DATABASE`` and ``SDM_LIVE_TEST_SCHEMA``."""
    schema = os.environ.get('SDM_LIVE_TEST_SCHEMA')
    database = os.environ.get('SDM_LIVE_TEST_DATABASE')
    if not schema or (leg != 'lakebase' and not database):
        raise SystemExit(
            'the write guard names no test schema (SDM_LIVE_TEST_*)'
        )
    if leg == 'lakebase':
        loader.WRITE_GUARD = _postgres_check(schema)
        return
    dialect = 'snowflake' if leg == 'snowflake' else 'databricks'
    try:
        check = guard.WriteGuard(dialect)
    except PermissionError as refused:
        raise SystemExit(f'the write guard cannot start: {refused}') from None
    loader.WRITE_GUARD = check.check
    if leg == 'databricks-spark':
        return
    opened = runner.connect_leg

    def connect_leg(*arguments: Any, **keywords: Any) -> Any:
        connection, close = opened(*arguments, **keywords)
        return guard.guard_connection(connection, check), close

    runner.connect_leg = connect_leg


def run(
    leg: str,
    reference: Path,
    out: Path,
    e0: bool = False,
    guard_path: Path | str | None = GUARD,
) -> int:
    r"""Records ``leg`` live (see the module's text), every write through
    the live write guard; returns the exit status."""
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
    install_guard(leg, guard, runner, loader)
    out.mkdir(parents=True, exist_ok=True)
    recording = out / f'{leg}.record.jsonl'
    observations = out / f'{leg}.python-ref.jsonl'
    try:
        status = runner.main(
            ['--mode', 'python-ref', '--reference', str(reference), '--database', leg,
             '--record', str(recording), '--out', str(observations),
             *planned['record']]
        )  # fmt: skip
    except runner.RunnerFailure as failure:
        print(f'{leg}: the recording failed: {failure}')
        return 1
    if status:
        print(f'{leg}: the recording failed (exit {status})')
        return 1
    for manifest in _module('captures').import_recording(recording, out):
        print(f'{leg}: wrote {manifest}')
    return 0


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


def job_environment(parameters: dict[str, str]) -> list[str]:
    r"""Sets the run's environment from a job's parameters
    (:data:`JOB_ENVIRONMENT`), each given one; no value is printed.
    Returns the variables set."""
    found = []
    for parameter, variable in JOB_ENVIRONMENT.items():
        value = (parameters.get(parameter) or '').strip()
        if value:
            os.environ[variable] = value
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
    )


if __name__ == '__main__':
    sys.exit(main())
