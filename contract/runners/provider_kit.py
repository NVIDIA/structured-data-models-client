#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The provider conformance kit's runner on the Python host (T2.5, T2.6).

    python -I contract/runners/provider_kit.py --database DATABASE \
        [--fault truncate|silent_capability] --out REPORT
    python -I contract/runners/provider_kit.py --load sqlite OUT

It builds ``contract/fixtures/provider-kit/dataset.sql`` on DATABASE
(DuckDB from the file; SQLite and Postgres loaded from that DuckDB by the
one contract loader, ``databases.py``, Postgres into a database of its own
on ``SDM_POSTGRES_TEST_DSN``'s server, dropped after), and runs each case
of ``expected.json`` beside it through the core on that database's Python
provider (``sdm_client._rust.providers``) and the Python run loop
(``sdm_client._rust.run``), as the file's ``about`` says: the core plans
the case's request for the provider (the test-only fetch, so a test-ops
build of ``sdm_client._native``), with what the case withholds taken out
of what the provider declares and the fallback's limits it declares, and
admits the answer. The provider path between the run loop and the
provider pages every answer in pages of ``page_rows`` rows, each but the
last a ``Page`` carrying a continuation; it fails the page a case's
``inject`` names, with an exception carrying ``retryable``; and it seeds
the fault named, as ``expected.json``'s ``faults`` say.

REPORT is the JSON the kit's tests read (``contract/tests/
test_t25_provider_kit.py``'s docstring): each case's ``fetched`` (the
request of every fetch the provider path answered or failed, one per
page), and its ``answer`` (the columns and rows the core admitted) or its
``error`` (the core's code, kind, retryable and fields). Each case is
judged by ``expected.json``'s rules; the runner exits 0 when every case
passes, 1 when any fails, 2 when it cannot run.

It needs a test-ops build of ``sdm_client._native``. Started on a build
without the test-only operations (the shipped wheel, in CI's
``contract-python`` jobs), it runs itself again under ``SDM_KIT_PYTHON``,
an interpreter whose build has them, where that is named, and otherwise
exits 2, naming the build.

``--load sqlite OUT`` writes the dataset into a new SQLite file at OUT
through the contract loader: the pure-Rust host's kit loads its SQLite so
(``sdm-contract-run --kit sqlite``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / 'contract/fixtures/provider-kit'
FORMAT = 'sdm-provider-kit/1'
DATABASES = ('duckdb', 'sqlite', 'postgres')
FAULTS = ('truncate', 'silent_capability')
#: Each database's Python provider: module and class.
PROVIDERS = {
    'duckdb': ('sdm_client._rust.providers.duckdb', 'DuckDBProvider'),
    'sqlite': ('sdm_client._rust.providers.sqlite', 'SqliteProvider'),
    'postgres': ('sdm_client._rust.providers.postgres', 'PostgresProvider'),
}


class Unrunnable(Exception):
    r"""The kit cannot run here, named."""


def expected() -> dict[str, Any]:
    return json.loads((KIT / 'expected.json').read_text(encoding='utf-8'))


def _loader() -> ModuleType:
    name = 'provider_kit_databases'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/runners/databases.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# --- the dataset, on each database ----------------------------------------------


def _duck() -> Any:
    import duckdb

    duck = duckdb.connect()
    duck.execute((KIT / 'dataset.sql').read_text(encoding='utf-8'))
    return duck


def load_sqlite(out: Path) -> None:
    r"""The dataset in a new SQLite file at ``out``, by the contract
    loader."""
    duck = _duck()
    try:
        _loader().load(duck, 'sqlite', out)
    finally:
        duck.close()


@contextmanager
def connection(database: str) -> Iterator[Any]:
    r"""A connection to the dataset on ``database``, as a user opens one."""
    if database == 'duckdb':
        duck = _duck()
        try:
            yield duck
        finally:
            duck.close()
        return
    if database == 'sqlite':
        import adbc_driver_sqlite.dbapi

        with tempfile.TemporaryDirectory(prefix='sdm-provider-kit-') as where:
            path = Path(where) / 'kit.sqlite'
            load_sqlite(path)
            with adbc_driver_sqlite.dbapi.connect(str(path)) as opened:
                yield opened
        return
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        raise Unrunnable('Postgres needs SDM_POSTGRES_TEST_DSN')
    import psycopg
    from psycopg.conninfo import make_conninfo

    name = f'sdm_kit_{uuid.uuid4().hex[:16]}'
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE {name}')
    try:
        scoped = make_conninfo(dsn, dbname=name)
        duck = _duck()
        try:
            _loader().load(duck, 'postgres', scoped)
        finally:
            duck.close()
        with psycopg.connect(scoped, autocommit=True) as opened:
            yield opened
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS {name} WITH (FORCE)')


# --- the provider path ----------------------------------------------------------


class InjectedFailure(Exception):
    r"""The failure a case's ``inject`` makes of one page's fetch, as a
    provider's exception that says whether trying again may succeed."""

    def __init__(self, retryable: bool) -> None:
        super().__init__('an injected provider failure')
        self.retryable = retryable


def _table(answer: Any) -> Any:
    import pyarrow as pa

    if isinstance(answer, pa.Table):
        return answer
    return pa.RecordBatchReader._import_from_c_capsule(
        answer.__arrow_c_stream__()
    ).read_all()


class ProviderPath:
    r"""Between the run loop and ``provider``: each answer in pages of
    ``page_rows`` rows, the fault named seeded, the page ``inject`` names
    failed, and the request of every fetch recorded in ``fetched``."""

    def __init__(
        self,
        provider: Callable[..., Any],
        page_rows: int,
        fault: str | None,
        inject: dict[str, Any] | None,
    ) -> None:
        self._provider = provider
        self._page_rows = page_rows
        self._fault = fault
        self._inject = inject
        self._held: dict[str, list[Any]] = {}
        self.fetched: list[str] = []

    def __call__(self, step: dict[str, Any], keys: Any) -> Any:
        from sdm_client._rust.run import Page

        name = step['request']['request']
        self.fetched.append(name)
        if self._inject and len(self.fetched) == self._inject['page']:
            raise InjectedFailure(self._inject['retryable'] is True)
        continuation = step.get('continuation')
        if continuation is None:
            table = _table(self._provider(step, keys))
            if self._fault == 'silent_capability' and name == 'neighbors':
                table = table.slice(0, 0)
            rows = table.num_rows
            pages = [
                table.slice(at, self._page_rows)
                for at in range(0, rows, self._page_rows)
            ] or [table]
            if self._fault == 'truncate' and len(pages) > 1:
                pages.pop()
            continuation = uuid.uuid4().hex
            self._held[continuation] = pages
        pages = self._held.pop(continuation)
        page = pages.pop(0)
        if not pages:
            return page
        self._held[continuation] = pages
        return Page(page, continuation)


# --- one case -------------------------------------------------------------------


def _resolved(case: dict[str, Any], by_id: dict[str, Any]) -> dict[str, Any]:
    named = case.get('same_as') or case.get('request_of')
    return by_id[named] if named else case


def _fetch(case: dict[str, Any], source: dict[str, Any], database: str) -> Any:
    r"""The test-only fetch of ``case``: its request, the provider whose
    plan the core makes, and what the case changes of what it declares."""
    request = dict(source['request'])
    if 'seeds' in source:
        request['seeds'] = {
            'attachment': 2,
            'columns': [
                {'name': name, 'type': {'type': 'int64'}, 'nullable': False}
                for name in ('key', 'batch')
            ],
        }
    capabilities = case.get('capabilities', {})
    fetch: dict[str, Any] = {
        'request': request,
        'provider': {'provider': 'sql', 'dialect': database},
        'echo': True,
        'undeclared': list(capabilities.get('withhold', [])),
    }
    if 'fallback' in capabilities:
        fetch['fallback'] = capabilities['fallback']
    return fetch


def _cell(value: Any) -> Any:
    from decimal import Decimal

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def run_case(
    case: dict[str, Any],
    by_id: dict[str, Any],
    database: str,
    connected: Any,
    page_rows: int,
    fault: str | None,
) -> dict[str, Any]:
    r"""What ``case`` gave on ``database``'s provider: the fetches, and the
    answer or the error."""
    import importlib

    import pyarrow as pa
    from sdm_client._rust import codec
    from sdm_client._rust.run import run
    from sdm_client.errors import StructuredDataError

    source = _resolved(case, by_id)
    module, name = PROVIDERS[database]
    provider = getattr(importlib.import_module(module), name)(connected)
    path = ProviderPath(provider, page_rows, fault, case.get('inject'))
    attachments: dict[int, Any] = {
        1: codec.attachment(_fetch(case, source, database))
    }
    operation: dict[str, Any] = {'op': 'test', 'test': 'fetch', 'fetch': 1}
    if 'seeds' in source:
        operation = {'op': 'test', 'test': 'keyed', 'fetch': 1, 'keys': 2}
        attachments[2] = pa.table(
            {
                'key': pa.array([s[0] for s in source['seeds']], pa.int64()),
                'batch': pa.array([s[1] for s in source['seeds']], pa.int64()),
            }
        )
    seen: dict[str, Any] = {'id': case['id'], 'fetched': path.fetched}
    try:
        finished = run(operation, attachments, provider=path)
    except StructuredDataError as error:
        details = dict(error.details)
        seen['error'] = {
            'code': error.code,
            'kind': details.pop('kind', None),
            'retryable': details.pop('retryable', None),
            'fields': {
                key: _cell(value)
                for key, value in details.items()
                if key != 'driver_class'
            },
        }
        return seen
    table = _table(finished.take(finished.result['data']))
    seen['answer'] = {
        'columns': table.column_names,
        'rows': [
            [_cell(value) for value in row.values()]
            for row in table.to_pylist()
        ],
    }
    return seen


# --- the judgment, by expected.json's rules ------------------------------------------


def _value(value: Any) -> Any:
    r"""A value as compared: a number by its value, a zero of either sign
    one value, anything else as it is."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) + 0.0
    return value


def _fetched_problem(case: dict[str, Any], fetched: list[str]) -> list[str]:
    r"""Where ``case`` names its fetches exactly, how ``fetched``
    differs."""
    if fetched != case['fetched']:
        return [f'fetched {fetched}, not exactly {case["fetched"]}']
    return []


def _key(row: list[Any]) -> str:
    return json.dumps([_value(v) for v in row])


def _answer_problem(got: dict[str, Any], wanted: dict[str, Any]) -> str | None:
    names = got['columns']
    missing = [c for c in wanted['columns'] if c not in names]
    if missing:
        return f'no column {missing} among {names}'
    at = [names.index(c) for c in wanted['columns']]
    rows = [[row[i] for i in at] for row in got['rows']]
    if 'any_of' in wanted:
        pool = [_key(r) for r in wanted['any_of']]
        if len(rows) != wanted['count']:
            return f'{len(rows)} rows, not {wanted["count"]}'
        for row in rows:
            if _key(row) not in pool:
                return f'{row} is not one of the rows it may be'
            pool.remove(_key(row))
        return None
    if wanted['ordered']:
        if [_key(r) for r in rows] != [_key(r) for r in wanted['rows']]:
            return f'rows {rows}, not {wanted["rows"]} in order'
    elif sorted(map(_key, rows)) != sorted(map(_key, wanted['rows'])):
        return f'rows {rows}, not the multiset {wanted["rows"]}'
    return None


def judge(
    case: dict[str, Any],
    seen: dict[str, Any],
    by_id: dict[str, Any],
    page_rows: int,
) -> list[str]:
    r"""Why ``seen`` does not pass ``case``; empty when it does."""
    fetched = seen['fetched']
    wanted = (
        by_id[case['same_as']]['answer']
        if 'same_as' in case
        else case.get('answer')
    )
    if wanted is not None:
        if 'answer' not in seen:
            return [f'an error, {seen.get("error")}, not an answer']
        problem = _answer_problem(seen['answer'], wanted)
        if problem:
            return [problem]
        if 'fetched' in case:
            return _fetched_problem(case, fetched)
        withheld = case.get('capabilities', {}).get('withhold', [])
        if withheld:
            if not fetched or set(fetched) & set(withheld):
                return [
                    f'fetched {fetched}: the fallback asks none of {withheld}'
                ]
            return []
        rows = wanted['count'] if 'any_of' in wanted else len(wanted['rows'])
        name = _resolved(case, by_id)['request']['request']
        if fetched.count(name) < math.ceil(rows / page_rows):
            return [
                f'fetched {fetched}: {rows} rows in fewer than pages of {page_rows}'
            ]
        return []
    if 'error' not in seen:
        return [f'an answer, not the error {case["error"]}']
    got, error = seen['error'], case['error']
    for name in ('code', 'kind', 'retryable'):
        if got[name] != error[name]:
            return [f'{name} {got[name]!r}, not {error[name]!r}']
    for name, value in error['fields'].items():
        if str(got['fields'].get(name)) != str(value):
            return [f'field {name} {got["fields"].get(name)!r}, not {value!r}']
    if 'fetched' in case:
        return _fetched_problem(case, fetched)
    injected = case.get('inject')
    if injected and len(fetched) < injected['page']:
        return [f'fetched {fetched}: page {injected["page"]} was never asked']
    return []


# --- the run --------------------------------------------------------------------


def run_kit(database: str, fault: str | None) -> tuple[dict[str, Any], bool]:
    r"""The report of every case on ``database``, ``fault`` seeded, and
    whether every case passed."""
    kit = expected()
    by_id = {case['id']: case for case in kit['cases']}
    page_rows = kit['page_rows']
    cases = []
    passed = True
    with connection(database) as connected:
        for case in kit['cases']:
            seen = run_case(case, by_id, database, connected, page_rows, fault)
            problems = judge(case, seen, by_id, page_rows)
            if problems:
                passed = False
                print(f'{case["id"]}: {"; ".join(problems)}', file=sys.stderr)
            cases.append(seen)
    report = {
        'format': FORMAT,
        'host': 'python-core',
        'database': database,
        'fault': fault,
        'cases': cases,
    }
    return report, passed


def _test_ops() -> bool:
    r"""Whether this interpreter's ``sdm_client._native`` has the
    test-only operations."""
    from sdm_client import _native

    return getattr(_native, 'TEST_OPS', False) is True


def _elsewhere(argv: list[str]) -> int:
    r"""This run, under ``SDM_KIT_PYTHON``'s interpreter, whose build has
    the test-only operations; 2 where none is named."""
    python = os.environ.get('SDM_KIT_PYTHON')
    # A venv's interpreter is a link: compared as named, not resolved.
    if not python or Path(python).absolute() == Path(sys.executable).absolute():
        print(
            'provider kit: it needs a test-ops build of sdm_client._native; '
            'SDM_KIT_PYTHON names an interpreter with one',
            file=sys.stderr,
        )
        return 2
    done = subprocess.run(
        [python, '-I', str(Path(__file__).resolve()), *argv],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return done.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--database', choices=DATABASES)
    parser.add_argument('--fault', choices=FAULTS)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--load', nargs=2, metavar=('DATABASE', 'OUT'))
    arguments = parser.parse_args(argv)
    if arguments.load:
        database, out = arguments.load
        if database != 'sqlite' or arguments.database or arguments.out:
            parser.error('--load takes sqlite and a file, alone')
        load_sqlite(Path(out))
        return 0
    if not arguments.database or not arguments.out:
        parser.error('--database and --out are required')
    if not _test_ops():
        return _elsewhere(argv if argv is not None else sys.argv[1:])
    try:
        report, passed = run_kit(arguments.database, arguments.fault)
    except Unrunnable as why:
        print(f'provider kit: {why}', file=sys.stderr)
        return 2
    arguments.out.write_text(
        json.dumps(report, indent=1) + '\n', encoding='utf-8'
    )
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
