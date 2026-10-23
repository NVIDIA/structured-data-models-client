# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L5 for Java (T3.12; design, choice 10, "Data access"; the brief's exit 2):
the same tables reach the core as identical Arrow through Java's JDBC
drivers and Python's drivers, and a seeded unit or scale change in one JDBC
path fails.

The table is :mod:`test_l5_conversion`'s (a column of each type of the type
map, built on DuckDB and loaded into SQLite and Postgres by the contract
loader). Each column is read through each database's Python provider on
the core's own plan (the test-only fetch, as L5 reads it), recording the
step the core asked and the provider's answer; the Java program
(``contract/runners/java/ContractRunner.java``, its ``fetch`` call) answers
the same steps through the JDBC provider on the same database, and each
answer must be the Python provider's, its schema's metadata included, value
for value. The Java program runs with the JVM's default zone set to a zone
other than UTC (``-Duser.timezone``), so no answer reads the JVM's zone.

It needs the JAR under test (``SDM_JAVA_JAR``, with ``target/lib``,
``target/jdbc`` and, for a test-ops native, ``SDM_JAVA_NATIVE`` beside it)
and Postgres (``SDM_POSTGRES_TEST_DSN``): without either it returns without
asserting, unless ``SDM_REQUIRE_JAVA`` or ``SDM_REQUIRE_POSTGRES`` is set,
when it fails (H26), as CI's Java L5 job sets them.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from sdm_client._rust import codec
from sdm_client._rust.run import run

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


L5 = _module('t312_l5_conversion', HERE / 'test_l5_conversion.py')
RUNNER = _module('t312_run_java', ROOT / 'contract/runners/run_java.py')

#: A zone other than UTC the Java program's JVM defaults to.
JVM_ZONE = 'Asia/Kolkata'


class Recording:
    r"""A Python provider whose steps and answers are kept, in order."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.steps: list[dict[str, Any]] = []
        self.answers: list[pa.Table] = []

    def __call__(self, step: dict[str, Any], keys: Any = None) -> Any:
        answer = self.inner(step, keys)
        table = pa.RecordBatchReader._import_from_c_capsule(
            answer.__arrow_c_stream__()
        ).read_all()
        self.steps.append(step)
        self.answers.append(table)
        return table


def _fetched(provider: Any, dialect: str, column: str) -> None:
    r"""``column`` (with ``id``) planned by the core and read through
    ``provider``; the core's refusal of what it read is no matter here."""
    from sdm_client.errors import StructuredDataError

    fetch = {
        'request': {
            'request': 'scan',
            'table': {'source': ['t']},
            'columns': ['id', column],
            'filters': [],
            'limit': None,
        },
        'provider': {'provider': 'sql', 'dialect': dialect},
        'echo': True,
    }
    try:
        run(
            {'op': 'test', 'test': 'fetch', 'fetch': 1},
            {1: codec.attachment(fetch)},
            provider=provider,
        )
    except StructuredDataError:
        pass


def _java() -> tuple[Path, Path | None] | None:
    jar = os.environ.get('SDM_JAVA_JAR')
    if not jar:
        assert not os.environ.get('SDM_REQUIRE_JAVA'), (
            'SDM_REQUIRE_JAVA is set but SDM_JAVA_JAR names no JAR'
        )
        return None
    native = os.environ.get('SDM_JAVA_NATIVE')
    return Path(jar), Path(native) if native else None


@pytest.fixture(scope='module')
def source(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Any | None]:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        yield None
        return
    made = L5.Source(tmp_path_factory.mktemp('l5java'), dsn)
    try:
        yield made
    finally:
        made.drop()


def _connect(source: Any, database: str) -> dict[str, Any]:
    if database == 'duckdb':
        return {}
    if database == 'sqlite':
        return {'path': str(source.path)}
    from psycopg.conninfo import conninfo_to_dict

    parts = conninfo_to_dict(source.scoped)
    return {
        'url': f'jdbc:postgresql://{parts.get("host", "localhost")}:'
        f'{parts.get("port", "5432")}/{parts["dbname"]}',
        'user': parts.get('user'),
        'password': parts.get('password'),
    }


def _python_answers(source: Any) -> dict[str, Recording]:
    r"""Every column through every database's Python provider, recorded."""
    recorded = {
        database: Recording(provider)
        for database, provider in source.providers().items()
    }
    for database, recording in recorded.items():
        for column in L5.COLUMNS:
            _fetched(recording, database, column)
    return recorded


def _java_answers(
    source: Any,
    java: tuple[Path, Path | None],
    database: str,
    steps: list[dict[str, Any]],
    seed: dict[str, str] | None = None,
) -> list[pa.Table | str]:
    jar, native = java
    program = RUNNER.Program(
        java=RUNNER.java_binary(None),
        path=RUNNER.classpath(jar, None),
        native=native,
        options=(f'-Duser.timezone={JVM_ZONE}',),
    )
    try:
        answer = program.ask(
            {
                'op': 'fetch',
                'case': f'l5@{database}',
                'database': database,
                'connect': _connect(source, database),
                # DuckDB's table is built by the Java program in memory, in
                # the zone the Python connection reads in.
                'setup': [f"SET GLOBAL TimeZone = '{L5.ZONE}'", *L5.SETUP]
                if database == 'duckdb'
                else [],
                'steps': steps,
                'seed': seed,
            }
        )
    finally:
        program.close()
    return [
        pa.ipc.open_stream(base64.b64decode(each['arrow'])).read_all()
        if 'arrow' in each
        else each['error']
        for each in answer['answers']
    ]


def _differences(python: Recording, java: list[pa.Table | str]) -> list[str]:
    found = []
    for step, mine, theirs in zip(
        python.steps, python.answers, java, strict=True
    ):
        columns = step['request'].get('columns')
        if isinstance(theirs, str):
            found.append(f'{columns}: Java failed ({theirs})')
            continue
        if not mine.schema.equals(theirs.schema, check_metadata=True):
            found.append(f'{columns}: schema {theirs.schema} not {mine.schema}')
        elif not mine.equals(theirs):
            found.append(f'{columns}: values differ')
    return found


@pytest.fixture(scope='module')
def python_answers(source: Any | None) -> dict[str, Recording] | None:
    return None if source is None else _python_answers(source)


@pytest.mark.parametrize('database', ['duckdb', 'sqlite', 'postgres'])
def test_each_column_reaches_the_core_as_python_s_driver_hands_it(
    source: Any | None,
    python_answers: dict[str, Recording] | None,
    database: str,
) -> None:
    java = _java()
    if source is None or java is None or python_answers is None:
        return
    python = python_answers[database]
    assert len(python.steps) == len(L5.COLUMNS)
    problems = _differences(
        python, _java_answers(source, java, database, python.steps)
    )
    assert not problems, '\n'.join(problems)


@pytest.mark.parametrize(
    ('column', 'read', 'read_as'),
    [
        # A unit: a timestamp read as an instant in UTC.
        ('ts', 'timestamp', 'timestamptz'),
        # A scale: a numeric read as a double.
        ('dec', 'numeric', 'float8'),
    ],
)
def test_a_unit_or_scale_change_in_one_jdbc_path_fails(
    source: Any | None,
    python_answers: dict[str, Recording] | None,
    column: str,
    read: str,
    read_as: str,
) -> None:
    java = _java()
    if source is None or java is None or python_answers is None:
        return
    python = python_answers['postgres']
    at = list(L5.COLUMNS).index(column)
    seeded = _java_answers(
        source,
        java,
        'postgres',
        [python.steps[at]],
        seed={'read': read, 'as': read_as},
    )
    single = Recording(None)
    single.steps, single.answers = [python.steps[at]], [python.answers[at]]
    assert _differences(single, seeded), (column, read, read_as)
