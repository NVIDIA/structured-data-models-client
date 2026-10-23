#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Java runner of the contract cases (design choice 10, "Tests").

    python3 -I contract/runners/run_java.py --out FILE [--jar JAR]
        [--classpath DIR] [--drivers DIR] [--native LIBRARY] [--java JAVA]
        [--database duckdb|sqlite|postgres] [CASE_OR_DIRECTORY...]

It reads each case with the Python runner's case reader
(``run_python.load_cases``), so both hosts read the same files the same
way, and hands the Java program (``contract/runners/java/ContractRunner.java``, run in
source-file mode against the JAR under test) only the calls to make and
what to make them with. The program makes each through the public Java
API and writes back what it sent the core and what it was answered; this
file turns each answer into an observation in
``contract/observation.schema.json``'s shape, runner ``java``, judged by
the checkpoint's comparator as rust-host's are.

A Java leg runs exactly where the Java host has a provider for it (design
choice 10): S0's cases, which read no data, from T3.11; from T3.12 an S1
case on DuckDB, SQLite and Postgres (``--database``), whose paths are all
Java's (``scope-java.yaml``: no ranking or forecast) and whose inputs Java's
types can say (:data:`UNSAID`). A case on no Java leg is not run, and is
named on stderr.

An S1 case's data is loaded as the Python runner loads it: on DuckDB the
Java program runs the case's base and setup statements itself, in memory;
SQLite and Postgres are loaded from that DuckDB by the one contract loader
(``databases.py``), SQLite into a new file and Postgres into a new database
on ``SDM_POSTGRES_TEST_DSN``'s server, which the program opens with its
JDBC driver (``--drivers``, Maven's ``target/jdbc``). Each case gets the
scripted endpoint (``scripted_endpoint.py``) the Python runner serves it,
and the program makes its public calls (``Schema.fromSource``, the edits,
``PredictiveQuery.materialize``, ``SdmClient.predict``), telling what
crossed the binding and what the provider ran; this file builds the
observation's ``core`` and ``driver`` sections from that with the Python
runner's own capture (``_Capture``) and canonical values
(``arrow_values``), so both hosts are recorded alike. Each call the
endpoint received must be, byte for byte, the call the core wrote: the
transport sends the core's bytes unchanged (choice 10, "Model calls"),
else a runner failure.

The Java outcome is the core's answer as the route maps it, as rust-host's
is: an accepted query a success, a refused one ``INVALID_REQUEST``, a raised
``StructuredDataException`` a refusal with its code (the model endpoint's
refusal its code, which the scripted endpoint gives none of); the stage and
the exception class are left null.

``--jar`` is the JAR under test (default: ``SDM_JAVA_JAR``) and
``--classpath`` the directory holding
its runtime dependencies (Maven's ``target/lib``, default: the JAR's
sibling ``lib``); ``--native`` names a native library to load in place of
the JAR's own (``sdm.native.library``); ``--java`` the ``java`` to run
(default: ``$JAVA_HOME/bin/java``, else ``java`` on the path). Every Java
run takes the two runtime flags choice 10 names (:data:`FLAGS`).

A runner failure (the Java program failing, an answer missing or not
holding to the observation schema) exits non-zero and writes no output, an
earlier run's included. No message text is recorded (decision H17).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PROGRAM = HERE / 'java' / 'ContractRunner.java'

sys.path.insert(0, str(HERE))
import run_python  # noqa: E402

#: The runtime flags of design choice 10: JNI's native access, and the
#: access Arrow Java's memory module needs.
FLAGS = (
    '--enable-native-access=ALL-UNNAMED',
    '--add-opens=java.base/java.nio=ALL-UNNAMED',
)
#: The databases the Java host has a provider for (T3.12).
PROVIDERS: tuple[str, ...] = ('duckdb', 'sqlite', 'postgres')
#: Each S1 case Java's types cannot say, and why: the call is refused at
#: compile time in Java, so the case has no Java leg (register JV-1).
UNSAID = {
    'wording.key-number': 'a primary key named by a number',
    'wording.time-number': 'a time column named by a number',
}
#: The paths Java's manifest names, for the slices it gates.
JAVA_MANIFEST = REPO / 'docs/design/rust-core/phase-0/scope-java.yaml'
#: How long one Java run may take, in seconds.
TIMEOUT = 900


class RunnerFailure(Exception):
    r"""The runner, not a case, failed; its text never quotes an input."""


def java_paths() -> frozenset[str]:
    r"""The S0 and S1 paths of Java's manifest."""
    import yaml

    manifest = yaml.safe_load(JAVA_MANIFEST.read_text(encoding='utf-8'))
    return frozenset(
        str(item['id'])
        for item in manifest['paths']
        if item['slice'] in ('S0', 'S1')
    )


def on_java(
    case: dict[str, Any], database: str = 'duckdb', paths: Any = None
) -> bool:
    r"""Whether the case runs on the Java leg of ``database``: an S0 case
    (validation, which reads no data) on DuckDB's, or an S1 case on a
    database it runs on, whose paths are all Java's, whose inputs Java's
    types can say (:data:`UNSAID`)."""
    core = case.get('core') or {}
    if case.get('slice') == 'S0' and 'validate_query' in core:
        return database == 'duckdb'
    if case.get('slice') != 'S1' or database not in PROVIDERS:
        return False
    if case['id'] in UNSAID:
        return False
    if database not in run_python.case_databases(case):
        return False
    known = java_paths() if paths is None else paths
    return all(
        run_python.on_database(path, database) in known
        for path in case['paths']
    )


def call_of(case: dict[str, Any]) -> dict[str, Any]:
    r"""What the Java program is to call for an S0 case: its validation,
    with the case's query, schema and profile."""
    spec = case['core']['validate_query']
    return {
        'op': 'validate_query',
        'case': case['id'],
        'query': case['input']['pql'],
        'schema': spec['schema'],
        'profile': spec['profile'],
    }


def s1_call(
    case: dict[str, Any],
    database: str,
    bases: dict[str, list[str]],
    connect: dict[str, Any],
    endpoint: str,
) -> dict[str, Any]:
    r"""What the Java program is to call for an S1 case on ``database``: the
    case's public calls, as run_python.py's ``_calls`` makes them, with what
    opens its data and the endpoint's URL."""
    given = case['input']
    validate = None
    if 'validate_query' in case.get('core', {}):
        # The case's validation, which the Python runner's forced route
        # asks before materialize, as rust-host asks the case's core
        # operations: the schema the case names as it is stored there.
        spec = case['core']['validate_query']
        validate = {
            'schema': _compare().stored_schema(
                spec['schema'], case, bases, database
            ),
            'profile': spec['profile'],
        }
    return {
        'op': 'case',
        'validate': validate,
        'case': case['id'],
        'database': database,
        'connect': connect,
        # DuckDB's data is built by the program, in memory, as the Python
        # runner builds it; elsewhere it is loaded before.
        'setup': [*bases[given['base']], *given.get('setup', [])]
        if database == 'duckdb'
        else [],
        'until': case['until'],
        'pql': given['pql'],
        'from_source': given.get('from_source', {}),
        'edits': given.get('edits', []),
        'materialize': given['materialize'],
        'outputs': given.get('outputs'),
        'endpoint': endpoint,
    }


def _compare() -> Any:
    r"""The comparator beside the runner, whose ``stored_schema`` gives the
    schema a case's validation is handed on a database."""
    import importlib.util

    path = REPO / 'scripts/rust_core/compare.py'
    spec = importlib.util.spec_from_file_location('sdm_compare', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('sdm_compare', module)
    spec.loader.exec_module(module)
    return module


def java_binary(given: str | None) -> str:
    if given:
        return given
    home = os.environ.get('JAVA_HOME')
    if home and (Path(home) / 'bin' / 'java').is_file():
        return str(Path(home) / 'bin' / 'java')
    found = shutil.which('java')
    if found is None:
        raise RunnerFailure('no java: pass --java or set JAVA_HOME')
    return found


def classpath(jar: Path, lib: Path | None, drivers: Path | None = None) -> str:
    directory = lib if lib is not None else jar.parent / 'lib'
    if not jar.is_file():
        raise RunnerFailure(f'{jar}: not a file')
    if not directory.is_dir():
        raise RunnerFailure(
            f'{directory}: no dependency directory (mvn package writes '
            'target/lib)'
        )
    parts = [str(jar), str(directory / '*')]
    jdbc = drivers if drivers is not None else jar.parent / 'jdbc'
    if jdbc.is_dir():
        parts.append(str(jdbc / '*'))
    return os.pathsep.join(parts)


class Program:
    r"""The Java program, running: one call a line in, one answer a line
    out, so one JVM answers every case."""

    def __init__(
        self,
        *,
        java: str,
        path: str,
        native: Path | None,
        options: tuple[str, ...] = (),
    ) -> None:
        command = [java, *FLAGS, *options]
        if native is not None:
            command.append(f'-Dsdm.native.library={native.resolve()}')
        command += ['-cp', path, str(PROGRAM)]
        self._process = subprocess.Popen(  # noqa: S603 - the runner's own JVM
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding='utf-8',
        )

    def ask(self, call: dict[str, Any]) -> dict[str, Any]:
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        try:
            self._process.stdin.write(json.dumps(call) + '\n')
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        except (BrokenPipeError, OSError):
            line = ''
        if not line:
            code = self._process.wait(timeout=TIMEOUT)
            raise RunnerFailure(
                f'the Java program failed with exit status {code}'
            )
        answer: dict[str, Any] = json.loads(line)
        if answer.get('case') != call['case']:
            raise RunnerFailure('the Java program answered another call')
        return answer

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            self._process.kill()
            raise RunnerFailure('the Java program did not finish') from None


def observation(
    case: dict[str, Any],
    case_digest: str,
    answer: dict[str, Any],
    commit: dict[str, Any],
) -> dict[str, Any]:
    r"""One observation of an S0 ``case`` from the Java program's
    ``answer``."""
    error = answer['error']
    if error is not None:
        outcome = {
            'result': 'refusal',
            'stage': None,
            'error': None,
            'code': error['code'],
        }
    elif answer['accepted']:
        outcome = {
            'result': 'success',
            'stage': None,
            'error': None,
            'code': None,
        }
    else:
        outcome = {
            'result': 'refusal',
            'stage': None,
            'error': None,
            'code': 'INVALID_REQUEST',
        }
    identity = answer['identity']
    if identity['providers'] != list(PROVIDERS):
        raise RunnerFailure(
            'the Java host reports providers this runner does not run legs for'
        )
    return {
        'format': 'sdm-observation/1',
        'case': case['id'],
        'identity': {
            'runner': 'java',
            'cell': None,
            'platform': identity['platform'],
            'commit': commit,
            'reference': None,
            'native': None,
            'native_loaded': None,
            'operations': answer['operations'],
            'contract': identity['contract'],
            'versions': {'java': identity['java']},
            'inputs': {'case': case_digest, 'bases': None},
        },
        'outcome': outcome,
        'host': None,
        'core': {
            'inputs': answer['inputs'],
            'fetches': [],
            'accepted': answer['accepted'],
            'described': None,
            'materialized': None,
            'predicted': None,
            'error': error,
        },
    }


def _arrow(text: str) -> Any:
    r"""An Arrow IPC stream the Java program sent, in base64, as a table."""
    import base64

    import pyarrow as pa

    return pa.ipc.open_stream(base64.b64decode(text)).read_all()


def captured(events: list[dict[str, Any]]) -> Any:
    r"""What crossed the binding, as the Python runner's capture records the
    Python host's (``_Capture``): each event the Java program was told,
    replayed into one, in order."""
    import base64
    from types import SimpleNamespace

    capture = run_python._Capture()
    for event in events:
        kind = event['event']
        if kind == 'started':
            attachments: dict[int, Any] = {}
            for key, value in event['attachments'].items():
                attachments[int(key)] = (
                    _arrow(value['arrow'])
                    if 'arrow' in value
                    else base64.b64decode(value['bytes'])
                )
            capture.started(base64.b64decode(event['message']), attachments)
        elif kind == 'stepped':
            capture.stepped(event['step'])
        elif kind == 'taken':
            value = (
                _arrow(event['arrow'])
                if 'arrow' in event
                else base64.b64decode(event['bytes'])
            )
            capture.taken(event['id'], value)
        elif kind == 'done':
            capture.done(event['result'])
        elif kind == 'failed':
            capture.failed(
                SimpleNamespace(kind=event['kind'], code=event['code'])
            )
        elif kind == 'warned':
            capture.warned(json.dumps({'warnings': event['warnings']}).encode())
    return capture


def driver_record(
    events: list[dict[str, Any]],
    logged: list[str] | None,
    database: str = 'duckdb',
) -> dict[str, Any]:
    r"""What the provider ran, as the Python runner's driver capture records
    it: each statement in the order run, with its cursor (each fetch's own,
    numbered from 1), the public call it ran in and its result read whole as
    Arrow (``query_arrow``, as rust-host records one), each key set bound
    with the statement that made it, and the query log: DuckDB's own on
    DuckDB, else the provider's record itself."""
    statements: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    cursors = 0
    for event in events:
        kind = event['event']
        if kind == 'cursor':
            cursors += 1
        elif kind == 'statement':
            statements.append(
                {
                    'handle': event['handle'],
                    'stage': event['stage'],
                    'sql': event['sql'],
                    'parameters': None,
                    'results': [],
                    'error': None,
                }
            )
        elif kind == 'answered' and statements:
            table = _arrow(event['arrow'])
            statements[-1]['results'].append(
                {
                    'method': 'query_arrow',
                    'rows': table.num_rows,
                    'schema': schema_of(table),
                    'digest': run_python._arrow_digest(table),
                    'values': run_python.arrow_values(
                        table, described(table, event['columns'], database)
                    ),
                }
            )
        elif kind == 'statement_failed' and statements:
            statements[-1]['error'] = event['error']
        elif kind == 'bound':
            table = _arrow(event['arrow'])
            bindings.append(
                {
                    'handle': event['handle'],
                    'stage': event['stage'],
                    'name': event['name'],
                    'type': 'arrow',
                    'sql': event['sql'],
                    'rows': table.num_rows,
                    'schema': schema_of(table),
                    'digest': run_python._arrow_digest(table),
                    'values': run_python.arrow_values(table),
                }
            )
    capture = {
        'statements': statements,
        'bindings': bindings,
        'events': [],
        'incomplete': [],
        'passed_through': [],
        'cursors': cursors,
    }
    texts = [statement['sql'] for statement in statements]
    if logged is None:
        # Off DuckDB the provider's own record is the log: each statement
        # it ran, and each key set it bound.
        query_log = {
            'statements': len(statements),
            'bindings': len(bindings),
            'compared': len(statements),
            'differing': [],
        }
    else:
        # DuckDB's log: a registered relation is '', and the statement that
        # keeps it as a temporary table is its binding's; each other text is
        # a statement, compared in order with the one recorded.
        made = {binding['sql'] for binding in bindings}
        ran = [text for text in logged if text and text not in made]
        bound = sum(1 for text in logged if text in made)
        compared, differing = 0, []
        if len(ran) == len(texts):
            for position, (sql, text) in enumerate(
                zip(texts, ran, strict=True)
            ):
                if sql.lstrip().upper().startswith('PRAGMA'):
                    continue
                compared += 1
                if sql != text:
                    differing.append(position)
        query_log = {
            'statements': len(ran),
            'bindings': bound,
            'compared': compared,
            'differing': differing,
        }
    return {'capture': capture, 'query_log': query_log}


def schema_of(table: Any) -> list[dict[str, str]]:
    r"""A result's Arrow schema as the capture records one: each field's
    name and type, a field marked as JSON text (the ``arrow.json``
    metadata, which pyarrow reads as its extension type) by the type it is
    stored as, as the Python providers' tables are recorded."""
    import pyarrow as pa

    return [
        {
            'name': field.name,
            'type': str(
                field.type.storage_type
                if isinstance(field.type, pa.BaseExtensionType)
                else field.type
            ),
        }
        for field in table.schema
    ]


def described(
    table: Any, columns: list[dict[str, Any]], database: str
) -> list[str]:
    r"""The type each column of a result is described by, as the Python
    runner's capture reads it from the reference's driver: on DuckDB its own
    names (the JDBC driver's description gives the same); on SQLite the
    Arrow type ADBC describes each column by, which is the result's own; on
    Postgres psycopg's display of the type (``type_display``): the name with
    a ``varchar``'s, ``bpchar``'s or ``numeric``'s declared modifier, an
    array's element name and ``[]``."""
    if database == 'sqlite':
        return [str(field.type) for field in table.schema]
    if database == 'duckdb':
        return [str(column['type']) for column in columns]
    shown = []
    for column in columns:
        name = str(column['type'])
        precision, scale = column['precision'], column['scale']
        array = name.startswith('_')
        base = name[1:] if array else name
        if (
            base in ('varchar', 'bpchar')
            and 0 < precision < 2147483647
            and not array
        ):
            base = f'{base}({precision})'
        elif base == 'numeric' and precision > 0 and not array:
            base = f'numeric({precision},{scale if scale < 1000 else scale - 2048})'
        shown.append(f'{base}[]' if array else base)
    return shown


def calls_sent(events: list[dict[str, Any]]) -> list[bytes]:
    r"""Each model call's body the core wrote, in order: the bytes
    attachment taken right after the step that asked for the call, by the
    id that step named."""
    import base64

    sent: list[bytes] = []
    asked: int | None = None
    for event in events:
        if event['event'] == 'stepped':
            step = event['step']
            asked = (
                step.get('body') if step.get('step') == 'call_model' else None
            )
        elif event['event'] == 'taken' and asked is not None:
            if event['id'] == asked and 'bytes' in event:
                sent.append(base64.b64decode(event['bytes']))
            asked = None
    return sent


def check_sent(
    case: dict[str, Any], events: list[dict[str, Any]], received: Any
) -> None:
    r"""Every call the endpoint received is the call the core wrote, byte
    for byte, in order (choice 10, "Model calls"): a runner failure
    otherwise."""
    if calls_sent(events) != [body for _, _, body in received.requests]:
        raise RunnerFailure(
            f'{case["id"]}: the endpoint received other bytes than the core '
            'wrote'
        )


def s1_observation(
    case: dict[str, Any],
    database: str,
    case_digest: str,
    bases_digest: str,
    answer: dict[str, Any],
    commit: dict[str, Any],
    received: Any,
) -> dict[str, Any]:
    r"""One observation of an S1 ``case`` on ``database`` from the Java
    program's ``answer``; every call the endpoint received is the call the
    core wrote, byte for byte."""
    identity = answer['identity']
    if identity['providers'] != list(PROVIDERS):
        raise RunnerFailure(
            'the Java host reports providers this runner does not run legs for'
        )
    check_sent(case, answer['events'], received)
    capture = captured(answer['events'])
    outcome = answer['outcome']
    versions = {'java': identity['java']}
    if database == 'duckdb':
        versions['duckdb'] = answer['duckdb']
    key = run_python.case_key(case['id'], database)
    return {
        'format': 'sdm-observation/1',
        'case': key,
        'identity': {
            'runner': 'java',
            'cell': None,
            'platform': identity['platform'],
            'commit': commit,
            'reference': None,
            'native': None,
            'native_loaded': None,
            'operations': capture.operations,
            'contract': identity['contract'],
            'versions': versions,
            'inputs': {'case': case_digest, 'bases': bases_digest},
        },
        'outcome': {
            'result': outcome['result'],
            'stage': None,
            'error': None,
            'code': outcome['code'],
        },
        'host': None,
        'core': capture.core(),
        'driver': driver_record(
            answer['events'],
            answer.get('logged') if database == 'duckdb' else None,
            database,
        ),
    }


@contextmanager
def loaded(
    case: dict[str, Any], bases: dict[str, list[str]], database: str
) -> Iterator[dict[str, Any]]:
    r"""What opens the case's data on ``database``: nothing on DuckDB, where
    the program builds it; a new SQLite file, or a new Postgres database on
    ``SDM_POSTGRES_TEST_DSN``'s server, loaded from the case's DuckDB tables
    by the contract loader, as the Python runner loads them; removed after."""
    if database == 'duckdb':
        yield {}
        return
    duck = run_python._connection(case, bases)
    loader = run_python._sibling('databases')
    try:
        if database == 'sqlite':
            with tempfile.TemporaryDirectory(
                prefix='sdm-java-case-'
            ) as scratch:
                path = Path(scratch) / 'case.sqlite'
                loader.load(duck, 'sqlite', path)
                duck.close()
                yield {'path': str(path)}
            return
        import uuid

        import psycopg
        from psycopg.conninfo import conninfo_to_dict, make_conninfo

        dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
        if not dsn:
            raise RunnerFailure(
                'a Postgres run needs SDM_POSTGRES_TEST_DSN, its server'
            )
        name = f'sdm_java_{uuid.uuid4().hex[:16]}'
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(f'CREATE DATABASE {name}')
        try:
            scoped = make_conninfo(dsn, dbname=name)
            loader.load(duck, 'postgres', scoped)
            duck.close()
            parts = conninfo_to_dict(scoped)
            yield {
                'url': (
                    f'jdbc:postgresql://{parts.get("host", "localhost")}:'
                    f'{parts.get("port", "5432")}/{name}'
                ),
                'user': parts.get('user'),
                'password': parts.get('password'),
            }
        finally:
            with psycopg.connect(dsn, autocommit=True) as admin:
                admin.execute(f'DROP DATABASE {name} WITH (FORCE)')
    finally:
        try:
            duck.close()
        except Exception:  # noqa: BLE001 - closed already
            pass


def run(
    paths: list[Path],
    *,
    jar: Path,
    lib: Path | None = None,
    native: Path | None = None,
    java: str | None = None,
    database: str = 'duckdb',
    drivers: Path | None = None,
) -> list[dict[str, Any]]:
    r"""The observations of the Java legs of the cases at ``paths`` on
    ``database``."""
    try:
        cases = run_python.load_cases(paths)
    except run_python.RunnerFailure as failed:
        raise RunnerFailure(str(failed)) from None
    known = java_paths()
    chosen = [
        (case, digest)
        for case, digest in cases
        if on_java(case, database, known)
    ]
    for case, _ in cases:
        if not on_java(case, database, known):
            print(f'{case["id"]}: no Java leg on {database}', file=sys.stderr)
    if not chosen:
        return []
    program = Program(
        java=java_binary(java),
        path=classpath(jar, lib, drivers),
        native=native,
    )
    commit = run_python._git(REPO)
    validate = run_python.validator('observation')
    bases = run_python.load_bases()
    bases_digest = run_python.digest(run_python.BASES.read_bytes())
    serve = run_python._sibling('scripted_endpoint').serve
    written = []
    try:
        for case, digest in chosen:
            if case.get('slice') == 'S0':
                made = observation(
                    case, digest, program.ask(call_of(case)), commit
                )
            else:
                given = case['input']
                with (
                    loaded(case, bases, database) as connect,
                    serve(
                        fail_call=given.get('fail_call'),
                        probabilities=given.get('probabilities', False),
                        quantiles=given.get('quantiles', False),
                        explanation=given.get('explanation', False),
                        item_metadata=given.get('item_metadata', False),
                        embeddings=given.get('embeddings', False),
                    ) as (url, received),
                ):
                    answer = program.ask(
                        s1_call(case, database, bases, connect, url)
                    )
                made = s1_observation(
                    case,
                    database,
                    digest,
                    bases_digest,
                    answer,
                    commit,
                    received,
                )
            problem = run_python._invalid(validate, made)
            if problem is not None:
                raise RunnerFailure(
                    f'{case["id"]}: an invalid observation: {problem}'
                )
            written.append(made)
    finally:
        program.close()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--jar', type=Path)
    parser.add_argument('--classpath', type=Path)
    parser.add_argument('--drivers', type=Path)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--java')
    parser.add_argument('--database', choices=PROVIDERS, default='duckdb')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('cases', nargs='*', type=Path)
    arguments = parser.parse_args(argv)
    arguments.out.unlink(missing_ok=True)
    jar = arguments.jar or (
        Path(os.environ['SDM_JAVA_JAR'])
        if os.environ.get('SDM_JAVA_JAR')
        else None
    )
    if jar is None:
        print(
            'run_java.py: no JAR: pass --jar or set SDM_JAVA_JAR',
            file=sys.stderr,
        )
        return 1
    paths = arguments.cases or [REPO / 'contract/cases/s0']
    try:
        written = run(
            paths,
            jar=jar,
            lib=arguments.classpath,
            native=arguments.native,
            java=arguments.java,
            database=arguments.database,
            drivers=arguments.drivers,
        )
    except RunnerFailure as failed:
        print(f'run_java.py: {failed}', file=sys.stderr)
        return 1
    arguments.out.write_text(
        ''.join(json.dumps(line, sort_keys=True) + '\n' for line in written),
        encoding='utf-8',
    )
    print(f'{len(written)} Java observation(s) written to {arguments.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
