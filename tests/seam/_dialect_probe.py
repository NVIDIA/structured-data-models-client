# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Mark the SQL every dialect function returns, then run something.

    _dialect_probe.py OUT corpus record --corpus DIR --only NAMES
    _dialect_probe.py OUT warehouse NAME
    _dialect_probe.py OUT local BACKEND WORKDIR

Every function of every ``rfm/backend/<db>/dialect.py`` is patched in place,
so a caller holding it under any name gets the patched one: a returned
``(sql, params)`` comes back as ``(sql + MARK, params)``. Then the run is
made, and ``OUT`` gets a JSON report: which functions were patched, how many
times each returned SQL, and, for ``warehouse`` and ``local``, every
statement the driver received.

A statement the driver receives without the mark at its end was not the text
a dialect function returned.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import runpy
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MARK = '\n/* sdm-seam-dialect */'
BACKENDS = ('databricks', 'snow', 'postgres', 'duckdb', 'sqlite')
PRAGMA_LOGGED = re.compile(r"^SELECT \* FROM pragma_\w+\('[^']*'\);$")

returned: Counter[str] = Counter()


def _marked(value: Any, key: str) -> Any:
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        returned[key] += 1
        items = [value[0] + MARK, *value[1:]]
        make = getattr(type(value), '_make', None)
        return make(items) if make is not None else tuple(items)
    if (
        isinstance(value, list)
        and value
        and all(
            isinstance(v, tuple) and v and isinstance(v[0], str) for v in value
        )
    ):
        return [_marked(v, key) for v in value]
    return value


def _written_in(func: Any, module: types.ModuleType) -> bool:
    r"""Whether ``func``'s code is in ``module``'s own source file, rather
    than made for it, as a named tuple's methods are."""
    code = getattr(inspect.unwrap(func), '__code__', None)
    return code is not None and code.co_filename == module.__file__


def _functions(module: types.ModuleType) -> list[tuple[str, Any]]:
    r"""The plain functions the module defines, its classes' included."""
    found: list[tuple[str, Any]] = []
    for name, value in vars(module).items():
        if inspect.isfunction(value) and _written_in(value, module):
            found.append((name, value))
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            for attr, member in vars(value).items():
                if attr.startswith('__') and attr.endswith('__'):
                    continue
                if isinstance(member, (staticmethod, classmethod)):
                    member = member.__func__
                if inspect.isfunction(member) and _written_in(member, module):
                    found.append((f'{name}.{attr}', member))
    return found


def _patch(module: types.ModuleType, qualname: str, func: Any) -> None:
    target = inspect.unwrap(func)
    if target.__closure__:
        raise RuntimeError(
            f'{module.__name__}.{qualname} is a closure, and cannot be probed'
        )
    original = types.FunctionType(
        target.__code__,
        target.__globals__,
        target.__name__,
        target.__defaults__,
        target.__closure__,
    )
    original.__kwdefaults__ = target.__kwdefaults__
    key = f'{module.__name__}.{qualname}'
    dispatch_name = '__sdm_seam_dispatch__'
    table = target.__globals__.setdefault(dispatch_name, {})

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        return _marked(original(*args, **kwargs), key)

    table[key] = dispatch
    namespace: dict[str, Any] = {}
    exec(
        f'def {target.__name__}(*args, **kwargs):\n'
        f'    return {dispatch_name}[{key!r}](*args, **kwargs)\n',
        {},
        namespace,
    )
    target.__code__ = namespace[target.__name__].__code__
    target.__defaults__ = None
    target.__kwdefaults__ = None


def install() -> dict[str, list[str]]:
    patched: dict[str, list[str]] = {}
    for backend in BACKENDS:
        name = f'sdm_relational_engine.rfm.backend.{backend}.dialect'
        module = importlib.import_module(name)
        functions = _functions(module)
        for qualname, func in functions:
            _patch(module, qualname, func)
        patched[name] = [q for q, _ in functions]
    return patched


def _corpus(args: list[str]) -> dict[str, Any]:
    scripts = REPO / 'scripts' / 'parity'
    sys.path.insert(0, str(scripts))
    sys.argv = [str(scripts / 'run_corpus.py'), *args]
    try:
        runpy.run_path(str(scripts / 'run_corpus.py'), run_name='__main__')
    except SystemExit as stop:
        return {'exit': stop.code if isinstance(stop.code, int) else 1}
    return {'exit': 0}


def _warehouse(name: str) -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import _warehouse as warehouse

    result = warehouse.run(name)
    return {'statements': [s.sql for s in result['statements']]}


def _local(backend: str, workdir: str) -> dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import _seam

    directory = Path(workdir)
    directory.mkdir(parents=True, exist_ok=True)
    if backend == 'sqlite':
        from sdm_connectors import SQLiteSource

        connection = _seam.sqlite_database(directory / 'shop.sqlite')
        with _seam.watch_sqlite() as log:
            _seam.predict(SQLiteSource(connection))
        return {'statements': [c.sql for c in log.statements]}
    if backend == 'duckdb':
        from sdm_connectors import DuckDBSource

        connection = _seam.duckdb_database(directory / 'shop.duckdb')
        connection.execute("CALL enable_logging('QueryLog')")
        _seam.predict(DuckDBSource(connection))
        rows = connection.execute(
            "SELECT message FROM duckdb_logs WHERE type = 'QueryLog'"
        ).fetchall()
        statements = [
            r[0]
            for r in rows
            if not r[0].startswith("CALL enable_logging('QueryLog')")
        ]
        # DuckDB logs a PRAGMA as the query it rewrites it to, without the
        # text it was given; the corpus recordings check those statements.
        rewritten = [s for s in statements if PRAGMA_LOGGED.match(s)]
        return {
            'statements': [s for s in statements if s not in rewritten],
            'rewritten_pragmas': rewritten,
        }
    if backend == 'postgres':
        import os

        from sdm_connectors import PostgresSource

        dsn = os.environ['SDM_POSTGRES_TEST_DSN']
        schema = f'p3_dialect_{os.getpid()}'
        connection = _seam.postgres_schema(dsn, schema, autocommit=False)
        try:
            with _seam.watch_postgres() as log:
                _seam.predict(PostgresSource(connection, schema=schema))
        finally:
            connection.close()
            _seam.drop_postgres_schema(dsn, schema)
        return {'statements': [c.sql for c in log.statements]}
    raise ValueError(backend)


def main(argv: list[str]) -> int:
    out, mode, *rest = argv[1:]
    report: dict[str, Any] = {'mark': MARK}
    try:
        report['patched'] = install()
        if mode == 'corpus':
            report.update(_corpus(rest))
        elif mode == 'warehouse':
            report.update(_warehouse(rest[0]))
        elif mode == 'local':
            report.update(_local(rest[0], rest[1]))
        else:
            raise ValueError(mode)
    except BaseException as error:  # reported, not raised: OUT says why
        import traceback

        report['error'] = ''.join(traceback.format_exception(error))
    report['returned'] = dict(returned)
    Path(out).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
