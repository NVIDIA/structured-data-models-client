# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The moved helpers: what the tables and samplers compute of SQL or its
parameters, which the dialects take over.

Each :class:`Helper` says how the capture describes a call of one attribute,
from the object it was called on, its arguments and what the database
answered while it ran, and how :func:`replay_helper` builds an object from
that description alone to call the attribute again.

The descriptions name the inputs the attribute decides on, and nothing of
the engine's objects: a dtype, a column's declared type, the rows a catalog
query returned, the class's thresholds.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
from _codec import decode, encode

ENGINE = 'sdm_relational_engine.rfm'


class _Refused(RuntimeError):
    r"""What a replayed query raises where the recorded one failed."""


@dataclass
class _Query:
    r"""A query a helper ran: through ``_run`` (``kind`` ``run``, giving a
    table) or ``_rows`` (``rows``, giving rows), and what came of it."""

    kind: str
    sql: str
    value: Any
    failed: bool


@contextlib.contextmanager
def _queries(obj: Any) -> Iterator[list[_Query]]:
    r"""The queries ``obj`` runs through ``_run`` and ``_rows`` in the
    block, each as it ran."""
    log: list[_Query] = []
    saved = {name: obj.__dict__.get(name) for name in ('_run', '_rows')}

    def logging(kind: str, original: Callable[[Any], Any]) -> Callable:
        def run(statement: Any) -> Any:
            try:
                value = original(statement)
            except Exception as error:
                log.append(_Query(kind, statement[0], error, True))
                raise
            log.append(_Query(kind, statement[0], value, False))
            return value

        return run

    run_original, rows_original = obj._run, obj._rows
    obj.__dict__['_run'] = logging('run', run_original)
    obj.__dict__['_rows'] = logging('rows', rows_original)
    try:
        yield log
    finally:
        for name, value in saved.items():
            if value is None:
                obj.__dict__.pop(name, None)
            else:
                obj.__dict__[name] = value


def _answer(log: list[_Query], kind: str, prefix: str) -> _Query | None:
    for query in log:
        if query.kind == kind and query.sql.lstrip().startswith(prefix):
            return query
    return None


def _rows_or_none(query: _Query | None) -> list[Any] | None:
    r"""The rows a query gave, or ``None`` where it failed or did not run."""
    if query is None or query.failed:
        return None
    return [tuple(row) for row in query.value]


def _stub(answers: dict[str, Any]) -> Callable[[Any], Any]:
    r"""A ``_run`` or ``_rows`` that answers a query by the start of its
    SQL: with the value given, or by failing where it is ``None``."""

    def run(statement: Any) -> Any:
        sql = statement[0].lstrip()
        for prefix, value in answers.items():
            if sql.startswith(prefix):
                if value is None:
                    raise _Refused(f'the recorded query failed: {sql[:60]}')
                return value
        raise AssertionError(f'no recorded answer for {sql[:80]!r}')

    return run


def _source_column(name: str) -> Any:
    from sdm_relational_engine.rfm.base.source import SourceColumn

    return SourceColumn(
        name=name,
        dtype=None,
        is_primary_key=False,
        is_unique_key=False,
        is_nullable=True,
    )


def _blank(cls: type, **overrides: Any) -> Any:
    r"""An instance of ``cls``, or of a subclass overriding ``overrides``,
    with nothing set."""
    if overrides:
        cls = type(f'_Replay{cls.__name__}', (cls,), overrides)
    return cls.__new__(cls)


# --- The helpers ----------------------------------------------------------------


Describe = Callable[[Any, inspect.BoundArguments, list[_Query]], dict]
Build = Callable[[dict[str, Any]], Callable[[], Any]]


@dataclass
class Helper:
    r"""One moved helper: ``owner.attribute`` of the backend's ``module``."""

    module: str
    owner: str
    attribute: str
    describe: Describe
    build: Build

    @property
    def name(self) -> str:
        return f'{self.owner}.{self.attribute}'

    def cls(self, backend: str) -> type:
        module = importlib.import_module(
            f'{ENGINE}.backend.{backend}.{self.module}'
        )
        return vars(module)[self.owner]

    def install(self, capture: Any, backend: str) -> tuple[Any, str, Any]:
        r"""Record each call of the attribute that returns; what undoes it."""
        cls = self.cls(backend)
        original = vars(cls)[self.attribute]
        function = (
            original.func
            if isinstance(original, functools.cached_property)
            else original
        )
        signature = inspect.signature(function)

        @functools.wraps(function)
        def recorded(obj: Any, *args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(obj, *args, **kwargs)
            bound.apply_defaults()
            with _queries(obj) if hasattr(obj, '_run') else _no_log() as log:
                value = function(obj, *args, **kwargs)
            described = self.describe(obj, bound, log)
            capture.add(
                backend,
                self.name,
                {key: encode(item) for key, item in described.items()},
                {'result': encode(value)},
            )
            return value

        replacement: Any = recorded
        if isinstance(original, functools.cached_property):
            replacement = functools.cached_property(recorded)
            replacement.__set_name__(cls, self.attribute)
        setattr(cls, self.attribute, replacement)
        return (cls, self.attribute, original)


@contextlib.contextmanager
def _no_log() -> Iterator[list[_Query]]:
    yield []


# Samplers: what a window bound on a time column is written as.


def _describe_dtype(
    obj: Any, bound: inspect.BoundArguments, _: list[_Query]
) -> dict[str, Any]:
    table_name, column = list(bound.arguments.values())[1:3]
    return {
        'table_name': table_name,
        'column': column,
        'dtype': obj.table_dtype_dict[table_name].get(column),
    }


def _build_dtype(backend: str, owner: str, attribute: str) -> Build:
    def build(args: dict[str, Any]) -> Callable[[], Any]:
        cls = vars(
            importlib.import_module(f'{ENGINE}.backend.{backend}.sampler')
        )[owner]
        obj = _blank(cls)
        table, column, dtype = args['table_name'], args['column'], args['dtype']
        obj._table_dtype_dict = {
            table: {} if dtype is None else {column: dtype}
        }
        return lambda: getattr(obj, attribute)(table, column)

    return build


def _describe_zone(
    obj: Any, bound: inspect.BoundArguments, _: list[_Query]
) -> dict[str, Any]:
    table_name, column = list(bound.arguments.values())[1:3]
    table = obj._table_dict.get(table_name)
    zoned = getattr(table, '_zoned_expr_columns', set())
    return {
        'table_name': table_name,
        'column': column,
        'source_column': column in obj.source_table_dict[table_name],
        'zoned_expr': column in zoned,
        'source_type': obj._snowflake_type_dict.get(table_name, {}).get(column),
    }


def _build_zone(attribute: str) -> Build:
    def build(args: dict[str, Any]) -> Callable[[], Any]:
        from sdm_relational_engine.rfm.backend.snow.sampler import SnowSampler

        obj = _blank(SnowSampler)
        table, column = args['table_name'], args['column']
        obj._source_table_dict = {
            table: {column: _source_column(column)}
            if args['source_column']
            else {}
        }
        obj._table_dict = {
            table: SimpleNamespace(
                _zoned_expr_columns={column} if args['zoned_expr'] else set()
            )
        }
        source_type = args['source_type']
        obj._snowflake_type_dict = {
            table: {} if source_type is None else {column: source_type}
        }
        return lambda: getattr(obj, attribute)(table, column)

    return build


# Tables: the scan caps and the zone of a time column.


def _describe_is_large(
    obj: Any, _: inspect.BoundArguments, log: list[_Query]
) -> dict[str, Any]:
    size = None
    detail = _answer(log, 'run', 'DESCRIBE DETAIL')
    if detail is not None and not detail.failed:
        table = detail.value
        if table.num_rows > 0 and 'sizeInBytes' in table.column_names:
            size = table.column('sizeInBytes')[0].as_py()
    described = None
    rows = _rows_or_none(_answer(log, 'rows', 'DESCRIBE TABLE EXTENDED'))
    if rows is not None:
        described = {
            str(row[0]).strip(): str(row[1]).strip()
            for row in rows
            if len(row) >= 2 and row[0] is not None
        }
        described = {
            key: value
            for key, value in described.items()
            if key in ('Type', 'Statistics')
        }
    return {
        'detail_size': size,
        'described': described,
        'min_bytes': obj._SAMPLE_BOUND_MIN_BYTES,
    }


def _build_is_large(args: dict[str, Any]) -> Callable[[], Any]:
    from sdm_relational_engine.rfm.backend.databricks.table import (
        DatabricksTable,
    )

    obj = _blank(DatabricksTable)
    obj._catalog, obj._schema, obj._source_name = 'c', 's', 't'
    obj._SAMPLE_BOUND_MIN_BYTES = args['min_bytes']
    size, described = args['detail_size'], args['described']
    detail = None if size is None else pa.table({'sizeInBytes': [size]})
    obj.__dict__['_run'] = _stub({'DESCRIBE DETAIL': detail})
    rows = None if described is None else list(described.items())
    obj.__dict__['_rows'] = _stub({'DESCRIBE TABLE EXTENDED': rows})
    return lambda: obj._is_large()


def _describe_prefiltered(
    obj: Any, _: inspect.BoundArguments, log: list[_Query]
) -> dict[str, Any]:
    return {
        'schema': obj._schema,
        'source_name': obj._source_name,
        'primary_key': obj._source_primary_key,
        'relation': _rows_or_none(_answer(log, 'rows', 'SELECT c.relkind')),
        'min_blocks': obj._SAMPLE_PREFILTER_MIN_BLOCKS,
        'target_blocks': obj._SAMPLE_PREFILTER_TARGET_BLOCKS,
    }


def _build_prefiltered(args: dict[str, Any]) -> Callable[[], Any]:
    from sdm_relational_engine.rfm.backend.postgres.table import PostgresTable

    obj = _blank(PostgresTable)
    obj._schema, obj._source_name = args['schema'], args['source_name']
    obj.__dict__['_source_primary_key'] = args['primary_key']
    obj._SAMPLE_PREFILTER_MIN_BLOCKS = args['min_blocks']
    obj._SAMPLE_PREFILTER_TARGET_BLOCKS = args['target_blocks']
    obj.__dict__['_rows'] = _stub({'SELECT c.relkind': args['relation']})
    return lambda: obj._prefiltered_sample_source()


def _describe_is_zoned_time(
    obj: Any, bound: inspect.BoundArguments, _: list[_Query]
) -> dict[str, Any]:
    name = bound.arguments['name']
    in_source = name in obj._source_column_dict
    return {
        'name': name,
        'source_column': in_source,
        'is_source': obj[name].is_source if in_source else None,
        'source_type': obj._source_duckdb_type_dict.get(name),
        'zoned_expr': name in obj._zoned_expr_columns,
    }


def _build_is_zoned_time(args: dict[str, Any]) -> Callable[[], Any]:
    from sdm_relational_engine.rfm.backend.duckdb.table import DuckDBTable

    name = args['name']
    column = SimpleNamespace(is_source=args['is_source'])
    obj = _blank(DuckDBTable, column=lambda self, _: column)
    obj.__dict__['_source_column_dict'] = (
        {name: _source_column(name)} if args['source_column'] else {}
    )
    source_type = args['source_type']
    obj._source_duckdb_type_dict = (
        {} if source_type is None else {name: source_type}
    )
    obj._zoned_expr_columns = {name} if args['zoned_expr'] else set()
    return lambda: obj._is_zoned_time(name)


def _describe_prefilter(
    obj: Any, bound: inspect.BoundArguments, _: list[_Query]
) -> dict[str, Any]:
    return {
        'rows': bound.arguments['rows'],
        'seed': bound.arguments['seed'],
        'listing': obj.__dict__.get('_listing'),
        'sampled_kinds': sorted(obj._SAMPLED_KINDS),
        'prefilter_partitions': obj._SAMPLE_PREFILTER_PARTITIONS,
        'micro_partition_bytes': obj._MICRO_PARTITION_BYTES,
    }


def _build_prefilter(args: dict[str, Any]) -> Callable[[], Any]:
    from sdm_relational_engine.rfm.backend.snow.table import SnowTable

    obj = _blank(SnowTable)
    obj.__dict__['_listing'] = args['listing']
    obj._SAMPLED_KINDS = frozenset(args['sampled_kinds'])
    obj._SAMPLE_PREFILTER_PARTITIONS = args['prefilter_partitions']
    obj._MICRO_PARTITION_BYTES = args['micro_partition_bytes']
    return lambda: obj._prefilter(args['rows'], seed=args['seed'])


#: What of a ``SHOW TABLES`` row the listing reads, besides its fields.
_LISTING_NAME = 'name'


def _describe_listing(
    obj: Any, _: inspect.BoundArguments, __: list[_Query]
) -> dict[str, Any]:
    fields = list(obj._LISTING_FIELDS)
    listed = obj.__dict__.get('_listed')
    if listed is not None:
        kept = {_LISTING_NAME, *fields}
        listed = [
            {key: value for key, value in row.items() if key in kept}
            for row in listed
        ]
    return {
        'source_name': obj._source_name,
        'listed': listed,
        'fields': fields,
    }


def _build_listing(args: dict[str, Any]) -> Callable[[], Any]:
    from sdm_relational_engine.rfm.backend.snow.table import SnowTable

    obj = _blank(SnowTable)
    obj._source_name = args['source_name']
    obj._database, obj._schema = 'd', 's'
    obj._LISTING_FIELDS = tuple(args['fields'])
    if args['listed'] is None:
        obj.__dict__['_run'] = _stub({'SHOW TABLES': None})
    else:
        obj.__dict__['_listed'] = args['listed']
    return lambda: obj._listing


def _describe_probe(
    obj: Any, bound: inspect.BoundArguments, log: list[_Query]
) -> dict[str, Any]:
    table_name = bound.arguments['table_name']
    return {
        'table_name': table_name,
        'time_column': obj.time_column_dict.get(table_name),
        'found': _rows_or_none(_answer(log, 'rows', 'SELECT')),
    }


def _build_probe(args: dict[str, Any]) -> Callable[[], Any]:
    from sdm_relational_engine.rfm.backend.sqlite.sampler import SQLiteSampler

    obj = _blank(SQLiteSampler)
    table, column = args['table_name'], args['time_column']
    obj._time_column_dict = {} if column is None else {table: column}
    obj._table_column_ref_dict = {
        table: {} if column is None else {column: 'c'}
    }
    obj._source_name_dict = {table: 't'}
    obj.__dict__['_rows'] = _stub({'SELECT': args['found']})
    return lambda: obj._probe_time_spelling(table)


def _describe_spell(
    obj: Any, bound: inspect.BoundArguments, _: list[_Query]
) -> dict[str, Any]:
    return {
        'separator': obj.separator,
        'digits': obj.digits,
        'suffix': obj.suffix,
        'times': bound.arguments['times'],
    }


def _build_spell(args: dict[str, Any]) -> Callable[[], Any]:
    from sdm_relational_engine.rfm.backend.sqlite.sampler import _TimeSpelling

    spelling = _TimeSpelling(args['separator'], args['digits'], args['suffix'])
    return lambda: spelling.spell(args['times'])


#: The moved helpers, by backend.
HELPERS: dict[str, tuple[Helper, ...]] = {
    'databricks': (
        Helper(
            'sampler',
            'DatabricksSampler',
            '_bound_format',
            _describe_dtype,
            _build_dtype('databricks', 'DatabricksSampler', '_bound_format'),
        ),
        Helper(
            'table',
            'DatabricksTable',
            '_is_large',
            _describe_is_large,
            _build_is_large,
        ),
    ),
    'duckdb': (
        Helper(
            'table',
            'DuckDBTable',
            '_is_zoned_time',
            _describe_is_zoned_time,
            _build_is_zoned_time,
        ),
    ),
    'postgres': (
        Helper(
            'sampler',
            'PostgresSampler',
            '_utc_bounds',
            _describe_dtype,
            _build_dtype('postgres', 'PostgresSampler', '_utc_bounds'),
        ),
        Helper(
            'table',
            'PostgresTable',
            '_prefiltered_sample_source',
            _describe_prefiltered,
            _build_prefiltered,
        ),
    ),
    'snow': (
        Helper(
            'sampler',
            'SnowSampler',
            '_zone',
            _describe_zone,
            _build_zone('_zone'),
        ),
        Helper(
            'sampler',
            'SnowSampler',
            '_bound_format',
            _describe_zone,
            _build_zone('_bound_format'),
        ),
        Helper(
            'table',
            'SnowTable',
            '_prefilter',
            _describe_prefilter,
            _build_prefilter,
        ),
        Helper(
            'table', 'SnowTable', '_listing', _describe_listing, _build_listing
        ),
    ),
    'sqlite': (
        Helper(
            'sampler',
            'SQLiteSampler',
            '_probe_time_spelling',
            _describe_probe,
            _build_probe,
        ),
        Helper(
            'sampler', '_TimeSpelling', 'spell', _describe_spell, _build_spell
        ),
    ),
}


def helper(backend: str, function: str) -> Helper:
    for found in HELPERS.get(backend, ()):
        if found.name == function:
            return found
    raise KeyError(f'{backend} has no moved helper {function}')


def replay_helper(backend: str, function: str, args: dict[str, Any]) -> Any:
    r"""Call the moved helper ``function`` of ``backend`` on an object built
    from ``args``, the description the capture recorded, and return what it
    returned."""
    return helper(backend, function).build(decode(args))()
