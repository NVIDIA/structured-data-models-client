# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The engine's graph over tables that stay in their source.

One place builds it, because two need it: `Schema.from_source` reads the
source's metadata through it, and the relational materializer samples through
it. Both must agree on which columns a table has and where it lives.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sdm_client.errors import MissingExtraError, StructuredDataError

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph


class EngineTables:
    r"""The engine's table objects for one schema's source, read once.

    Describing a table in place costs queries: its columns, its keys, a preview
    of up to 1,000 rows, and on Databricks a row count. The engine keeps each
    answer on the table object that asked, so the objects are kept here, by
    their name in the source, and every later gather is given copies of them
    rather than asking again.

    Copies, and a fresh graph each time, because a gather writes onto what it
    is given -- keys, time columns, semantic types, and a composite key's
    derived column -- and two gathers, or a schema and its snapshot, must not
    see each other's writes. What the copies share is only what was read.

    The time range of each table is not kept. It decides the anchor a question
    defaults to, the latest moment in the data, and a schema that outlives new
    rows has to see them.
    """

    def __init__(self) -> None:
        self._tables: dict[str, Any] = {}

    def keep(self, tables: Mapping[str, Any]) -> None:
        r"""Keep engine tables, each by the name the schema gives it in the
        source.

        Not by the engine's own ``source_name``: on Databricks that is the
        table qualified by catalog and schema, which is not the name a schema
        maps to, so nothing kept that way was ever found again.
        """
        for key, table in tables.items():
            self._tables.setdefault(key, table)

    def missing(self, source_names: Sequence[str]) -> list[str]:
        return [name for name in source_names if name not in self._tables]

    def graph(
        self, tables: Mapping[str, tuple[str, Sequence[str] | None]]
    ) -> Graph:
        r"""A fresh graph over copies of the kept tables, named as the schema
        names them and holding only the columns it kept, with no keys, time
        columns or edges: the materializer writes every one of those.
        """
        from sdm_relational_engine.rfm import Graph

        return Graph(
            tables=[
                _copy_for_one_gather(self._tables[source_name], name, columns)
                for name, (source_name, columns) in tables.items()
            ],
            edges=[],
        )


def _copy_for_one_gather(
    table: Any, name: str, columns: Sequence[str] | None
) -> Any:
    copied = copy.copy(table)
    copied._column_dict = {
        key: copy.copy(column) for key, column in table._column_dict.items()
    }
    # The engine names a table once, when it is built, and has no rename.
    copied._name = name
    copied.end_time_column = None
    copied.time_column = None
    copied.primary_key = None
    if columns is not None:
        for column in [c.name for c in copied.columns if c.name not in columns]:
            copied.remove_column(column)
    return copied


# The engine's dtypes a scalar wire column can carry, as the wire names them.
# Anything else -- lists, binary, durations -- has no scalar form on the wire.
_WIRE_DTYPE = {
    'bool': 'bool',
    'int': 'int64',
    'byte': 'int32',
    'int16': 'int32',
    'int32': 'int32',
    'int64': 'int64',
    'float': 'float64',
    'float32': 'float32',
    'float64': 'float64',
    'string': 'string',
    'date': 'timestamp[us]',
    'time': 'timestamp[us]',
}


def wire_dtype(engine_dtype: Any) -> str | None:
    r"""The wire dtype for an engine column, or ``None`` if it has none."""
    return _WIRE_DTYPE.get(str(engine_dtype))


def engine_graph(
    source: Any,
    tables: Mapping[str, tuple[str, Sequence[str] | None]] | None,
    *,
    infer_metadata: bool,
    cache: EngineTables | None = None,
) -> Graph:
    r"""Build the engine's graph over ``tables`` in ``source``.

    Args:
        source: The ``sdm_connectors.Source`` the tables are in, or a
            ``SemanticModel`` kept in one.
        tables: For each name the schema uses, the table's name in the source
            and the columns to read, or ``None`` for all of them. ``None`` in
            place of the mapping reads every table the source describes.
        infer_metadata: Whether the engine infers keys, time columns and
            semantic types. Only the first read of a source asks; afterwards
            the schema says what they are.

        cache: Tables already read from this source. Given, only a table not
            in it is read, and the graph is a fresh one over copies of the
            kept tables; see :class:`EngineTables`.

    Edges are never added here. The schema decides them, and the materializer
    links exactly those.
    """
    if cache is not None and tables is not None:
        wanted = [source_name for source_name, _ in tables.values()]
        missing = cache.missing(wanted)
        if missing:
            if is_semantic_model(source):
                model = semantic_model_graph(source)
                cache.keep({name: model[name] for name in model.tables})
            else:
                # Read under their source names, so each is kept by one.
                read = engine_graph(
                    source,
                    {name: (name, None) for name in missing},
                    infer_metadata=infer_metadata,
                )
                cache.keep({name: read[name] for name in missing})
        return cache.graph(tables)

    if is_semantic_model(source):
        graph = semantic_model_graph(source)
        return graph if tables is None else _selected(graph, tables)

    try:
        from sdm_relational_engine.rfm import Graph
    except ImportError as error:  # pragma: no cover - optional extra
        raise MissingExtraError(
            'relational', 'sdm-relational-engine'
        ) from error

    specs: list[dict[str, Any]] | None = None
    if tables is not None:
        specs = []
        for name, (source_name, columns) in tables.items():
            spec: dict[str, Any] = {'name': name, 'source_name': source_name}
            if columns is not None:
                spec['columns'] = list(columns)
            specs.append(spec)

    backend = source.backend
    namespace = source.namespace
    connection = source.connection
    if backend == 'databricks':
        return Graph.from_databricks(
            connection,
            tables=specs,
            catalog=namespace.get('catalog'),
            schema=namespace.get('schema'),
            edges=[],
            infer_metadata=infer_metadata,
            verbose=False,
        )
    if backend == 'snowflake':
        return Graph.from_snowflake(
            connection,
            tables=specs,
            database=namespace.get('database'),
            schema=namespace.get('schema'),
            edges=[],
            infer_metadata=infer_metadata,
            verbose=False,
        )
    if backend == 'postgres':
        return Graph.from_postgres(
            connection,
            tables=specs,
            schema=namespace.get('schema'),
            edges=[],
            infer_metadata=infer_metadata,
            verbose=False,
        )
    if backend == 'duckdb':
        return Graph.from_duckdb(
            connection,
            tables=specs,
            edges=[],
            infer_metadata=infer_metadata,
            verbose=False,
        )
    if backend == 'sqlite':
        return Graph.from_sqlite(
            connection,
            tables=specs,
            edges=[],
            infer_metadata=infer_metadata,
            verbose=False,
        )
    raise StructuredDataError(  # pragma: no cover - Source refuses others
        f'no engine backend samples a {backend!r} source',
        code='INVALID_REQUEST',
    )


def is_semantic_model(source: Any) -> bool:
    r"""Whether ``source`` is a semantic model rather than a namespace."""
    try:
        from sdm_connectors import SemanticModel
    except ImportError:  # pragma: no cover - nothing could have built one
        return False
    return isinstance(source, SemanticModel)


def semantic_model_graph(model: Any) -> Graph:
    r"""The engine's graph of a semantic model: a Databricks metric view or a
    Snowflake semantic view.

    Named after the model's own tables, with its links, keys and time columns
    as the model declares them.
    """
    try:
        from sdm_relational_engine.rfm import Graph
    except ImportError as error:  # pragma: no cover - optional extra
        raise MissingExtraError(
            'relational', 'sdm-relational-engine'
        ) from error

    namespace = model.namespace
    if model.backend == 'databricks':
        return Graph.from_databricks_metric_view(
            model.name,
            model.connection,
            catalog=namespace.get('catalog'),
            schema=namespace.get('schema'),
            verbose=False,
        )
    if model.backend == 'snowflake':
        # The engine resolves an unqualified name against the connection's
        # current database and schema, so the source's own are written in.
        parts = [
            namespace[part]
            for part in ('database', 'schema')
            if part in namespace
        ]
        name = (
            model.name if '.' in model.name else '.'.join([*parts, model.name])
        )
        return Graph.from_snowflake_semantic_view(
            name, model.connection, verbose=False
        )
    raise StructuredDataError(  # pragma: no cover - SemanticModel refuses
        f'no semantic model is kept in a {model.backend!r} source',
        code='INVALID_REQUEST',
    )


def _selected(
    graph: Graph, tables: Mapping[str, tuple[str, Sequence[str] | None]]
) -> Graph:
    r"""``graph`` holding only the model tables the schema kept, under the
    names the schema gives them, with the columns it kept, and no edges.

    Keys and time columns are cleared as well: the materializer sets every one
    of them from the schema, and a column the model keyed on may be one the
    schema left out.
    """
    for edge in list(graph.edges):
        graph.unlink(edge.src_table, edge.fkey, edge.dst_table)
    kept = {source_name: name for name, (source_name, _) in tables.items()}
    engine_tables = {name: graph[name] for name in list(graph.tables)}
    for name in engine_tables:
        graph.remove_table(name)
    for source_name, table in engine_tables.items():
        if source_name not in kept:
            continue
        # The engine names a table once, when it is built, and has no rename;
        # a model's tables are named by the model, and the schema may call
        # them something else.
        table._name = kept[source_name]
        graph.add_table(table)
    for name, (_, columns) in tables.items():
        table = graph[name]
        # The engine's setter is declared apart from its getter, which mypy
        # reads as a read-only property.
        table.primary_key = None  # type: ignore[misc]
        table.time_column = None
        if columns is None:
            continue
        for column in [c.name for c in table.columns if c.name not in columns]:
            table.remove_column(column)
    return graph
