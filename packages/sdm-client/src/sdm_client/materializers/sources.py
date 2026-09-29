# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The engine's graph over tables that stay in their source.

One place builds it, because two need it: `Schema.from_source` reads the
source's metadata through it, and the relational materializer samples through
it. Both must agree on which columns a table has and where it lives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sdm_client.errors import MissingExtraError, StructuredDataError

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph

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
    tables: Mapping[str, tuple[str, Sequence[str] | None]],
    *,
    infer_metadata: bool,
    metric_view: str | None = None,
) -> Graph:
    r"""Build the engine's graph over ``tables`` in ``source``.

    Args:
        source: The :class:`sdm_connectors.Source` the tables are in.
        tables: For each name the schema uses, the table's name in the source
            and the columns to read, or ``None`` for all of them.
        infer_metadata: Whether the engine infers keys, time columns and
            semantic types. Only the first read of a source asks; afterwards
            the schema says what they are.
        metric_view: The metric view the schema was built from. Its tables are
            rebuilt from the view, since a dimension can be an expression the
            view alone defines, and then cut down to what the schema kept.

    Edges are never added here. The schema decides them, and the materializer
    links exactly those.
    """
    if metric_view is not None:
        return _trimmed_to(metric_view_graph(source, metric_view), tables)

    try:
        from sdm_relational_engine.rfm import Graph
    except ImportError as error:  # pragma: no cover - optional extra
        raise MissingExtraError(
            'relational', 'sdm-relational-engine'
        ) from error

    specs: list[dict[str, Any]] = []
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


def metric_view_graph(source: Any, metric_view: str) -> Graph:
    r"""The engine's graph of a Databricks metric view in ``source``."""
    try:
        from sdm_relational_engine.rfm import Graph
    except ImportError as error:  # pragma: no cover - optional extra
        raise MissingExtraError(
            'relational', 'sdm-relational-engine'
        ) from error

    namespace = source.namespace
    return Graph.from_databricks_metric_view(
        metric_view,
        source.connection,
        catalog=namespace.get('catalog'),
        schema=namespace.get('schema'),
        verbose=False,
    )


def _trimmed_to(
    graph: Graph, tables: Mapping[str, tuple[str, Sequence[str] | None]]
) -> Graph:
    r"""``graph`` holding only the tables and columns the schema kept, and no
    edges.

    Keys and time columns are cleared as well: the materializer sets every one
    of them from the schema, and a column the view keyed on may be one the
    schema left out.
    """
    for edge in list(graph.edges):
        graph.unlink(edge.src_table, edge.fkey, edge.dst_table)
    for name in list(graph.tables):
        if name not in tables:
            graph.remove_table(name)
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
