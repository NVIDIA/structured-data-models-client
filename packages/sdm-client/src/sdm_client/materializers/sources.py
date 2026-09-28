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
) -> Graph:
    r"""Build the engine's graph over ``tables`` in ``source``.

    Args:
        source: The :class:`sdm_connectors.Source` the tables are in.
        tables: For each name the schema uses, the table's name in the source
            and the columns to read, or ``None`` for all of them.
        infer_metadata: Whether the engine infers keys, time columns and
            semantic types. Only the first read of a source asks; afterwards
            the schema says what they are.

    Edges are never added here. The schema decides them, and the materializer
    links exactly those.
    """
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
