# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The engine's graph over tables that stay in their source.

One place builds it, because two need it: `Schema.from_source` reads the
source's metadata through it, and the relational materializer samples through
it. Both must agree on which columns a table has and where it lives.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import MissingExtraError, StructuredDataError

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph

    from sdm_client._rust.results import Reading


class EngineTables:
    r"""Engine tables read once per source.

    Each gather gets copies, because it writes keys, time columns and stypes
    onto them. Time ranges are not kept, so a schema sees new rows.
    """

    def __init__(self) -> None:
        self._tables: dict[str, Any] = {}
        self.read_under: dict[str, Reading] = {}

    def readings(self) -> dict[str, Reading]:
        r"""Each table the core read and kept, by the schema's source name
        (the contract's ``HeldSchema.read_under``, T3.4): where it read it,
        and its columns with what the source said of each then, kept here,
        with the engine's own tables, so the two share one lifetime; one
        pickled before it held any starts empty."""
        return self.__dict__.setdefault('read_under', {})

    def keep(self, tables: Mapping[str, Any]) -> None:
        r"""Keyed by the schema's source name; Databricks and Postgres qualify
        the engine's.
        """
        for key, table in tables.items():
            self._tables.setdefault(key, table)

    def graph(
        self, source: Any, tables: Mapping[str, tuple[str, Sequence[str]]]
    ) -> Graph:
        r"""A fresh graph over copies of ``tables`` in ``source``, without
        edges: the schema decides those.

        Args:
            source: A ``Source``, or a ``SemanticModel`` kept in one.
            tables: Schema name to (source name, columns kept). Only tables
                not read yet are read.
        """
        from sdm_relational_engine.rfm import Graph

        missing = [
            source_name
            for source_name, _ in tables.values()
            if source_name not in self._tables
        ]
        if missing:
            if is_semantic_model(source):
                model = semantic_model_graph(source)
                self.keep({name: model[name] for name in model.tables})
            else:
                read = engine_graph(
                    source,
                    {name: name for name in missing},
                    infer_metadata=False,
                )
                self.keep({name: read[name] for name in missing})
        return Graph(
            tables=[
                _copy_for_one_gather(self._tables[source_name], name, columns)
                for name, (source_name, columns) in tables.items()
            ],
            edges=[],
        )


def _copy_for_one_gather(table: Any, name: str, columns: Sequence[str]) -> Any:
    copied = table.copy(name)
    copied.end_time_column = None
    copied.time_column = None
    copied.primary_key = None
    for column in [c.name for c in copied.columns if c.name not in columns]:
        copied.remove_column(column)
    return copied


# The engine's dtypes a scalar wire column can carry, as the wire names them.
# Lists, binary and durations have no scalar wire form.
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


@contextmanager
def _schema_says_time_columns() -> Iterator[None]:
    r"""Keep the engine from warning about a time column it kept unchecked:
    the schema says that, with its own override, in one notice.
    """
    from sdm_relational_engine.rfm import TimeColumnWarning

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', TimeColumnWarning)
        yield


def engine_graph(
    source: Any, tables: Mapping[str, str] | None, *, infer_metadata: bool
) -> Graph:
    r"""The engine's graph over ``tables`` in ``source``, without edges: the
    schema decides those.

    Args:
        source: A ``Source``.
        tables: Schema name to source name; ``None`` reads every table.
        infer_metadata: Let the engine infer keys, time columns and stypes.
    """
    try:
        from sdm_relational_engine import (
            MetricViewAsTableError,
            NoTablesFoundError,
            TableNotFoundError,
        )
    except ImportError as error:  # pragma: no cover - optional extra
        raise MissingExtraError(
            'relational', 'structured-data-models-relational-engine'
        ) from error

    specs = (
        None
        if tables is None
        else [
            {'name': name, 'source_name': source_name}
            for name, source_name in tables.items()
        ]
    )
    try:
        with _schema_says_time_columns():
            return _read(source, specs, infer_metadata)
    except NoTablesFoundError as error:
        # The engine's own advice names its API, which cannot fill a namespace.
        raise StructuredDataError(
            f'{source!r} holds no table to describe', code='INVALID_REQUEST'
        ) from error
    except TableNotFoundError as error:
        raise StructuredDataError(str(error), code='INVALID_REQUEST') from error
    except MetricViewAsTableError as error:
        # The engine's advice names its own API; here a source reads the view.
        raise StructuredDataError(
            f'{error.name!r} is a metric view, not a table: its measures are '
            f'aggregations with no value per row. Read it as a semantic model '
            f'instead, with source.semantic_model({error.name!r}).',
            code='INVALID_REQUEST',
            details={'metric_view': error.name},
        ) from error


def _read(
    source: Any, specs: list[dict[str, Any]] | None, infer_metadata: bool
) -> Graph:
    from sdm_relational_engine.rfm import Graph

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
    as the model declares them. Discovery and gathering both read it here, so
    they keep the same columns.

    What the model has that a graph cannot hold is said in one
    :class:`SDMWarning`, and nothing is said for a model that loses nothing:
    how it was read is in the graph's ``conversion_notes``, which the schema
    renders as text. A model the
    engine refuses, for a filter on a joined table say, is refused with a
    :class:`StructuredDataError` that carries the engine's reason.
    """
    try:
        from sdm_relational_engine import (
            TableNotFoundError,
            ViewConversionError,
        )
        from sdm_relational_engine.rfm import Graph
    except ImportError as error:  # pragma: no cover - optional extra
        raise MissingExtraError(
            'relational', 'structured-data-models-relational-engine'
        ) from error

    try:
        # The engine is told not to warn, rather than its warnings filtered,
        # since a filter holds for every thread: what it would say is said
        # below, and by the schema, to the caller.
        graph = _model_graph(Graph, model)
    except (ViewConversionError, TableNotFoundError) as error:
        # The engine's refusals of the model, each saying what to change in
        # it, and a table it reads that is not there, as for a source's own
        # tables. Any other failure is not the model's, and is not one.
        raise StructuredDataError(
            str(error),
            code='INVALID_REQUEST',
            details={'semantic_model': model.name},
        ) from error
    if graph.conversion_messages:
        warnings.warn(
            f'{model!r} has what a graph cannot hold, so it is read in '
            f'part:\n'
            + '\n'.join(
                f'- {message}' for message in graph.conversion_messages
            ),
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
    return graph


def _model_graph(graph_type: type[Graph], model: Any) -> Graph:
    namespace = model.namespace
    if model.backend == 'databricks':
        return graph_type.from_databricks_metric_view(
            model.name,
            model.connection,
            catalog=namespace.get('catalog'),
            schema=namespace.get('schema'),
            verbose=False,
            trust_expressions=model.trust_expressions,
            warn=False,
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
        return graph_type.from_snowflake_semantic_view(
            name,
            model.connection,
            verbose=False,
            trust_expressions=model.trust_expressions,
            warn=False,
        )
    raise StructuredDataError(  # pragma: no cover - SemanticModel refuses
        f'no semantic model is kept in a {model.backend!r} source',
        code='INVALID_REQUEST',
    )
