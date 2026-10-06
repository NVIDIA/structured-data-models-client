# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import io
import re
import warnings
from collections import defaultdict
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import pandas as pd
from predictive_query.schema.definition import (
    ColumnKey,
    ColumnKeyGroup,
    SchemaDefinition,
)
from predictive_query.schema.definition import (
    Relationship as PredictiveQueryEdge,
)
from predictive_query.schema.table import TableDefinition
from predictive_query.schema.typing import Stype
from sdm_connectors import quote_ident
from typing_extensions import Self

import sdm_relational_engine.utils.display as display
from sdm_relational_engine.exceptions import (
    GraphConstructionError,
    KumoRelationalError,
    MetricViewAsTableError,
    NoTablesFoundError,
    ViewConversionError,
)
from sdm_relational_engine.rfm.base import (
    Column,
    ColumnSpec,
    DataBackend,
    Table,
    composite_key,
    count_keys,
)
from sdm_relational_engine.rfm.base.casting import CastMixin
from sdm_relational_engine.rfm.base.utils import Timedelta
from sdm_relational_engine.rfm.view_expr import (
    _unsafe_expr_reason,
    _untrusted_call_reason,
)
from sdm_relational_engine.utils.environment import (
    in_jupyter_notebook,
    in_streamlit_notebook,
    in_tmux,
)

if TYPE_CHECKING:
    from adbc_driver_sqlite.dbapi import AdbcSqliteConnection
    from databricks.sql.client import Connection as DatabricksConnection
    from duckdb import DuckDBPyConnection
    from psycopg import Connection as PostgresConnection
    from sdm_connectors.backends.spark import SparkConnection
    from snowflake.connector import SnowflakeConnection


@dataclass(frozen=True, eq=True)
class Edge(CastMixin, PredictiveQueryEdge):
    r"""A relationship between two tables in an RFM graph."""

    src_table: str
    fkey: str
    dst_table: str

    def __iter__(self) -> Iterator[str]:
        return iter((self.src_table, self.fkey, self.dst_table))

    def __hash__(self) -> int:
        return hash((self.src_table, self.fkey, self.dst_table))


EdgeLike: TypeAlias = Edge | dict[str, str] | tuple[str, str, str]


@dataclass
class SqliteConnectionConfig(CastMixin):
    uri: str | Path
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DuckDBConnectionConfig(CastMixin):
    uri: str | Path | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


_STRING_LITERAL = re.compile(r"'(?:''|[^'])*'")


def _mask_literals(expr: str) -> str:
    r"""Blanks out single-quoted literals so a pattern can be searched against
    SQL text alone.

    Only single quotes: a double-quoted or backticked span is a *quoted
    identifier*, which a table qualifier legitimately uses (``"ORDERS".X``),
    so those must stay in scope.

    Used to decide whether a view expression references another table. Another
    table's name inside a literal is data rather than a reference, and dropping
    the column for it reports "references other tables" untruthfully.
    """
    return _STRING_LITERAL.sub(lambda match: ' ' * len(match.group()), expr)


def _sub_outside_literals(pattern: re.Pattern[str], expr: str) -> str:
    r"""Applies ``pattern.sub('', ...)`` to the SQL text of ``expr``, leaving
    its single-quoted literals byte-for-byte intact.

    A qualifier pattern is a piece of text like ``ORDERS.``, and a literal is
    free to contain that text as data: stripping it out of
    ``CASE WHEN CHANNEL = 'ORDERS.WEB' THEN ...`` rewrites the expression to
    compare against ``'WEB'``, which is still valid SQL and so fails silently
    with different values than the view declares.
    """
    out: list[str] = []
    last = 0
    for match in _STRING_LITERAL.finditer(expr):
        out.append(pattern.sub('', expr[last : match.start()]))
        out.append(match.group())
        last = match.end()
    out.append(pattern.sub('', expr[last:]))
    return ''.join(out)


def _qualifier_pattern(table_name: str) -> re.Pattern[str]:
    r"""Returns a pattern that matches a ``<table_name>.`` qualifier within a
    SQL expression, ignoring case just like SQL identifier resolution does.

    Args:
        table_name: The name of the qualifying table.
    """
    return re.compile(
        rf'(?<![A-Za-z0-9_$."])'
        rf'(?:{re.escape(quote_ident(table_name))}|{re.escape(table_name)})'
        rf'\s*\.',
        flags=re.IGNORECASE,
    )


def _view_expr_message(
    column: str,
    expr: str,
    view: str,
    dialect: str,
    trust_expressions: bool,
    columns: Collection[str],
) -> str | None:
    r"""Why a view's column expression is dropped, or ``None`` to keep it.

    Args:
        column: How the dropped column is named in the message.
        expr: The expression as written in the view definition.
        view: ``'metric view'`` or ``'semantic view'``.
        dialect: ``'databricks'`` or ``'snowflake'``.
        trust_expressions: Whether to call any function the view names.
        columns: The columns of the table the expression is evaluated on.
    """
    hatch = ''
    reason = _unsafe_expr_reason(expr, dialect)
    if reason is None and not trust_expressions:
        reason = _untrusted_call_reason(expr, dialect, columns=columns)
        hatch = (
            '. Pass trust_expressions=True to keep it if you trust whoever '
            'authored the view'
        )
    if reason is None:
        return None
    return (
        f'Failed to add {column} since {reason}, and expressions taken from '
        f'the {view} are executed as SQL under your warehouse '
        f'credentials{hatch}'
    )


_METRIC_VIEW_ON_METRIC_VIEW = (
    'a metric view built on another metric view is not supported'
)


def _add_view_columns(table: Table, columns: Sequence[ColumnSpec]) -> list[str]:
    r"""Adds a view's columns to ``table``, leaving out each one that cannot
    be added on its own, and returns why for each.

    A table's expressions are sampled in one query, so a single failing
    expression, or a column of a type with no graph equivalent, fails the
    others with it. In that case nothing of the batch is kept, and the
    columns are added one at a time instead.

    Args:
        table: The table, whose rows can already be read.
        columns: The columns to add.
    """
    before = set(table._column_dict)
    try:
        table.add_columns(columns)
        return []
    except Warning:
        raise
    except Exception:
        for name in list(table._column_dict):
            if name not in before:
                table.remove_column(name)

    msgs: list[str] = []
    for column in columns:
        try:
            table.add_column(column)
        except Warning:
            raise
        except Exception as error:
            msgs.append(
                f"Failed to add column '{column.name}' to table "
                f"'{table.name}': {error}"
            )
    return msgs


def _join_key_reason(table: Table, key: str) -> str | None:
    r"""Why ``key`` cannot be the primary key a view's join gives ``table``,
    or ``None`` if it can.

    A many-to-one join promises one row of ``table`` per key, and neither
    Databricks nor a view checks it. A key that repeats would make one entity
    of several rows, so it is counted where the rows are. A row with no key
    is one no join reaches, and sampling leaves it out, so it is not counted
    against the key.
    """
    (count,) = count_keys([(table, (key,))])
    if count.rows is None or count.distinct is None:
        return None
    if count.distinct == 0:
        return f"its key '{key}' holds no value in table '{table.name}'"
    if count.repeats > 0:
        example = '' if count.example is None else f', e.g. {count.example!r}'
        return (
            f"its key '{key}' is not unique in table '{table.name}': "
            f'{count.repeats:,} of {count.rows:,} rows repeat a value'
            f'{example}. A graph table needs one row per key, so point the '
            f"join at a table or view with one row per '{key}'"
        )
    return None


def _infer_view_time_column(
    table: Table, window_orders: Sequence[str]
) -> list[str]:
    r"""Infers the time column of a metric view's table, and says how when
    the view's own columns would have led it elsewhere.

    A view often holds a timestamp only as a truncation or cast of a source
    column, such as ``DATE_TRUNC('MONTH', order_date)``, which says when a
    row happened only to the month, and starts at or before the column it
    truncates, so the shared rule would take it. Each such column is offered
    to the rule as the source column it derives from instead, added to the
    table for the purpose and kept only if taken. Where a dimension holds
    the source column's name, as ``order_date: DATE_TRUNC('MONTH',
    source.order_date)`` does, the source column is added under a name of
    its own, ``order_date_raw``, and the dimension keeps its name and its
    truncated value. A dimension a window measure orders by is the time
    axis the view's author chose, so it is offered alone first, and the
    rule decides among them all only if it refuses that.

    Args:
        table: The table, its columns and links added.
        window_orders: The dimensions the view's window measures order by.
    """
    from predictive_query.schema.typing import Dtype

    from sdm_relational_engine.rfm.backend.databricks.metric_view import (
        derived_column,
    )

    timestamps = [
        column.name
        for column in table.columns
        if column.stype == Stype.timestamp
    ]
    source = {
        name.lower(): column
        for name, column in table._source_column_dict.items()
    }
    derived_from: dict[str, str] = {}
    for name in timestamps:
        expr = table[name].expr
        under = None if expr is None else derived_column(str(expr))
        column = None if under is None else source.get(under.lower())
        if column is not None and column.dtype in (Dtype.date, Dtype.time):
            derived_from[name] = column.name

    added: list[str] = []
    # The source column each added column is, where a dimension has its
    # name and it is added under another:
    renamed: dict[str, str] = {}
    for raw in dict.fromkeys(derived_from.values()):
        taken = {column.name.lower() for column in table.columns}
        name = raw
        if table.has_column(raw) and table[raw].is_source:
            usable = table[raw].stype == Stype.timestamp
        else:
            if raw.lower() in taken:
                name = f'{raw}_raw'
                suffix = 2
                while name.lower() in taken:
                    name = f'{raw}_raw_{suffix}'
                    suffix += 1
            try:
                if name == raw:
                    table.add_column(raw)
                else:
                    table.add_column(
                        ColumnSpec(name=name, expr=quote_ident(raw, char='`'))
                    )
                    renamed[name] = raw
                added.append(name)
                usable = table[name].stype == Stype.timestamp
            except Warning:
                raise
            except Exception:
                usable = False
        derived_from = {
            k: name if v == raw else v
            for k, v in derived_from.items()
            if v != raw or usable
        }

    candidates = list(dict.fromkeys(derived_from.get(n, n) for n in timestamps))
    axes = list(
        dict.fromkeys(
            derived_from.get(name, name)
            for name in window_orders
            if name in timestamps
        )
    )
    chosen = None
    if axes and len(candidates) > 1:
        chosen = table._infer_time_column_among(axes)
    if chosen is None and candidates:
        chosen = table._infer_time_column_among(candidates)

    choice = table._time_column_choice
    kept = set() if choice is None else {choice.column, choice.fallback}
    for raw in added:
        if raw not in kept:
            table.remove_column(raw)

    reasons: list[str] = []
    if chosen is not None and chosen in axes and len(candidates) > 1:
        reasons.append('a window measure of the view orders by it')
    instead_of = tuple(n for n, raw in derived_from.items() if raw == chosen)
    if instead_of:
        assert choice is not None
        table._time_column_choice = replace(choice, instead_of=instead_of)
        verbs = (
            'truncates or casts' if len(instead_of) == 1 else 'truncate or cast'
        )
        reasons.append(
            f'{_quoted_names(instead_of)} {verbs} it, and it says when a row '
            f'happened more precisely'
        )
    if not reasons:
        return []
    added_note = ', and added it as a column' if chosen in added else ''
    if chosen in renamed:
        added_note = (
            f", the source column '{renamed[chosen]}' added under a name of "
            f'its own, since a dimension has its name'
        )
    return [
        f"Took '{chosen}' as the time column of table '{table.name}'"
        f'{added_note}, since {" and ".join(reasons)}'
    ]


def _measures_message(fact_table: Table, measures: Sequence[Any]) -> str:
    r"""Says that a view's measures are left to a predictive query, and the
    aggregation each simple one is, over 30 days for example.

    Args:
        fact_table: The view's source table, the columns of its measures
            added.
        measures: The view's
            :class:`~sdm_relational_engine.rfm.backend.databricks.metric_view.MetricViewMeasure`.
    """

    def _name(name: str) -> str:
        return name if re.fullmatch(r'[A-Za-z_]\w*', name) else f'`{name}`'

    # A measure names a column in any case, as Databricks reads it, and a
    # dimension that renames a column stands for it.
    columns = {
        column.name.lower(): column.name for column in fact_table.columns
    }
    for column in fact_table.columns:
        if column.expr is not None and (
            renamed := _renamed_column(str(column.expr))
        ):
            columns.setdefault(renamed.lower(), column.name)
    expressed: list[str] = []
    others: list[str] = []
    for measure in measures:
        if measure.aggregation is None:
            others.append(measure.name)
            continue
        function, column = measure.aggregation
        if column is not None and column.lower() not in columns:
            others.append(measure.name)
            continue
        target = '*' if column is None else _name(columns[column.lower()])
        expressed.append(
            f"'{measure.name}' is {function}({_name(fact_table.name)}."
            f'{target}, 0, 30, days)'
        )
    said = [
        'Left the measures to the predictive query, which aggregates the '
        'columns they read, added to their tables where missing'
    ]
    if expressed:
        said.append(f'over the next 30 days, {", ".join(expressed)}')
    if others:
        noun = 'has' if len(others) == 1 else 'have'
        said.append(f'{_quoted_names(others)} {noun} no single aggregation')
    return '; '.join(said)


def _renamed_column(expr: str) -> str | None:
    r"""The column an expression is, if it is one column and nothing else."""
    match = re.fullmatch(r'`((?:[^`]|``)+)`|([A-Za-z_]\w*)', expr.strip())
    if match is None:
        return None
    quoted, bare = match.groups()
    return bare if quoted is None else quoted.replace('``', '`')


def _real_name(table: Table, name: str) -> str:
    r"""``name`` as ``table``'s source spells it, which Databricks matches
    to ``name`` without regard to case, or ``name`` if it spells none.
    """
    if name in table._source_column_dict:
        return name
    folded = {column.lower(): column for column in table._source_column_dict}
    return folded.get(name.lower(), name)


def _quoted_names(names: Sequence[str]) -> str:
    quoted = [f"'{name}'" for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return f'{", ".join(quoted[:-1])} and {quoted[-1]}'


class ViewConversionWarning(UserWarning):
    r"""Warn that parts of a view could not be represented in a :class:`Graph`.

    Metric and semantic views normally convert only partially because measures
    and cross-table expressions have no graph equivalent. The
    dedicated category lets a project silence this warning without silencing
    every client warning. The same diagnostics appear on the returned graph as
    :attr:`Graph.conversion_messages` for programmatic handling.
    """


def _warn_view_conversion(message: str, owned_connection: Any) -> None:
    r"""Report what a view conversion dropped without leaking a connection.

    Under a warnings-as-errors policy ``warnings.warn`` raises, so the caller
    never receives the graph or the connection it would have owned. Closing the
    connection here preserves the constructor contract: the client closes any
    connection it opened when it does not return the graph.

    Args:
        message: The aggregated conversion diagnostics.
        owned_connection: The connection this constructor opened itself, or
            ``None`` when the caller supplied one (theirs stays open).
    """
    try:
        warnings.warn(message, ViewConversionWarning, stacklevel=3)
    except BaseException:
        if owned_connection is not None:
            owned_connection.close()
        raise


def _reraise_as_graph_error(error: BaseException, source: str) -> None:
    r"""Raise :class:`GraphConstructionError` for a foreign driver failure.

    Returns without raising for anything this package owns, for a built-in,
    and for a :class:`Warning` the caller's filters escalated into an error, so
    the caller's bare ``raise`` re-raises those unchanged with their original
    traceback. Only a third-party driver's exception is translated, which is
    the one class of failure that otherwise escapes graph construction with no
    base in common with the rest of this client.

    Deliberately called from the existing cleanup handler rather than applied
    as a decorator: a wrapper frame shifts the stack depth that
    :func:`_warn_view_conversion` relies on, which silently re-attributes every
    conversion warning to this module instead of to the caller.
    """
    if isinstance(error, KumoRelationalError | Warning):
        return
    if type(error).__module__ == 'builtins':
        return
    raise GraphConstructionError(
        f'Reading the {source} source failed while building the graph: '
        f'{type(error).__name__}: {error}'
    ) from error


def _require_discovered_tables(tables: Sequence[Any], where: str) -> None:
    r"""Reject a discovery query that found no tables.

    Naming a table that does not exist raises on every backend, but a mistyped
    database path or schema name is the more common slip and used to return a
    valid-looking empty graph. It failed much later with "At least one table
    needs to be added to the graph", pointing at the table list rather than the
    typo.

    Args:
        tables: The table names the discovery query returned.
        where: What was searched, for the message.
    """
    if len(tables) == 0:
        raise NoTablesFoundError(
            f'No tables found in {where}. Check the name, or '
            f'pass `tables=[...]` explicitly.'
        )


class Graph:
    r"""A relational graph of :class:`Table` objects and links.

    Use :class:`Graph` to finalize data and initialize :class:`KumoRelational`.

    .. code-block:: python

        >>> # doctest: +SKIP
        >>> import pandas as pd
        >>> import sdm_relational_engine.rfm as rfm

        >>> # Load data frames into memory:
        >>> df1 = pd.DataFrame(...)
        >>> df2 = pd.DataFrame(...)
        >>> df3 = pd.DataFrame(...)

        >>> # Define tables from data frames:
        >>> table1 = rfm.LocalTable(df1, name="table1")
        >>> table2 = rfm.LocalTable(df2, name="table2")
        >>> table3 = rfm.LocalTable(df3, name="table3")

        >>> # Create a graph from its tables:
        >>> graph = rfm.Graph([table1, table2, table3])

        >>> # Infer table metadata:
        >>> graph.infer_metadata()

        >>> # Infer links/edges:
        >>> graph.infer_links()

        >>> # Inspect table metadata:
        >>> for table in graph.tables.values():
        ...     table.print_metadata()

        >>> # Visualize graph:
        >>> print(graph.to_mermaid())

        >>> # Add/Remove edges between tables:
        >>> graph.link(src_table="table1", fkey="id1", dst_table="table2")
        >>> graph.unlink(src_table="table1", fkey="id1", dst_table="table2")

        >>> # Validate graph:
        >>> graph.validate()

    A graph holds the tables it is given, not copies, so a table put in two
    graphs is one table in both: a key or semantic type set through one is
    set in the other. The one exception is a lone edit stamp, which
    inference takes on a table nothing links to and refuses on one
    something does; it is settled for the graph the table is read through.
    Pass ``table.copy()`` to give a graph a table of its own.

    Build directly from a data source instead of :class:`Table` objects with
    :meth:`from_data`, :meth:`from_snowflake`,
    :meth:`from_snowflake_semantic_view`, :meth:`from_databricks`,
    :meth:`from_databricks_metric_view`, :meth:`from_postgres`, or
    :meth:`from_relbench`.
    """

    # Constructors ############################################################

    def __init__(
        self,
        tables: Sequence[Table],
        edges: Sequence[EdgeLike] | None = None,
    ) -> None:
        r"""Build a graph from ``tables``.

        ``edges=None`` adds the foreign keys the source catalog declares, on
        backends that have one. An explicit sequence, including an empty one,
        becomes the complete edge set. A caller that pins the graph's shape
        gets only that shape. Edges change what the model sees, so an
        unrequested catalog link changes predictions.
        """
        self._tables: dict[str, Table] = {}
        self._edges: list[Edge] = []
        self._conversion_messages: tuple[str, ...] = ()
        self._conversion_notes: tuple[str, ...] = ()
        self._connection: (
            AdbcSqliteConnection
            | DuckDBPyConnection
            | SnowflakeConnection
            | DatabricksConnection
            | PostgresConnection
            | SparkConnection
            | None
        ) = None

        for table in tables:
            self.add_table(table)

        for table in tables if edges is None else ():  # Source metadata:
            if not any(column.is_source for column in table.columns):
                continue
            for fkey in table._source_foreign_key_dict.values():
                if fkey.name not in table:
                    continue
                if not table[fkey.name].is_source:
                    continue
                dst_table_names = [
                    table.name
                    for table in self.tables.values()
                    if table.source_name == fkey.dst_table
                ]
                if len(dst_table_names) != 1:
                    continue
                dst_table = self[dst_table_names[0]]
                if dst_table._primary_key != fkey.primary_key:
                    continue
                if not dst_table[fkey.primary_key].is_source:
                    continue
                self.link(table.name, fkey.name, dst_table.name)

        for edge in edges or []:
            _edge = Edge._cast(edge)
            assert _edge is not None
            if _edge not in self._edges:
                self.link(*_edge)

    @classmethod
    def from_data(
        cls,
        df_dict: dict[str, pd.DataFrame],
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Creates a :class:`Graph` from a dictionary of
        :class:`pandas.DataFrame` objects.

        Automatically infers table metadata and links by default.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import pandas as pd
            >>> import sdm_relational_engine.rfm as rfm

            >>> # Load data frames into memory:
            >>> df1 = pd.DataFrame(...)
            >>> df2 = pd.DataFrame(...)
            >>> df3 = pd.DataFrame(...)

            >>> # Create a graph from a dictionary of data frames:
            >>> graph = rfm.Graph.from_data({
            ...     "table1": df1,
            ...     "table2": df2,
            ...     "table3": df3,
            ... })

        Args:
            df_dict: A dictionary of data frames, where the keys are the names
                of the tables and the values hold table data.
            edges: An optional list of
                :class:`~sdm_relational_engine.rfm.Edge` objects to add to the
                graph. If not provided (:obj:`None`), edges will be
                automatically inferred from the data in case
                ``infer_metadata=True``. An empty sequence is not the same
                thing: it means "these edges and no others", and suppresses
                inference.
            infer_metadata: Whether to infer metadata for all tables in the
                graph.
            verbose: Whether to print verbose output.
        """
        from sdm_relational_engine.rfm.backend.local import LocalTable

        graph = cls(
            tables=[LocalTable(df, name) for name, df in df_dict.items()],
            edges=edges,
        )

        if infer_metadata:
            graph.infer_metadata(verbose=False)

            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_sqlite(
        cls,
        connection: AdbcSqliteConnection
        | SqliteConnectionConfig
        | str
        | Path
        | dict[str, Any],
        tables: Sequence[str | dict[str, Any]] | None = None,
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Create a :class:`Graph` from :class:`sqlite`, inferring table
        metadata and links by default.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import sdm_relational_engine.rfm as rfm

            >>> # Create a graph from all tables in a SQLite database:
            >>> graph = rfm.Graph.from_sqlite('data.db')

            >>> # Select specific tables and override table metadata:
            >>> graph = rfm.Graph.from_sqlite('data.db', tables=[
            ...     'USERS',
            ...     {'name': 'ORDERS', 'source_name': 'ORDERS_SNAPSHOT'},
            ...     {'name': 'ITEMS', 'primary_key': 'ITEM_ID'},
            ... ])

            >>> # Provide known edges while still inferring table metadata:
            >>> graph = rfm.Graph.from_sqlite(
            ...     'data.db',
            ...     tables=[
            ...         {'name': 'USERS', 'primary_key': 'USER_ID'},
            ...         {'name': 'ORDERS', 'primary_key': 'ORDER_ID'},
            ...         {'name': 'ITEMS', 'primary_key': 'ITEM_ID'},
            ...     ],
            ...     edges=[
            ...         ('ORDERS', 'USER_ID', 'USERS'),
            ...         ('ORDERS', 'ITEM_ID', 'ITEMS'),
            ...     ],
            ... )

        Args:
            connection: An open connection from
                :meth:`~sdm_relational_engine.rfm.backend.sqlite.connect` or
                the path to the database file.
            tables: Set of table names or :class:`SQLiteTable` keyword
                arguments to include, such as ``name``, ``source_name``,
                ``primary_key``, or ``time_column``. If ``None``, will add all
                tables present in the database.
            edges: Optional edge-like objects to add to the graph. Each item
                may be a :class:`~sdm_relational_engine.rfm.Edge`, a
                dictionary, or a tuple such as
                ``(src_table, fkey, dst_table)``. If not provided
                (:obj:`None`), edges will be automatically inferred from the
                data in case ``infer_metadata=True``, and the foreign keys
                the source catalog declares are added as well. An empty
                sequence is not the same thing: it means "these edges and no
                others", suppressing both.
            infer_metadata: Whether to infer missing metadata for all tables in
                the graph.
            verbose: Whether to print verbose output.
        """
        from sdm_relational_engine.rfm.backend.sqlite import (
            Connection,
            SQLiteTable,
            connect,
        )

        internal_connection = False
        if not isinstance(connection, Connection):
            config = SqliteConnectionConfig._cast(connection)
            assert isinstance(config, SqliteConnectionConfig)
            connection = connect(config.uri, **config.kwargs)
            internal_connection = True
        assert isinstance(connection, Connection)

        try:
            if tables is None:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    tables = [row[0] for row in cursor.fetchall()]
                _require_discovered_tables(tables, 'the SQLite database')

            table_kwargs: list[dict[str, Any]] = []
            for table in tables:
                kwargs = dict(name=table) if isinstance(table, str) else table
                table_kwargs.append(kwargs)

            graph = cls(
                tables=[
                    SQLiteTable(connection=connection, **kwargs)
                    for kwargs in table_kwargs
                ],
                edges=edges,
            )
        except BaseException as error:
            if internal_connection:
                connection.close()
            _reraise_as_graph_error(error, 'SQLite')
            raise

        if internal_connection:
            graph._connection = connection

        if infer_metadata:
            graph.infer_metadata(verbose=False)

            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_duckdb(
        cls,
        connection: DuckDBPyConnection
        | DuckDBConnectionConfig
        | str
        | Path
        | dict[str, Any]
        | None = None,
        tables: Sequence[str | dict[str, Any]] | None = None,
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Creates a :class:`Graph` from a :class:`duckdb` database.

        Automatically infers table metadata and links by default.

        Args:
            connection: An open connection from
                :meth:`~sdm_relational_engine.rfm.backend.duckdb.connect`, the
                database file path, a connection config, or ``None`` for an
                in-memory database.
            tables: Set of table names or :class:`DuckDBTable` keyword
                arguments to include. If ``None``, will add all non-temporary
                tables present in the database.
            edges: An optional list of
                :class:`~sdm_relational_engine.rfm.Edge` objects to add to the
                graph. If not provided (:obj:`None`), edges will be
                automatically inferred from the data in case
                ``infer_metadata=True``, and the foreign keys the source
                catalog declares are added as well. An empty sequence is not
                the same thing: it means "these edges and no others",
                suppressing both.
            infer_metadata: Whether to infer missing metadata for all tables in
                the graph.
            verbose: Whether to print verbose output.
        """
        from sdm_relational_engine.rfm.backend.duckdb import (
            Connection,
            DuckDBTable,
            connect,
        )

        internal_connection = False
        if not isinstance(connection, Connection):
            if connection is None:
                config = DuckDBConnectionConfig()
            else:
                config = DuckDBConnectionConfig._cast(connection)
            assert isinstance(config, DuckDBConnectionConfig)
            connection = connect(config.uri, **config.kwargs)
            internal_connection = True
        assert isinstance(connection, Connection)

        try:
            if tables is None:
                with connection.cursor() as cursor:
                    # Only the schema an unqualified name resolves in, since
                    # that is where each table is read from.
                    cursor.execute(
                        'SELECT table_name FROM duckdb_tables() '
                        'WHERE NOT temporary AND NOT internal '
                        'AND database_name = current_database() '
                        'AND schema_name = current_schema()'
                    )
                    tables = [row[0] for row in cursor.fetchall()]
                _require_discovered_tables(tables, 'the DuckDB database')

            table_kwargs: list[dict[str, Any]] = []
            for table in tables:
                kwargs = dict(name=table) if isinstance(table, str) else table
                table_kwargs.append(kwargs)

            graph = cls(
                tables=[
                    DuckDBTable(connection=connection, **kwargs)
                    for kwargs in table_kwargs
                ],
                edges=edges,
            )
        except BaseException as error:
            if internal_connection:
                connection.close()
            _reraise_as_graph_error(error, 'DuckDB')
            raise

        if internal_connection:
            graph._connection = connection

        if infer_metadata:
            graph.infer_metadata(verbose=False)

            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_snowflake(
        cls,
        connection: SnowflakeConnection | dict[str, Any] | None = None,
        tables: Sequence[str | dict[str, Any]] | None = None,
        database: str | None = None,
        schema: str | None = None,
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Create a :class:`Graph` from a :class:`snowflake` database and
        schema, inferring table metadata and links by default.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import sdm_relational_engine.rfm as rfm

            >>> # Create a graph directly in a Snowflake notebook:
            >>> graph = rfm.Graph.from_snowflake(
            ...     database='MY_DB',
            ...     schema='MY_SCHEMA',
            ... )

            >>> # Create a graph using explicit Snowflake credentials:
            >>> graph = rfm.Graph.from_snowflake(
            ...     connection={
            ...         'user': '<snowflake_user>',
            ...         'password': '<snowflake_password>',
            ...         'account': '<snowflake_account>',
            ...         'warehouse': '<snowflake_warehouse>',
            ...         'database': 'MY_DB',
            ...         'schema': 'MY_SCHEMA',
            ...     },
            ...     database='MY_DB',
            ...     schema='MY_SCHEMA',
            ...     tables=[
            ...         'USERS',
            ...         'ORDERS',
            ...         {'name': 'ITEMS', 'source_name': 'ITEMS_SNAPSHOT'},
            ...     ],
            ... )

            >>> # Fine-grained control over table specification:
            >>> graph = rfm.Graph.from_snowflake(tables=[
            ...     'USERS',
            ...     dict(name='ORDERS', source_name='ORDERS_SNAPSHOT'),
            ...     dict(name='ITEMS', schema='OTHER_SCHEMA'),
            ... ], database='DEFAULT_DB', schema='DEFAULT_SCHEMA')

        Args:
            connection: An open connection from
                :meth:`~sdm_relational_engine.rfm.backend.snow.connect` or the
                :class:`snowflake` connector keyword arguments to open a new
                connection. If ``None``, will re-use an active session in case
                it exists, or create a new connection from credentials stored
                in environment variables.
            tables: Set of table names or :class:`SnowTable` keyword arguments
                to include. If ``None``, will add all tables present in the
                current database and schema.
            database: The database.
            schema: The schema.
            edges: An optional list of
                :class:`~sdm_relational_engine.rfm.Edge` objects to add to the
                graph. If not provided (:obj:`None`), edges will be
                automatically inferred from the data in case
                ``infer_metadata=True``, and the foreign keys the source
                catalog declares are added as well. An empty sequence is not
                the same thing: it means "these edges and no others",
                suppressing both.
            infer_metadata: Whether to infer metadata for all tables in the
                graph.
            verbose: Whether to print verbose output.
        """
        from sdm_relational_engine.rfm.backend.snow import (
            Connection,
            SnowTable,
            connect,
            paramstyle,
        )

        internal_connection = False
        if not isinstance(connection, Connection):
            connection = connect(**(connection or {}))
            internal_connection = True
        assert isinstance(connection, Connection)

        try:
            if database is None or schema is None:
                with connection.cursor() as cursor:
                    cursor.execute(
                        'SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()'
                    )
                    result = cursor.fetchone()
                    assert result is not None
                    database = database or result[0]
                    assert database is not None
                    schema = schema or result[1]

            if tables is None:
                if schema is None:
                    raise ValueError(
                        "No current 'schema' set. Please specify "
                        'the Snowflake schema manually'
                    )

                with paramstyle(connection), connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                    SELECT TABLE_NAME
                    FROM {quote_ident(database)}.INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = ?
                    """,
                        (schema,),
                    )
                    tables = [row[0] for row in cursor.fetchall()]
                _require_discovered_tables(
                    tables, f"schema '{schema}' of database '{database}'"
                )

            table_kwargs: list[dict[str, Any]] = []
            for table in tables:
                if isinstance(table, str):
                    kwargs = dict(name=table, database=database, schema=schema)
                else:
                    kwargs = copy.copy(table)
                    kwargs.setdefault('database', database)
                    kwargs.setdefault('schema', schema)
                table_kwargs.append(kwargs)

            graph = cls(
                tables=[
                    SnowTable(connection=connection, **kwargs)
                    for kwargs in table_kwargs
                ],
                edges=edges,
            )
        except BaseException as error:
            if internal_connection:
                connection.close()
            _reraise_as_graph_error(error, 'Snowflake')
            raise

        if internal_connection:
            graph._connection = connection

        if infer_metadata:
            graph.infer_metadata(verbose=False)

            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_databricks(
        cls,
        connection: DatabricksConnection
        | SparkConnection
        | dict[str, Any]
        | None = None,
        tables: Sequence[str | dict[str, Any]] | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Create a :class:`Graph` from a :class:`databricks` SQL warehouse,
        inferring table metadata and links by default.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import sdm_relational_engine.rfm as rfm

            >>> # Create a graph using explicit Databricks credentials:
            >>> graph = rfm.Graph.from_databricks(
            ...     connection={
            ...         'server_hostname': '<workspace>.cloud.databricks.com',
            ...         'http_path': '/sql/1.0/warehouses/<warehouse_id>',
            ...         'access_token': '<access_token>',
            ...     },
            ...     catalog='MY_CATALOG',
            ...     schema='MY_SCHEMA',
            ...     tables=[
            ...         'USERS',
            ...         'ORDERS',
            ...         {'name': 'ITEMS', 'source_name': 'ITEMS_SNAPSHOT'},
            ...     ],
            ... )

            >>> # Fine-grained control over table specification:
            >>> graph = rfm.Graph.from_databricks(tables=[
            ...     'USERS',
            ...     dict(name='ORDERS', source_name='ORDERS_SNAPSHOT'),
            ...     dict(name='ITEMS', schema='OTHER_SCHEMA'),
            ... ], catalog='DEFAULT_CATALOG', schema='DEFAULT_SCHEMA')

        Args:
            connection: An open connection from
                :func:`~sdm_connectors.backends.databricks.connect`,
                a notebook's Spark session wrapped in
                :class:`~sdm_connectors.backends.spark.SparkConnection`, or the
                :class:`databricks.sql` connector keyword arguments to open a
                new connection. If ``None``, will open a connection from
                credentials stored in environment variables.
            tables: Set of table names or :class:`DatabricksTable` keyword
                arguments to include. If ``None``, will add all tables present
                in the current catalog and schema.
            catalog: The Unity Catalog catalog.
            schema: The schema.
            edges: An optional list of
                :class:`~sdm_relational_engine.rfm.Edge` objects to add to the
                graph. If not provided (:obj:`None`), edges will be
                automatically inferred from the data in case
                ``infer_metadata=True``, and the foreign keys the source
                catalog declares are added as well. An empty sequence is not
                the same thing: it means "these edges and no others",
                suppressing both.
            infer_metadata: Whether to infer metadata for all tables in the
                graph.
            verbose: Whether to print verbose output.
        """
        from sdm_relational_engine.rfm.backend.databricks import (
            CONNECTION_TYPES,
            DatabricksTable,
        )

        internal_connection = False
        if not isinstance(connection, CONNECTION_TYPES):
            from sdm_connectors.backends.databricks import connect

            connection = connect(**(connection or {}))
            internal_connection = True
        assert isinstance(connection, CONNECTION_TYPES)

        try:
            if catalog is None or schema is None:
                with connection.cursor() as cursor:
                    cursor.execute('SELECT current_catalog(), current_schema()')
                    result = cursor.fetchone()
                    assert result is not None
                    catalog = catalog or result[0]
                    assert catalog is not None
                    schema = schema or result[1]

            if tables is None:
                if schema is None:
                    raise ValueError(
                        "No current 'schema' set. Please specify "
                        'the Databricks schema manually'
                    )

                quoted_catalog = quote_ident(catalog, char='`')
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                    SELECT table_name
                    FROM {quoted_catalog}.information_schema.tables
                    WHERE table_schema = ?
                      AND table_type != 'METRIC_VIEW'
                    """,
                        parameters=[schema],
                    )
                    tables = [row[0] for row in cursor.fetchall()]
                _require_discovered_tables(
                    tables, f"schema '{schema}' of catalog '{catalog}'"
                )

            table_kwargs: list[dict[str, Any]] = []
            for table in tables:
                if isinstance(table, str):
                    kwargs = dict(name=table, catalog=catalog, schema=schema)
                else:
                    kwargs = copy.copy(table)
                    kwargs.setdefault('catalog', catalog)
                    kwargs.setdefault('schema', schema)
                table_kwargs.append(kwargs)

            graph = cls(
                tables=[
                    DatabricksTable(connection=connection, **kwargs)
                    for kwargs in table_kwargs
                ],
                edges=edges,
            )
        except BaseException as error:
            if internal_connection:
                connection.close()
            _reraise_as_graph_error(error, 'Databricks')
            raise

        if internal_connection:
            graph._connection = connection

        if infer_metadata:
            graph.infer_metadata(verbose=False)

            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_postgres(
        cls,
        connection: PostgresConnection | str | dict[str, Any] | None = None,
        tables: Sequence[str | dict[str, Any]] | None = None,
        schema: str | None = None,
        edges: Sequence[EdgeLike] | None = None,
        infer_metadata: bool = True,
        verbose: bool = True,
    ) -> Self:
        r"""Creates a graph from a PostgreSQL database.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import sdm_relational_engine.rfm as rfm
            >>> graph = rfm.Graph.from_postgres(
            ...     connection={
            ...         'host': 'postgres.example.com',
            ...         'dbname': 'analytics',
            ...         'user': 'analyst',
            ...         'password': '<password>',
            ...     },
            ...     schema='public',
            ... )

        Args:
            connection: An open psycopg connection, PostgreSQL URI/libpq
                conninfo string, or psycopg connection keyword arguments. If
                ``None``, PostgreSQL ``PG*`` environment variables are used.
            tables: Table names or :class:`PostgresTable` keyword arguments.
                If ``None``, all base tables in ``schema`` are included.
            schema: PostgreSQL schema. Defaults to ``current_schema()``.
            edges: Optional explicit graph edges. If ``None``, edges are
                inferred when ``infer_metadata=True``, and the foreign keys
                PostgreSQL declares are added as well. An empty sequence is not
                the same thing: it means "these edges and no others",
                suppressing both.
            infer_metadata: Whether to infer metadata for every table.
            verbose: Whether to print inferred metadata and links.
        """
        from sdm_relational_engine.rfm.backend.postgres import (
            Connection,
            PostgresTable,
            connect,
        )
        from sdm_relational_engine.rfm.backend.postgres.table import _reading

        internal_connection = False
        if not isinstance(connection, Connection):
            if isinstance(connection, str):
                connection = connect(connection)
            else:
                connection = connect(**(connection or {}))
            internal_connection = True
        assert isinstance(connection, Connection)

        try:
            if schema is None:
                with _reading(connection) as cursor:
                    cursor.execute('SELECT current_schema()')
                    result = cursor.fetchone()
                    assert result is not None
                    schema = result[0]

            if schema is None:
                raise ValueError(
                    "No current 'schema' set. Please specify the PostgreSQL "
                    'schema manually'
                )

            if tables is None:
                with _reading(connection) as cursor:
                    cursor.execute(
                        'SELECT table_name\n'
                        'FROM information_schema.tables\n'
                        'WHERE table_schema = %s\n'
                        "  AND table_type = 'BASE TABLE'\n"
                        'ORDER BY table_name',
                        (schema,),
                    )
                    tables = [row[0] for row in cursor.fetchall()]
                _require_discovered_tables(tables, f"schema '{schema}'")

            table_kwargs: list[dict[str, Any]] = []
            for table in tables:
                if isinstance(table, str):
                    kwargs = dict(name=table, schema=schema)
                else:
                    kwargs = copy.copy(table)
                    kwargs.setdefault('schema', schema)
                table_kwargs.append(kwargs)

            graph = cls(
                tables=[
                    PostgresTable(connection=connection, **kwargs)
                    for kwargs in table_kwargs
                ],
                edges=edges,
            )
        except BaseException as error:
            if internal_connection:
                connection.close()
            _reraise_as_graph_error(error, 'PostgreSQL')
            raise

        if internal_connection:
            graph._connection = connection

        if infer_metadata:
            graph.infer_metadata(verbose=False)
            if edges is None:
                graph.infer_links(verbose=False)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def from_databricks_metric_view(
        cls,
        metric_view_name: str,
        connection: DatabricksConnection
        | SparkConnection
        | dict[str, Any]
        | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        verbose: bool = True,
        *,
        trust_expressions: bool = False,
        warn: bool = True,
    ) -> Self:
        r"""Create a :class:`Graph` from a Databricks Unity Catalog metric
        view.

        Its star or snowflake schema becomes a graph: ``source`` becomes the
        fact table, each join a linked table named for its alias, and each
        dimension an expression column on its referenced table. Measures define
        aggregations, not row-level values, so the graph omits them; express
        aggregations with a predictive query, as
        :attr:`conversion_notes` spells each simple one. Other
        unrepresentable elements are dropped with a warning.

        A joined table that cannot be read, because it is missing, empty or
        not readable to you, drops its join with a warning, and a column that
        cannot be added, such as an expression that fails, drops alone. Only a
        source table that cannot be read fails the conversion.

        The view's ``filter`` is applied to the fact table, so every read of it
        sees only the rows the view holds. A filter on a joined table cannot
        be, and the view is refused rather than read unfiltered: create a
        regular view with that ``WHERE`` clause and point the metric view's
        source at it.

        A parameter the filter or a dimension references is read as its
        default, as a query of the view without arguments reads it. A filter
        on a parameter with no default refuses the view, and a dimension on
        one is dropped with a warning.

        Dimension expressions later run as SQL under your warehouse
        credentials, making the view definition executable input. Expressions
        with a statement separator, comment, or sub-query are not plain scalar
        expressions and are dropped with a warning. So are expressions calling
        anything but a pure built-in function, such as a user-defined function
        or ``secret()``, unless ``trust_expressions`` is set. Load only trusted
        metric views.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import sdm_relational_engine.rfm as rfm

            >>> graph = rfm.Graph.from_databricks_metric_view(
            ...     'MY_CATALOG.MY_SCHEMA.MY_METRIC_VIEW',
            ...     connection={
            ...         'server_hostname': '<workspace>.cloud.databricks.com',
            ...         'http_path': '/sql/1.0/warehouses/<warehouse_id>',
            ...         'access_token': '<access_token>',
            ...     },
            ... )

        Args:
            metric_view_name: The name of the metric view, either fully
                qualified or resolved against ``catalog`` and ``schema``.
            connection: An open connection from
                :func:`~sdm_connectors.backends.databricks.connect` or the
                :class:`databricks.sql` connector keyword arguments to open a
                new connection. If ``None``, will open a connection from
                credentials stored in environment variables.
            catalog: The Unity Catalog catalog of the metric view. If set to
                ``None``, an unqualified ``metric_view_name`` is resolved
                against the connection defaults.
            schema: The schema of the metric view. If set to ``None``, an
                unqualified ``metric_view_name`` is resolved against the
                connection defaults.
            verbose: Whether to print verbose output.
            trust_expressions: Whether to keep dimensions, and accept a
                filter, that call functions other than pure built-ins.
                Sub-queries, comments and statement separators are refused
                regardless.
            warn: Whether to warn of what the view loses, as a
                :class:`ViewConversionWarning`, and of a time column taken
                on the rows read alone, as a
                :class:`~sdm_relational_engine.rfm.TimeColumnWarning`. A
                caller that says them itself turns this off, rather than
                filter warnings for every thread:
                :attr:`conversion_messages` and each table's
                ``time_column_choice`` hold them either way.

        Raises:
            ViewConversionError: If the view is refused for what it is, with
                what to change: a query or a metric view as its source, a
                join name that collides with the source table's, or one of
                the below. A ``ValueError`` as well.
            UnsupportedFilterError: If the view's filter references a joined
                table or a parameter with no default, or is not a safe scalar
                condition. A :class:`ViewConversionError`.
        """
        from sdm_relational_engine.rfm.backend.databricks import (
            CONNECTION_TYPES,
            DatabricksTable,
        )
        from sdm_relational_engine.rfm.backend.databricks.metric_view import (
            SOURCE_ALIAS,
            MetricViewJoin,
            MetricViewWildcard,
            UnsupportedJoinError,
            _skipped_join_message,
            parse_metric_view,
            parse_table_reference,
            read_metric_view_definition,
            resolve_join_keys,
            unquote_ident,
        )

        internal_connection = False
        if not isinstance(connection, CONNECTION_TYPES):
            from sdm_connectors.backends.databricks import connect

            connection = connect(**(connection or {}))
            internal_connection = True
        assert isinstance(connection, CONNECTION_TYPES)

        try:
            name_parts = parse_table_reference(metric_view_name)
            if name_parts is None:
                raise ViewConversionError(
                    f"Invalid metric view name '{metric_view_name}'"
                )
            if len(name_parts) == 1 and schema is not None:
                name_parts = (schema, *name_parts)
            if len(name_parts) == 2 and catalog is not None:
                name_parts = (catalog, *name_parts)
            quoted_name = '.'.join(
                quote_ident(part, char='`') for part in name_parts
            )

            description = read_metric_view_definition(connection, quoted_name)
            view_catalog, view_schema = description.catalog, description.schema

            def _table_parts(parts: tuple[str, ...]) -> tuple[str, str, str]:
                if len(parts) == 2 and view_catalog is not None:
                    parts = (view_catalog, *parts)
                elif len(parts) == 1 and view_catalog and view_schema:
                    parts = (view_catalog, view_schema, *parts)
                if len(parts) != 3:
                    raise ViewConversionError(
                        f'Could not fully qualify table '
                        f"'{'.'.join(parts)}' referenced by "
                        f"metric view '{metric_view_name}'"
                    )
                return parts[0], parts[1], parts[2]

            tables_by_alias: dict[str, DatabricksTable] = {}

            def _fact_table(parts: tuple[str, ...]) -> DatabricksTable:
                r"""The source table, its columns read. Tables start empty,
                so the checks below see their columns before anything reads
                a row.
                """
                if SOURCE_ALIAS not in tables_by_alias:
                    catalog_, schema_, source_ = _table_parts(parts)
                    table = DatabricksTable(
                        connection,
                        name=source_,
                        source_name=source_,
                        catalog=catalog_,
                        schema=schema_,
                        columns=[],
                        primary_key=None,
                    )
                    table._warns_unchecked = warn
                    try:
                        table._source_column_dict  # noqa: B018
                    except MetricViewAsTableError as error:
                        raise ViewConversionError(
                            f"'{table.source_name}' is a metric view; "
                            f'{_METRIC_VIEW_ON_METRIC_VIEW}, point its source '
                            f'at the underlying table'
                        ) from error
                    tables_by_alias[SOURCE_ALIAS] = table
                return tables_by_alias[SOURCE_ALIAS]

            spec = parse_metric_view(
                description.definition,
                description.parameters,
                source_columns=lambda parts: (
                    _fact_table(parts)._source_column_dict
                ),
            )
            msgs = list(spec.messages)
            notes = list(spec.notes)

            dtypes = {
                name: DatabricksTable._to_dtype(data_type)
                for name, data_type in description.dtypes.items()
            }

            fact_table = _fact_table(spec.source)
            if any(join.alias == fact_table.name for join in spec.joins):
                raise ViewConversionError(
                    f"Cannot convert metric view '{metric_view_name}' into a "
                    f"graph since the join name '{fact_table.name}' collides "
                    f'with the name of the source table'
                )
            joins_by_alias = {join.alias: join for join in spec.joins}

            def _table(alias: str) -> DatabricksTable:
                if alias not in tables_by_alias:
                    join = joins_by_alias[alias]
                    catalog_, schema_, source_ = _table_parts(join.table)
                    tables_by_alias[alias] = DatabricksTable(
                        connection,
                        name=join.alias,
                        source_name=source_,
                        catalog=catalog_,
                        schema=schema_,
                        columns=[],
                        primary_key=None,
                    )
                    tables_by_alias[alias]._warns_unchecked = warn
                return tables_by_alias[alias]

            def _columns_of(alias: str) -> list[str]:
                if alias == SOURCE_ALIAS:
                    return list(fact_table._source_column_dict)
                # A join that cannot be read is dropped below with its reason,
                # so here it shows no columns.
                try:
                    return list(_table(alias)._source_column_dict)
                except Warning:
                    raise
                except Exception:
                    return []

            # Every read of the fact table runs the filter, so it is checked
            # against the table's columns and attached before the first one.
            if spec.filter is not None:
                fact_table.row_filter = (
                    spec.filter.trusted()
                    if trust_expressions
                    else spec.filter.checked(_columns_of(SOURCE_ALIAS))
                )

            columns_by_alias: dict[str, list[ColumnSpec]] = defaultdict(list)
            renames: dict[str, dict[str, str]] = defaultdict(dict)
            for column_spec in spec.columns:
                if isinstance(column_spec, MetricViewWildcard):
                    # Databricks refuses a view whose wildcards name a column
                    # twice, so one that exists names each once.
                    excluded = {name.lower() for name in column_spec.excluded}
                    columns_by_alias[column_spec.alias].extend(
                        ColumnSpec(name=name)
                        for name in _columns_of(column_spec.alias)
                        if name.lower() not in excluded
                    )
                elif (
                    real := {
                        name.lower(): name
                        for name in _columns_of(column_spec.alias)
                    }.get(unquote_ident(column_spec.expr).lower())
                ) is not None and real.lower() == column_spec.name.lower():
                    # A column named as itself in any case, as Databricks
                    # reads it, is that column, named as the view names it.
                    if real != column_spec.name:
                        renames[column_spec.alias][real] = column_spec.name
                    columns_by_alias[column_spec.alias].append(
                        ColumnSpec(name=column_spec.name)
                    )
                elif msg := _view_expr_message(
                    f"column '{column_spec.name}'",
                    column_spec.expr,
                    'metric view',
                    'databricks',
                    trust_expressions,
                    _columns_of(column_spec.alias),
                ):
                    msgs.append(msg)
                else:
                    columns_by_alias[column_spec.alias].append(
                        ColumnSpec(
                            name=column_spec.name,
                            expr=column_spec.expr,
                            dtype=dtypes.get(column_spec.name),
                        )
                    )

            # Measures are left to a predictive query, which aggregates
            # columns, so the columns they read are added where missing. A
            # measure is only read here, never run.
            for measure in spec.measures:
                for alias, name in measure.inputs:
                    columns = {c.lower(): c for c in _columns_of(alias)}
                    if (column := columns.get(name.lower())) is None:
                        continue
                    taken = {
                        spec_.name.lower(): spec_
                        for spec_ in columns_by_alias[alias]
                    }
                    # A dimension renaming the column stands for it.
                    if any(
                        (renamed := _renamed_column(str(spec_.expr)))
                        and renamed.lower() == column.lower()
                        for spec_ in columns_by_alias[alias]
                        if spec_.expr is not None
                    ):
                        continue
                    if (same := taken.get(column.lower())) is None:
                        columns_by_alias[alias].append(ColumnSpec(name=column))
                    elif not same.is_source:
                        msgs.append(
                            f"Did not add the column '{column}', which the "
                            f"measure '{measure.name}' reads, since the "
                            f"dimension '{same.name}' has its name"
                        )

            # A join key is named as the view's condition spells it, where
            # no dimension, wildcard or measure already names it in any
            # case: Databricks names the column as those do.
            for join in spec.joins:
                try:
                    other_alias, other_key, child_key = resolve_join_keys(
                        join,
                        child_columns=_columns_of(join.alias),
                        parent_columns=_columns_of(join.parent_alias),
                    )
                except UnsupportedJoinError:
                    continue  # Dropped below, with its reason.
                for alias, key in (
                    (join.alias, child_key),
                    (other_alias, other_key),
                ):
                    folded = {name.lower(): name for name in _columns_of(alias)}
                    real = folded.get(key.lower())
                    named = any(
                        spec_.name.lower() == key.lower()
                        for spec_ in columns_by_alias[alias]
                    )
                    if (
                        real is not None
                        and real != key
                        and real not in renames[alias]
                        and not named
                    ):
                        renames[alias][real] = key

            # Before a row is read, so every read names them as the view does.
            for alias, names in renames.items():
                try:
                    _table(alias)._name_columns(names)
                except Warning:
                    raise
                except Exception:
                    if alias == SOURCE_ALIAS:
                        raise
                    # A join that cannot be read is dropped below.

            graph = cls(tables=[])

            # A failure to read the fact table aborts. A joined table that
            # cannot be read, or a column that cannot be added, is dropped
            # below with its reason instead.
            fact_table._source_sample_df  # noqa: B018
            msgs.extend(
                _add_view_columns(
                    fact_table, columns_by_alias.get(SOURCE_ALIAS, [])
                )
            )
            graph.add_table(fact_table)
            table_dict: dict[str, Table] = {SOURCE_ALIAS: fact_table}

            def _drop_join(join: MetricViewJoin, reason: str) -> None:
                r"""Reports a join dropped, with the table and dimensions
                that go with it, which a question may have been asked of.
                """
                dimensions = [
                    column.name for column in columns_by_alias[join.alias]
                ]
                msgs.append(
                    _skipped_join_message(
                        join.alias, reason, [join.alias], dimensions
                    )
                )

            for join in spec.joins:
                parent_table = table_dict.get(join.parent_alias)
                if parent_table is None:
                    _drop_join(
                        join,
                        f"its parent join '{join.parent_alias}' was not added",
                    )
                    continue

                try:
                    table = _table(join.alias)
                    table._source_column_dict  # noqa: B018
                    table._source_sample_df  # noqa: B018
                except Warning:
                    raise
                except MetricViewAsTableError:
                    _drop_join(
                        join,
                        f"'{table.source_name}' is a metric view; "
                        f"{_METRIC_VIEW_ON_METRIC_VIEW}, point the join's "
                        f'source at the underlying table',
                    )
                    continue
                except Exception as error:
                    _drop_join(join, f'its table could not be read: {error}')
                    continue

                # Adding a join mutates tables that are already part of the
                # graph, so record every mutation and undo it in full in case
                # the join is rejected - a dropped join must not leave any
                # trace behind:
                undo_columns: list[tuple[Table, str, Column | None]] = []
                undo_primary_key: tuple[Table, str, Stype] | None = None
                join_msgs = _add_view_columns(
                    table, columns_by_alias.get(join.alias, [])
                )

                try:
                    other_alias, other_key, child_key = resolve_join_keys(
                        join,
                        child_columns=table._source_column_dict,
                        parent_columns=parent_table._source_column_dict,
                    )
                    other_table = table_dict[other_alias]
                    other_key = _real_name(other_table, other_key)
                    child_key = _real_name(table, child_key)

                    if join.cardinality == 'many_to_one':
                        one_table, one_key = table, child_key
                        many_table, many_key = other_table, other_key
                    else:
                        one_table, one_key = other_table, other_key
                        many_table, many_key = table, child_key

                    if (
                        one_table._primary_key is not None
                        and one_table._primary_key != one_key
                    ):
                        raise UnsupportedJoinError(
                            f"Failed to add join '{join.alias}' since table "
                            f"'{one_table.name}' already uses "
                            f"'{one_table._primary_key}' as its primary key"
                        )

                    # A key set here was counted then, so it is counted once.
                    if one_table._primary_key is None and (
                        reason := _join_key_reason(one_table, one_key)
                    ):
                        raise UnsupportedJoinError(
                            f"Failed to add join '{join.alias}' since {reason}"
                        )

                    for key_table, key in (
                        (one_table, one_key),
                        (many_table, many_key),
                    ):
                        if not key_table.has_column(key):
                            key_table.add_column(key)
                            undo_columns.append((key_table, key, None))
                        elif not key_table[key].is_source:
                            undo_columns.append(
                                (key_table, key, key_table[key])
                            )
                            key_table.remove_column(key)
                            key_table.add_column(key)
                            join_msgs.append(
                                f"Replaced the derived column '{key}' of table "
                                f"'{key_table.name}' with its physical source "
                                f"column since join '{join.alias}' references "
                                f'it as a key'
                            )

                    if one_table._primary_key is None:
                        undo_primary_key = (
                            one_table,
                            one_key,
                            one_table[one_key].stype,
                        )
                        one_table.primary_key = one_key

                    graph.add_table(table)
                    table_dict[join.alias] = table
                    graph.link(many_table.name, many_key, one_table.name)
                    msgs.extend(join_msgs)
                except Warning:
                    raise
                except Exception as error:
                    if table_dict.get(join.alias) is table:
                        graph.remove_table(join.alias)
                        del table_dict[join.alias]
                    if undo_primary_key is not None:
                        undo_table, undo_key, undo_stype = undo_primary_key
                        undo_table.primary_key = None
                        undo_table[undo_key].stype = undo_stype
                    for undo_table, undo_key, undo_column in reversed(
                        undo_columns
                    ):
                        if undo_table.has_column(undo_key):
                            undo_table.remove_column(undo_key)
                        if undo_column is not None:
                            undo_table._column_dict[undo_key] = undo_column
                    if isinstance(error, UnsupportedJoinError):
                        # Its message names the join, which `_drop_join`
                        # names again.
                        prefix = f"Failed to add join '{join.alias}' since "
                        _drop_join(join, str(error).removeprefix(prefix))
                    else:
                        _drop_join(
                            join, f'its keys could not be linked: {error}'
                        )

            if spec.measures:
                notes.append(_measures_message(fact_table, spec.measures))

            for table in graph.tables.values():
                notes.extend(_infer_view_time_column(table, spec.window_orders))

            msgs.extend(graph._drop_invalid_edges())
            graph._settle_time_columns()
            graph.validate()

            if verbose:
                graph.print_metadata()
                graph.print_links()

            graph._conversion_messages = tuple(msgs)
            graph._conversion_notes = tuple(notes)
        except BaseException as error:
            if internal_connection:
                connection.close()
            _reraise_as_graph_error(error, 'Databricks')
            raise

        if internal_connection:
            graph._connection = connection

        if warn and len(msgs) > 0:
            title = (
                f'Could not fully convert the metric view definition '
                f"'{metric_view_name}' into a graph:\n"
            )
            _warn_view_conversion(
                title + '\n'.join(f'- {msg}' for msg in msgs),
                connection if internal_connection else None,
            )

        return graph

    @classmethod
    def from_snowflake_semantic_view(
        cls,
        semantic_view_name: str,
        connection: SnowflakeConnection | dict[str, Any] | None = None,
        verbose: bool = True,
        *,
        trust_expressions: bool = False,
        warn: bool = True,
    ) -> Self:
        r"""Create a :class:`Graph` from a Snowflake semantic view.

        Logical tables become graph tables backed by ``base_table`` and
        ``relationships`` become edges. Dimensions, time dimensions, and facts
        become columns. An ``expr`` containing only the column name, optionally
        qualified by its table, is plain; any other ``expr`` is an expression
        column. The following are dropped with a warning listing each one:

        - **Metrics.** They define aggregations rather than row-level values;
          express aggregations with a predictive query instead.
        - **Composite primary keys** and **composite relationship keys**.
        - **Columns whose expression references another logical table.**
        - **Columns whose expression calls anything but a pure built-in
          function**, such as a user-defined function or ``SYSTEM$WAIT()``,
          unless ``trust_expressions`` is set. Expressions run as SQL under
          your warehouse credentials every time the graph samples.
        - **Relationships whose referencing (left) key is itself a primary
          key**, i.e. one-to-one relationships. Note a view made only of such
          relationships yields a graph with no edges at all.

        Each time column is inferred, preferring ``time_dimensions``. Correct a
        mistake with ``graph['table'].time_column = 'column'``.
        ``semantic_view_name`` reaches Snowflake unchanged, so an unqualified
        name uses the connection's current database and schema; qualify it as
        ``'MY_DB.MY_SCHEMA.MY_VIEW'`` to reach another one.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import sdm_relational_engine.rfm as rfm

            >>> graph = rfm.Graph.from_snowflake_semantic_view(
            ...     'MY_DB.MY_SCHEMA.MY_SEMANTIC_VIEW',
            ...     connection={
            ...         'account': '<account>',
            ...         'user': '<user>',
            ...         'password': '<password>',
            ...         'warehouse': '<warehouse>',
            ...     },
            ... )

        Args:
            semantic_view_name: The name of the semantic view, either fully
                qualified or resolved against the connection's current database
                and schema.
            connection: An open connection from
                :meth:`~sdm_relational_engine.rfm.backend.snow.connect` or the
                :class:`snowflake.connector` keyword arguments to open a new
                connection. If ``None``, will open a connection from
                credentials stored in environment variables.
            verbose: Whether to print verbose output.
            trust_expressions: Whether to keep columns that call functions
                other than pure built-ins. Sub-queries, comments and
                statement separators are refused regardless.
            warn: Whether to warn of what the view loses, and of a time
                column taken on the rows read alone, as
                :meth:`from_databricks_metric_view` does.

        Raises:
            GraphConstructionError: If the semantic view does not exist or
                the role cannot read it. The driver's own error is
                translated, and travels as the cause.
        """
        import yaml

        from sdm_relational_engine.rfm.backend.snow import (
            Connection,
            SnowTable,
            connect,
            paramstyle,
        )

        internal_connection = False
        if not isinstance(connection, Connection):
            connection = connect(**(connection or {}))
            internal_connection = True
        assert isinstance(connection, Connection)

        try:
            with paramstyle(connection), connection.cursor() as cursor:
                sql = 'SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)'
                cursor.execute(sql, (semantic_view_name,))
                result = cursor.fetchone()
                assert result is not None
                cfg = yaml.safe_load(result[0])

            graph = cls(tables=[])
        except BaseException as error:
            if internal_connection:
                connection.close()
            _reraise_as_graph_error(error, 'Snowflake')
            raise

        if internal_connection:
            graph._connection = connection

        msgs = []
        table_names = {table_cfg['name'] for table_cfg in cfg['tables']}
        for table_cfg in cfg['tables']:
            table_name = table_cfg['name']
            source_table_name = table_cfg['base_table']['table']
            database = table_cfg['base_table']['database']
            schema = table_cfg['base_table']['schema']

            primary_key: str | None = None
            if 'primary_key' in table_cfg:
                primary_key_cfg = table_cfg['primary_key']
                if len(primary_key_cfg['columns']) == 1:
                    primary_key = primary_key_cfg['columns'][0]
                elif len(primary_key_cfg['columns']) > 1:
                    msgs.append(
                        f'Failed to add primary key for table '
                        f"'{table_name}' since composite primary keys "
                        f'are not yet supported'
                    )

            self_pattern = _qualifier_pattern(table_name)
            other_patterns = [
                _qualifier_pattern(name)
                for name in table_names
                if name != table_name
            ]

            # The table starts empty, so the checks below see its columns
            # before anything reads a row.
            table = SnowTable(
                connection,
                name=table_name,
                source_name=source_table_name,
                database=database,
                schema=schema,
                columns=[],
                primary_key=None,
            )
            table._warns_unchecked = warn

            columns: list[ColumnSpec] = []
            unsupported_columns: list[str] = []
            for column_cfg in chain(
                table_cfg.get('dimensions', []),
                table_cfg.get('time_dimensions', []),
                table_cfg.get('facts', []),
            ):
                column_name = column_cfg['name']
                column_expr = column_cfg.get('expr', None)
                column_data_type = column_cfg.get('data_type', None)

                if column_expr is None:
                    columns.append(ColumnSpec(name=column_name))
                    continue

                column_expr = _sub_outside_literals(
                    self_pattern, column_expr
                ).strip()

                if column_expr == column_name:
                    columns.append(ColumnSpec(name=column_name))
                    continue

                # Drop expressions that reference other tables (for now):
                masked_expr = _mask_literals(column_expr)
                if any(
                    pattern.search(masked_expr) for pattern in other_patterns
                ):
                    unsupported_columns.append(column_name)
                    continue

                if msg := _view_expr_message(
                    f"column '{column_name}' of table '{table_name}'",
                    column_expr,
                    'semantic view',
                    'snowflake',
                    trust_expressions,
                    list(table._source_column_dict),
                ):
                    msgs.append(msg)
                    continue

                column = ColumnSpec(
                    name=column_name,
                    expr=column_expr,
                    dtype=SnowTable._to_dtype(column_data_type),
                )
                columns.append(column)

            if len(unsupported_columns) == 1:
                msgs.append(
                    f"Failed to add column '{unsupported_columns[0]}' "
                    f"of table '{table_name}' since its expression "
                    f'references other tables'
                )
            elif len(unsupported_columns) > 1:
                msgs.append(
                    f"Failed to add columns '{unsupported_columns}' "
                    f"of table '{table_name}' since their expressions "
                    f'reference other tables'
                )

            # A column or key the view declares and its table cannot hold,
            # one missing from it say, refuses the view, saying which.
            try:
                table.add_columns(columns)
                if primary_key is not None:
                    if primary_key not in table:
                        table.add_column(primary_key)
                    table.primary_key = primary_key
            except ValueError as error:
                if isinstance(error, KumoRelationalError):
                    raise
                raise ViewConversionError(
                    f"Cannot convert semantic view '{semantic_view_name}' "
                    f"into a graph since its table '{table_name}' cannot "
                    f'hold what it declares: {error}'
                ) from error

            # TODO: allow registering time columns explicitly, without the
            # heuristic.
            time_candidates = [  # Prioritize columns in `time_dimensions`:
                column_cfg['name']
                for column_cfg in table_cfg.get('time_dimensions', [])
                if table.has_column(column_cfg['name'])
                and table[column_cfg['name']].stype == Stype.timestamp
            ]
            if len(time_candidates) == 0:
                time_candidates = [
                    column.name
                    for column in table.columns
                    if column.stype == Stype.timestamp
                ]
            if len(time_candidates) > 0:
                table._infer_time_column_among(time_candidates)

            graph.add_table(table)

        for relation_cfg in cfg.get('relationships', []):
            name = relation_cfg['name']
            if len(relation_cfg['relationship_columns']) != 1:
                msgs.append(
                    f"Failed to add relationship '{name}' since "
                    f'composite key references are not yet supported'
                )
                continue

            left_table = relation_cfg['left_table']
            left_key = relation_cfg['relationship_columns'][0]['left_column']
            right_table = relation_cfg['right_table']
            right_key = relation_cfg['relationship_columns'][0]['right_column']

            added_column = False
            try:
                if graph[right_table]._primary_key != right_key:
                    msgs.append(
                        f"Failed to add relationship '{name}' since "
                        f"the referenced key '{right_key}' of table "
                        f"'{right_table}' is not a primary key"
                    )
                    continue

                if graph[left_table]._primary_key == left_key:
                    msgs.append(
                        f"Failed to add relationship '{name}' since "
                        f"the referencing key '{left_key}' of table "
                        f"'{left_table}' is a primary key"
                    )
                    continue

                if left_key not in graph[left_table]:
                    graph[left_table].add_column(left_key)
                    added_column = True

                graph.link(left_table, left_key, right_table)
            except (ValueError, KeyError) as error:
                if added_column:
                    graph[left_table].remove_column(left_key)
                msgs.append(
                    f"Failed to add relationship '{name}' since its "
                    f'keys could not be linked: {error}'
                )

        msgs.extend(graph._drop_invalid_edges())
        graph._settle_time_columns()
        graph.validate()

        if verbose:
            graph.print_metadata()
            graph.print_links()

        graph._conversion_messages = tuple(msgs)
        if warn and len(msgs) > 0:
            title = (
                f'Could not fully convert the semantic view definition '
                f"'{semantic_view_name}' into a graph:\n"
            )
            _warn_view_conversion(
                title + '\n'.join(f'- {msg}' for msg in msgs),
                connection if internal_connection else None,
            )

        return graph

    @classmethod
    def from_relbench(
        cls,
        dataset: str,
        verbose: bool = True,
    ) -> Graph:
        r"""Loads a `RelBench <https://relbench.stanford.edu>`_ dataset into a
        :class:`Graph` instance.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import sdm_relational_engine.rfm as rfm

            >>> graph = rfm.Graph.from_relbench("f1")

        Args:
            dataset: The RelBench dataset name.
            verbose: Whether to print verbose output.
        """
        from sdm_relational_engine.rfm.relbench import from_relbench

        graph = from_relbench(dataset, verbose=verbose)

        if verbose:
            graph.print_metadata()
            graph.print_links()

        return graph

    @classmethod
    def graph_and_pquery_from_timeseries(
        cls,
        df: pd.DataFrame,
        timeseries_col: str,
        timestamps_col: str | None = None,
        time_delta: pd.Timedelta | None = None,
        anchor_time: pd.Timestamp | None = None,
        entity_col: str | None = None,
        num_timeframes: int = 1,
    ) -> tuple[Self, str]:
        r"""Create a :class:`Graph` and predictive query from a flat time-series
        table.

        Each row is an entity whose ``timeseries_col`` holds an array of
        historical observations. This produces the two-table *(entity +
        target)* structure Kumo Relational expects and a ready-to-use query.

        The input ``df`` is split into:

        * An entity table with one row per input row and all columns except
          ``timeseries_col`` and, when given, ``timestamps_col``.
        * A target table with one row per observation, a foreign key to the
          entity table, a timestamp, and the observed value.

        .. code-block:: python

            >>> # doctest: +SKIP
            >>> import pandas as pd
            >>> import sdm_relational_engine.rfm as rfm

            >>> df = pd.DataFrame({
            ...     'customer_id': [1, 2, 3],
            ...     'sales': [[10, 20, 15, 30], [5, 8, 12], [100, 95, 80]],
            ... })

            >>> anchor = pd.Timestamp('2024-01-10')
            >>> graph, pquery = rfm.Graph.graph_and_pquery_from_timeseries(
            ...     df,
            ...     timeseries_col='sales',
            ...     entity_col='customer_id',
            ...     time_delta=pd.Timedelta('1D'),
            ...     anchor_time=anchor,
            ...     num_timeframes=4,
            ... )
            >>> # pquery == ("PREDICT MAX(target.value, 0, 1, days) "
            >>> #            "FORECAST 4 TIMEFRAMES "
            >>> #            "FOR EACH entity.customer_id")

            >>> model = rfm.KumoRelational(graph)
            >>> result = model.predict(pquery, anchor_time=anchor)

        Args:
            df: Input DataFrame where each row represents one entity and
                ``timeseries_col`` holds a list or array of scalar
                observations for that entity.
            timeseries_col: Name of the column containing per-entity
                observation arrays.
            timestamps_col: Optional name of the column containing per-entity
                timestamp arrays (one timestamp per observation value).  Must
                be the same length as ``timeseries_col`` for every row.  When
                ``None``, synthetic timestamps are generated backwards from
                ``anchor_time`` using ``time_delta``.
            time_delta: Step size between consecutive observations.  Required
                when ``timestamps_col`` is ``None``.  Also controls the
                prediction-window size in the generated pquery.  When
                ``timestamps_col`` is provided and ``time_delta`` is ``None``,
                the typical step is inferred from the median inter-observation
                interval across all entities.
            anchor_time: The forecast cutoff timestamp.  Required when
                ``timestamps_col`` is ``None``; used to place synthetic
                timestamps so that the last observation falls at
                ``anchor_time - time_delta``.  Pass the same value to
                :meth:`KumoRelational.predict` as ``anchor_time``.
            entity_col: Name of an existing column to use as the entity
                primary key.  When ``None``, integer IDs ``[0, 1, ..., n-1]``
                are generated and stored in a new column named ``entity_id``.
            num_timeframes: Number of timeframes to forecast.

        Returns:
            A tuple ``(graph, pquery)`` where *graph* is a :class:`Graph`
            ready for use with :class:`KumoRelational` and *pquery* is a
            predictive-query string for single-step forecasting.  Pass
            *pquery* to :meth:`KumoRelational.predict` with ``anchor_time`` to
            generate forecasts.
        """
        from sdm_relational_engine.rfm.backend.local import LocalTable

        target_table_name = 'target'
        entity_table_name = 'entity'
        value_col_name = 'value'
        timestamp_col_name = 'timestamp'

        if timeseries_col not in df.columns:
            raise ValueError(
                f"timeseries_col '{timeseries_col}' not found in the DataFrame"
            )

        if timestamps_col is not None and timestamps_col not in df.columns:
            raise ValueError(
                f"timestamps_col '{timestamps_col}' not found in the DataFrame"
            )

        if timestamps_col is None and time_delta is None:
            raise ValueError(
                "Either 'timestamps_col' or 'time_delta' must be provided. "
                'Supply a column of per-row timestamp lists via '
                "'timestamps_col', or a constant step size via 'time_delta' "
                'to generate synthetic timestamps.'
            )

        if timestamps_col is None and anchor_time is None:
            raise ValueError(
                "'anchor_time' is required when 'timestamps_col' is not "
                'provided.  Pass the forecast cutoff timestamp so that '
                'synthetic timestamps can be placed correctly.'
            )

        if entity_col is not None and entity_col not in df.columns:
            raise ValueError(
                f"entity_col '{entity_col}' not found in the DataFrame"
            )

        pk_col = entity_col if entity_col is not None else 'entity_id'

        if entity_col is None and pk_col in df.columns:
            raise ValueError(
                f"entity_id '{pk_col}' conflicts with an existing "
                f'column in the DataFrame.  Rename the existing column or '
                f"pass a custom 'entity_col'."
            )

        if time_delta is not None:
            td = time_delta
        else:
            assert timestamps_col is not None
            td = _infer_timedelta_from_timestamps(df[timestamps_col])

        pquery_amount, pquery_unit = _timedelta_to_pquery(td)

        df = df.reset_index(drop=True)

        drop_cols = [timeseries_col]
        if timestamps_col is not None:
            drop_cols.append(timestamps_col)

        entity_df = df.drop(columns=drop_cols)
        if entity_col is None:
            entity_df.insert(0, pk_col, pd.RangeIndex(len(df)))

        series_list = df[timeseries_col].tolist()

        if timestamps_col is not None:
            timestamps_list = df[timestamps_col].tolist()
        else:
            assert anchor_time is not None
            timestamps_list = [
                [anchor_time - (len(s) - i) * td for i in range(len(s))]
                for s in series_list
            ]

        fk_vals: list = []
        ts_vals: list = []
        val_vals: list = []
        entity_ids = entity_df[pk_col].tolist()

        for eid, series, timestamps in zip(
            entity_ids, series_list, timestamps_list
        ):
            n = len(series)
            fk_vals.extend([eid] * n)
            ts_vals.extend(timestamps)
            val_vals.extend(series)

        target_df = pd.DataFrame(
            {
                pk_col: fk_vals,
                timestamp_col_name: pd.to_datetime(ts_vals),
                value_col_name: pd.array(val_vals, dtype=float),
            }
        )

        entity_table = LocalTable(
            entity_df,
            entity_table_name,
            primary_key=pk_col,
        )
        target_table = LocalTable(
            target_df,
            target_table_name,
            time_column=timestamp_col_name,
        )
        target_table[value_col_name].stype = Stype.numerical

        graph = cls([entity_table, target_table])
        graph.link(target_table_name, pk_col, entity_table_name)

        pquery = (
            f'PREDICT MAX({target_table_name}.{value_col_name}, '
            f'0, {pquery_amount}, {pquery_unit}) '
            f'FORECAST {num_timeframes} TIMEFRAMES '
            f'FOR EACH {entity_table_name}.{pk_col}'
        )

        return graph, pquery

    # Backend #################################################################

    @property
    def backend(self) -> DataBackend | None:
        backends = [table.backend for table in self._tables.values()]
        return backends[0] if len(backends) > 0 else None

    # Tables ##################################################################

    def has_table(self, name: str) -> bool:
        r"""Returns ``True`` if the graph has a table with name ``name``;
        ``False`` otherwise.
        """
        return name in self.tables

    def table(self, name: str) -> Table:
        r"""Returns the table with name ``name`` in the graph.

        Raises:
            KeyError: If ``name`` is not present in the graph.
        """
        if not self.has_table(name):
            raise KeyError(f"Table '{name}' not found in graph")
        return self.tables[name]

    @property
    def tables(self) -> dict[str, Table]:
        r"""Returns the dictionary of table objects.

        The tables are the ones given, not copies. A lone edit stamp that
        inference left to the links is settled for this graph as they are
        read, so a table in two graphs has each graph's time column when
        read through it.
        """
        self._settle_time_columns()
        return self._tables

    def add_table(self, table: Table) -> Self:
        r"""Adds a table to the graph.

        Args:
            table: The table to add.

        Raises:
            KeyError: If a table with the same name already exists in the
                graph.
            ValueError: If the table belongs to a different backend than the
                rest of the tables in the graph.
        """
        if table.name in self._tables:
            raise KeyError(
                f"Cannot add table with name '{table.name}' to "
                f'this graph; table names must be globally unique.'
            )

        if self.backend is not None and table.backend != self.backend:
            raise ValueError(
                f'Cannot register a table with backend '
                f"'{table.backend}' to this graph since other "
                f"tables have backend '{self.backend}'."
            )

        self._tables[table.name] = table
        # A table settled in another graph may be linked differently here.
        self._settle_time_columns()

        return self

    def remove_table(self, name: str) -> Self:
        r"""Removes a table with ``name`` from the graph.

        Args:
            name: The table to remove.

        Raises:
            KeyError: If no such table is present in the graph.
        """
        if not self.has_table(name):
            raise KeyError(f"Table '{name}' not found in the graph")

        del self._tables[name]

        self._edges = [
            edge
            for edge in self._edges
            if edge.src_table != name and edge.dst_table != name
        ]
        # Its edges may have been the last links to another table.
        self._settle_time_columns()

        return self

    @property
    def metadata(self) -> pd.DataFrame:
        r"""Returns a :class:`pandas.DataFrame` object containing metadata
        information about the tables in this graph.

        The returned dataframe has columns ``"Name"``, ``"Primary Key"``,
        ``"Time Column"``, and ``"End Time Column"``, which provide an
        aggregated view of the properties of the tables of this graph.

        Example:
            >>> # doctest: +SKIP
            >>> import sdm_relational_engine.rfm as rfm
            >>> graph = rfm.Graph(tables=...).infer_metadata()
            >>> graph.metadata  # doctest: +SKIP
                Name   Primary Key  Time Column  End Time Column
            0   users  user_id      -            -
        """
        tables = list(self.tables.values())

        return pd.DataFrame(
            {
                'Name': pd.Series(dtype=str, data=[t.name for t in tables]),
                'Primary Key': pd.Series(
                    dtype=str, data=[t._primary_key or '-' for t in tables]
                ),
                'Time Column': pd.Series(
                    dtype=str, data=[t._time_column or '-' for t in tables]
                ),
                'End Time Column': pd.Series(
                    dtype=str,
                    data=[t._end_time_column or '-' for t in tables],
                ),
            }
        )

    def print_metadata(self) -> None:
        r"""Prints the :meth:`~Graph.metadata` of the graph."""
        msg = 'Graph Metadata'
        if not in_tmux():
            msg = f'🗂️ {msg}'
        display.title(msg)

        display.dataframe(self.metadata)

    def infer_metadata(self, verbose: bool = True) -> Self:
        r"""Infers metadata for all tables in the graph.

        Args:
            verbose: Whether to print verbose output.

        Note:
            For more information, please see
            :meth:`sdm_relational_engine.rfm.Table.infer_metadata`.
        """
        for table in self.tables.values():
            table.infer_metadata(verbose=False)
        self._settle_time_columns()

        if verbose:
            self.print_metadata()

        return self

    def _settle_time_columns(self) -> None:
        r"""Refuse a lone edit stamp on a table an edge points at, and take
        it on one none does: the first is an entity edited in place, the
        second a log of changes.

        Called whenever a table or an edge is added or removed, since either
        can change which tables something links to, and whenever the tables
        are read through this graph: a table put in two graphs is one object,
        linked to in one and maybe not the other, and the graph it is read
        through decides.
        """
        waiting = [
            (name, table)
            for name, table in self._tables.items()
            if table._time_column_waits_on_links
        ]
        if not waiting:
            return
        linked = {edge.dst_table for edge in self._edges}
        for name, table in waiting:
            table._settle_time_column(name in linked)

    # Edges ###################################################################

    @property
    def edges(self) -> list[Edge]:
        r"""Returns the edges of the graph."""
        return self._edges

    @property
    def conversion_messages(self) -> tuple[str, ...]:
        r"""What :meth:`from_databricks_metric_view` or
        :meth:`from_snowflake_semantic_view` could not represent in this graph.

        Empty for every other constructor, and empty when a view converted
        cleanly. The same messages are also raised as a
        :class:`ViewConversionWarning`; this is the form to assert on, e.g.
        before running predictions that depend on a relationship the view
        declared. How a view was read where nothing was lost is in
        :attr:`conversion_notes` instead.
        """
        return self._conversion_messages

    @property
    def conversion_notes(self) -> tuple[str, ...]:
        r"""How :meth:`from_databricks_metric_view` read the view where
        nothing was lost, which no warning repeats.

        The aggregation each of the view's measures is in a predictive
        query, the time column taken in place of a date the view truncates,
        and each parameter read as its default. Empty for every other
        constructor.
        """
        return self._conversion_notes

    def print_links(self) -> None:
        r"""Prints the :meth:`~Graph.edges` of the graph."""
        edges = sorted(
            [
                (
                    edge.dst_table,
                    self[edge.dst_table]._primary_key,
                    edge.src_table,
                    edge.fkey,
                )
                for edge in self.edges
            ]
        )

        if in_tmux():
            display.title('Graph Links (FK <> PK)')
        else:
            display.title('🕸️ Graph Links (FK ↔️ PK)')
        if len(edges) > 0:
            items: list[str] = []
            for edge in edges:
                fkey = f'`{edge[2]}.{edge[3]}`'
                pkey = f'`{edge[0]}.{edge[1]}`'
                arrow = '<>' if in_tmux() else '↔️'
                items.append(f'{fkey} {arrow} {pkey}')
            display.unordered_list(items)
        else:
            display.italic('No links registered')

    def _resolve_foreign_key(
        self,
        src_table: str,
        fkey: str | Sequence[str],
        dst_table: str,
    ) -> str:
        r"""The single column an edge joins on.

        A reference naming one column is that column, however it was written,
        so a one-element sequence takes the ordinary path rather than folding
        a column that already identifies a row on its own.
        """
        names = (fkey,) if isinstance(fkey, str) else tuple(fkey)
        if len(names) > 1:
            return self._derive_composite_foreign_key(
                src_table, names, dst_table
            )

        name = names[0]
        if self.has_table(dst_table):
            destination = self[dst_table]
            identity = destination.primary_key_columns
            if (
                destination.has_composite_primary_key
                and name != destination.primary_key.name
            ):
                raise ValueError(
                    f"Cannot link '{src_table}' to '{dst_table}' on "
                    f"'{name}': '{dst_table}' identifies a row by "
                    f'{list(identity)} together, so one column cannot name '
                    f'one of its rows. Reference every part of the identity.'
                )
        return name

    def _derive_composite_foreign_key(
        self,
        src_table: str,
        fkey: Sequence[str],
        dst_table: str,
    ) -> str:
        r"""Folds a multi-column reference into the key the edge joins on.

        The destination's identity is carried by a derived column, so a table
        referencing it needs the same fold over its own columns. The two are
        then checked to actually meet: a fold that agrees on neither side
        produces an edge that matches no rows, which trains a graph with a
        relationship silently missing rather than failing.
        """
        fkey = tuple(fkey)
        if not self.has_table(src_table):
            raise ValueError(
                f"Source table '{src_table}' does not exist in the graph"
            )
        if not self.has_table(dst_table):
            raise ValueError(
                f"Destination table '{dst_table}' does not exist in the graph"
            )

        source, destination = self[src_table], self[dst_table]
        expected = destination.primary_key_columns
        if len(expected) != len(fkey):
            raise ValueError(
                f"Cannot link '{src_table}' to '{dst_table}' on {list(fkey)}: "
                f"the destination's identity is {list(expected)}, so the "
                f'reference has to name {len(expected)} column(s), not '
                f'{len(fkey)}'
            )

        missing = [name for name in fkey if not source.has_column(name)]
        if missing:
            raise ValueError(
                f"Cannot link '{src_table}' to '{dst_table}': column(s) "
                f"{missing} are not present in '{src_table}'. A table "
                f'referencing a composite identity has to carry every part '
                f'of it.'
            )
        composite_key.refuse_unfoldable_dtypes(
            [(name, source[name].dtype) for name in fkey]
        )

        self._assert_composite_link_meets(source, destination, fkey)

        derived = source._derived_key_column_name(fkey)
        if not source.has_column(derived):
            source._materialize_derived_key(derived, fkey)
        source[derived].stype = Stype.ID
        return derived

    def _assert_composite_link_meets(
        self,
        source: Table,
        destination: Table,
        fkey: Sequence[str],
    ) -> None:
        r"""Refuses a composite link whose two sides share no value.

        Checked before the derived column is built, so a rejected link leaves
        the source exactly as it found it.
        """
        destination_key = destination.primary_key
        if destination_key is None:
            return
        left_frame = source._source_sample_df
        if any(name not in left_frame.columns for name in fkey):
            return
        right = destination._sample_values(destination_key.name)
        if right is None:
            return
        left = composite_key.encode_frame(left_frame, list(fkey))
        if len(left) == 0 or len(right) == 0:
            return
        if len(set(left).intersection(set(right))) > 0:
            return
        raise ValueError(
            f"Linking '{source.name}' to '{destination.name}' on "
            f'{list(fkey)} matches no rows: no value of {list(fkey)} in '
            f"'{source.name}' identifies a row of '{destination.name}'. "
            f'Check that the columns are named in the same order as '
            f"'{destination.name}' declares its identity "
            f'({list(destination.primary_key_columns)}), and that they hold '
            f'the same values.'
        )

    def link(
        self,
        src_table: str | Table,
        fkey: str | Sequence[str],
        dst_table: str | Table,
    ) -> Self:
        r"""Links two tables (``src_table`` and ``dst_table``) from the foreign
        key ``fkey`` in the source table to the primary key in the destination
        table.

        The link is treated as bidirectional.

        Args:
            src_table: The name of the source table of the edge. This table
                must have a foreign key with name :obj:`fkey` that links to the
                primary key in the destination table.
            fkey: The name of the foreign key in the source table.
            dst_table: The name of the destination table of the edge. This
                table must have a primary key that links to the source table's
                foreign key.

        Raises:
            ValueError: if the edge is already present in the graph, if the
                source table does not exist in the graph, if the destination
                table does not exist in the graph, if the source key does not
                exist in the source table.
        """
        if isinstance(src_table, Table):
            src_table = src_table.name
        assert isinstance(src_table, str)

        if isinstance(dst_table, Table):
            dst_table = dst_table.name
        assert isinstance(dst_table, str)

        fkey = self._resolve_foreign_key(src_table, fkey, dst_table)

        edge = Edge(src_table, fkey, dst_table)

        if edge in self.edges:
            raise ValueError(f'{edge} already exists in the graph')

        if not self.has_table(src_table):
            raise ValueError(
                f"Source table '{src_table}' does not exist in the graph"
            )

        if not self.has_table(dst_table):
            raise ValueError(
                f"Destination table '{dst_table}' does not exist in the graph"
            )

        if not self[src_table].has_column(fkey):
            raise ValueError(
                f"Source key '{fkey}' does not exist as a column "
                f"in source table '{src_table}'"
            )

        if not Stype.ID.supports_dtype(self[src_table][fkey].dtype):
            raise ValueError(
                f"Cannot use '{fkey}' in source table "
                f"'{src_table}' as a foreign key due to its "
                f'incompatible data type. Foreign keys must have '
                f"data type 'int', 'float' or 'string' "
                f"(got '{self[src_table][fkey].dtype}')"
            )

        self[src_table][fkey].stype = Stype.ID
        self._edges.append(edge)
        self._inferred_edges = tuple(
            marked
            for marked in getattr(self, '_inferred_edges', ())
            if marked != (src_table, fkey, dst_table)
        )
        self._settle_time_columns()

        return self

    def unlink(
        self,
        src_table: str | Table,
        fkey: str | Sequence[str],
        dst_table: str | Table,
    ) -> Self:
        r"""Removes an :class:`~sdm_relational_engine.rfm.Edge` from the graph.

        Args:
            src_table: The name of the source table of the edge.
            fkey: The name of the foreign key in the source table.
            dst_table: The name of the destination table of the edge.

        Raises:
            ValueError: if the edge is not present in the graph.
        """
        if isinstance(src_table, Table):
            src_table = src_table.name
        assert isinstance(src_table, str)
        if not isinstance(fkey, str):
            names = tuple(fkey)
            fkey = (
                names[0]
                if len(names) == 1
                else self[src_table]._derived_key_column_name(names)
            )

        if isinstance(dst_table, Table):
            dst_table = dst_table.name
        assert isinstance(dst_table, str)

        edge = Edge(src_table, fkey, dst_table)

        if edge not in self.edges:
            raise ValueError(f'{edge} is not present in the graph')

        self._edges.remove(edge)
        self._settle_time_columns()

        return self

    def infer_links(self, verbose: bool = True) -> Self:
        r"""Infers missing links for the tables and adds them as edges to the
        graph.

        Args:
            verbose: Whether to print verbose output.
        """
        declared = {
            (edge.src_table, edge.fkey, edge.dst_table) for edge in self.edges
        }
        known_edges = {(edge.src_table, edge.fkey) for edge in self.edges}

        for table in self.tables.values():  # Use links from source metadata:
            if not any(column.is_source for column in table.columns):
                continue
            for fkey in table._source_foreign_key_dict.values():
                if fkey.name not in table:
                    continue
                if not table[fkey.name].is_source:
                    continue
                if (table.name, fkey.name) in known_edges:
                    continue
                dst_table_names = [
                    table.name
                    for table in self.tables.values()
                    if table.source_name == fkey.dst_table
                ]
                if len(dst_table_names) != 1:
                    continue
                dst_table = self[dst_table_names[0]]
                if dst_table._primary_key != fkey.primary_key:
                    continue
                if not dst_table[fkey.primary_key].is_source:
                    continue
                self.link(table.name, fkey.name, dst_table.name)
                known_edges.add((table.name, fkey.name))

        # A list of primary key candidates (+score) for every column:
        candidate_dict: dict[
            tuple[str, str],
            list[tuple[str, float]],
        ] = defaultdict(list)

        for dst_table in self.tables.values():
            dst_key = dst_table.primary_key

            if dst_key is None:
                continue

            assert dst_key.dtype is not None
            dst_number = dst_key.dtype.is_int() or dst_key.dtype.is_float()
            dst_string = dst_key.dtype.is_string()

            dst_table_name = dst_table.name.lower()
            dst_key_name = dst_key.name.lower()

            for src_table in self.tables.values():
                src_table_name = src_table.name.lower()

                for src_key in src_table.columns:
                    if (src_table.name, src_key.name) in known_edges:
                        continue

                    if src_key in {
                        src_table.primary_key,
                        src_table.time_column,
                        src_table.end_time_column,
                    }:
                        continue  # Cannot link to special columns.

                    src_number = (
                        src_key.dtype.is_int() or src_key.dtype.is_float()
                    )
                    src_string = src_key.dtype.is_string()

                    if src_number != dst_number or src_string != dst_string:
                        continue  # Non-compatible data types.

                    src_key_name = src_key.name.lower()

                    score = 0.0

                    # Name similarity:
                    if src_key_name == dst_key_name:
                        score += 7.0
                    elif (
                        (
                            dst_key_name != 'id'
                            and src_key_name.endswith(dst_key_name)
                        )
                        or src_key_name.endswith(  # e.g., user.id -> user_id
                            f'{dst_table_name}_{dst_key_name}'
                        )
                        or src_key_name.endswith(  # e.g., user.id -> userid
                            f'{dst_table_name}{dst_key_name}'
                        )
                        or (
                            dst_table_name.endswith('s')
                            and src_key_name.endswith(  # e.g., users.id -> user_id
                                f'{dst_table_name[:-1]}_{dst_key_name}'
                            )
                        )
                        or (
                            dst_table_name.endswith('s')
                            and src_key_name.endswith(  # e.g., users.id -> userid
                                f'{dst_table_name[:-1]}{dst_key_name}'
                            )
                        )
                    ):
                        score += 4.0
                    elif src_key_name.endswith(dst_table_name):
                        score += 4.0  # e.g., users -> users
                    elif (
                        dst_table_name.endswith('s')  # e.g., users -> user
                        and src_key_name.endswith(dst_table_name[:-1])
                    ):
                        score += 4.0
                    elif (
                        (
                            src_key_name == 'parentid'
                            or src_key_name == 'parent_id'
                        )
                        and src_table_name == dst_table_name
                    ) or (
                        src_table.name == 'posts'
                        and src_key.name == 'AcceptedAnswerId'
                        and dst_table.name == 'posts'
                    ):
                        score += 2.0
                    elif (
                        src_table.name == 'user_friends'
                        and src_key.name == 'friend'
                        and dst_table.name == 'users'
                    ):
                        score += 3.0

                    # For non-exact matching, at least one additional
                    # requirement needs to be met.

                    # Exact data type compatibility:
                    if src_key.stype == Stype.ID:
                        score += 2.0

                    if src_key.dtype == dst_key.dtype:
                        score += 1.0

                    # Cardinality ratio:
                    if (
                        src_table._num_rows is not None
                        and dst_table._num_rows is not None
                        and src_table._num_rows > dst_table._num_rows
                    ):
                        score += 1.0

                    if score < 5.0:
                        continue

                    candidate_dict[(src_table.name, src_key.name)].append(
                        (dst_table.name, score)
                    )

        for (src_table_name, src_key_name), scores in candidate_dict.items():
            scores.sort(key=lambda x: x[-1], reverse=True)

            if len(scores) > 1 and scores[0][1] == scores[1][1]:
                continue  # Cannot uniquely infer link.

            dst_table_name = scores[0][0]
            self.link(src_table_name, src_key_name, dst_table_name)

        self._inferred_edges = tuple(
            sorted(
                set(self.inferred_edges)
                | {
                    (edge.src_table, edge.fkey, edge.dst_table)
                    for edge in self.edges
                    if (edge.src_table, edge.fkey, edge.dst_table)
                    not in declared
                }
            )
        )

        self._warn_unlinkable_tables()
        self._settle_time_columns()

        if verbose:
            self.print_links()

        return self

    @property
    def inferred_edges(self) -> tuple[tuple[str, str, str], ...]:
        r"""The edges :meth:`infer_links` chose, as ``(src, fkey, dst)``.

        Inference reads names and values to guess relationships, so a question
        runs against a partly guessed graph. Recording guessed edges lets a
        caller explain a prediction and lets a cache distinguish guessed
        shapes instead of serving whichever was inferred first.

        Empty before inference runs, and empty when every edge was declared.
        An edge stays marked for as long as it is there: a later inference pass
        that adds nothing new does not un-guess what an earlier one guessed.

        Reported against the edges the graph currently HAS, so an edge that
        inference chose and the caller then removed stops being reported.
        Otherwise a graph would carry the mark of a link it no longer has, and
        fingerprint differently from an identical graph that never had it.
        """
        marked = getattr(self, '_inferred_edges', ())
        if not marked:
            return ()
        present = {
            (edge.src_table, edge.fkey, edge.dst_table) for edge in self.edges
        }
        return tuple(edge for edge in marked if edge in present)

    def _warn_unlinkable_tables(self) -> None:
        r"""Report a key-like column on a table with no primary key.

        Primary-key inference weighs the column name far more heavily than the
        data, so it declines a unique key whose name does not echo its table,
        such as ``sales_orders.order_id`` or ``order_lines.line_id``. Nothing
        else reports this. No other table can link to a table without a primary
        key, so the graph silently loses the edge and predictions worsen.

        Deliberately deferred until link inference has run. Before that a
        genuine key is indistinguishable from a unique foreign key, such as one
        row per order in a returns table, and warning then fires on every
        junction table. A candidate since claimed as a foreign key is
        explained, so only the rest are reported.
        """
        fkeys: set[tuple[str, str]] = {
            (edge.src_table, edge.fkey) for edge in self.edges
        }
        for table in self.tables.values():
            if table.has_primary_key():
                continue
            declined = [
                name
                for name in table._declined_primary_keys
                if table.has_column(name) and (table.name, name) not in fkeys
            ]
            if len(declined) == 0:
                continue
            sampled = table._declined_primary_key_rows
            evidence = (
                'hold a unique value per row'
                if sampled is None
                else f'hold a unique value in each of the {sampled:,} rows sampled '
                f'from it, and may or may not be unique overall'
            )
            warnings.warn(
                f"No primary key was inferred for table '{table.name}', so no "
                f'other table can link to it. Column(s) {declined} {evidence}; '
                f'pass `primary_key=` explicitly to use one of them.',
                stacklevel=2,
            )

    # Metadata ################################################################

    def validate(self) -> Self:
        r"""Validates the graph to ensure that all relevant metadata is
        specified for its tables and edges.

        Concretely, validation ensures that edges properly link foreign keys to
        primary keys between valid tables.
        It additionally ensures that primary and foreign keys between tables
        in an :class:`~sdm_relational_engine.rfm.Edge` have the same data type.

        Raises:
            ValueError: if validation fails.
        """
        if len(self.tables) == 0:
            raise ValueError(
                'At least one table needs to be added to the graph'
            )

        backends = {table.backend for table in self._tables.values()}
        if len(backends) != 1:
            raise ValueError('Found multiple table backends in the graph')

        for edge in self.edges:
            if (reason := self._edge_error(edge)) is not None:
                raise ValueError(reason)

        return self

    def _edge_error(self, edge: Edge) -> str | None:
        r"""Return why ``edge`` is unusable, or ``None`` if it is valid.

        Split out of :meth:`validate` so that a caller converting a warehouse
        view can drop an offending relationship and report it through
        :attr:`conversion_messages` rather than lose the whole graph to one bad
        edge. View conversion is best-effort and partial by contract.
        :meth:`validate` still raises on the first failure.
        """
        src_table, fkey, dst_table = edge

        # `Table.remove_column` knows nothing about the graph's edges, so
        # dropping a linked foreign key leaves the edge behind. Caught
        # here rather than as a `KeyError` out of the lookup below, which
        # names neither the edge nor the fix.
        if not self[src_table].has_column(fkey):
            return (
                f"Edge {edge} is invalid since table '{src_table}' no "
                f"longer has a column '{fkey}'. Remove the link with "
                f'`unlink()` before removing the column.'
            )

        src_key = self[src_table][fkey]
        dst_key = self[dst_table].primary_key

        # Check that the destination table defines a primary key:
        if dst_key is None:
            return (
                f"Edge {edge} is invalid since table '{dst_table}' does "
                f'not have a primary key. Add either a primary key or '
                f'remove the link before proceeding.'
            )

        # Ensure that foreign key is not a primary key:
        src_pkey = self[src_table].primary_key
        if src_pkey is not None and src_pkey.name == fkey:
            return (
                f"Cannot treat the primary key of table '{src_table}' as "
                f'a foreign key. Remove either the primary key or the '
                f'link before proceeding.'
            )

        # Check that fkey/pkey have valid and consistent data types. Every
        # backend populates data types from its own catalog, so this check
        # applies to remote tables just as much as to local ones:
        assert src_key.dtype is not None
        src_number = src_key.dtype.is_int() or src_key.dtype.is_float()
        src_string = src_key.dtype.is_string()
        assert dst_key.dtype is not None
        dst_number = dst_key.dtype.is_int() or dst_key.dtype.is_float()
        dst_string = dst_key.dtype.is_string()

        if not src_number and not src_string:
            return (
                f'{edge} is invalid as foreign key must be a number or '
                f"string (got '{src_key.dtype}')"
            )

        if src_number != dst_number or src_string != dst_string:
            return (
                f"{edge} is invalid as foreign key '{fkey}' and primary "
                f"key '{dst_key.name}' have incompatible data types (got "
                f"foreign key data type '{src_key.dtype}' and primary key "
                f"data type '{dst_key.dtype}')"
            )

        return None

    def _drop_invalid_edges(self) -> list[str]:
        r"""Remove and describe every edge :meth:`_edge_error` rejects.

        The view constructors convert partially by contract: an element with no
        graph equivalent is dropped and reported, not raised. A relationship
        with mismatched keys belongs in that channel. Most often, the foreign
        and primary keys have incompatible data types that the caller cannot
        fix in a view they do not own. One bad relationship then costs one edge
        instead of every table declared by the view.
        """
        msgs = []
        for edge in list(self.edges):
            if (reason := self._edge_error(edge)) is not None:
                self.unlink(*edge)
                msgs.append(
                    f'Failed to add the relationship between '
                    f"'{edge.src_table}' and '{edge.dst_table}' "
                    f'since {reason}'
                )
        return msgs

    # Visualization ###########################################################

    def _to_mermaid(self, show_columns: bool = True) -> str:
        r"""Returns a Mermaid ER diagram string representation of the graph.

        Args:
            show_columns: Whether to show all columns of every table in the
                graph. If ``False``, will only show the primary key, foreign
                key(s), and time column of each table.

        Returns:
            A string containing the Mermaid ER diagram.
        """
        fkeys_dict: dict[str, list[str]] = defaultdict(list)
        for src_table_name, fkey_name, _ in self.edges:
            fkeys_dict[src_table_name].append(fkey_name)

        lines = ['erDiagram']

        for table_name, table in self.tables.items():
            lines.append(f'{" " * 4}{table_name} {{')
            if pkey := table.primary_key:
                lines.append(f'{" " * 8}{pkey.stype} {pkey.name} PK')
            for fkey_name in fkeys_dict[table_name]:
                fkey = table[fkey_name]
                lines.append(f'{" " * 8}{fkey.stype} {fkey.name} FK')
            if time_col := table.time_column:
                lines.append(f'{" " * 8}{time_col.stype} {time_col.name}')
            if time_col := table.end_time_column:
                lines.append(f'{" " * 8}{time_col.dtype} {time_col.name}')

            if show_columns:
                for column in table.columns:
                    if column.name in fkeys_dict[table_name]:
                        continue
                    if column.name == table._primary_key:
                        continue
                    if column.name == table._time_column:
                        continue
                    if column.name == table._end_time_column:
                        continue
                    lines.append(f'{" " * 8}{column.stype} {column.name}')

            lines.append(f'{" " * 4}}}')

        if len(self.edges) > 0:
            lines.append('')

        for src_table, fkey, dst_table in self.edges:
            lines.append(f'{" " * 4}{dst_table} o|--o{{ {src_table} : {fkey}')

        return '\n'.join(lines)

    def visualize(
        self,
        path: Path | str | io.BytesIO | None = None,
        show_columns: bool = True,
        height: int = 540,
    ) -> None:
        r"""Visualizes the tables and edges in this graph as a Mermaid
        entity-relationship diagram.

        Display is fully self-contained: the bundled mermaid.js is inlined
        into the produced HTML, so no system executables (graphviz), CDN
        scripts, or web services are needed inside notebook environments.
        Only PNG/SVG export renders via the mermaid.ink web service and
        therefore requires network access.

        Args:
            path: Where to write the visualization. ``None`` displays the
                graph inline (Jupyter, Databricks, Colab, and Snowflake
                notebooks) or prints the Mermaid source in terminals.
                A path ending in ``.html`` writes a standalone offline HTML
                file, ``.mmd`` writes the Mermaid source, and ``.png`` or
                ``.svg`` writes an image (network required). A
                :class:`io.BytesIO` receives PNG bytes (network required).
            show_columns: Whether to show all columns of every table in the
                graph. If ``False``, will only show the primary key, foreign
                key(s), and time column of each table.
            height: The pixel height of the inline notebook rendering.
        """
        from sdm_relational_engine.rfm import graph_visualization

        source = self._to_mermaid(show_columns)

        path = Path(path) if isinstance(path, str) else path

        if isinstance(path, Path):
            suffix = path.suffix.removeprefix('.').lower()
            if suffix == '':
                raise ValueError(f"Missing file extension in path '{path}'")
            if suffix == 'html':
                path.write_text(
                    graph_visualization.to_html(source), encoding='utf-8'
                )
            elif suffix == 'mmd':
                path.write_text(source + '\n', encoding='utf-8')
            elif suffix in ('png', 'svg'):
                path.write_bytes(
                    graph_visualization.render_image(source, suffix)
                )
            else:
                raise ValueError(
                    f"File extension '{suffix}' not supported "
                    f'for visualization. Expected one of '
                    f"'html', 'mmd', 'png' or 'svg'."
                )

        elif isinstance(path, io.BytesIO):
            path.write(graph_visualization.render_image(source, 'png'))

        elif in_streamlit_notebook():
            import streamlit as st

            try:
                st.components.v1.html(
                    graph_visualization.to_html(source),
                    height=height,
                    scrolling=True,
                )
            except Exception:  # Custom components are unavailable.
                st.code(source)

        elif in_jupyter_notebook():
            from IPython.display import HTML
            from IPython.display import display as ipython_display

            ipython_display(
                HTML(graph_visualization.to_iframe(source, height=height))
            )

        else:
            warnings.warn(
                'Cannot display the rendered graph outside of a '
                'notebook environment - printing the Mermaid '
                "source instead. Use `visualize(path='graph.html')`"
                ' to write a standalone offline rendering.'
            )
            print(source)

    # Helpers #################################################################

    def to_mermaid(self, show_columns: bool = True) -> str:
        r"""This graph as a Mermaid entity-relationship diagram.

        Any Mermaid renderer can draw the source, including a notebook, a
        documentation site, or mermaid.live. Use :meth:`visualize` to draw it
        here instead. That path renders offline from the bundled Mermaid, so it
        works where a CDN fetch would not.

        Args:
            show_columns: Whether to show every column, or only the primary
                key, foreign keys and time column of each table.
        """
        return self._to_mermaid(show_columns)

    def _materialization_signature(self) -> tuple[Any, ...]:
        r"""What a materialization of this graph would be built from.

        Materializing is expensive and depends on the graph rather than the
        query, so a caller may reuse one across predictions. This is what tells
        it the graph has moved on: the schema, plus each table's row count
        where a backend can give one for nothing.

        Row counts are best effort. They catch rows appended or dropped, and
        they do not catch a value edited in place, so this narrows the window
        in which a reused materialization is stale rather than closing it.
        Backends that would have to run a query contribute nothing here, since
        this is evaluated on every prediction.
        """
        counts = tuple(
            (name, table._local_row_count())
            for name, table in sorted(self.tables.items())
        )
        return (self._to_api_graph_definition(), counts)

    def _to_api_graph_definition(self) -> SchemaDefinition:
        tables: dict[str, TableDefinition] = {}
        col_groups: list[ColumnKeyGroup] = []
        for table_name, table in self.tables.items():
            tables[table_name] = table._to_api_table_definition()
            if table.primary_key is None:
                continue
            keys = [ColumnKey(table_name, table.primary_key.name)]
            for edge in self.edges:
                if edge.dst_table == table_name:
                    keys.append(ColumnKey(edge.src_table, edge.fkey))
            keys = sorted(
                list(set(keys)),
                key=lambda x: f'{x.table_name}.{x.col_name}',
            )
            if len(keys) > 1:
                col_groups.append(ColumnKeyGroup(tuple(keys)))
        return SchemaDefinition(tables, col_groups)

    def update_connection(
        self,
        connection: (
            AdbcSqliteConnection
            | DuckDBPyConnection
            | SnowflakeConnection
            | DatabricksConnection
            | PostgresConnection
            | SparkConnection
        ),
    ) -> None:
        r"""Updates the connection to a database."""
        if self._connection is not None:
            self._connection.close()

        self._connection = None

        for table in self.tables.values():
            if table.backend == DataBackend.SQLITE:
                from adbc_driver_sqlite.dbapi import AdbcSqliteConnection

                from sdm_relational_engine.rfm.backend.sqlite import (
                    SQLiteTable,
                )

                assert isinstance(table, SQLiteTable)
                assert isinstance(connection, AdbcSqliteConnection)
                table._connection = connection
            if table.backend == DataBackend.DUCKDB:
                from sdm_relational_engine.rfm.backend.duckdb import (
                    Connection as DuckDBConnection,
                )
                from sdm_relational_engine.rfm.backend.duckdb import (
                    DuckDBTable,
                )

                assert isinstance(table, DuckDBTable)
                assert isinstance(connection, DuckDBConnection)
                table._connection = connection
            if table.backend == DataBackend.SNOWFLAKE:
                from snowflake.connector import SnowflakeConnection

                from sdm_relational_engine.rfm.backend.snow import SnowTable

                assert isinstance(table, SnowTable)
                assert isinstance(connection, SnowflakeConnection)
                table._connection = connection

            if table.backend == DataBackend.DATABRICKS:
                from sdm_relational_engine.rfm.backend.databricks import (
                    CONNECTION_TYPES,
                    DatabricksTable,
                )

                assert isinstance(table, DatabricksTable)
                assert isinstance(connection, CONNECTION_TYPES)
                table._connection = connection

            if table.backend == DataBackend.POSTGRES:
                from sdm_relational_engine.rfm.backend.postgres import (
                    Connection as PgConnection,
                )
                from sdm_relational_engine.rfm.backend.postgres import (
                    PostgresTable,
                )

                assert isinstance(table, PostgresTable)
                assert isinstance(connection, PgConnection)
                table._connection = connection

    # Class properties ########################################################

    def __hash__(self) -> int:
        return hash((tuple(self.edges), tuple(sorted(self.tables.keys()))))

    def __contains__(self, name: str) -> bool:
        return self.has_table(name)

    def __getitem__(self, name: str) -> Table:
        return self.table(name)

    def __delitem__(self, name: str) -> None:
        self.remove_table(name)

    def __repr__(self) -> str:
        tables = '\n'.join(f'    {table},' for table in self.tables)
        tables = f'[\n{tables}\n  ]' if len(tables) > 0 else '[]'
        edges = '\n'.join(
            f'    {edge.src_table}.{edge.fkey}'
            f' ⇔ {edge.dst_table}.{self[edge.dst_table]._primary_key},'
            for edge in self.edges
        )
        edges = f'[\n{edges}\n  ]' if len(edges) > 0 else '[]'
        return (
            f'{self.__class__.__name__}(\n'
            f'  tables={tables},\n'
            f'  edges={edges},\n'
            f')'
        )

    def __del__(self) -> None:
        if self._connection is not None:
            self._connection.close()


def _timedelta_to_pquery(td: pd.Timedelta) -> tuple[int, str]:
    r"""Convert a positive Timedelta to a ``(amount, unit)`` pair for a
    pquery time range.
    """
    total_seconds = int(td.total_seconds())
    if total_seconds <= 0:
        raise ValueError(f'time_delta must be positive, got {td}')
    if total_seconds % (24 * 3600) == 0:
        return total_seconds // (24 * 3600), 'days'
    if total_seconds % 3600 == 0:
        return total_seconds // 3600, 'hours'
    if total_seconds % 60 == 0:
        return total_seconds // 60, 'minutes'
    raise ValueError(
        f'time_delta {td} must be a whole number of minutes, hours, or days'
    )


def _infer_timedelta_from_timestamps(
    timestamps_series: pd.Series,
) -> pd.Timedelta:
    r"""Infer the median step size from a Series of per-entity timestamp lists.

    Emits a warning when the observed step sizes are not all equal.
    """
    deltas: list[pd.Timedelta] = []
    for ts_list in timestamps_series:
        ts = pd.to_datetime(list(ts_list)).sort_values()
        if len(ts) >= 2:
            deltas.extend(ts[1:] - ts[:-1])
    if not deltas:
        raise ValueError(
            "Cannot infer 'time_delta' from 'timestamps_col': all series "
            "have fewer than 2 observations. Please provide 'time_delta' "
            'explicitly.'
        )
    delta_series = pd.Series(deltas)
    median_td = Timedelta(delta_series.median())
    if delta_series.nunique() > 1:
        warnings.warn(
            f"Observed step sizes in 'timestamps_col' are not all equal "
            f'(found {delta_series.nunique()} distinct intervals). '
            f"Using the median ({median_td}) as 'time_delta' for the "
            f"predictive query. Pass 'time_delta' explicitly to override.",
            stacklevel=3,
        )
    return median_td
