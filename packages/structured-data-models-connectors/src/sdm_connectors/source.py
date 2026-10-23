# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Where a schema's tables are, and how to reach them.

A source pairs a connection with the namespace its tables live in. It reads
nothing itself: it is what a schema over tables that stay where they are is
built from, so that sampling can run as SQL against the tables in place.

Everything that differs between warehouses lives here, on a class of its own
per backend: how a table is qualified, which sessions stand in for a
connection, and whether the warehouse keeps semantic models. A schema built
from any of them is then described the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard, overload

from sdm_connectors.sql import ConnectorError

if TYPE_CHECKING:
    from sdm_connectors._types import (
        DatabricksLike,
        DuckDBLike,
        PostgresLike,
        SnowflakeLike,
        SnowparkLike,
        SnowparkSession,
        SparkLike,
        SparkSession,
        SQLiteLike,
    )
    from sdm_connectors.backends.databricks import (
        Connection as DatabricksConnection,
    )
    from sdm_connectors.backends.duckdb import Connection as DuckDBConnection
    from sdm_connectors.backends.postgres import (
        Connection as PostgresConnection,
    )
    from sdm_connectors.backends.snowflake import (
        Connection as SnowflakeConnection,
    )
    from sdm_connectors.backends.spark import SparkConnection
    from sdm_connectors.backends.sqlite import Connection as SQLiteConnection

# Recognised by the module that defines the connection's class, or any of its
# bases, so a driver is never imported just to be ruled out.
_DRIVER_MODULES: tuple[tuple[str, str], ...] = (
    ('databricks.sql', 'databricks'),
    ('sdm_connectors.backends.spark', 'databricks'),
    ('snowflake.connector', 'snowflake'),
    ('psycopg', 'postgres'),
    ('duckdb', 'duckdb'),
    ('_duckdb', 'duckdb'),
    ('adbc_driver_sqlite', 'sqlite'),
)

_ACCEPTED = (
    'a Databricks SQL connection or a Spark session, a Snowflake connection '
    'or Snowpark session, a psycopg connection, a DuckDB connection, or an '
    "ADBC SQLite connection, each opened with its driver's own connect"
)


def _modules(value: Any) -> list[str]:
    return [cls.__module__ or '' for cls in type(value).__mro__]


def _is_spark_session(value: object) -> TypeGuard[SparkSession]:
    return any(
        cls.__name__ == 'SparkSession'
        and (cls.__module__ or '').startswith('pyspark.')
        for cls in type(value).__mro__
    )


def _is_snowpark_session(value: object) -> TypeGuard[SnowparkSession]:
    return any(
        cls.__name__ == 'Session'
        and (cls.__module__ or '').startswith('snowflake.snowpark')
        for cls in type(value).__mro__
    )


def _is_async_psycopg(value: object) -> bool:
    # By name, like the sessions, so psycopg is never imported to rule it out.
    return any(
        cls.__name__ == 'AsyncConnection'
        and (cls.__module__ or '').startswith('psycopg')
        for cls in type(value).__mro__
    )


def _backend_of(connection: Any) -> str | None:
    if _is_spark_session(connection):
        return 'databricks'
    if _is_snowpark_session(connection):
        return 'snowflake'
    for module in _modules(connection):
        for prefix, backend in _DRIVER_MODULES:
            if module == prefix or module.startswith(prefix + '.'):
                return backend
    return None


class Source:
    r"""A connection, and the namespace its tables live in.

    Build one with the class for its warehouse, which names that warehouse's
    namespace and offers what only it has:

    >>> source = DatabricksSource(spark, catalog='main', schema='sales')
    >>> source = SnowflakeSource(session, database='SALES', schema='PUBLIC')
    >>> source = PostgresSource(psycopg.connect(...), schema='public')
    >>> source = DuckDBSource(duckdb.connect('sales.duckdb', read_only=True))

    ``Source(connection, ...)`` works too, and returns the class that fits the
    connection, which is also the type a type checker gives it.

    The source never closes the connection. It belongs to whoever opened it.

    Raises:
        ConnectorError: ``UNKNOWN_CONNECTOR`` for a connection this package
            cannot sample through, ``INVALID_CONNECTOR_ARGS`` for a namespace
            the warehouse does not have.
    """

    #: Which engine the tables are sampled with.
    _BACKEND: ClassVar[str] = ''
    #: The names this warehouse qualifies a table with, outermost first.
    _NAMESPACE: ClassVar[tuple[str, ...]] = ()
    #: What the class accepts, for the message when it is given anything else.
    _TAKES: ClassVar[str] = _ACCEPTED

    _backend: str
    _connection: Any
    _namespace: dict[str, str]

    # So that `Source(connection)` is typed as the class it returns. Each takes
    # a shape from _types rather than the driver's class, which may be Any.
    @overload
    def __new__(
        cls,
        connection: DatabricksLike,
        *,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> DatabricksSource: ...
    @overload
    def __new__(
        cls,
        connection: SnowflakeLike,
        *,
        database: str | None = None,
        schema: str | None = None,
    ) -> SnowflakeSource: ...
    @overload
    def __new__(
        cls, connection: PostgresLike, *, schema: str | None = None
    ) -> PostgresSource: ...
    @overload
    def __new__(cls, connection: DuckDBLike) -> DuckDBSource: ...
    @overload
    def __new__(cls, connection: SQLiteLike) -> SQLiteSource: ...
    @overload
    def __new__(
        cls,
        connection: SparkLike,
        *,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> DatabricksSource: ...
    @overload
    def __new__(
        cls,
        connection: SnowparkLike,
        *,
        database: str | None = None,
        schema: str | None = None,
    ) -> SnowflakeSource: ...
    def __new__(
        cls, connection: Any = None, *args: Any, **namespace: Any
    ) -> Source:
        if cls is not Source:
            return super().__new__(cls)
        backend = _backend_of(connection)
        chosen = _BY_BACKEND.get(backend or '')
        if chosen is None:
            raise ConnectorError(
                f'a Source needs {_ACCEPTED}; got {type(connection).__name__}',
                code='UNKNOWN_CONNECTOR',
            )
        chosen._check_namespace_names(namespace)
        return super().__new__(chosen)

    @classmethod
    def _check_namespace_names(cls, namespace: Mapping[str, Any]) -> None:
        given = [name for name, value in namespace.items() if value is not None]
        foreign = sorted(set(given) - set(cls._NAMESPACE))
        if foreign:
            has = (
                f'is qualified by {" and ".join(cls._NAMESPACE)}'
                if cls._NAMESPACE
                else 'has no namespace to name'
            )
            raise ConnectorError(
                f'a {cls._BACKEND} source {has}, so {", ".join(foreign)} '
                f'cannot be given',
                code='INVALID_CONNECTOR_ARGS',
            )

    def _bind(self, connection: Any, namespace: Mapping[str, Any]) -> None:
        backend = _backend_of(connection)
        if backend != self._BACKEND:
            raise ConnectorError(
                f'{type(self).__name__} needs {self._TAKES}; got '
                f'{type(connection).__name__}',
                code='UNKNOWN_CONNECTOR',
            )
        self._check_namespace_names(namespace)
        given = {name: v for name, v in namespace.items() if v is not None}
        for name, value in given.items():
            if not isinstance(value, str) or not value:
                raise ConnectorError(
                    f'{name} must be a non-empty string, got {value!r}',
                    code='INVALID_CONNECTOR_ARGS',
                )
        self._backend = backend
        self._connection = connection
        self._namespace = given

    @property
    def backend(self) -> str:
        r"""Which engine the tables are sampled with: ``'databricks'``,
        ``'snowflake'``, ``'postgres'``, ``'duckdb'`` or ``'sqlite'``.
        """
        return self._backend

    @property
    def connection(self) -> Any:
        r"""The connection sampling runs through."""
        return self._connection

    @property
    def kind(self) -> str:
        r"""Which kind of source this is, finer than its backend: what a
        call over it is routed by, and which provider reads it.

        Both Databricks connections have the backend ``'databricks'``; a SQL
        warehouse connection is ``'databricks-sql'`` and a notebook's Spark
        session ``'databricks-spark'``. A semantic model is its own kind,
        ``'databricks-metric-view'`` or ``'snowflake-semantic-view'``. Any
        other source is its backend.
        """
        if self._backend == 'databricks':
            from sdm_connectors.backends.spark import SparkConnection

            if isinstance(self._connection, SparkConnection):
                return 'databricks-spark'
            return 'databricks-sql'
        return self._backend

    def namespace_parts(self) -> list[str | None]:
        r"""The namespace as the name parts before a table's own, outermost
        first, in the class's order (``catalog`` then ``schema``,
        ``database`` then ``schema``, ``schema``), each ``None`` where the
        connection's current one is meant; empty for a source with no
        namespace (DuckDB, SQLite).
        """
        return [self._namespace.get(name) for name in self._NAMESPACE]

    @property
    def namespace(self) -> dict[str, str]:
        r"""The namespace given, by name: ``catalog``, ``database``,
        ``schema``. Anything left out is the connection's current one.
        """
        return dict(self._namespace)

    def __reduce__(self) -> Any:
        raise TypeError(
            'a Source holds an open connection, which cannot be pickled'
        )

    def __repr__(self) -> str:
        where = ', '.join(f'{k}={v!r}' for k, v in self._namespace.items())
        return f'{type(self).__name__}({where})'


class DatabricksSource(Source):
    r"""Tables in a Databricks catalog and schema.

    >>> source = DatabricksSource(spark, catalog='main', schema='sales')
    >>> source = DatabricksSource(databricks.sql.connect(...), catalog='main')

    A notebook's ``spark`` is sampled through ``spark.sql``, so no SQL warehouse
    is needed; that takes ``DataFrame.toArrow()``, which Databricks Runtime
    17.3 LTS and serverless environment 4 have. On an older runtime, pass a SQL
    warehouse connection instead.

    A connection or session passed here is the caller's and is used as it is,
    so it carries no ``structured-data-models-client`` partner identifier or
    query tag. One opened with :func:`sdm_connectors.databricks.warehouse`
    carries both.

    Args:
        connection: A Databricks SQL connection, or a notebook's Spark session.
        catalog: The catalog. Defaults to the connection's current one.
        schema: The schema. Defaults to the connection's current one.
    """

    _BACKEND = 'databricks'
    _NAMESPACE = ('catalog', 'schema')
    _TAKES = (
        'a Databricks SQL connection from databricks.sql.connect(), or a '
        "notebook's Spark session"
    )

    def __init__(
        self,
        connection: DatabricksConnection | SparkSession,
        *,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> None:
        bound: object = connection
        if _is_spark_session(connection):
            from sdm_connectors.backends.spark import SparkConnection

            bound = SparkConnection(connection)
        self._bind(bound, {'catalog': catalog, 'schema': schema})

    @property
    def connection(self) -> DatabricksConnection | SparkConnection:
        r"""The connection sampling runs through: the SQL connection, or the
        Spark session as a connection.
        """
        return self._connection

    def semantic_model(
        self, name: str, *, trust_expressions: bool = False
    ) -> SemanticModel:
        r"""A metric view in this catalog and schema, as a source of its own.

        >>> model = source.semantic_model('sales_metrics')

        A metric view already names the tables a question is about and how
        they join. Its source table becomes a table named as it is in the
        catalog, each join a table named after the join, and each dimension a
        column of the table it reads. Measures are aggregations rather than
        row values, so they are left out. The view's ``filter`` limits the
        rows of its source table; a filter on a joined table is refused.

        Args:
            name: The metric view, qualified or resolved against this source's
                catalog and schema.
            trust_expressions: Whether to keep dimensions that call functions
                other than pure built-ins. They run as SQL under your
                credentials each time a request is gathered, so by default
                such a dimension is left out with a message.
        """
        return SemanticModel(self, name, trust_expressions=trust_expressions)


class SnowflakeSource(Source):
    r"""Tables in a Snowflake database and schema.

    >>> source = SnowflakeSource(session, database='SALES', schema='PUBLIC')

    A Snowpark ``session`` is reached through its own connector connection, so
    a Snowflake notebook passes it as it is.

    Args:
        connection: A Snowflake connection, or a Snowpark session.
        database: The database. Defaults to the connection's current one.
        schema: The schema. Defaults to the connection's current one.
    """

    _BACKEND = 'snowflake'
    _NAMESPACE = ('database', 'schema')
    _TAKES = (
        'a Snowflake connection from snowflake.connector.connect(), or a '
        'Snowpark session'
    )

    def __init__(
        self,
        connection: SnowflakeConnection | SnowparkSession,
        *,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        if _is_snowpark_session(connection):
            connection = connection.connection
        self._bind(connection, {'database': database, 'schema': schema})

    @property
    def connection(self) -> SnowflakeConnection:
        r"""The connection sampling runs through, a Snowpark session's own."""
        return self._connection

    def semantic_model(
        self, name: str, *, trust_expressions: bool = False
    ) -> SemanticModel:
        r"""A semantic view in this database and schema, as a source of its
        own.

        >>> model = source.semantic_model('SALES_SEMANTICS')

        Its logical tables become tables, its relationships the links between
        them, and its dimensions and facts their columns. Metrics are
        aggregations rather than row values, so they are left out.

        Args:
            name: The semantic view, qualified or resolved against this
                source's database and schema.
            trust_expressions: Whether to keep columns that call functions
                other than pure built-ins. They run as SQL under your
                credentials each time a request is gathered, so by default
                such a column is left out with a message.
        """
        return SemanticModel(self, name, trust_expressions=trust_expressions)


class PostgresSource(Source):
    r"""Tables in a PostgreSQL schema, Lakebase included.

    >>> source = PostgresSource(psycopg.connect(...), schema='public')

    Args:
        connection: A psycopg connection.
        schema: The schema. Defaults to the connection's current one.
    """

    _BACKEND = 'postgres'
    _NAMESPACE = ('schema',)
    _TAKES = 'a psycopg connection from psycopg.connect()'

    def __init__(
        self, connection: PostgresConnection, *, schema: str | None = None
    ) -> None:
        if _is_async_psycopg(connection):
            raise ConnectorError(
                f'PostgresSource needs a synchronous psycopg connection, '
                f'since the engine reads synchronously; got '
                f'{type(connection).__name__}',
                code='UNKNOWN_CONNECTOR',
            )
        self._bind(connection, {'schema': schema})

    @property
    def connection(self) -> PostgresConnection:
        r"""The connection sampling runs through."""
        return self._connection


class DuckDBSource(Source):
    r"""Tables in a DuckDB database.

    >>> source = DuckDBSource(duckdb.connect('sales.duckdb', read_only=True))
    """

    _BACKEND = 'duckdb'
    _TAKES = 'a DuckDB connection from duckdb.connect()'

    def __init__(self, connection: DuckDBConnection) -> None:
        self._bind(connection, {})

    @property
    def connection(self) -> DuckDBConnection:
        r"""The connection sampling runs through."""
        return self._connection


class SQLiteSource(Source):
    r"""Tables in a SQLite database.

    >>> source = SQLiteSource(adbc_driver_sqlite.dbapi.connect('sales.sqlite'))
    """

    _BACKEND = 'sqlite'
    _TAKES = 'an ADBC SQLite connection from adbc_driver_sqlite.dbapi.connect()'

    def __init__(self, connection: SQLiteConnection) -> None:
        self._bind(connection, {})

    @property
    def connection(self) -> SQLiteConnection:
        r"""The connection sampling runs through."""
        return self._connection


class SemanticModel(Source):
    r"""A warehouse's semantic model: tables, and how they join, it describes.

    Got from :meth:`DatabricksSource.semantic_model` (a metric view) or
    :meth:`SnowflakeSource.semantic_model` (a semantic view). Both are
    what the Open Semantic Interchange and dbt call a semantic model, and a
    schema reads the same thing from each: tables, the links between them, and
    row-level columns. Their aggregations, the measures and metrics, are not
    rows, so a predictive query expresses them instead.

    Args:
        source: The source the model is kept in.
        name: The model's name, qualified or resolved against ``source``'s
            namespace.
        trust_expressions: Whether to keep columns whose expressions call
            functions other than pure built-ins.
    """

    def __init__(
        self,
        source: DatabricksSource | SnowflakeSource,
        name: str,
        *,
        trust_expressions: bool = False,
    ) -> None:
        if not isinstance(source, DatabricksSource | SnowflakeSource):
            raise ConnectorError(
                f'a semantic model is kept in Databricks or Snowflake, not '
                f'{type(source).__name__}',
                code='INVALID_CONNECTOR_ARGS',
            )
        if not isinstance(name, str) or not name:
            raise ConnectorError(
                f'a semantic model needs its name, got {name!r}',
                code='INVALID_CONNECTOR_ARGS',
            )
        if not isinstance(trust_expressions, bool):
            raise ConnectorError(
                f'trust_expressions is True or False, got '
                f'{trust_expressions!r}',
                code='INVALID_CONNECTOR_ARGS',
            )
        self._source = source
        self._name = name
        self._trust_expressions = trust_expressions
        self._backend = source.backend
        self._connection = source.connection
        self._namespace = source.namespace

    @property
    def source(self) -> DatabricksSource | SnowflakeSource:
        r"""The source the model is kept in."""
        return self._source

    @property
    def name(self) -> str:
        r"""The model's name, as given."""
        return self._name

    @property
    def trust_expressions(self) -> bool:
        r"""Whether columns calling functions other than pure built-ins are
        kept.
        """
        return self._trust_expressions

    @property
    def kind(self) -> str:
        r"""``'databricks-metric-view'`` or ``'snowflake-semantic-view'``,
        whichever connection keeps it."""
        return f'{self._backend}-{_SEMANTIC_KIND[self._backend]}'

    def namespace_parts(self) -> list[str | None]:
        r"""Its source's namespace, which its name resolves against."""
        return self._source.namespace_parts()

    def __repr__(self) -> str:
        trusted = ', trust_expressions=True' if self._trust_expressions else ''
        return f'SemanticModel({self._name!r} in {self._source!r}{trusted})'


#: What a semantic model is called in each warehouse that keeps one.
_SEMANTIC_KIND: Mapping[str, str] = {
    'databricks': 'metric-view',
    'snowflake': 'semantic-view',
}


_BY_BACKEND: Mapping[str, type[Source]] = {
    'databricks': DatabricksSource,
    'snowflake': SnowflakeSource,
    'postgres': PostgresSource,
    'duckdb': DuckDBSource,
    'sqlite': SQLiteSource,
}
