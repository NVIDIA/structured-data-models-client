# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The built-in sources: tables in a database the library reads with its
own SQL, through a connection the caller opened.

Each is a :class:`~sdm_client.Source` that pairs a connection with the
namespace its tables live in, and answers the typed requests the core
sends every source through the same :meth:`~sdm_client.Source.answer`:
each with the SQL the library writes for that database, run on the
caller's connection, in the contract's shape. Asked directly, a built-in
source answers exactly what an operation over it receives.

>>> source = DatabricksSource(spark, catalog='main', schema='sales')
>>> source = SnowflakeSource(session, database='SALES', schema='PUBLIC')
>>> source = PostgresSource(psycopg.connect(...), schema='public')
>>> source = DuckDBSource(duckdb.connect('sales.duckdb', read_only=True))

A source never closes its connection. It belongs to whoever opened it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, TypeGuard

from sdm_client.errors import MissingExtraError
from sdm_client.provider import Source

if TYPE_CHECKING:
    from typing import TypeAlias

    import pyarrow as pa
    import pyspark.sql
    import pyspark.sql.connect.session
    import snowflake.snowpark
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

    from sdm_client._rust.sql_kit import KitRun
    from sdm_client.provider import (
        ArrowStream,
        Capabilities,
        DataRequest,
        ProviderFailure,
    )

    #: A notebook's ``spark``: classic, or Spark Connect as serverless runs it.
    SparkSession: TypeAlias = (
        pyspark.sql.SparkSession | pyspark.sql.connect.session.SparkSession
    )
    SnowparkSession: TypeAlias = snowflake.snowpark.Session

__all__ = [
    'DatabricksSource',
    'DuckDBSource',
    'PostgresSource',
    'SQLiteSource',
    'SnowflakeSource',
]

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
    r"""Which database ``connection`` is a driver's connection to."""
    if _is_spark_session(connection):
        return 'databricks'
    if _is_snowpark_session(connection):
        return 'snowflake'
    for module in _modules(connection):
        for prefix, backend in _DRIVER_MODULES:
            if module == prefix or module.startswith(prefix + '.'):
                return backend
    return None


class _BuiltInSource(Source):
    r"""What every built-in source shares: a connection, the namespace its
    tables live in, the semantic model it reads, and the driver that runs
    the SQL the library writes for its database. Not part of the API.

    Each operation's answers are its own: the core reaches a built-in
    source through :meth:`_operation`, a fresh SQL kit for that operation,
    which settles a namespace left to the connection once and reads an
    identical catalog read once, within that operation only. Asked
    directly, :meth:`capabilities`, :meth:`answer` and :meth:`failure` each
    answer through a kit of their own.
    """

    #: The database recognised by the connection (``_backend_of``).
    _BACKEND: ClassVar[str] = ''
    #: The extra that installs the database's driver.
    _EXTRA: ClassVar[str] = ''
    #: The core's SQL dialect for the database.
    _DIALECT: ClassVar[str] = ''
    #: The names this database qualifies a table with, outermost first.
    _NAMESPACE: ClassVar[tuple[str, ...]] = ()
    #: What the class accepts, for the message when it is given anything else.
    _TAKES: ClassVar[str] = ''

    _connection: Any
    _namespace: dict[str, str]
    #: The provider class (``sdm_client._rust.providers``) running the
    #: statements on the connection.
    _driver: str
    #: The semantic model read, by name, and whether its expressions are
    #: trusted; ``None`` for the tables themselves.
    _view: tuple[str, bool] | None = None

    def _bind(self, connection: Any, namespace: Mapping[str, Any]) -> None:
        try:
            import sdm_connectors  # noqa: F401
        except ImportError:
            raise MissingExtraError(
                self._EXTRA, 'structured-data-models-connectors'
            ) from None
        if _backend_of(connection) != self._BACKEND:
            _refuse(
                f'{type(self).__name__} needs {self._TAKES}; got '
                f'{type(connection).__name__}',
                'UNKNOWN_CONNECTOR',
            )
        given = {name: v for name, v in namespace.items() if v is not None}
        for name, value in given.items():
            if not isinstance(value, str) or not value:
                _refuse(
                    f'{name} must be a non-empty string, got {value!r}',
                    'INVALID_CONNECTOR_ARGS',
                )
        self._connection = connection
        self._namespace = given

    @property
    def connection(self) -> Any:
        r"""The connection the tables are read through."""
        return self._connection

    @property
    def namespace(self) -> dict[str, str]:
        r"""The namespace given, by name: ``catalog``, ``database``,
        ``schema``. Anything left out is the connection's current one.
        """
        return dict(self._namespace)

    def capabilities(self) -> Capabilities:
        r"""What this source answers, as the library's SQL for its database
        declares it."""
        return self._operation().capabilities()

    def answer(
        self, request: DataRequest, expected: pa.Schema | None
    ) -> ArrowStream:
        r"""Answers ``request`` with the library's SQL for this database, run
        on the connection, in the contract's shape (``expected``, where the
        request fixes one).

        Raises:
            StructuredDataError: A request this database's SQL cannot
                answer, refused by name.
            Exception: The driver's failure, as it raised it.
        """
        return self._operation().answer(request, expected)

    def failure(self, error: Exception) -> ProviderFailure:
        r"""What the driver's failure ``error`` tells the core: its SQLSTATE
        and vendor code, and what it means on this database."""
        return self._operation().failure(error)

    def _operation(self) -> KitRun:
        r"""One operation's answers: a fresh SQL kit over the connection,
        the source the core asks for that operation."""
        from sdm_client._rust import providers
        from sdm_client._rust.sql_kit import KitRun

        driver = getattr(providers, self._driver)
        return KitRun(self._sql_choice(), driver(self._connection))

    def _sql_choice(self) -> dict[str, Any]:
        r"""The contract's SQL choice of this source: its dialect, the
        namespace given (each part ``None`` where the connection's current
        one is meant; left out where none is given), and the semantic model
        read, where one is."""
        choice: dict[str, Any] = {'provider': 'sql', 'dialect': self._DIALECT}
        parts = [self._namespace.get(name) for name in self._NAMESPACE]
        if any(part is not None for part in parts):
            choice['namespace'] = parts
        if self._view is not None:
            name, trusted = self._view
            choice['view'] = {'name': name, 'trust_expressions': trusted}
        return choice

    def _semantic_model(self, name: str, trust_expressions: bool) -> Any:
        r"""This source, reading the semantic model ``name``."""
        if not isinstance(name, str) or not name:
            _refuse(
                f'a semantic model needs its name, got {name!r}',
                'INVALID_CONNECTOR_ARGS',
            )
        if not isinstance(trust_expressions, bool):
            _refuse(
                f'trust_expressions is True or False, got '
                f'{trust_expressions!r}',
                'INVALID_CONNECTOR_ARGS',
            )
        model = object.__new__(type(self))
        model.__dict__.update(self.__dict__)
        model._view = (name, trust_expressions)
        return model

    def __repr__(self) -> str:
        where = ', '.join(f'{k}={v!r}' for k, v in self._namespace.items())
        source = f'{type(self).__name__}({where})'
        if self._view is None:
            return source
        name, trusted = self._view
        trust = ', trust_expressions=True' if trusted else ''
        return f'SemanticModel({name!r} in {source}{trust})'


def _refuse(message: str, code: str) -> NoReturn:
    from sdm_connectors import ConnectorError

    raise ConnectorError(message, code=code)


class DatabricksSource(_BuiltInSource):
    r"""Tables in a Databricks catalog and schema.

    >>> source = DatabricksSource(spark, catalog='main', schema='sales')
    >>> source = DatabricksSource(databricks.sql.connect(...), catalog='main')

    A notebook's ``spark`` is read through ``spark.sql``, so no SQL warehouse
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

    Raises:
        ConnectorError: ``UNKNOWN_CONNECTOR`` for any other connection,
            ``INVALID_CONNECTOR_ARGS`` for a name that is not a non-empty
            string.
    """

    _BACKEND = 'databricks'
    _EXTRA = 'databricks'
    _DIALECT = 'databricks'
    _NAMESPACE = ('catalog', 'schema')
    _TAKES = (
        'a Databricks SQL connection from databricks.sql.connect(), or a '
        "notebook's Spark session"
    )

    def __init__(
        self,
        connection: DatabricksConnection
        | pyspark.sql.SparkSession
        | pyspark.sql.connect.session.SparkSession,
        *,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> None:
        bound: object = connection
        self._driver = 'DatabricksSqlProvider'
        if _is_spark_session(connection):
            from sdm_connectors.backends.spark import SparkConnection

            bound = SparkConnection(connection)
            self._driver = 'DatabricksSparkProvider'
        self._bind(bound, {'catalog': catalog, 'schema': schema})

    @property
    def connection(self) -> DatabricksConnection | SparkConnection:
        r"""The connection the tables are read through: the SQL connection,
        or the Spark session as a connection.
        """
        return self._connection

    def semantic_model(
        self, name: str, *, trust_expressions: bool = False
    ) -> DatabricksSource:
        r"""A metric view in this catalog and schema, read as a source of
        the same class.

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
        model: DatabricksSource = self._semantic_model(name, trust_expressions)
        return model


class SnowflakeSource(_BuiltInSource):
    r"""Tables in a Snowflake database and schema.

    >>> source = SnowflakeSource(session, database='SALES', schema='PUBLIC')

    A Snowpark ``session`` is reached through its own connector connection, so
    a Snowflake notebook passes it as it is.

    Args:
        connection: A Snowflake connection, or a Snowpark session.
        database: The database. Defaults to the connection's current one.
        schema: The schema. Defaults to the connection's current one.

    Raises:
        ConnectorError: ``UNKNOWN_CONNECTOR`` for any other connection,
            ``INVALID_CONNECTOR_ARGS`` for a name that is not a non-empty
            string.
    """

    _BACKEND = 'snowflake'
    _EXTRA = 'snowflake'
    _DIALECT = 'snowflake'
    _NAMESPACE = ('database', 'schema')
    _TAKES = (
        'a Snowflake connection from snowflake.connector.connect(), or a '
        'Snowpark session'
    )
    _driver = 'SnowflakeProvider'

    def __init__(
        self,
        connection: SnowflakeConnection | snowflake.snowpark.Session,
        *,
        database: str | None = None,
        schema: str | None = None,
    ) -> None:
        if _is_snowpark_session(connection):
            connection = connection.connection
        self._bind(connection, {'database': database, 'schema': schema})

    @property
    def connection(self) -> SnowflakeConnection:
        r"""The connection the tables are read through, a Snowpark
        session's own."""
        return self._connection

    def semantic_model(
        self, name: str, *, trust_expressions: bool = False
    ) -> SnowflakeSource:
        r"""A semantic view in this database and schema, read as a source of
        the same class.

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
        model: SnowflakeSource = self._semantic_model(name, trust_expressions)
        return model


class PostgresSource(_BuiltInSource):
    r"""Tables in a PostgreSQL schema, Lakebase included.

    >>> source = PostgresSource(psycopg.connect(...), schema='public')

    Args:
        connection: A synchronous psycopg connection.
        schema: The schema. Defaults to the connection's current one.

    Raises:
        ConnectorError: ``UNKNOWN_CONNECTOR`` for any other connection,
            ``INVALID_CONNECTOR_ARGS`` for a name that is not a non-empty
            string.
    """

    _BACKEND = 'postgres'
    _EXTRA = 'postgres'
    _DIALECT = 'postgres'
    _NAMESPACE = ('schema',)
    _TAKES = 'a psycopg connection from psycopg.connect()'
    _driver = 'PostgresProvider'

    def __init__(
        self, connection: PostgresConnection, *, schema: str | None = None
    ) -> None:
        if _is_async_psycopg(connection):
            _refuse(
                f'PostgresSource needs a synchronous psycopg connection, '
                f'since the engine reads synchronously; got '
                f'{type(connection).__name__}',
                'UNKNOWN_CONNECTOR',
            )
        self._bind(connection, {'schema': schema})

    @property
    def connection(self) -> PostgresConnection:
        r"""The connection the tables are read through."""
        return self._connection


class DuckDBSource(_BuiltInSource):
    r"""Tables in a DuckDB database.

    >>> source = DuckDBSource(duckdb.connect('sales.duckdb', read_only=True))

    Raises:
        ConnectorError: ``UNKNOWN_CONNECTOR`` for any other connection.
    """

    _BACKEND = 'duckdb'
    _EXTRA = 'duckdb'
    _DIALECT = 'duckdb'
    _TAKES = 'a DuckDB connection from duckdb.connect()'
    _driver = 'DuckDBProvider'

    def __init__(self, connection: DuckDBConnection) -> None:
        self._bind(connection, {})

    @property
    def connection(self) -> DuckDBConnection:
        r"""The connection the tables are read through."""
        return self._connection


class SQLiteSource(_BuiltInSource):
    r"""Tables in a SQLite database.

    >>> source = SQLiteSource(adbc_driver_sqlite.dbapi.connect('sales.sqlite'))

    Raises:
        ConnectorError: ``UNKNOWN_CONNECTOR`` for any other connection.
    """

    _BACKEND = 'sqlite'
    _EXTRA = 'sqlite'
    _DIALECT = 'sqlite'
    _TAKES = 'an ADBC SQLite connection from adbc_driver_sqlite.dbapi.connect()'
    _driver = 'SqliteProvider'

    def __init__(self, connection: SQLiteConnection) -> None:
        self._bind(connection, {})

    @property
    def connection(self) -> SQLiteConnection:
        r"""The connection the tables are read through."""
        return self._connection
