# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from numbers import Integral
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import pandas as pd
import pyarrow as pa

from sdm_connectors.backends import connect, owns_connection
from sdm_connectors.sql import (
    ConnectorError,
    driver_guard,
    require_driver,
    resolve_sql,
)

_SQL_SOURCES = ('sqlite', 'duckdb', 'postgres', 'snowflake', 'databricks')
_FILE_SOURCES = ('local', 's3')

_PARQUET_SUFFIXES = ('.parquet', '.pq', '.parq')
_CSV_SUFFIXES = ('.csv', '.txt')
_COMPRESSION_SUFFIXES = ('.gz', '.bz2', '.zip', '.xz', '.zst', '.tar')
_FILE_FORMATS = ('csv', 'parquet')

# The range `Int64` holds. Beyond it pandas keeps an object column, which is
# exact, so there is nothing to rescue and forcing a dtype would only raise.
_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)

# What a plain read gives a string column on this pandas, so narrowing one back
# lands where it would have landed without the null-aware backend.
_PLAIN_STRING_DTYPE = pd.Series(['a']).dtype


_NULLABLE_INTEGER_DTYPES = {
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.uint8(): pd.UInt8Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
}


def read(source: str, **kwargs: Any) -> pd.DataFrame:
    r"""Reads one flat table from any supported source.

    Args:
        source: ``'local'`` (``data=`` a DataFrame/dict, or ``path=`` a CSV or
            Parquet file), ``'s3'`` (``path='s3://...'``, optional
            ``storage_options=``), or a SQL backend -- ``'sqlite'``,
            ``'duckdb'``, ``'postgres'``, ``'snowflake'``, ``'databricks'``.
        **kwargs: For a SQL source, exactly one of ``table=`` / ``query=`` plus
            the connection arguments :func:`connect` takes for that backend
            (``sqlite`` / ``duckdb`` also accept the database path as
            ``database=`` or ``uri=``). Connections opened here are closed
            again before returning.

    Returns:
        The rows as a ``pd.DataFrame``.

    Raises:
        ConnectorError: With a stable ``code`` -- ``UNKNOWN_CONNECTOR``,
            ``INVALID_CONNECTOR_ARGS``, ``CONNECT_FAILED``, ``QUERY_FAILED``,
            ``READ_FAILED`` or ``NOT_FOUND`` -- and the driver exception
            chained as ``__cause__``.
        MissingBackendError: If the source's optional driver is not installed.
    """
    if source == 'local':
        return _read_local(**kwargs)
    if source == 's3':
        return _read_s3(**kwargs)
    if source not in _SQL_SOURCES:
        raise ConnectorError(
            f'Unknown connector {source!r}; supported: '
            f'{sorted((*_FILE_SOURCES, *_SQL_SOURCES))}',
            code='UNKNOWN_CONNECTOR',
        )

    table = kwargs.pop('table', None)
    query = kwargs.pop('query', None)
    if source in ('sqlite', 'duckdb'):
        if 'database' in kwargs and 'uri' in kwargs:
            raise ConnectorError(
                f"{source} connector accepts exactly one of 'database' or "
                f"'uri', not both",
                code='INVALID_CONNECTOR_ARGS',
            )
        uri = kwargs.pop('database', None)
        uri = kwargs.pop('uri', uri)
        connection = connect(source, uri, **kwargs)
    else:
        connection = connect(source, **kwargs)

    try:
        return read_table(connection, table=table, query=query)
    finally:
        if owns_connection(connection):
            with suppress(Exception):
                connection.close()


def _query_handle(connection: Any) -> tuple[Any, bool]:
    r"""Return the ``(handle, owned)`` to run a query on.

    ``DuckDBPyConnection.cursor()`` hands back a *new independent connection*
    rather than a cursor on the same session, so the caller's temp tables and
    registered DataFrames are invisible to it. A driver whose ``cursor()``
    yields an object of the connection's own type is duplicating the
    connection, so the query runs on the handle the caller passed in, and only
    a genuine cursor we created is ours to close.
    """
    cursor = connection.cursor()
    if isinstance(cursor, type(connection)):
        with suppress(Exception):
            cursor.close()
        return connection, False
    return cursor, True


def _columnar_fetch(cursor: Any, method: str) -> Any:
    r"""Call a columnar fetch, or return ``None`` if it does not apply here.

    Snowflake answers a metadata statement (``SHOW``, ``DESC``, ``LIST``) with
    a result set it cannot render as Arrow or pandas, and *raises*
    ``NotSupportedError`` rather than returning ``None``. Treating that as
    "this strategy does not apply" lets the caller fall through to the
    row-based fetch, which reads those statements fine; without it every such
    query failed as ``QUERY_FAILED: Unknown error``, naming neither the cause
    nor a way forward. Any other failure is a real one and propagates.
    """
    try:
        return getattr(cursor, method)()
    except Exception as error:
        if type(error).__name__ == 'NotSupportedError':
            return None
        raise


def read_table(
    connection: Any,
    *,
    table: str | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    r"""Fetches one flat table over an already-open connection.

    Driver-agnostic: uses whichever of Arrow, pandas or plain cursor fetch the
    connection offers. The connection stays open and owned by the caller.

    Args:
        connection: An open connection, e.g. from :func:`connect`.
        table: A table identifier; see :func:`resolve_sql` for what is allowed.
        query: A complete SQL statement. Exactly one of ``table`` / ``query``.

    Returns:
        The rows as a ``pd.DataFrame``.

    Raises:
        ConnectorError: ``INVALID_CONNECTOR_ARGS`` for a bad argument pair,
            ``QUERY_FAILED`` if the statement does not execute.
    """
    sql = resolve_sql(table=table, query=query)
    with driver_guard('QUERY_FAILED', 'query execution failed', sql=sql):
        cursor, owned = _query_handle(connection)
    try:
        with driver_guard('QUERY_FAILED', 'query execution failed', sql=sql):
            cursor.execute(sql)
        with driver_guard(
            'QUERY_FAILED',
            'reading the result failed',
            sql=sql,
            type_error_is_the_caller_s=False,
        ):
            # Arrow first, every time. A driver's own pandas conversion decides
            # for itself what a wide integer becomes, and DuckDB's `fetchdf`
            # turns a HUGEINT -- which is what `sum(BIGINT)` widens to -- into a
            # float64, rounding it. Arrow carries the exact value through to
            # `_arrow_to_pandas`, which knows not to widen it.
            for method in (
                'fetch_arrow_all',
                'to_arrow_table',
                'fetch_arrow_table',
                'fetchall_arrow',
            ):
                if hasattr(cursor, method):
                    arrow_table = _columnar_fetch(cursor, method)
                    if arrow_table is not None:
                        return _arrow_to_pandas(arrow_table)
            if hasattr(cursor, 'fetch_pandas_all'):
                frame = _columnar_fetch(cursor, 'fetch_pandas_all')
                if frame is not None:
                    return frame
            if hasattr(cursor, 'fetchdf'):
                return cursor.fetchdf()
            return _frame_from_rows(cursor.fetchall(), cursor.description)
    finally:
        if owned:
            with suppress(Exception):
                cursor.close()


def _arrow_to_pandas(arrow_table: Any) -> pd.DataFrame:
    r"""Convert an Arrow table without widening null-bearing integers.

    The numpy default represents nulls in an integer column by promoting it to
    ``float64``, which silently rounds values beyond 2**53. Such columns use a
    null-aware integer dtype instead, matching what DuckDB already returns.
    Integer columns without nulls keep their numpy dtype.

    A decimal column needs nothing here: Arrow hands it to pandas as ``Decimal``
    objects, which the wire layer types by value. It is the *driver's* own
    pandas conversion that rounds those, which is why the caller reaches for
    Arrow first.
    """
    df = arrow_table.to_pandas(types_mapper=_NULLABLE_INTEGER_DTYPES.get)
    for position, dtype in enumerate(df.dtypes):
        if isinstance(
            dtype, pd.api.extensions.ExtensionDtype
        ) and pd.api.types.is_integer_dtype(dtype):
            column = df.iloc[:, position]
            if not column.isna().any():
                df.isetitem(position, column.astype(dtype.numpy_dtype))
    return df


def _read_local(
    *,
    data: pd.DataFrame | dict[str, Any] | None = None,
    path: str | None = None,
    format: str | None = None,
    **extra: Any,
) -> pd.DataFrame:
    if extra:
        raise ConnectorError(
            f'local connector got unexpected arguments {sorted(extra)}; '
            "supported: 'data', 'path' and 'format'",
            code='INVALID_CONNECTOR_ARGS',
            details={'arguments': sorted(extra)},
        )
    if data is not None and path is not None:
        raise ConnectorError(
            "local connector accepts exactly one of 'data' or 'path', not both",
            code='INVALID_CONNECTOR_ARGS',
        )
    if data is not None:
        if isinstance(data, pd.DataFrame):
            return data
        try:
            return _narrow_nullable_dtypes(
                pd.DataFrame(_integer_columns_exactly(data))
            )
        except (ValueError, TypeError) as error:
            raise ConnectorError(
                f'local connector could not build a DataFrame from '
                f"'data': {error}",
                code='INVALID_CONNECTOR_ARGS',
            ) from error
    if path is None:
        raise ConnectorError(
            "local connector requires either 'data' or 'path'",
            code='INVALID_CONNECTOR_ARGS',
        )
    path = _as_path_str('local', path)
    _require_local_path(path)
    return _read_file(path, format=format)


def _as_path_str(backend: str, path: Any) -> str:
    r"""Coerce a filesystem path argument to ``str`` before validation.

    File readers parse schemes, prefixes and suffixes as strings. Coercing
    first makes valid ``PathLike`` inputs work and turns invalid path objects
    into the public ``ConnectorError`` contract instead of an internal
    ``AttributeError`` from ``urllib``.
    """
    if isinstance(path, bytes):
        return path.decode()
    try:
        coerced = os.fspath(path)
    except TypeError as error:
        raise ConnectorError(
            f'{backend} connector requires a filesystem path as a string or '
            f'os.PathLike, got {type(path).__name__}',
            code='INVALID_CONNECTOR_ARGS',
            details={'path': repr(path)},
        ) from error
    return coerced.decode() if isinstance(coerced, bytes) else coerced


def _require_local_path(path: str) -> None:
    r"""Reject a URI handed to the connector named ``local``.

    ``pandas`` resolves any fsspec/urllib-supported URL, so without this the
    'safe, no-network' source reaches the network and returns the response as
    a DataFrame. A single-character scheme is a Windows drive letter, not a
    URI. A ``file://`` URI naming a host is refused too: ``urllib`` drops the
    authority and reads the path locally, so the URI does not mean what it
    says.
    """
    parsed = urlparse(path)
    if len(parsed.scheme) > 1 and parsed.scheme != 'file':
        raise ConnectorError(
            f'the local connector reads local filesystem paths, got a '
            f'{parsed.scheme!r} URI: {path!r}. Use the matching connector, '
            f"e.g. source='s3'.",
            code='INVALID_CONNECTOR_ARGS',
            details={'path': path},
        )
    if parsed.scheme == 'file' and parsed.netloc.lower() not in (
        '',
        'localhost',
    ):
        raise ConnectorError(
            f'the local connector reads local filesystem paths, got a '
            f'file:// URI naming the host {parsed.netloc!r}: {path!r}',
            code='INVALID_CONNECTOR_ARGS',
            details={'path': path},
        )


def _read_s3(
    *,
    path: str | None = None,
    storage_options: dict[str, Any] | None = None,
    format: str | None = None,
    **extra: Any,
) -> pd.DataFrame:
    if extra:
        raise ConnectorError(
            f's3 connector got unexpected arguments {sorted(extra)}; '
            "supported: 'path', 'storage_options' and 'format'",
            code='INVALID_CONNECTOR_ARGS',
            details={'arguments': sorted(extra)},
        )
    if path is None:
        raise ConnectorError(
            "s3 connector requires 'path' (an s3:// object URI)",
            code='INVALID_CONNECTOR_ARGS',
        )
    path = _as_path_str('s3', path)
    if not path.startswith('s3://'):
        raise ConnectorError(
            f's3 connector requires an s3:// URI, got {path!r}',
            code='INVALID_CONNECTOR_ARGS',
            details={'path': path},
        )
    require_driver('s3', 's3fs', 's3fs')
    return _read_file(path, storage_options=storage_options, format=format)


def _is_directory(
    path: str, storage_options: dict[str, Any] | None = None
) -> tuple[bool, Exception | None]:
    r"""Whether ``path`` names a directory, local or remote.

    Built with the caller's ``storage_options``, so a private endpoint or an
    explicit credential is used here as it is for the read itself.

    ``os.path.isdir`` answers only for the local filesystem, so an ``s3://``
    prefix was never a directory to it and a partitioned Parquet dataset --
    which is how tabular data lands in object storage -- was refused unless the
    caller happened to write a trailing slash. The filesystem that owns the
    path is asked instead.

    Reached only for a path whose name gives no answer. A failure to reach the
    store is not an answer either, and there is no read afterwards to report
    it: for a suffix-less path this call is the last word, so swallowing the
    error made bad credentials on ``s3://bucket/parts`` surface as an
    unsupported file format. The reason is returned alongside the verdict and
    the caller says which happened.
    """
    parsed = urlparse(path)
    scheme = parsed.scheme
    if len(scheme) <= 1 or scheme == 'file':
        # `os.path.isdir` is always False for the URI text itself, so a
        # `file://` parquet directory without a trailing slash was refused as
        # an unsupported format even though the reader accepts that form.
        local = url2pathname(parsed.path) if scheme == 'file' else path
        return os.path.isdir(local), None
    try:
        import fsspec

        return bool(
            fsspec.filesystem(scheme, **(storage_options or {})).isdir(path)
        ), None
    except Exception as error:  # noqa: BLE001 - reported by the caller
        return False, error


def _resolve_format(
    path: str, storage_options: dict[str, Any] | None = None
) -> str:
    r"""Decide the file format from ``path``, or reject it.

    Falling back to CSV for any unrecognised suffix does not fail loudly: a
    ``.json`` file parses into junk column names and zero rows, and a ``.tsv``
    into a single column. Only the documented formats are read, and a directory
    (with or without a trailing separator) is a Parquet dataset.

    The suffix is read first, so a named file is decided without touching the
    filesystem at all. Only a path that carries no answer of its own is asked
    about, which is the case a partitioned dataset falls into.
    """
    if path.endswith(('/', os.sep)):
        return 'parquet'
    suffix = PurePosixPath(path.lower()).suffix
    if suffix in _COMPRESSION_SUFFIXES:
        suffix = PurePosixPath(path.lower()).suffixes[-2:-1]
        suffix = suffix[0] if suffix else ''
    if suffix in _PARQUET_SUFFIXES:
        return 'parquet'
    if suffix in _CSV_SUFFIXES:
        return 'csv'
    is_directory, unreachable = _is_directory(path, storage_options)
    if is_directory:
        return 'parquet'
    if unreachable is not None:
        raise ConnectorError(
            f'could not determine the format of {path!r}: it names no known '
            f'suffix, and the store could not be asked whether it is a '
            f'directory ({type(unreachable).__name__}: {unreachable}). Fix '
            f'the access problem, or pass format= explicitly',
            code='CONNECT_FAILED',
            details={'path': path},
        ) from unreachable
    raise ConnectorError(
        f'unsupported file format {suffix or path!r} for {path!r}; supported: '
        f'{sorted(_PARQUET_SUFFIXES + _CSV_SUFFIXES)} (optionally compressed) '
        f'and Parquet directories. Pass format="csv" or format="parquet" to '
        f'read a file whose name does not carry one of these suffixes.',
        code='INVALID_CONNECTOR_ARGS',
        details={'path': path, 'suffix': suffix},
    )


def _narrow_nullable_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    r"""Put columns with no nulls back into their plain numpy dtype.

    A null-aware dtype is only needed where a null actually is. Narrowing the
    rest keeps an ordinary frame looking ordinary -- ``int64`` rather than
    ``Int64``, ``object`` rather than ``string`` -- which is what a caller
    comparing dtypes expects, and what the Arrow path already does.
    """
    for position, dtype in enumerate(frame.dtypes):
        if not isinstance(dtype, pd.api.extensions.ExtensionDtype):
            continue
        column = frame.iloc[:, position]
        if column.isna().any():
            continue
        if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(
            dtype
        ):
            frame.isetitem(position, column.astype(dtype.numpy_dtype))
        elif isinstance(dtype, pd.StringDtype):
            # Whatever a plain read would have produced for a string column,
            # which differs across pandas versions.
            frame.isetitem(position, column.astype(_PLAIN_STRING_DTYPE))
        elif isinstance(dtype, pd.BooleanDtype):
            frame.isetitem(position, column.astype(bool))
    return frame


def _frame_from_rows(
    rows: Sequence[Any], description: Sequence[Any] | None
) -> pd.DataFrame:
    r"""The rows of a driver that answers in rows rather than in columns.

    Everything this path can know about a column beyond its values is in
    ``description``, so it is read here rather than thrown away: the name, and
    the type the server declared, which is the only thing that can type a
    column holding nothing but nulls.
    """
    entries = list(description or ())
    columns = [str(entry[0]) for entry in entries]
    frame = _frame_by_position(rows, columns, entries)
    return _narrow_nullable_dtypes(frame)


def _frame_by_position(
    rows: Sequence[Any],
    columns: Sequence[str],
    description: Sequence[Any] = (),
) -> pd.DataFrame:
    r"""The rows as a frame, keyed by where a value sits rather than by its name.

    A join returns two columns of one name, and keying by name collapses them:
    same shape, same names, the last one's values, nothing said. Only a driver
    that answers in rows arrives here -- Databricks and DuckDB answer in Arrow,
    which refuses such a result before this sees it.

    Built a column at a time for the reason :func:`_integer_columns_exactly`
    exists: pandas promotes a null-bearing integer column as it builds.

    A row factory that hands back a mapping -- psycopg's ``dict_row``, which
    the connector's own allow-list admits -- has no positions to read, and
    indexing one raised ``KeyError: 0``, surfacing as a query failure naming
    only ``0``. Those are read by name, which is all such a row has; a driver
    that collapsed two columns of one name before this saw them is said so
    rather than guessed at.

    A result set of no columns still has rows -- ``SELECT FROM
    generate_series(1, 2)`` is legal -- and returning an empty frame lost
    them, so a caller counting rows counted none.
    """
    if not columns:
        return pd.DataFrame(index=pd.RangeIndex(len(rows)))
    values = _values_by_column(rows, columns)
    built = _integer_columns_exactly(dict(enumerate(values)))
    frame = pd.DataFrame(built, columns=list(range(len(columns))))
    frame.columns = pd.Index(columns)
    return _typed_from_the_server(frame, description)


def _values_by_column(
    rows: Sequence[Any], columns: Sequence[str]
) -> list[list[Any]]:
    r"""One list per column, read the way this driver's rows can be read."""
    if rows and isinstance(rows[0], Mapping):
        if len(set(columns)) != len(columns):
            raise ConnectorError(
                'this connection returns rows as mappings, and the result has '
                'two columns of one name, which the driver collapsed before '
                'this saw them. Name them apart with AS, or connect without '
                'a mapping row factory',
                code='QUERY_FAILED',
            )
        return [[row.get(name) for row in rows] for name in columns]
    return [[row[position] for row in rows] for position in range(len(columns))]


# What the server's own name for a type means here, for the one case values
# cannot answer. Only the names a driver on this path actually reports; an
# array or a name not listed is left alone rather than guessed at.
_DTYPE_BY_SERVER_TYPE = {
    'int': 'Int64',
    'int2': 'Int64',
    'int4': 'Int64',
    'int8': 'Int64',
    'integer': 'Int64',
    'bigint': 'Int64',
    'smallint': 'Int64',
    'tinyint': 'Int64',
    'long': 'Int64',
    'short': 'Int64',
    'float4': 'float64',
    'float8': 'float64',
    'float': 'float64',
    'double': 'float64',
    'real': 'float64',
    'bool': 'boolean',
    'boolean': 'boolean',
    'date': 'datetime64[ns]',
    'timestamp': 'datetime64[ns]',
    'timestamptz': 'datetime64[ns, UTC]',
    'text': 'string',
    'varchar': 'string',
    'char': 'string',
    'bpchar': 'string',
    'string': 'string',
}

# A server's type name may carry a length, as `varchar(42)` does. The length
# does not change what the column holds, so it is dropped before the lookup.
# Brackets are not: `int8[]` is an array and nothing here describes one.
_A_LENGTH = re.compile(r'\(\d+(?:,\s*\d+)?\)$')


def _typed_from_the_server(
    frame: pd.DataFrame, description: Sequence[Any]
) -> pd.DataFrame:
    r"""Type a column of nothing but nulls from what the server declared it to be.

    Values decide every other column, and a column with no values decides
    nothing, so an all-null ``bigint`` was built as ``object`` and described to
    the model as text -- and so was an all-null ``float8``, ``bool``, ``date``
    or ``timestamp``. Every other backend reaches this frame through Arrow,
    which carries the declared type, so postgres was the only one that lost it.

    Read only where the driver offers a name for the type and that name is one
    this knows; where it does not, nothing is claimed.
    """
    for position, entry in enumerate(description):
        named = getattr(entry, 'type_display', None)
        if not isinstance(named, str):
            continue
        dtype = _DTYPE_BY_SERVER_TYPE.get(_A_LENGTH.sub('', named.lower()))
        if dtype is None:
            continue
        column = frame.iloc[:, position]
        if not column.isna().all():
            continue
        frame.isetitem(position, pd.array([None] * len(column), dtype=dtype))
    return frame


def _integer_columns_exactly(data: Any) -> Any:
    r"""Build integer columns as ``Int64`` before pandas can widen them.

    ``pd.DataFrame({'id': [big, None]})`` promotes to ``float64``, which rounds
    every value past 2**53 as the frame is built. By then the value is gone --
    casting back to an integer afterwards cannot recover it -- so the column
    has to be constructed with a dtype that holds a null.

    Only columns that ``Int64`` can hold are converted. A column holding a
    whole number wider than that is built as ``object`` instead, which is what
    pandas does for it anyway when nothing is missing -- but not when something
    is: ``[2 ** 63 + 1, None]`` becomes ``float64`` and the value is gone.
    """
    if not isinstance(data, Mapping):
        return data
    built: dict[Any, Any] = {}
    for name, values in data.items():
        if _fits_nullable_int64(values):
            built[name] = pd.array(list(values), dtype='Int64')
        elif _holds_wide_integer(values):
            built[name] = pd.Series(list(values), dtype=object)
        else:
            built[name] = values
    return built


def _holds_wide_integer(values: Any) -> bool:
    r"""Whether ``values`` holds a whole number too wide for ``Int64``.

    Such a column cannot be built as ``Int64``, and left to pandas alongside a
    null it becomes ``float64``, which rounds it away. Built as ``object`` the
    integer stays an integer and the wire rules can see it.
    """
    if not isinstance(values, (list, tuple)):
        return False
    return any(
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and not _INT64_MIN <= int(value) <= _INT64_MAX
        for value in values
        if not _is_null(value)
    )


def _fits_nullable_int64(values: Any) -> bool:
    r"""Whether ``values`` is a list of whole numbers ``Int64`` can hold.

    ``bool`` is excluded although it is an ``int``: a flag column is not a
    number, and pandas keeps it a bool without help.
    """
    if not isinstance(values, (list, tuple)):
        return False
    present = [value for value in values if not _is_null(value)]
    if not present:
        return False
    return all(
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and _INT64_MIN <= int(value) <= _INT64_MAX
        for value in present
    )


def _is_null(value: Any) -> bool:
    r"""``None``, ``NaN`` and ``pd.NA`` alike, without comparing ``pd.NA``."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _read_file(
    path: str,
    storage_options: dict[str, Any] | None = None,
    format: str | None = None,
) -> pd.DataFrame:
    if format is None:
        resolved = _resolve_format(path, storage_options)
    elif format in _FILE_FORMATS:
        resolved = format
    else:
        raise ConnectorError(
            f'unknown format {format!r}; supported: {sorted(_FILE_FORMATS)}',
            code='INVALID_CONNECTOR_ARGS',
            details={'format': format},
        )
    with driver_guard(
        'READ_FAILED',
        f'failed to read {path!r}',
        missing_as_not_found=True,
        path=path,
    ):
        # A blank in an integer column makes pandas promote the whole column
        # to float64 as it parses, rounding every value past 2**53 before the
        # frame exists. A null-aware backend keeps the integer an integer.
        # Parquet needs this as much as CSV does: a file whose integer column
        # is nullable arrives as float64 whatever its values, so an id past
        # 2**53 is already wrong by the time the wire rules could catch it.
        if resolved == 'parquet':
            return _narrow_nullable_dtypes(
                pd.read_parquet(
                    path,
                    storage_options=storage_options,
                    dtype_backend='numpy_nullable',
                )
            )
        return _narrow_nullable_dtypes(
            pd.read_csv(
                path,
                storage_options=storage_options,
                dtype_backend='numpy_nullable',
            )
        )
