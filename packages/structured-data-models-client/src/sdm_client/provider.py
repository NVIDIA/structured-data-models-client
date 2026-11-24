# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A caller's own source of data, answering the core's data requests.

The core never sends SQL to a source. It sends one of a fixed set of
typed requests (:data:`DataRequest`), and the source answers each with
Arrow rows. Every request a source receives is one its
:class:`Capabilities` declared, or :class:`Describe` and :class:`Scan`,
which every source answers. Subclass :class:`Source` and pass an instance
wherever a source is taken:

>>> schema = Schema.from_source(MySource(), primary_keys=...)

The core checks every row a source returns against the query's own
rules, so a source may return more rows than a value filter holds for,
but never fewer. A filter kind the source does not declare
(``Capabilities.filters``) is never sent to it; the core applies it to the
rows read. The seeds of a :class:`Neighbors` or :class:`Lookup` holding
more keys than ``Capabilities.max_keys_per_request`` reach the source
already split into several requests.

Values in requests are plain Python values, exactly as the core holds
them: ``int``, ``float``, ``bool``, ``str``, :class:`~decimal.Decimal`,
``bytes``, :class:`datetime.date`, :class:`datetime.time`, and
:class:`datetime.datetime` and :class:`datetime.timedelta` for timestamps
and durations. A nanosecond timestamp or duration is a
:class:`pandas.Timestamp` or :class:`pandas.Timedelta`, which are those
types keeping nanoseconds, and a time of day with nanoseconds a
:class:`pandas.Timedelta` since midnight; a date or timestamp outside the
years :mod:`datetime` holds is a :class:`numpy.datetime64`. A timestamp
without a zone is a wall clock; one with a zone, an instant in that zone.

The library calls a source from the thread that runs the operation, one
request at a time.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    'BATCH_COLUMN',
    'MATCHES_COLUMN',
    'SEED_BATCH',
    'SEED_KEY',
    'AllBut',
    'AllOf',
    'AnyOf',
    'Bound',
    'Capabilities',
    'ColumnRanges',
    'Compare',
    'CompareOp',
    'Count',
    'CountRows',
    'DataRequest',
    'Describe',
    'DescribePart',
    'Fallback',
    'Filter',
    'FilterKind',
    'HasRows',
    'In',
    'IsNull',
    'KeyCounts',
    'KeyIn',
    'KeySet',
    'Lookup',
    'NeighborLimit',
    'Neighbors',
    'NotNull',
    'NullOrder',
    'OptionalRequest',
    'Preview',
    'ProviderFailure',
    'Relation',
    'RepeatedKey',
    'SampleEntities',
    'Sampling',
    'Scan',
    'SeedBound',
    'Source',
    'TableColumn',
    'TableKey',
    'TieTerm',
    'TimeBounds',
    'TimeRange',
    'Window',
]

#: The answer column naming the seed a :class:`Neighbors` or :class:`Lookup`
#: row answers.
BATCH_COLUMN = '__kumo_batch__'
#: The answer column of a :class:`Lookup` counting the rows that hold the
#: seed's key.
MATCHES_COLUMN = '__kumo_matches__'
#: The column of a seed key set holding each seed's key (``key``; ``key_1``,
#: ``key_2``, ... for a key of several columns).
SEED_KEY = 'key'
#: The column of a seed key set holding each seed's batch position.
SEED_BATCH = 'batch'


# --- enumerations ---------------------------------------------------------


class OptionalRequest(str, enum.Enum):
    r"""A request a source may answer itself; the core answers an
    undeclared one in memory, from the rows a :class:`Scan` reads, within
    ``Capabilities.fallback``, or refuses it."""

    PREVIEW = 'preview'
    COUNT_ROWS = 'count_rows'
    KEY_COUNTS = 'key_counts'
    TIME_BOUNDS = 'time_bounds'
    COLUMN_RANGES = 'column_ranges'
    REPEATED_KEY = 'repeated_key'
    SAMPLE_ENTITIES = 'sample_entities'
    NEIGHBORS = 'neighbors'
    LOOKUP = 'lookup'


class FilterKind(str, enum.Enum):
    r"""A kind of value filter a source may declare it applies
    (``Capabilities.filters``)."""

    COMPARE = 'compare'
    IN = 'in'
    IS_NULL = 'is_null'
    ANY_OF = 'any_of'
    ALL_OF = 'all_of'
    HAS_ROWS = 'has_rows'


class CompareOp(str, enum.Enum):
    r"""How a :class:`Compare` compares."""

    EQ = 'eq'
    NE = 'ne'
    LT = 'lt'
    LE = 'le'
    GT = 'gt'
    GE = 'ge'


class DescribePart(str, enum.Enum):
    r"""Which part of the catalog a :class:`Describe` asks for; its answer
    has the schema the request's ``expected`` gives."""

    TABLES = 'tables'
    COLUMNS = 'columns'
    KEYS = 'keys'
    FOREIGN_KEYS = 'foreign_keys'
    INDEXES = 'indexes'
    INDEX_COLUMNS = 'index_columns'
    NAMESPACE = 'namespace'
    KEY_ENFORCEMENT = 'key_enforcement'
    VIEW_DEFINITION = 'view_definition'


class Sampling(str, enum.Enum):
    r"""Whether a sample drawn with the same seed is the same rows every
    time."""

    SEEDED_STABLE = 'seeded_stable'
    UNSEEDED = 'unseeded'


class NullOrder(str, enum.Enum):
    r"""Where nulls go when the source orders rows."""

    FIRST = 'first'
    LAST = 'last'


# --- capabilities and failures --------------------------------------------


@dataclass(frozen=True)
class Fallback:
    r"""The most rows and bytes the core reads to answer a request in
    memory."""

    max_rows: int
    max_bytes: int

    def __post_init__(self) -> None:
        if not (_positive(self.max_rows) and _positive(self.max_bytes)):
            raise ValueError('fallback limits must be positive integers')


@dataclass(frozen=True)
class Capabilities:
    r"""What a :class:`Source` answers, and how.

    Every source answers :class:`Describe` and :class:`Scan`. Each other
    request is answered by the source when declared in ``optional``;
    otherwise the core answers it itself, in memory, from the rows a scan
    reads, within ``fallback``'s limits, or refuses it with
    ``UNSUPPORTED_REQUEST`` when no fallback is declared.

    Attributes:
        optional: The optional requests the source answers itself.
        max_keys_per_request: The most seeds one :class:`Neighbors` or
            :class:`Lookup` may carry; more reach the source as several
            requests, whose answers the core joins. ``None`` is no limit.
        sampling: Whether a sample drawn with the same seed is the same
            rows every time.
        max_seed: The largest sampling seed the source takes.
        null_order: Where the source puts nulls when it orders rows;
            ``None``, and the core never asks it for rows in an order.
        fallback: The limits within which the core answers undeclared
            optional requests in memory; ``None`` declares no fallback.
        filters: The value filters the source applies to a sample or
            scan; the core sends it no other kind, and checks every row it
            keeps itself, so a source may return more rows than a value
            filter holds for but never fewer.
    """

    optional: frozenset[OptionalRequest] = frozenset()
    max_keys_per_request: int | None = None
    sampling: Sampling = Sampling.SEEDED_STABLE
    max_seed: int = 2**63 - 1
    null_order: NullOrder | None = NullOrder.LAST
    fallback: Fallback | None = None
    filters: frozenset[FilterKind] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'optional', _members(self.optional, OptionalRequest)
        )
        object.__setattr__(self, 'filters', _members(self.filters, FilterKind))
        object.__setattr__(self, 'sampling', Sampling(self.sampling))
        if self.null_order is not None:
            object.__setattr__(self, 'null_order', NullOrder(self.null_order))
        if self.max_keys_per_request is not None and not _positive(
            self.max_keys_per_request
        ):
            raise ValueError('max_keys_per_request must be a positive integer')
        if not _integer(self.max_seed) or self.max_seed < 0:
            raise ValueError('max_seed must be a non-negative integer')
        if self.fallback is not None and not isinstance(
            self.fallback, Fallback
        ):
            raise TypeError('fallback must be a Fallback')


@dataclass(frozen=True)
class ProviderFailure:
    r"""What a source's failure tells the core: each field where the
    failure says one.

    Attributes:
        sqlstate: The SQLSTATE, as a database driver gives it.
        vendor_code: The database's own error code.
        error_class: The source's own error class name, where it has one.
        message: The internal diagnostic; logged by the core, never shown
            to end users.
    """

    sqlstate: str | None = None
    vendor_code: str | None = None
    error_class: str | None = None
    message: str | None = None


# --- parts of requests ----------------------------------------------------


@dataclass(frozen=True)
class Relation:
    r"""A table, by its name's parts, each quoted alone
    (``('main', 't')`` is ``t`` of ``main``)."""

    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'parts', tuple(self.parts))
        if not self.parts:
            raise ValueError('a relation has at least one name part')

    @property
    def table(self) -> str:
        r"""The last part: the table's own name."""
        return self.parts[-1]


@dataclass(frozen=True, eq=False)
class KeySet:
    r"""A set of keys the core sent: its column names, and its rows as a
    pyarrow stream, which can be read once."""

    columns: tuple[str, ...]
    reader: pa.RecordBatchReader


@dataclass(frozen=True)
class Bound:
    r"""A bound on a value."""

    value: Any
    inclusive: bool


@dataclass(frozen=True)
class SeedBound:
    r"""A bound read from the seed's own time column ``seed_column``."""

    seed_column: str
    inclusive: bool


@dataclass(frozen=True)
class Window:
    r"""Which of a seed's rows count: the row's ``column`` against the
    seed's own time columns, named in the seed key set."""

    column: str
    lower: SeedBound | None
    upper: SeedBound | None


@dataclass(frozen=True)
class Count:
    r"""At most ``count`` rows per seed."""

    count: int


@dataclass(frozen=True)
class AllBut:
    r"""All but the last ``|value|`` rows per seed, as a negative limit in
    the reference means; ``value`` is the number as written."""

    value: str


NeighborLimit = Count | AllBut


@dataclass(frozen=True)
class TieTerm:
    r"""One term of a row order."""

    column: str
    as_timestamp: bool
    descending: bool
    nulls_last: bool
    string_like: bool


@dataclass(frozen=True)
class TableColumn:
    r"""A column of a table."""

    table: Relation
    column: str


@dataclass(frozen=True)
class TableKey:
    r"""A key of a table, with the label its answer row carries."""

    table: Relation
    columns: tuple[str, ...]
    label: int
    unique: bool


# --- filters --------------------------------------------------------------


@dataclass(frozen=True)
class KeyIn:
    r"""The row's ``columns`` equal one of the key set's rows."""

    columns: tuple[str, ...]
    keys: KeySet


@dataclass(frozen=True)
class TimeRange:
    r"""The row's time ``column`` within the bounds given."""

    column: str
    lower: Bound | None
    upper: Bound | None


@dataclass(frozen=True)
class NotNull:
    r"""The row holds a value in ``column``."""

    column: str


@dataclass(frozen=True)
class Compare:
    r"""The row's ``column`` compared with ``value`` by ``op``; a null
    meets none."""

    column: str
    op: CompareOp
    value: Any


@dataclass(frozen=True)
class In:
    r"""The row's ``column`` equals one of ``values``."""

    column: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class IsNull:
    r"""The row holds no value in ``column``."""

    column: str


@dataclass(frozen=True)
class AnyOf:
    r"""At least one of ``filters`` holds."""

    filters: tuple[Filter, ...]


@dataclass(frozen=True)
class AllOf:
    r"""Every one of ``filters`` holds."""

    filters: tuple[Filter, ...]


@dataclass(frozen=True)
class HasRows:
    r"""At least ``at_least`` rows of ``table`` whose ``key`` is the row's
    ``column``, among the rows meeting every one of ``filters``."""

    column: str
    table: Relation
    key: str
    filters: tuple[Filter, ...]
    at_least: int


Filter = (
    KeyIn
    | TimeRange
    | NotNull
    | Compare
    | In
    | IsNull
    | AnyOf
    | AllOf
    | HasRows
)


# --- requests -------------------------------------------------------------


class _Request:
    r"""What every request has: its name in the core's contract."""

    #: The request's name in the core's contract, such as ``'neighbors'``.
    name: str = ''


@dataclass(frozen=True)
class Describe(_Request):
    r"""One part of the catalog, for the tables named (every table when
    ``tables`` is empty)."""

    tables: tuple[Relation, ...]
    part: DescribePart
    name = 'describe'


@dataclass(frozen=True)
class Scan(_Request):
    r"""Every row of ``table`` meeting all ``filters``, ``columns`` in
    order. With a ``limit``, any ``limit`` of those rows."""

    table: Relation
    columns: tuple[str, ...]
    filters: tuple[Filter, ...]
    limit: int | None
    name = 'scan'


@dataclass(frozen=True)
class Preview(_Request):
    r"""At most ``limit`` rows of ``table``, ``columns`` in order."""

    table: Relation
    columns: tuple[str, ...]
    limit: int
    name = 'preview'


@dataclass(frozen=True)
class CountRows(_Request):
    r"""One row, one ``int64`` column: the rows of ``table`` meeting all
    filters."""

    table: Relation
    filters: tuple[Filter, ...]
    name = 'count_rows'


@dataclass(frozen=True)
class KeyCounts(_Request):
    r"""One row per key, in order: its label, the table's rows, the key's
    distinct values among rows holding all of it, and the rows missing any
    part of it (schema given as ``expected``)."""

    keys: tuple[TableKey, ...]
    name = 'key_counts'


@dataclass(frozen=True)
class TimeBounds(_Request):
    r"""One row per column, in order: its position, and its smallest and
    largest value as a timestamp in microseconds (schema given as
    ``expected``)."""

    columns: tuple[TableColumn, ...]
    name = 'time_bounds'


@dataclass(frozen=True)
class ColumnRanges(_Request):
    r"""One row: the rows of ``table``, then for each column in order its
    smallest value, largest value (each of the column's own type, null when
    no row holds one) and the rows holding a value."""

    table: Relation
    columns: tuple[str, ...]
    name = 'column_ranges'


@dataclass(frozen=True)
class RepeatedKey(_Request):
    r"""At most one row: a value of ``key`` that more than one row of
    ``table`` holds."""

    table: Relation
    key: tuple[str, ...]
    name = 'repeated_key'


@dataclass(frozen=True)
class SampleEntities(_Request):
    r"""A sample of ``n`` entities of ``table`` by ``key``, drawn with
    ``seed``: ``columns`` in order, from the rows meeting every filter.
    ``rows`` is the table's row count, where the core counted it."""

    table: Relation
    key: tuple[str, ...]
    n: int
    seed: int
    columns: tuple[str, ...]
    filters: tuple[Filter, ...]
    rows: int | None
    name = 'sample_entities'


@dataclass(frozen=True)
class Neighbors(_Request):
    r"""The rows of ``table`` whose ``key`` equals a seed's key: first
    :data:`BATCH_COLUMN`, the seed's ``batch`` position (of the seed key
    set's ``batch`` column's type), then ``columns`` in order.

    A row is kept for a seed only within the seed's ``window``, where one is
    given. With ``per_seed``, at most that many rows per seed, the first in
    ``order``, and the answer is ordered by ``(batch, order)``; without it,
    there is no order.
    """

    table: Relation
    key: tuple[str, ...]
    columns: tuple[str, ...]
    seeds: KeySet
    window: Window | None
    per_seed: NeighborLimit | None
    order: tuple[TieTerm, ...]
    name = 'neighbors'


@dataclass(frozen=True)
class Lookup(_Request):
    r"""One row per seed whose key ``table`` holds: :data:`BATCH_COLUMN`,
    the seed's ``batch`` position; :data:`MATCHES_COLUMN`, the rows holding
    that key (non-null ``int64``); then ``columns`` in order. Which row of a
    key several rows hold is not defined."""

    table: Relation
    key: tuple[str, ...]
    columns: tuple[str, ...]
    seeds: KeySet
    name = 'lookup'


DataRequest = (
    Describe
    | Scan
    | Preview
    | CountRows
    | KeyCounts
    | TimeBounds
    | ColumnRanges
    | RepeatedKey
    | SampleEntities
    | Neighbors
    | Lookup
)


# --- the source -----------------------------------------------------------


class Source(abc.ABC):
    r"""Where a schema's data is read, answering the core's data requests.

    Subclass it to read data your own way: implement :meth:`capabilities`
    and :meth:`answer`, and pass an instance wherever a source is taken:

    >>> schema = Schema.from_source(MySource(), primary_keys=...)

    The built-in sources (:class:`~sdm_client.DuckDBSource`,
    :class:`~sdm_client.PostgresSource`, :class:`~sdm_client.SQLiteSource`,
    :class:`~sdm_client.DatabricksSource`,
    :class:`~sdm_client.SnowflakeSource`) are sources too, each answered by
    the library's own SQL for its database.

    A source holds live data access, so a source, and a schema over one,
    cannot be pickled. The library never closes it.
    """

    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        r"""What this source answers; read once per operation."""

    @abc.abstractmethod
    def answer(self, request: DataRequest, expected: pa.Schema | None) -> Any:
        r"""Answers one request with every row it asks for.

        Args:
            request: The request.
            expected: The exact schema the answer must have, where the
                request fixes one (:class:`Describe`, :class:`KeyCounts`,
                :class:`TimeBounds`); otherwise ``None``, and the answer
                holds the columns the request names, in order, typed as the
                source stores them.

        Returns:
            Any object exporting an Arrow stream (``__arrow_c_stream__``):
            a pyarrow ``Table`` or ``RecordBatchReader``, or a DuckDB or
            Polars result.

        Raises:
            Exception: Any failure; the operation then fails with
                ``PROVIDER_ERROR``.
        """

    def failure(self, error: Exception) -> ProviderFailure | None:
        r"""What one of this source's failures tells the core beyond its
        class.

        By default ``None``: the failure's ``sqlstate`` and ``vendor_code``
        attributes, as database drivers set them, and its message are read.
        """
        return None

    def _sdm_chosen(self) -> tuple[dict[str, Any], Any]:
        r"""The core's provider choice for this source, and the run loop's
        provider answering its fetches. Not part of the API: a built-in
        source answers through its SQL kit instead."""
        from sdm_client._rust import host_providers

        return host_providers.chosen(self)

    def __reduce__(self) -> Any:
        raise TypeError(
            'a Source holds an open connection, which cannot be pickled'
        )


# --- helpers --------------------------------------------------------------


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive(value: object) -> bool:
    return _integer(value) and value > 0  # type: ignore[operator]


def _members(values: Iterable[Any], kind: type[enum.Enum]) -> frozenset[Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f'expected a collection of {kind.__name__}')
    return frozenset(kind(value) for value in values)
