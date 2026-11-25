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

Each value in a request is a :class:`Value`: a plain Python value with
the Arrow type the core holds it as, which a source needs to write it
exactly (an integer's width and sign, a decimal's precision, a time's
unit and zone, the type of a null). The plain values are ``None``,
``int``, ``float``, ``bool``, ``str``, :class:`~decimal.Decimal`,
``bytes``, :class:`datetime.date`, :class:`datetime.time`, and
:class:`datetime.datetime` and :class:`datetime.timedelta` for timestamps
and durations. A nanosecond timestamp or duration is a
:class:`pandas.Timestamp` or :class:`pandas.Timedelta`, which are those
types keeping nanoseconds, and a time of day with nanoseconds a
:class:`pandas.Timedelta` since midnight; a date or timestamp outside the
years :mod:`datetime` holds is a :class:`numpy.datetime64`. A timestamp
without a zone is a wall clock; one with a zone, an instant in that zone.

The library reads a source's :meth:`Source.capabilities` once at the start
of each operation, before any request of it, and then calls
:meth:`Source.answer` from the thread that runs the operation, one request
at a time.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
    'ArrowStream',
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
    'FailureFacts',
    'FailureReason',
    'Fallback',
    'Filter',
    'FilterKind',
    'HasRows',
    'In',
    'IndexAdvice',
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
    'RelationColumn',
    'RepeatedKey',
    'SampleEntities',
    'Sampling',
    'Scan',
    'SeedBound',
    'SemanticModel',
    'Source',
    'SqlLimit',
    'TableColumn',
    'TableKey',
    'TextOrder',
    'TextTimes',
    'TieTerm',
    'TimeBounds',
    'TimeRange',
    'Unusable',
    'Value',
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


class TextTimes(str, enum.Enum):
    r"""How the core reads a time the source stores as text."""

    #: Each value on its own, as pandas reads one (``to_datetime``).
    VALUE_BY_VALUE = 'value_by_value'
    #: By the client's date rules, mixed offsets read at UTC.
    CLIENT_RULES = 'client_rules'


class IndexAdvice(str, enum.Enum):
    r"""Which of the source's indexes serve the sampler's reads, for its
    warning of the indexes it finds missing."""

    #: An index over exactly the one column read.
    SINGLE_COLUMN = 'single_column'
    #: Any index whose first column is the read's first column.
    LEADING_COLUMN = 'leading_column'


class TextOrder(str, enum.Enum):
    r"""How the source compares text, where the core compares it for the
    source in memory (``Capabilities.fallback_text_order``)."""

    #: By the exact UTF-8 bytes: no locale, case folding or normalization.
    BINARY_UTF8 = 'binary_utf8'


class Unusable(str, enum.Enum):
    r"""Why a table the source failed to read cannot be used."""

    #: It does not exist, or the caller may not see it.
    MISSING = 'missing'
    #: The caller may not read it.
    FORBIDDEN = 'forbidden'


class FailureReason(str, enum.Enum):
    r"""What went wrong, where a source's failure says."""

    VALUE_NOT_CONVERTED = 'value_not_converted'
    DIVISION_BY_ZERO = 'division_by_zero'
    OUT_OF_RANGE = 'out_of_range'
    COLUMN_UNRESOLVED = 'column_unresolved'
    TABLE_UNRESOLVED = 'table_unresolved'
    PERMISSION_DENIED = 'permission_denied'
    SYNTAX_REJECTED = 'syntax_rejected'
    TYPES_INCOMPATIBLE = 'types_incompatible'


@runtime_checkable
class ArrowStream(Protocol):
    r"""Any object exporting an Arrow stream through the Arrow PyCapsule
    interface: a pyarrow ``Table`` or ``RecordBatchReader``, a DuckDB or
    Polars result."""

    def __arrow_c_stream__(
        self, requested_schema: object | None = None
    ) -> object: ...


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
class SemanticModel:
    r"""The semantic model a source is read through: its name, and whether
    its expressions calling functions other than pure built-ins are kept
    (they run as SQL under the caller's credentials)."""

    name: str
    trust_expressions: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('a semantic model needs its name')
        if not isinstance(self.trust_expressions, bool):
            raise TypeError('trust_expressions is True or False')


@dataclass(frozen=True)
class Capabilities:
    r"""What a :class:`Source` answers, and how.

    Every source answers :class:`Describe` and :class:`Scan`. Each other
    request is answered by the source when declared in ``optional``;
    otherwise the core answers it itself, in memory, from the rows a scan
    reads, within ``fallback``'s limits, or refuses it with
    ``UNSUPPORTED_REQUEST`` when no fallback is declared.

    The rest describe the source to the core: how it names its tables, what
    its column types mean, and how it reads what the core asks. Each
    default is a plain source of tables; the built-in sources declare their
    engine's.

    Attributes:
        optional: The optional requests the source answers itself.
        max_keys_per_request: The most seeds one :class:`Neighbors` or
            :class:`Lookup` may carry; more reach the source as several
            requests, whose answers the core joins. ``None`` is no limit.
        sampling: Whether a sample drawn with the same seed is the same
            rows every time.
        max_seed: The largest sampling seed the source takes, an unsigned
            64-bit integer.
        null_order: Where the source puts nulls when it orders rows;
            ``None``, and the core never asks it for rows in an order.
        fallback: The limits within which the core answers undeclared
            optional requests in memory; ``None`` declares no fallback.
        filters: The value filters the source applies to a sample or
            scan; the core sends it no other kind, and checks every row it
            keeps itself, so a source may return more rows than a value
            filter holds for but never fewer.
        fallback_text_order: How the source compares text, where the core's
            in-memory fallback compares text for it; ``None``, and such a
            comparison is refused by name.
        text_times: How the core reads a time the source stores as text;
            ``None``, and it refuses to.
        index_advice: Which indexes serve the sampler's reads; ``None``,
            and the core neither reads the source's indexes nor warns of
            missing ones.
        engine: The database engine the source is, by the name messages
            name it by (``'duckdb'``, ``'sqlite'``, ``'postgres'``,
            ``'databricks'``, ``'snowflake'``); its semantic models are read
            in that engine's SQL. ``None`` for any other.
        namespace: The namespace a table named by one part is read under,
            outermost first, such as ``('main', 'sales')``.
        ask_namespace_first: Whether the core asks the source its namespace
            (a :class:`Describe` of :attr:`DescribePart.NAMESPACE`, answered
            in full) before it reads a table named by one part.
        retry_upper_case_names: Whether a table the catalog does not find
            as named is tried again upper-cased, as an engine folding
            unquoted names does.
        column_types_are_affinities: Whether a column's declared type is
            only an affinity, so its type is read from its values.
        reread_tables_every_run: Whether each run reads the tables as they
            stand, rather than as a schema given to it holds them.
        semantic_model: The semantic model the source is read through.
        derived_relations: Whether the source reads a derived
            :class:`Relation`.
        sample_by_row_count: Whether the source sizes a sample by the
            table's rows, which the core counts first and names on each
            :class:`SampleEntities`.
        sample_by_key_type: Whether the source orders a seeded sample by
            the key's data type, which the core names on each
            :class:`SampleEntities`.
        neighbor_lookback: Whether neighbor reads by a foreign key over a
            table with a time column look back from each seed only as far
            as the table's rate of rows says, as warehouse samplers do.
        keys_optional: Whether a failed read of a table's declared keys is
            read as no declared keys, rather than failing the description.
        key_enforcement: Whether the catalog answers which tables enforce
            their keys (:attr:`DescribePart.KEY_ENFORCEMENT`).
        compares_stored_times: Whether the source compares a window's
            bounds with the stored text of the window's column, which the
            core reads once and names on each bound's seed column.
        nan_is_missing: Whether a NaN the source answers is read as
            missing, and ``-0.0`` as ``0.0``.
        opaque_vendor: The vendor whose opaque Arrow extension types
            (``arrow.opaque``) the source hands over, read by that vendor's
            layout; ``None``, and such a column is read as its storage.
    """

    optional: frozenset[OptionalRequest] = frozenset()
    max_keys_per_request: int | None = None
    sampling: Sampling = Sampling.SEEDED_STABLE
    max_seed: int = 2**63 - 1
    null_order: NullOrder | None = NullOrder.LAST
    fallback: Fallback | None = None
    filters: frozenset[FilterKind] = frozenset()
    fallback_text_order: TextOrder | None = None
    text_times: TextTimes | None = None
    index_advice: IndexAdvice | None = None
    engine: str | None = None
    namespace: tuple[str, ...] = ()
    ask_namespace_first: bool = False
    retry_upper_case_names: bool = False
    column_types_are_affinities: bool = False
    reread_tables_every_run: bool = False
    semantic_model: SemanticModel | None = None
    derived_relations: bool = False
    sample_by_row_count: bool = False
    sample_by_key_type: bool = False
    neighbor_lookback: bool = False
    keys_optional: bool = False
    key_enforcement: bool = False
    compares_stored_times: bool = False
    nan_is_missing: bool = False
    opaque_vendor: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'optional', _members(self.optional, OptionalRequest)
        )
        object.__setattr__(self, 'filters', _members(self.filters, FilterKind))
        object.__setattr__(self, 'sampling', Sampling(self.sampling))
        for name, kind in (
            ('null_order', NullOrder),
            ('fallback_text_order', TextOrder),
            ('text_times', TextTimes),
            ('index_advice', IndexAdvice),
        ):
            given = getattr(self, name)
            if given is not None:
                object.__setattr__(self, name, kind(given))
        if self.max_keys_per_request is not None and not _positive(
            self.max_keys_per_request
        ):
            raise ValueError('max_keys_per_request must be a positive integer')
        if not _integer(self.max_seed) or not 0 <= self.max_seed < 2**64:
            raise ValueError('max_seed must be an unsigned 64-bit integer')
        if self.fallback is not None and not isinstance(
            self.fallback, Fallback
        ):
            raise TypeError('fallback must be a Fallback')
        if isinstance(self.namespace, str):
            raise TypeError('namespace is a sequence of name parts')
        object.__setattr__(self, 'namespace', tuple(self.namespace))
        for part in self.namespace:
            if not isinstance(part, str) or not part:
                raise ValueError('each namespace part is a non-empty string')
        for name in ('engine', 'opaque_vendor'):
            given = getattr(self, name)
            if given is not None and (not isinstance(given, str) or not given):
                raise ValueError(f'{name} is a non-empty string or None')
        if self.semantic_model is not None and not isinstance(
            self.semantic_model, SemanticModel
        ):
            raise TypeError('semantic_model must be a SemanticModel')
        for name in (
            'ask_namespace_first',
            'retry_upper_case_names',
            'column_types_are_affinities',
            'reread_tables_every_run',
            'derived_relations',
            'sample_by_row_count',
            'sample_by_key_type',
            'neighbor_lookback',
            'keys_optional',
            'key_enforcement',
            'compares_stored_times',
            'nan_is_missing',
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'{name} is True or False')


@dataclass(frozen=True)
class FailureFacts:
    r"""What a source's failure means, as the core reads it.

    Attributes:
        not_found: The table asked for does not exist under the name asked;
            a name tried in another spelling is tried next.
        refuses_expression: The source refused an expression it was asked
            to read, for what it says or reads.
        unusable: Why a table that failed to be read cannot be used.
        denied: The source denied the caller access to what was asked.
        reason: What went wrong, where the failure says.
    """

    not_found: bool = False
    refuses_expression: bool = False
    unusable: Unusable | None = None
    denied: bool = False
    reason: FailureReason | None = None

    def __post_init__(self) -> None:
        if self.unusable is not None:
            object.__setattr__(self, 'unusable', Unusable(self.unusable))
        if self.reason is not None:
            object.__setattr__(self, 'reason', FailureReason(self.reason))


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
        facts: What the failure means, where the source reads it itself;
            ``None``, and the core reads only what the SQL standard's codes
            say (a SQLSTATE of ``42501`` is a denial).
    """

    sqlstate: str | None = None
    vendor_code: str | None = None
    error_class: str | None = None
    message: str | None = None
    facts: FailureFacts | None = None

    def __post_init__(self) -> None:
        if self.facts is not None and not isinstance(self.facts, FailureFacts):
            raise TypeError('facts must be a FailureFacts')


# --- parts of requests ----------------------------------------------------


@dataclass(frozen=True)
class Value:
    r"""A value of a request: the plain Python value, and the Arrow type
    the core holds it as.

    Attributes:
        value: The value (see the module's description of each type's).
        type: Its Arrow type, such as ``pa.int64()`` or
            ``pa.timestamp('us', tz='UTC')``.
    """

    value: Any
    type: pa.DataType


@dataclass(frozen=True)
class RelationColumn:
    r"""A column of a derived :class:`Relation`: its name, and what it
    reads, exactly one of a column of the source (``column``) or an
    expression in the source's own SQL (``expression``)."""

    name: str
    column: str | None = None
    expression: str | None = None


@dataclass(frozen=True)
class Relation:
    r"""A table, by its name's parts, each quoted alone
    (``('main', 't')`` is ``t`` of ``main``).

    A source declaring ``Capabilities.derived_relations`` may be asked for a
    derived relation, as a semantic model's definition gives one: its
    ``columns``, each a name with the source column or SQL expression it
    reads, and a ``filter``, an SQL condition every row of it holds. Both
    are text of the source's own SQL. With ``found``, the columns are the
    ones the source's catalog found, each read by its own name, and the
    source may read the table through them.

    Attributes:
        parts: The name's parts, outermost first.
        columns: The relation's columns, where it is derived; empty for
            the table's own.
        filter: The condition every row holds; ``None`` for every row.
        found: Whether ``columns`` are the ones the catalog found.
    """

    parts: tuple[str, ...]
    columns: tuple[RelationColumn, ...] = ()
    filter: str | None = None
    found: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, 'parts', tuple(self.parts))
        object.__setattr__(self, 'columns', tuple(self.columns))
        if not self.parts:
            raise ValueError('a relation has at least one name part')

    @property
    def table(self) -> str:
        r"""The last part: the table's own name."""
        return self.parts[-1]


@dataclass(frozen=True, eq=False)
class KeySet:
    r"""A set of keys the core sent: its columns, and its rows as a
    pyarrow stream, which can be read once.

    Attributes:
        schema: The key set's columns, as the core declares them, each with
            what the core notes of it in its field metadata (such as the
            joined key's declared type), which a source may ignore.
        reader: The key set's rows.
    """

    schema: pa.Schema
    reader: pa.RecordBatchReader

    @property
    def columns(self) -> tuple[str, ...]:
        r"""The key set's column names, in order."""
        return tuple(self.schema.names)


@dataclass(frozen=True)
class Bound:
    r"""A bound on a value."""

    value: Value
    inclusive: bool


@dataclass(frozen=True)
class SeedBound:
    r"""A bound read from the seed's own time column ``seed_column``."""

    seed_column: str
    inclusive: bool


@dataclass(frozen=True)
class Window:
    r"""Which of a seed's rows count: the row's ``column`` against the
    seed's own time columns, named in the seed key set. ``source_type`` is
    the column's type as the source's catalog declares it, where the core
    read it."""

    column: str
    lower: SeedBound | None
    upper: SeedBound | None
    source_type: str | None = None


@dataclass(frozen=True)
class Count:
    r"""At most ``count`` rows per seed."""

    count: int


@dataclass(frozen=True)
class AllBut:
    r"""All but the last ``|value|`` rows per seed, as a negative limit in
    the reference means; ``value`` is the number as written."""

    value: str


@dataclass(frozen=True)
class SqlLimit:
    r"""As many rows per seed as ``expression``, in the source's own SQL,
    says, as a query writes one."""

    expression: str


NeighborLimit = Count | AllBut | SqlLimit


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
    value: Value


@dataclass(frozen=True)
class In:
    r"""The row's ``column`` equals one of ``values``."""

    column: str
    values: tuple[Value, ...]


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
    r"""One row per column, in order: ``table_index``, its position (an
    ``int32`` or ``int64``), then ``min_date`` and ``max_date``, its
    smallest and largest value, null where no row holds one, each of the
    column's own type: a timestamp of any unit or zone, a date, or text as
    a source with no time type stores it (read by
    ``Capabilities.text_times``)."""

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
    ``rows`` is the table's row count, where the core counted it
    (``Capabilities.sample_by_row_count``); ``key_dtype`` the schema's data
    type of a key of one column, where the source samples by it
    (``Capabilities.sample_by_key_type``)."""

    table: Relation
    key: tuple[str, ...]
    n: int
    seed: Value
    columns: tuple[str, ...]
    filters: tuple[Filter, ...]
    rows: int | None
    key_dtype: str | None = None
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
    :class:`~sdm_client.SnowflakeSource`) are sources like any other: each
    answers these requests with the library's own SQL for its database, and
    the core reaches every source the same way.

    A source holds live data access, so a source, and a schema over one,
    cannot be pickled. The library never closes it.
    """

    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        r"""What this source answers; read once at the start of each
        operation, before any request of it."""

    @abc.abstractmethod
    def answer(
        self, request: DataRequest, expected: pa.Schema | None
    ) -> ArrowStream:
        r"""Answers one request with every row it asks for.

        Args:
            request: The request.
            expected: The exact schema the answer must have, where the
                request fixes one (:class:`Describe`, :class:`KeyCounts`);
                otherwise ``None``, and the answer holds the columns the
                request names, in order, typed as the source stores them.

        Returns:
            Any object exporting an Arrow stream (:class:`ArrowStream`):
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
