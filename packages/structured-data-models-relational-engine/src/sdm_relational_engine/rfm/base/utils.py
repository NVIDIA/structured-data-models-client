# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.schema.typing import Dtype

from sdm_relational_engine.rfm.base.source import SourceColumn


def Timestamp(*args: Any, **kwargs: Any) -> pd.Timestamp:
    r"""Constructs a timezone-naive :class:`pandas.Timestamp` (UTC), raising
    on ``NaT``.
    """
    result = pd.Timestamp(*args, **kwargs)
    if result is pd.NaT:
        raise TypeError('Expected pd.Timestamp, got NaT')
    if result.tzinfo is not None:
        result = result.tz_convert('UTC').tz_localize(None)
    return cast(pd.Timestamp, result)


def Timedelta(*args: Any, **kwargs: Any) -> pd.Timedelta:
    r"""Constructs a :class:`pandas.Timedelta`, raising on ``NaT``."""
    result = pd.Timedelta(*args, **kwargs)
    if result is pd.NaT:
        raise TypeError('Expected pd.Timedelta, got NaT')
    return cast(pd.Timedelta, result)


def is_datetime(ser: pd.Series) -> bool:
    r"""Check whether a :class:`pandas.Series` holds datetime values."""
    if isinstance(ser.dtype, pd.ArrowDtype):
        dtype = ser.dtype.pyarrow_dtype
        return (
            pa.types.is_timestamp(dtype)
            or pa.types.is_date(dtype)
            or pa.types.is_time(dtype)
        )

    return pd.api.types.is_datetime64_any_dtype(ser)


def to_naive_utc(value: Any) -> Any:
    r"""Strips a timezone by converting to UTC first, never by discarding it.

    Anchor timestamps are compared against, and joined to, the graph's own
    timezone-naive UTC timestamps, so a timezone-aware one has to be converted
    rather than refused. ``predict`` returns ``ANCHOR_TIMESTAMP`` with a
    timezone. This conversion makes that output usable as the next call's
    context.

    Anything that is not a timezone-aware timestamp or series is returned
    unchanged.
    """
    if isinstance(value, pd.Series):
        if isinstance(value.dtype, pd.DatetimeTZDtype):
            return value.dt.tz_convert('UTC').dt.tz_localize(None)
        return value
    if isinstance(value, pd.Timestamp) and value.tzinfo is not None:
        return value.tz_convert('UTC').tz_localize(None)
    return value


def to_datetime(ser: pd.Series, column_name: str | None = None) -> pd.Series:
    r"""Converts a :class:`pandas.Series` to ``datetime64[ns]`` format.

    Unparseable values become ``NaT``. Pass ``column_name`` to warn about them
    instead of dropping them silently; leave it unset when probing candidate
    columns, where failing to parse is an expected outcome.
    """
    if isinstance(ser.dtype, pd.ArrowDtype):
        ser = pd.Series(ser.to_numpy(), index=ser.index, name=ser.name)

    if not pd.api.types.is_datetime64_any_dtype(ser):
        source = ser
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Could not infer format',
            )
            ser = pd.to_datetime(ser, unit='ns', errors='coerce')
        if column_name is not None:
            coerced = ser.isna() & source.notna()
            num_coerced = int(coerced.sum())
            if num_coerced > 0:
                examples = source[coerced].unique()[:3].tolist()
                warnings.warn(
                    f'Could not parse {num_coerced:,} of {len(source):,} '
                    f"value(s) of time column '{column_name}' (for example "
                    f"{examples}); they were dropped as 'NaT'",
                    stacklevel=2,
                )

    if isinstance(ser.dtype, pd.DatetimeTZDtype):
        ser = ser.dt.tz_convert('UTC').dt.tz_localize(None)

    if ser.dtype != 'datetime64[ns]':
        ser = ser.astype('datetime64[ns]')

    return ser


_MIN_DAYS_IN_MONTH = 28
_MAX_DAYS_IN_MONTH = 31
_MIN_DAYS_IN_YEAR = 365
_MAX_DAYS_IN_YEAR = 366
_SECONDS_IN_MINUTE = 60
_SECONDS_IN_HOUR = 60 * _SECONDS_IN_MINUTE
_SECONDS_IN_DAY = 24 * _SECONDS_IN_HOUR

TimeBound = Literal['earliest', 'latest']


def _elastic_days(
    scaled_value: int,
    bound: TimeBound,
    shortest: int,
    longest: int,
) -> int:
    r"""Pick the day count that pushes an elastic unit towards :obj:`bound`.

    A negative offset reaches further back the longer its unit is, so the sign
    decides which of the two lengths lands on the requested side.
    """
    if (scaled_value > 0) == (bound == 'latest'):
        return longest
    return shortest


def date_offset_to_seconds(
    offset: pd.DateOffset,
    bound: TimeBound = 'latest',
) -> int:
    r"""Convert a :class:`pandas.DateOffset` into a number of seconds.

    Months and years have no fixed length, so the result can only bracket the
    calendar offset: :obj:`'latest'` returns the most seconds the offset can
    ever span and :obj:`'earliest'` the fewest.

    .. note::
        The sampler uses this to pre-select candidate rows, and label
        computation then re-applies the exact calendar bounds to what it is
        given. A bound that lands inside the calendar window therefore drops
        rows that no later step can recover, so widen both ends: a window's
        start asks for :obj:`'earliest'` and its end for :obj:`'latest'`.
    """
    total_sec = 0
    multiplier = getattr(offset, 'n', 1)  # The multiplier (if present).

    for attr, value in offset.__dict__.items():
        if value is None or value == 0:
            continue
        scaled_value = value * multiplier
        if attr == 'years':
            days = _elastic_days(
                scaled_value, bound, _MIN_DAYS_IN_YEAR, _MAX_DAYS_IN_YEAR
            )
            total_sec += scaled_value * days * _SECONDS_IN_DAY
        elif attr == 'months':
            days = _elastic_days(
                scaled_value, bound, _MIN_DAYS_IN_MONTH, _MAX_DAYS_IN_MONTH
            )
            total_sec += scaled_value * days * _SECONDS_IN_DAY
        elif attr == 'weeks':
            total_sec += scaled_value * 7 * _SECONDS_IN_DAY
        elif attr == 'days':
            total_sec += scaled_value * _SECONDS_IN_DAY
        elif attr == 'hours':
            total_sec += scaled_value * _SECONDS_IN_HOUR
        elif attr == 'minutes':
            total_sec += scaled_value * _SECONDS_IN_MINUTE
        elif attr == 'seconds':
            total_sec += scaled_value

    return total_sec


_SEED_WORD = 2**32


def random_state(random_seed: int | None) -> np.random.RandomState:
    r"""The random stream a ``random_seed`` stands for.

    All client-side randomness draws from numpy's legacy MT19937 generator,
    :class:`numpy.random.RandomState`, whose streams NEP 19 freezes: a seed
    gives the same numbers in every numpy release, and the generator is the
    one the C++ neighbor sampler uses. These are all the streams, and the
    seeds they start from:

    * The example shuffle in :meth:`Sampler.sample_target`: one stream of
      ``random_seed`` per call, shuffling the test candidates first and then,
      when they are drawn separately, the training candidates.
    * The entity-table sample of the in-memory backend: a stream of
      ``random_seed`` of its own, handed to :meth:`pandas.DataFrame.sample`.
    * The seeded SQLite entity-table sample: the coefficients of the hash it
      orders rows by, drawn from a stream of ``random_seed`` of its own.
    * The tabular context sample in ``sdm_client``, which does not depend on
      this package: ``RandomState(42)``, its ``_CONTEXT_SEED``.

    The C++ neighbor sampler seeds its own ``std::mt19937`` the way this
    function seeds numpy's generator, so for any seed it starts from the state
    ``random_state(random_seed)`` holds, and the other SQL backends hand the
    seed to the database:
    DuckDB and Databricks to ``REPEATABLE``, Postgres to ``setseed``,
    Databricks and Snowflake to a hash of the seed and the key that picks the
    rows kept. None of these draws from a stream here.

    A seed below ``2**32`` seeds the stream directly, as
    ``RandomState(random_seed)``. A larger one, which ``RandomState`` cannot
    take as a number, seeds it with its 32-bit words, least significant
    first. ``None`` draws a fresh seed from the operating system.

    Args:
        random_seed: The seed, a whole number from 0 upwards, or ``None``.
    """
    if random_seed is None or random_seed < _SEED_WORD:
        return np.random.RandomState(random_seed)
    words: list[int] = []
    rest = int(random_seed)
    while rest > 0:
        words.append(rest % _SEED_WORD)
        rest //= _SEED_WORD
    return np.random.RandomState(np.array(words, dtype=np.uint32))


def ordered_sample_sql(
    source: str,
    projections: Sequence[str],
    order_terms: Sequence[str],
    num_rows: int,
) -> str:
    r"""Selects ``projections`` of the first ``num_rows`` rows of ``source``,
    in the order ``order_terms`` give.

    A bare ``LIMIT`` keeps whichever rows the warehouse reads first, which on
    Databricks and Snowflake depends on which files finish first, and those
    rows decide the type each column is inferred as. The terms are selected
    ahead of the projections and ordered by position, so a projection aliased
    like a source column cannot take that column's place in the order. A
    sample of expressions then holds the rows of the sample of columns, in
    the same order, as the two are joined by position. The caller drops the
    first ``len(order_terms)`` columns of the result.

    Args:
        source: What the rows are read from.
        projections: What is read of each row.
        order_terms: The terms that order the rows, if any.
        num_rows: How many rows to read.
    """
    if len(order_terms) == 0:
        return f'SELECT {", ".join(projections)} FROM {source} LIMIT {num_rows}'
    selected = [
        *(
            f'{term} AS __kumo_order_{i}__'
            for i, term in enumerate(order_terms)
        ),
        *projections,
    ]
    positions = ', '.join(str(i + 1) for i in range(len(order_terms)))
    return (
        f'SELECT {", ".join(selected)} FROM {source} '
        f'ORDER BY {positions} LIMIT {num_rows}'
    )


#: The largest fraction a block prefilter is applied at: below an eighth of
#: the table it saves less than the scan it replaces is worth risking.
MAX_PREFILTER_FRACTION = Fraction(1, 8)


def prefilter_fraction(wanted: Fraction) -> Fraction | None:
    r"""The fraction of a table's blocks a repeatable block sample keeps, so
    that it holds at least ``wanted`` of them, or ``None`` where reading the
    whole table is cheap enough.

    The fraction is ``wanted`` rounded up to a power of two: its percentage
    is then an exact decimal every warehouse reads as written, and a table
    growing or shrinking a little keeps the same one. Above
    :data:`MAX_PREFILTER_FRACTION` no prefilter applies.

    Args:
        wanted: The least fraction of the table the sample must read.
    """
    if wanted <= 0:
        return None
    exponent = math.ceil(math.log2(wanted))
    # `log2` of a fraction is a float; settle the boundary exactly.
    while Fraction(2) ** (exponent - 1) >= wanted:
        exponent -= 1
    while Fraction(2) ** exponent < wanted:
        exponent += 1
    fraction = Fraction(2) ** exponent
    if fraction > MAX_PREFILTER_FRACTION:
        return None
    return fraction


def percent_literal(fraction: Fraction) -> str:
    r"""``fraction`` of a table as the percentage a ``SAMPLE`` clause takes,
    such as ``1.5625``: exact, as a power of two times 100 is.
    """
    return repr(100 * float(fraction))


def key_count_select(
    label: int,
    source: str,
    columns: Sequence[str],
    unique: bool,
    count_distinct: str,
) -> str:
    r"""One ``SELECT`` of a key count's ``UNION ALL``: the rows of
    ``source``, the distinct values of its key, and the rows missing part of
    the key.

    Args:
        label: The table's position in the count, selected as ``k``.
        source: What the rows are read from.
        columns: The key's columns, quoted.
        unique: Whether the source keeps the key unique, so that only the
            nulls cost anything to count.
        count_distinct: The dialect's count of the key's distinct values,
            leaving out a row missing any part.
    """
    missing = ' OR '.join(f'{column} IS NULL' for column in columns)
    nulls = f'COUNT(CASE WHEN {missing} THEN 1 END)'
    distinct = f'COUNT(*) - {nulls}' if unique else count_distinct
    return (
        f'SELECT {label} AS k, COUNT(*) AS n, {distinct} AS d, '
        f'{nulls} AS z\nFROM {source}'
    )


def count_distinct_values(columns: Sequence[str]) -> str:
    r"""The distinct values of a key, leaving out a row missing any part, as
    Spark and Snowflake count several columns: skipping such a row
    themselves.
    """
    return f'COUNT(DISTINCT {", ".join(columns)})'


def repeated_key_select(source: str, columns: Sequence[str]) -> str:
    r"""One value of the key ``columns`` that more than one row of ``source``
    holds.
    """
    quoted = ', '.join(columns)
    present = ' AND '.join(f'{column} IS NOT NULL' for column in columns)
    return (
        f'SELECT {quoted}\nFROM {source}\n'
        f'WHERE {present}\nGROUP BY {quoted}\nHAVING COUNT(*) > 1\nLIMIT 1'
    )


#: How many columns the inference sample of a table without a declared
#: primary key hashes, at most.
_NUM_SAMPLE_HASH_COLUMNS = 8


def sample_order_terms(
    columns: Sequence[SourceColumn],
    primary_key: str | None,
    quote: Callable[[str], str],
    hash_function: str,
) -> list[str]:
    r"""The terms that pick and order the rows a warehouse table's types are
    inferred from, for :func:`ordered_sample_sql`.

    The rows are those a hash picks, so they are spread over the table rather
    than taken from wherever it was read first, then ordered by every scalar
    column, so the order is total. A declared primary key is the only column
    hashed, then ordered by. Without one, at most
    ``_NUM_SAMPLE_HASH_COLUMNS`` columns are hashed, numbers, booleans and
    times before strings and binary values, as they are cheaper to hash.
    Lists, unsupported and unknown types are neither hashed nor ordered by,
    since not every warehouse hashes or orders them.

    Any such order still reads every row of the projected columns: only an
    order tied to the files read could stop early, and that is not the same
    on every run. Hashing fewer columns makes that read cheaper to order.

    Args:
        columns: The table's source columns, in source order.
        primary_key: The column the source declares as primary key, if any.
        quote: Quotes a column name for the warehouse.
        hash_function: The warehouse's hash of several values, such as
            ``xxhash64`` or ``HASH``.
    """
    scalars = [
        column
        for column in columns
        if column.dtype is not None
        and column.dtype != Dtype.unsupported
        and not column.dtype.is_list()
    ]
    if len(scalars) == 0:
        return []
    refs = [quote(column.name) for column in scalars]
    names = [column.name for column in scalars]
    if primary_key is not None and primary_key in names:
        key = refs[names.index(primary_key)]
        # The source may not enforce the key, so rows sharing one are still
        # ordered by the rest:
        rest = [ref for ref in refs if ref != key]
        return [f'{hash_function}({key})', key, *rest]
    by_cost = sorted(
        range(len(scalars)),
        key=lambda i: (cast(Dtype, scalars[i].dtype).is_string(), i),
    )
    hashed = sorted(by_cost[:_NUM_SAMPLE_HASH_COLUMNS])
    return [
        f'{hash_function}({", ".join(refs[i] for i in hashed)})',
        *refs,
    ]
