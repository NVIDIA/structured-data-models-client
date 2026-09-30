# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Literal, cast

import pandas as pd
import pyarrow as pa


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
