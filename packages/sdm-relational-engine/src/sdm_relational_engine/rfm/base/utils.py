# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, cast

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
    rather than refused. ``predict`` returns ``ANCHOR_TIMESTAMP`` timezone-aware,
    which makes this the difference between a prediction output being usable as
    the next call's context and not.

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
