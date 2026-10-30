# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inert facts for exact civil/instant index owners, never comparison policy."""

from datetime import date, datetime, timedelta, timezone, tzinfo
from inspect import getattr_static
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .._routing import RouteError


def _micros(value: timedelta) -> int:
    return (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds


def _pytz_offset(zone: Any) -> int:
    # pytz is optional on newer pandas. Only its actual standard class shapes
    # enter this branch; module names are not an admission criterion.
    try:
        import pytz
        from pytz.tzinfo import DstTzInfo, StaticTzInfo
    except ImportError as error:
        raise RouteError(
            'this temporal timezone owner is not represented'
        ) from error
    cls = type(zone)
    if type(cls) is not type:
        raise RouteError('custom timezone metaclasses are not represented')
    base: type[Any]
    field: str | None
    if cls is pytz.UTC.__class__:
        base, field = cls, None
    elif cls is vars(pytz)['_FixedOffset']:
        base, field = cls, '_offset'
    elif cls.__bases__ == (DstTzInfo,):
        base, field = DstTzInfo, '_utcoffset'
    elif cls.__bases__ == (StaticTzInfo,):
        base, field = StaticTzInfo, '_utcoffset'
    else:
        raise RouteError('this temporal timezone owner is not represented')
    if getattr_static(zone, '__getattribute__') is not tzinfo.__getattribute__:
        raise RouteError('custom timezone attribute access is not represented')
    for name in ('utcoffset', 'dst'):
        if getattr_static(zone, name) is not base.__dict__[name]:
            raise RouteError('custom timezone methods are not represented')
    if field is None:
        return 0
    offset = getattr_static(zone, field)
    if type(offset) is not timedelta:
        raise RouteError('custom timezone offset owners are not represented')
    return _micros(offset)


def _offsets(value: datetime, zone: Any) -> tuple[int, int]:
    if type(zone) in (timezone, ZoneInfo):
        first = value.utcoffset()
        second = value.replace(fold=1 - value.fold).utcoffset()
        if type(first) is not timedelta or type(second) is not timedelta:
            raise RouteError('this temporal offset is not represented')
        return _micros(first), _micros(second)
    offset = _pytz_offset(zone)
    return offset, offset


def observe(values: Any, missing: list[bool]) -> list[dict[str, Any]]:
    identities: dict[int, int] = {}
    atoms: list[dict[str, Any]] = []
    for value, absent in zip(values, missing, strict=True):
        if absent:
            atoms.append({'kind': 'integer', 'value': '0'})
            continue
        owner = type(value)
        if owner is date:
            atoms.append({'kind': 'date', 'ordinal': value.toordinal()})
            continue
        if owner not in (datetime, pd.Timestamp):
            raise RouteError(
                'this mixed temporal comparison family is not yet represented by the core'
            )
        zone = value.tzinfo
        identity = (
            None
            if zone is None
            else identities.setdefault(id(zone), len(identities) + 1)
        )
        micros = (
            value.hour * 3600 + value.minute * 60 + value.second
        ) * 1_000_000 + value.microsecond
        if owner is pd.Timestamp:
            year = value.year
            alternate = None
            if zone is not None:
                if type(zone) not in (timezone, ZoneInfo):
                    # Validate the native owner without invoking descriptors or
                    # asking a wide-year Timestamp for utcoffset/toordinal.
                    alternate = _pytz_offset(zone)
                elif 1 <= year <= 9999:
                    # Constructing this exact civil owner is inert. Timestamp
                    # replace would re-localize an ambiguous wall time.
                    civil = datetime(
                        year,
                        value.month,
                        value.day,
                        value.hour,
                        value.minute,
                        value.second,
                        value.microsecond,
                        tzinfo=zone,
                        fold=value.fold,
                    )
                    _, alternate = _offsets(civil, zone)
            atoms.append(
                {
                    'kind': 'timestamp',
                    'ticks': int(value.asm8.view('i8')),
                    'unit': {
                        's': 'second',
                        'ms': 'millisecond',
                        'us': 'microsecond',
                        'ns': 'nanosecond',
                    }[value.unit],
                    'year': year,
                    'month': value.month,
                    'day': value.day,
                    'micros': micros,
                    'nanosecond': value.nanosecond,
                    'timezone': identity,
                    'alternate_offset_micros': alternate,
                }
            )
        elif zone is None or type(zone) is timezone:
            offset = value.utcoffset()
            atoms.append(
                {
                    'kind': 'date_time',
                    'ordinal': value.toordinal(),
                    'micros': micros,
                    'offset_micros': None
                    if offset is None
                    else _micros(offset),
                }
            )
        else:
            offset, alternate = _offsets(value, zone)
            atoms.append(
                {
                    'kind': 'zoned_date_time',
                    'ordinal': value.toordinal(),
                    'micros': micros,
                    'timezone': identity,
                    'offset_micros': offset,
                    'alternate_offset_micros': alternate,
                }
            )
    return atoms
