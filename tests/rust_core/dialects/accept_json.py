# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P6.json: JSON written into SQL parameters is Python's ``json.dumps``,
byte for byte, as each dialect calls it.

Generated values (seed 20261005) go through today's Python and through
``sdm_core._native.dialect_render``, and the output must be identical:

- PostgreSQL's and Databricks' ``dumps`` (compact, ``default`` taking a
  numpy scalar's ``item()`` and ``str`` of anything else) of lists of ints of
  any size, floats of every kind (``1e16``, subnormals, ``-0.0``, NaN and the
  infinities), Unicode text (``ensure_ascii``), nested lists, tuples and
  objects, numpy scalars, pandas timestamps to the nanosecond, decimals and
  dates;
- Snowflake's ``json.dumps`` (default separators, no ``default``) of plain
  JSON values, through ``rows_by_key``;
- every backend's ``holds_any`` over generated text;
- PostgreSQL's ``serialize_time`` (``pd.Timestamp.isoformat``) of times at
  second to nanosecond precision, naive or zoned, with and without ``utc``;
- DuckDB's and SQLite's ``_json_values`` (``pc.strftime`` text and more) of
  generated Arrow arrays.

A generated input today's Python refuses is dropped; at most 2% may be.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import struct
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from _dialects import BACKENDS, encode, mismatch, python_call, rust_call

SEED = 20261005

_ZONES = (
    None,
    'UTC',
    'America/New_York',
    'Asia/Kolkata',
    'Europe/London',
    '+05:30',
    '-03:00',
)

#: Values every run includes, whatever the generator draws.
MUST = (
    1e16,
    1e15,
    9999999999999998.0,
    1e22,
    1e-5,
    1e-7,
    0.0001,
    5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
    -0.0,
    0.1,
    1 / 3,
    float('nan'),
    float('inf'),
    float('-inf'),
    2**63,
    2**64,
    -(2**64) - 1,
    10**40,
    'é中\U0001f600 \x00"\\/\x7f﻿',
    [[1, [2.5, ['x']]], []],
)


# --- Generators ------------------------------------------------------------------


def _float(rng: random.Random) -> float:
    kind = rng.randrange(6)
    if kind == 0:
        return struct.unpack('<d', rng.getrandbits(64).to_bytes(8, 'little'))[0]
    if kind == 1:
        return rng.choice(MUST[:16])
    if kind == 2:
        return rng.uniform(-1, 1) * 10 ** rng.randint(-30, 30)
    if kind == 3:
        return float(rng.randint(-(10**17), 10**17))
    if kind == 4:
        return float(f'{rng.randint(1, 99)}e{rng.randint(-330, 306)}')
    return round(rng.uniform(-1000, 1000), rng.randint(0, 8))


def _int(rng: random.Random) -> int:
    bits = rng.choice((4, 8, 16, 31, 32, 53, 63, 64, 65, 100, 300))
    return rng.choice((1, -1)) * rng.getrandbits(bits)


def _char(rng: random.Random) -> str:
    kind = rng.randrange(8)
    if kind < 3:
        return chr(rng.randint(0x20, 0x7E))
    if kind == 3:
        return rng.choice('"\\/\b\f\n\r\t\x00\x01\x1f\x7f')
    if kind == 4:
        return chr(rng.randint(0x80, 0xFF))
    if kind == 5:
        code = rng.randint(0x100, 0xFFFD)
        return '?' if 0xD800 <= code <= 0xDFFF else chr(code)
    if kind == 6:
        return chr(rng.randint(0x10000, 0x10FFFF))
    return rng.choice('  ﻿ ́')


def _text(rng: random.Random) -> str:
    return ''.join(_char(rng) for _ in range(rng.randint(0, 12)))


def _timestamp(rng: random.Random) -> pd.Timestamp:
    zone = rng.choice(_ZONES)
    if zone is None:
        nanos = rng.randint(-(2**63) + 10**10, 2**63 - 10**10)
    else:
        nanos = rng.randint(0, 4102444800 * 10**9)
    unit = rng.choice(('s', 'ms', 'us', 'ns'))
    stamp = pd.Timestamp(nanos, unit='ns')
    if rng.random() < 0.5:
        stamp = stamp.floor(
            {'s': 's', 'ms': 'ms', 'us': 'us', 'ns': 'ns'}[unit]
        )
    if zone is not None:
        stamp = stamp.tz_localize('UTC').tz_convert(_tz(zone))
    return stamp.as_unit(unit)


def _tz(zone: str) -> Any:
    if zone[0] in '+-':
        sign = 1 if zone[0] == '+' else -1
        hours, minutes = int(zone[1:3]), int(zone[4:6])
        return dt.timezone(sign * dt.timedelta(hours=hours, minutes=minutes))
    return zone


def _decimal(rng: random.Random) -> Decimal:
    if rng.random() < 0.05:
        return Decimal(
            rng.choice(('NaN', 'Infinity', '-Infinity', '-0', '0E-7'))
        )
    return Decimal(f'{_int(rng) % 10**30}E{rng.randint(-40, 40)}')


def _numpy(rng: random.Random) -> Any:
    kind = rng.choice(
        (
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint32',
            'uint64',
            'float16',
            'float32',
            'float64',
            'bool',
        )
    )
    if kind == 'bool':
        return np.bool_(rng.random() < 0.5)
    if kind.startswith('float'):
        with np.errstate(over='ignore'):
            return np.dtype(kind).type(_float(rng))
    info = np.iinfo(kind)
    return np.dtype(kind).type(rng.randint(int(info.min), int(info.max)))


def plain(rng: random.Random, depth: int = 0) -> Any:
    r"""A value plain ``json.dumps`` writes."""
    kind = rng.randrange(9 if depth < 3 else 6)
    if kind == 0:
        return _int(rng)
    if kind == 1:
        return _float(rng)
    if kind == 2:
        return _text(rng)
    if kind == 3:
        return rng.choice((True, False, None))
    if kind == 4:
        return rng.randint(-1000, 1000)
    if kind == 5:
        return rng.choice(MUST)
    if kind == 6:
        return [plain(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    if kind == 7:
        return tuple(plain(rng, depth + 1) for _ in range(rng.randint(0, 3)))
    return {_text(rng): plain(rng, depth + 1) for _ in range(rng.randint(0, 3))}


def anything(rng: random.Random, depth: int = 0) -> Any:
    r"""A value ``dumps`` writes, through its ``default`` where JSON has
    none."""
    kind = rng.randrange(12)
    if kind < 6:
        return plain(rng, depth)
    if kind == 6:
        return _numpy(rng)
    if kind == 7:
        return _timestamp(rng)
    if kind == 8:
        return _decimal(rng)
    if kind == 9:
        return rng.choice(
            (
                dt.date(
                    rng.randint(1, 9999), rng.randint(1, 12), rng.randint(1, 28)
                ),
                dt.datetime(2024, 3, 1, 12, 30, 5, rng.randint(0, 999999)),
                pd.NaT,
            )
        )
    if kind == 10 and depth < 3:
        return [anything(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    if kind == 11 and depth < 3:
        return {
            _text(rng): anything(rng, depth + 1)
            for _ in range(rng.randint(0, 3))
        }
    return _int(rng)


def _array(rng: random.Random, backend: str) -> pa.Array:
    size = rng.randint(0, 8)
    kind = rng.randrange(10 if backend == 'duckdb' else 8)

    def maybe(value: Any) -> Any:
        return None if rng.random() < 0.15 else value

    if kind < 3:
        unit = rng.choice(('s', 'ms', 'us', 'ns'))
        zone = rng.choice(_ZONES)
        per = {'s': 1, 'ms': 10**3, 'us': 10**6, 'ns': 10**9}[unit]
        if zone is not None:
            low, high = 0, 4102444800 * per
        elif unit == 'ns':
            low, high = -(2**63) + 1, 2**63 - 1
        else:
            low, high = -62135596800 * per, 253402300799 * per
        storage = pa.array(
            [maybe(rng.randint(low, high)) for _ in range(size)], pa.int64()
        )
        return storage.view(pa.timestamp(unit, tz=zone))
    if kind == 3:
        storage = pa.array(
            [maybe(rng.randint(-719162, 2932896)) for _ in range(size)],
            pa.int32(),
        )
        return storage.view(pa.date32())
    if kind == 4:
        return pa.array(
            [maybe(_float(rng)) for _ in range(size)],
            rng.choice((pa.float32(), pa.float64())),
        )
    if kind == 5:
        return pa.array(
            [maybe(rng.randint(-(2**63), 2**63 - 1)) for _ in range(size)],
            pa.int64(),
        )
    if kind == 6:
        return pa.array(
            [maybe(_text(rng)) for _ in range(size)],
            rng.choice((pa.string(), pa.large_string())),
        )
    if kind == 7:
        return pa.array([maybe(rng.random() < 0.5) for _ in range(size)])
    if kind == 8:
        scale = rng.randint(0, 6)
        return pa.array(
            [
                maybe(Decimal(rng.randint(-(10**20), 10**20)).scaleb(-scale))
                for _ in range(size)
            ],
            pa.decimal128(38, scale),
        )
    storage = pa.array(
        [maybe(rng.randint(0, 86400 * 10**6 - 1)) for _ in range(size)],
        pa.int64(),
    )
    return storage.view(pa.time64('us'))


# --- The cases ---------------------------------------------------------------------


def cases() -> list[tuple[str, str, dict[str, Any]]]:
    r"""``(backend, function, args)`` of every generated call."""
    from predictive_query.schema.typing import Dtype

    rng = random.Random(SEED)
    out: list[tuple[str, str, dict[str, Any]]] = []
    musts = [list(MUST)] + [[value] for value in MUST]
    for i in range(6_000):
        obj = (
            musts[i]
            if i < len(musts)
            else [anything(rng) for _ in range(rng.randint(0, 6))]
        )
        for backend in ('postgres', 'databricks'):
            out.append((backend, 'dumps', {'obj': encode(obj)}))
    for i in range(3_000):
        ids = (
            musts[i]
            if i < len(musts)
            else [plain(rng) for _ in range(rng.randint(0, 6))]
        )
        args = {
            'table': '"T"',
            'projections': ['"X"'],
            'key': '"K"',
            'dtype': Dtype.int,
            'ids': ids,
            'order_by': [],
        }
        out.append(('snow', 'rows_by_key', encode(args)))
    for i in range(1_000):
        values = [_text(rng) for _ in range(rng.randint(0, 5))]
        args = {'table': 't', 'identity': 'k', 'values': values}
        out.append((BACKENDS[i % len(BACKENDS)], 'holds_any', encode(args)))
    for _ in range(4_000):
        value = _timestamp(rng) if rng.random() < 0.95 else pd.NaT
        args = {'value': value, 'utc': rng.random() < 0.5}
        out.append(('postgres', 'serialize_time', encode(args)))
    for i in range(3_000):
        backend = ('duckdb', 'sqlite')[i % 2]
        out.append(
            (backend, '_json_values', encode({'values': _array(rng, backend)}))
        )
    return out


def _walk(value: Any, depth: int = 0) -> Any:
    r"""Every encoded value inside ``value``, with its nesting depth."""
    yield value, depth
    if isinstance(value, list):
        for item in value:
            yield from _walk(item, depth + 1)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk(item, depth + 1)


def _covers(generated: list[tuple[str, str, dict[str, Any]]]) -> None:
    r"""The generated inputs hold what P6.json names, plenty of each."""
    seen = {
        'big ints': 0,
        '1e16': 0,
        'subnormals': 0,
        'nanosecond timestamps': 0,
        'microsecond timestamps': 0,
        'nanosecond arrays': 0,
        'astral text': 0,
        'nested lists': 0,
    }
    for _, _, args in generated:
        for value, depth in _walk(args):
            if isinstance(value, float):
                seen['1e16'] += value == 1e16
                seen['subnormals'] += 0 < abs(value) < 2.2250738585072014e-308
            elif isinstance(value, str):
                seen['astral text'] += any(ord(c) > 0xFFFF for c in value)
            elif isinstance(value, list):
                seen['nested lists'] += depth >= 4
            elif isinstance(value, dict) and len(value) == 1:
                ((tag, body),) = value.items()
                if tag == '$int':
                    seen['big ints'] += 1
                elif tag == '$timestamp':
                    fraction = body[19:].split('+')[0].split('-')[0]
                    seen['nanosecond timestamps'] += len(fraction) == 10
                    seen['microsecond timestamps'] += len(fraction) == 7
                elif tag == '$arrow':
                    seen['nanosecond arrays'] += body['type'].startswith(
                        'timestamp[ns'
                    )
    scarce = {what: n for what, n in seen.items() if n < 50}
    assert not scarce, f'too few generated inputs hold {scarce}'


def test_json_matches_python() -> None:
    generated = cases()
    _covers(generated)
    failures, dropped, checked = [], 0, 0
    for backend, function, args in generated:
        try:
            want = python_call(backend, function, args)
        except Exception:  # noqa: BLE001 - an input today's Python refuses
            dropped += 1
            continue
        checked += 1
        try:
            got = rust_call(backend, function, args)
        except Exception as error:  # noqa: BLE001 - reported per case
            failures.append(
                f'{backend} {function} {args}: raised {error!r}'[:600]
            )
            continue
        problem = mismatch(want, got)
        if problem:
            failures.append(
                f'{backend} {function} {str(args)[:300]}: {problem}'
            )
    assert dropped <= math.ceil(0.02 * len(generated)), (
        f"today's Python refused {dropped} of {len(generated)} inputs"
    )
    assert not failures, (
        f'{len(failures)} of {checked} generated calls differ:\n'
        + '\n'.join(failures[:30])
    )
