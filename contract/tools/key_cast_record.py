#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's cast of the entities to the key's type into
``contract/key-cast-vectors.json`` (T1.32a's rule-table vectors).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/key_cast_record.py

Written by T1.32a's independent test author before the implementation.
Each vector is one key type crossed with one list of entities:

- the key type is ``contract/bases.yaml``'s ``shop`` with ``users.user_id``
  and ``orders.user_id`` changed to a DuckDB type (``KEYS``), the type the
  source hands over as the key's Arrow type;
- the entities are Python values a caller passes as ``indices`` (``INPUTS``
  and each key's own ``extra``): integers in and out of every width's
  range, integral and non-integral floats, digit and other text, mixed
  lists, booleans, None, NaN and ``pd.NA``, NumPy scalars, decimals,
  timestamps and dates, bytes.

For each, the reference runs ``Schema.from_source`` and
``PredictiveQuery.materialize(indices=..., anchor_time=2024-03-01)`` and
this script keeps:

- ``reference.public``: what the public call gave, a success with the
  entities as the payload's predicted rows carry them, or the refusal's
  class, ``StructuredDataError.code`` and stage (no message text, H17);
  this is what both hosts are held to;
- ``reference.cast``: the reference's own cast where the engine makes it
  (``rfm.py``'s ``pd.Series(indices, dtype=train_pkey.dtype)``), observed
  in place: the key's dtype and the values cast, or the exception it
  raised. It documents the rule table; no host is compared on it;
- ``host_arrow``: the entities as one Arrow column by the host's natural
  inference (``pa.array(values)``) or, when that fails (mixed kinds, an
  integer past int64), as text: each value's text (``host_text``: a float
  as a float entity alone is written, T1.32g's rule, in the reference's
  text since T1.36, pyarrow's: ``1e16`` as ``1e+16``, ``1.5e-07`` as
  ``1.5e-7``, ``1.0`` as ``1``; any other value its ``str()``), None, NaN,
  ``pd.NA`` and ``pd.NaT`` as null, a column of them alone Arrow's null
  type (design, "Host-held semantics", the ledger's row for this cast).
  ``rust-host`` hands the core this column; ``python-core`` hands the
  bridge the Python values themselves;
- ``register``: whether a host may differ from the reference only under
  a register entry naming the vector and the transition it makes
  (``REGISTER_REF``), why, and the disposition it is held to
  (``derive_marks``). It is derived from the recorded answers, never
  chosen: the reference fails with ``INTERNAL_ERROR`` (E66(b)'s class, a
  clearer answer is a change); it accepts an entity by changing its kind
  (text read as a number, a number or a time written as text, a boolean
  or a time read as a number) or its value (``1.5`` cast to ``1``); the
  host hands the core the same column as another vector the reference
  answers otherwise; a derived expectation is not the reference's answer
  (disposition ``expected``); or the host hands a mix of kinds over as
  text the reference answers otherwise given as text (``as_text``, the
  reference's answer to that text recorded beside it, disposition
  ``as_text``). Every other vector is held to the reference exactly. This
  marks where the core is expected to differ; it decides no policy;
- ``expected`` (round 2, ``ROUND_2`` and the decimal keys' exponents):
  where the rule is the core's own, derived from the exact value by
  integer arithmetic, never through a double: a text or decimal entity of
  a float key is the nearest float of the key's width, ties to even, past
  the largest refused by name; an exponent past any key and a decimal no
  key holds are refused by name; a 38-digit decimal is held exactly.
  Integer entities of float keys are the reference's, recorded.

Nothing here is written by hand but the inputs: every expected value is
the reference's or derived here. Decimals are written from their digits,
never through the caller's decimal context (``unscaled``). ``observe``, ``decode`` and ``canonical`` are imported by
``contract/tests/test_t132a_key_cast.py``, so both routes are observed by
the same code.
"""

from __future__ import annotations

import base64
import json
import math
import platform
import struct
import subprocess
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/key-cast-vectors.json'
FORMAT = 'sdm-key-cast-vectors/1'
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ANCHOR = '2024-03-01'
#: How a register entry names the vectors it covers, in its ``refs``: this
#: prefix and a vector's id, or a glob of ids (``fnmatch``), for example
#: ``contract/key-cast-vectors.json#decimal128-18-0.*``.
REGISTER_REF = 'contract/key-cast-vectors.json#'


def _alter(sql_type: str, using: str | None = None) -> list[str]:
    tail = f' USING {using}' if using else ''
    return [
        f'ALTER TABLE users ALTER user_id TYPE {sql_type}'
        + tail.replace('X', 'user_id'),
        f'ALTER TABLE orders ALTER user_id TYPE {sql_type}'
        + tail.replace('X', 'user_id'),
    ]


#: The key types: ``shop``'s ``user_id`` (0 to 7, BIGINT) changed to each
#: DuckDB type a key may have in the reference, the Arrow type it hands
#: over named by ``id``. Each ``extra`` holds entities of the table's own
#: values in that type's natural Python form. DATE, TIMESTAMP and BOOLEAN
#: are not here: the reference refuses them as keys before any cast
#: (``Schema.from_source``: "incompatible semantic type").
KEYS: list[dict[str, Any]] = [
    {'id': 'int8', 'setup': _alter('TINYINT')},
    {'id': 'int16', 'setup': _alter('SMALLINT')},
    {'id': 'int32', 'setup': _alter('INTEGER')},
    {'id': 'int64', 'setup': []},
    {'id': 'uint8', 'setup': _alter('UTINYINT')},
    {'id': 'uint16', 'setup': _alter('USMALLINT')},
    {'id': 'uint32', 'setup': _alter('UINTEGER')},
    {'id': 'uint64', 'setup': _alter('UBIGINT')},
    {
        'id': 'uint64-past-int64',
        'setup': _alter('UBIGINT', 'X::UBIGINT + 9223372036854775808::UBIGINT'),
        'extra': {
            'own.past-int64': [2**63 + 1, 2**63 + 2],
            'own.past-int64-text': [str(2**63 + 1)],
            'own.past-int64-float': [float(2**63)],
        },
    },
    {'id': 'float32', 'setup': _alter('FLOAT')},
    {
        'id': 'float64',
        'setup': _alter('DOUBLE'),
        'extra': {'own.half-key': [0.5]},
    },
    {'id': 'utf8', 'setup': _alter('VARCHAR', 'X::VARCHAR')},
    {
        'id': 'large-utf8',
        'setup': [
            'SET arrow_large_buffer_size = true',
            *_alter('VARCHAR', 'X::VARCHAR'),
        ],
    },
    {
        'id': 'decimal128-18-0',
        'setup': _alter('DECIMAL(18, 0)'),
        'derived': lambda: _exponent_vectors(),
    },
    {
        'id': 'decimal128-38-10',
        'setup': _alter('DECIMAL(38, 10)', 'X + 0.5'),
        'extra': {
            'own.decimal': [Decimal('1.5')],
            'own.text': ['1.5'],
            'own.float': [1.5],
        },
        'derived': lambda: _exponent_vectors(),
    },
    {'id': 'decimal128-38-0', 'setup': _alter('HUGEINT')},
]


# --- the vectors whose expectation is derived, not recorded -----------------
#
# Round 2 (the reviews of T1.32a part 1): float keys whose table holds the
# neighbours a rounded entity could land on, decimal exponents at the
# extremes, and decimals 38 and 76 digits wide. The reference is recorded
# for each as for every vector, and where the rule is the core's own (a
# text or decimal entity of a float key takes the nearest float of the
# key's width; an exponent past any key is refused by name; a wide decimal
# key holds its entity exactly) the vector also carries ``expected``,
# derived here from the exact value by integer arithmetic, never through a
# double. A host is held to ``expected`` first; where it differs from the
# reference's answer, a register entry must name the transition.

F32_MAX = Fraction((2**24 - 1) * 2**104)
F64_MAX = Fraction((2**53 - 1) * 2**971)


def nearest(q: Fraction, bits: int) -> Fraction | None:
    r"""The float of ``bits`` (24 for float32, 53 for float64) nearest the
    exact ``q``, ties to even, or None past the largest finite one. Normal
    numbers only (the vectors hold none smaller)."""
    if q == 0:
        return Fraction(0)
    magnitude = abs(q)
    e = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** e > magnitude:
        e -= 1
    while Fraction(2) ** (e + 1) <= magnitude:
        e += 1
    ulp = Fraction(2) ** (e - bits + 1)
    rounded = round(magnitude / ulp) * ulp  # Fraction rounds ties to even
    if rounded > (F32_MAX if bits == 24 else F64_MAX):
        return None
    return rounded if q > 0 else -rounded


def float_bits(q: Fraction, bits: int) -> str:
    r"""The IEEE-754 bits of a value ``nearest`` gave, in hex (exact: the
    value is a float of that width, so a double holds it)."""
    return struct.pack('>f' if bits == 24 else '>d', float(q)).hex()


def exact_text(q: Fraction) -> str:
    r"""A dyadic rational's exact decimal expansion."""
    k = q.denominator.bit_length() - 1
    if q.denominator != 1 << k:
        raise ValueError('not dyadic')
    digits = str(abs(q.numerator) * 5**k).rjust(k + 1, '0')
    whole, fraction = (digits[:-k], digits[-k:]) if k else (digits, '')
    text = whole + ('.' + fraction.rstrip('0') if fraction.rstrip('0') else '')
    return ('-' if q < 0 else '') + text


def decimal_text(text: str) -> str:
    r"""A plain decimal's text normalized: no sign on zero, no leading
    zeros, no trailing fractional zeros, no point left alone."""
    text = text.strip()
    sign = '-' if text.startswith('-') else ''
    text = text.lstrip('+-')
    whole, _, fraction = text.partition('.')
    whole = whole.lstrip('0') or '0'
    fraction = fraction.rstrip('0')
    if whole == '0' and not fraction:
        return '0'
    return sign + whole + ('.' + fraction if fraction else '')


def _floats(values: list[Fraction], kind: str) -> list[str]:
    r"""The ``CASE`` giving ``user_id`` 0 to 7 these exact float values."""
    whens = ' '.join(
        f"WHEN {i} THEN '{exact_text(v)}'::DOUBLE" for i, v in enumerate(values)
    )
    return _alter(kind, f'CASE X {whens} END')


def _float_vectors(
    pairs: list[tuple[Fraction, Fraction, Fraction]], bits: int, signs: bool
) -> dict[str, tuple[list[Any], dict[str, Any]]]:
    r"""Text and decimal entities just above, below and at the midpoint of
    each neighbouring pair ``(lo, hi)``, ``tiny`` apart, both signs, each
    expecting the nearest float of the width (``nearest``)."""
    out: dict[str, tuple[list[Any], dict[str, Any]]] = {}
    for lo, hi, tiny in pairs:
        mid = (lo + hi) / 2
        label = f'{exact_text(lo).split(".")[0]}'
        for sign in (1, -1) if signs else (1,):
            for where, value in (
                ('above', mid + tiny),
                ('below', mid - tiny),
                ('midpoint', mid),
            ):
                exact = sign * value
                text = _plain(exact)
                expected = expect_float(exact, bits)
                tag = f'{"minus-" if sign < 0 else ""}{label}.{where}'
                out[f'near.{tag}.text'] = ([text], expected)
                out[f'near.{tag}.decimal'] = ([Decimal(text)], expected)
    return out


def _plain(q: Fraction) -> str:
    r"""A rational with a terminating decimal expansion, as plain text."""
    n, d = q.numerator, q.denominator
    twos = fives = 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        raise ValueError('no terminating expansion')
    k = max(twos, fives)
    digits = str(abs(n) * (10**k // q.denominator)).rjust(k + 1, '0')
    whole, fraction = (digits[:-k], digits[-k:]) if k else (digits, '')
    fraction = fraction.rstrip('0')
    return ('-' if q < 0 else '') + whole + ('.' + fraction if fraction else '')


def expect_float(exact: Fraction, bits: int) -> dict[str, Any]:
    r"""What a float key of ``bits`` answers an entity of value ``exact``:
    its nearest float, or a refusal by name past the largest."""
    rounded = nearest(exact, bits)
    if rounded is None:
        return {'outcome': 'refusal', 'code': 'INVALID_REQUEST', 'named': True}
    kind = 'f32' if bits == 24 else 'f64'
    return {
        'outcome': 'success',
        'entities': [[kind, float_bits(rounded, bits)]],
    }


def expect_decimal(text: str) -> dict[str, Any]:
    return {'outcome': 'success', 'entities': [['dec', decimal_text(text)]]}


REFUSED = {'outcome': 'refusal', 'code': 'INVALID_REQUEST', 'named': True}
_ONE = Fraction(1)
_E23 = Fraction(1, 2**23)
_NEAR_ONE = [_ONE, _ONE + _E23, Fraction(2**24), Fraction(2**24 + 2)]
_LARGE = [
    Fraction(2**60),
    Fraction(2**60 + 2**37),
    F32_MAX,
    F32_MAX - 2**104,
]
_OVER = F32_MAX + 2**103
_EXPONENTS = [
    '100e2147483647',
    '-100e2147483647',
    '1.000e2147483647',
    '1e2147483648',
    '100e-2147483648',
    '1e-2147483648',
    '1e-2147483649',
    '10e99999999999999999999',
    '1e-99999999999999999999',
]
WIDE = 12345678901234567890123456789012345670


def _exponent_vectors() -> dict[str, tuple[list[Any], dict[str, Any]]]:
    out: dict[str, tuple[list[Any], dict[str, Any]]] = {}
    for i, text in enumerate(_EXPONENTS):
        out[f'exponent.{i}.text'] = ([text], REFUSED)
    out['exponent.huge.decimal'] = ([Decimal('100E+2147483647')], REFUSED)
    out['exponent.tiny.decimal'] = ([Decimal('1E-2147483648')], REFUSED)
    return out


def _large_vectors() -> dict[str, tuple[list[Any], dict[str, Any]]]:
    out = _float_vectors(
        [
            (_LARGE[0], _LARGE[1], Fraction(1, 1000)),
            (_LARGE[3], _LARGE[2], Fraction(1)),
        ],
        24,
        True,
    )
    for sign in (1, -1):
        tag = 'minus-' if sign < 0 else ''
        for where, value in (
            ('below', _OVER - 1),
            ('at', _OVER),
            ('above', _OVER + 1),
        ):
            exact = sign * value
            expected = expect_float(exact, 24)
            out[f'overflow.{tag}{where}.text'] = ([_plain(exact)], expected)
            out[f'overflow.{tag}{where}.decimal'] = (
                [Decimal(_plain(exact))],
                expected,
            )
        out[f'overflow.{tag}1e39.text'] = (
            [f'{"-" if sign < 0 else ""}1e39'],
            expect_float(sign * Fraction(10**39), 24),
        )
    return out


#: The key types whose table holds the neighbours a cast could land on,
#: or values 38 digits wide, each with its own entities only.
#: ``utf8-float-text``'s keys: ``user_id`` 1 to 3 as the text of three
#: floats a host writes in a mix (``host_text``, the reference's text of a
#: float entity since T1.36), every other its digits.
FLOAT_TEXT_KEYS = (
    "CASE X WHEN 1 THEN '1e+16' WHEN 2 THEN '1.5e-7' "
    "WHEN 3 THEN '1.0594382859262542e+15' ELSE X::VARCHAR END"
)
#: ``utf8-float-rust-text``'s keys: the same floats as Rust's ``{}`` writes
#: them, the core's text of a float entity before T1.36, which neither a
#: float alone nor one in a mix may reach any more.
RUST_FLOAT_TEXT_KEYS = (
    "CASE X WHEN 1 THEN '10000000000000000' WHEN 2 THEN '0.00000015' "
    "WHEN 3 THEN '1059438285926254.3' ELSE X::VARCHAR END"
)
#: ``utf8-float-pyarrow-text``'s keys besides ``0`` to ``7``: pyarrow's
#: text (``pc.cast(..., pa.string())``) of the floats of
#: ``_pyarrow_float_entities``, written here as the reference's cast gives
#: them (pandas 3.0.6 with pyarrow 25.0.1, pandas 2.2.0 with 14.0.1 and
#: pandas 2.2.3 with 19.0.1 alike). ``3.0``'s, ``3``, is ``user_id`` 3's.
PYARROW_FLOAT_TEXTS = (
    '1e+16',
    '1.2e+15',
    '1e+15',
    '1e+21',
    '1e+22',
    '1.2345678901234568e+17',
    '1000000000',
    '1e+10',
    '1.5e-7',
    '0.000001',
    '1e-7',
    '0.00001',
    '0.0001',
    '0.0001234',
    '0.1',
    '2.5',
    '-0',
    '5e-324',
    '1.7976931348623157e+308',
    '1.0594382859262542e+15',
    'inf',
    '-inf',
    '16777216',
    '0.10000000149011612',
    '1.0000000272564224e+16',
)


def _numpy() -> Any:
    import numpy as np

    return np


def _pyarrow_float_entities() -> dict[str, list[Any]]:
    r"""``utf8-float-pyarrow-text``'s entities: floats whose text the core
    and the reference may write otherwise (an exponent each way, the
    extremes, the boundaries of the plain form, a tie of the shortest
    digits, infinities and NaN), controls written alike (``0.1``, ``2.5``,
    ``-0.0``, ``3.0``), float32 entities, and the text itself."""
    import numpy as np

    nan = float('nan')
    floats = {
        '1e16': 1e16,
        '1-2e15': 1.2e15,
        '1e15': 1e15,
        '1e21': 1e21,
        '1e22': 1e22,
        '1-2345678901234568e17': 123456789012345680.0,
        '1e9': 1e9,
        '1e10': 1e10,
        '1-5e-7': 1.5e-07,
        '1e-6': 1e-06,
        '1e-7': 1e-07,
        '1e-5': 1e-05,
        '1e-4': 1e-04,
        '1-234e-4': 0.0001234,
        '0-1': 0.1,
        '2-5': 2.5,
        'minus-zero': -0.0,
        'integral-3': 3.0,
        'smallest-subnormal': 5e-324,
        'largest': 1.7976931348623157e308,
        'tie': 1059438285926254.2,
        'inf': math.inf,
        'minus-inf': -math.inf,
        'nan': nan,
        # A float32 column's values as Python floats (``tolist``).
        'widened-float32-0-1': float(np.float32(0.1)),
        'widened-float32-1e16': float(np.float32(1e16)),
    }
    out: dict[str, list[Any]] = {
        f'float.{name}': [value] for name, value in floats.items()
    }
    out |= {
        'float.several': [3.0, 1e16, 1.5e-07],
        'float.1e16-nan': [1e16, nan],
        'float.1e16-none': [1e16, None],
        'mixed.int-float-1e16': [1, 1e16],
        'numpy.float64-1e16': [np.float64(1e16)],
        'numpy.float32-0-1': [np.float32(0.1)],
        'numpy.float32-2pow24-plus-1': [np.float32(2**24 + 1)],
        'numpy.float32-1e16': [np.float32(1e16)],
        'numpy.float32-1-5e-7': [np.float32(1.5e-07)],
        'text.1e16': ['1e+16'],
    }
    return out


#: ``utf8-numeric-ids``'s keys besides ``0`` to ``7``.
NUMERIC_IDS = ('10000000000000000', '10000000000000001', '10000000000000002')


def _ids(texts: tuple[str, ...]) -> list[str]:
    r"""Statements adding a user of each of these keys."""
    return [f"INSERT INTO users VALUES ('{text}', 30, 'US')" for text in texts]


def _numeric_id_entities() -> dict[str, list[Any]]:
    r"""``utf8-numeric-ids``'s entities: ``1e16`` as a float, as NumPy's
    (a float column's values, a pandas column with its NaN dropped), with
    a NaN beside it, and as an integer and text, which reach the first ID
    whatever a float's spelling."""
    np = _numpy()
    return {
        'float.1e16': [1e16],
        'numpy.float64-1e16': [np.float64(1e16)],
        'numpy.float32-1e16': [np.float32(1e16)],
        'float.1e16-nan': [1e16, float('nan')],
        'float.1e16-and-next': [1e16, 10000000000000002.0],
        'int.1e16': [10**16],
        'text.1e16': ['10000000000000000'],
    }


ROUND_2: list[dict[str, Any]] = [
    {
        'id': 'float32-near-one',
        'setup': _floats(_NEAR_ONE + [-v for v in _NEAR_ONE], 'FLOAT'),
        'extra': {
            f'int.{tag}{n}': [sign * n]
            for sign, tag in ((1, ''), (-1, 'minus-'))
            for n in (2**24, 2**24 + 1, 2**24 + 2)
        }
        | {'float.2pow24-plus-1': [float(2**24 + 1)]},
        'derived': lambda: _float_vectors(
            [
                (_ONE, _ONE + _E23, Fraction(1, 10**30)),
                (Fraction(2**24), Fraction(2**24 + 2), Fraction(1, 10**6)),
            ],
            24,
            True,
        ),
    },
    {
        'id': 'float32-large',
        'setup': _floats(_LARGE + [-v for v in _LARGE], 'FLOAT'),
        'extra': {
            f'int.{tag}{name}': [sign * n]
            for sign, tag in ((1, ''), (-1, 'minus-'))
            for name, n in (
                ('2pow60', 2**60),
                ('2pow60-mid-plus-1', 2**60 + 2**36 + 1),
                ('2pow60-mid-minus-1', 2**60 + 2**36 - 1),
            )
        },
        'derived': _large_vectors,
    },
    {
        'id': 'float32-past-2pow24',
        'setup': _alter('FLOAT', 'X * 2 + 16777216'),
        'extra': {f'int.{n}': [n] for n in (2**24 + 1, 2**24 + 2, 2**24 + 3)},
        'derived': lambda: {
            f'text.{n}': ([str(n)], expect_float(Fraction(n), 24))
            for n in (2**24 + 1, 2**24 + 3)
        },
    },
    {
        'id': 'float64-past-2pow53',
        'setup': _alter('DOUBLE', 'X * 2 + 9007199254740992'),
        'extra': {f'int.{n}': [n] for n in (2**53 + 1, 2**53 + 2, 2**53 + 3)}
        | {'float.2pow53-plus-2': [float(2**53 + 2)]},
        'derived': lambda: (
            {
                f'text.{n}': ([str(n)], expect_float(Fraction(n), 53))
                for n in (2**53 + 1, 2**53 + 3)
            }
            | {
                'decimal.2pow53-plus-1-and-a-bit': (
                    [Decimal('9007199254740993.0000001')],
                    expect_float(
                        Fraction(Decimal('9007199254740993.0000001')), 53
                    ),
                )
            }
        ),
    },
    {
        'id': 'float64-past-minus-2pow53',
        'setup': _alter('DOUBLE', 'X * -2 - 9007199254740992'),
        'extra': {'int.minus-2pow53-minus-1': [-(2**53) - 1]},
        'derived': lambda: {
            'text.minus-2pow53-minus-1': (
                [str(-(2**53) - 1)],
                expect_float(Fraction(-(2**53) - 1), 53),
            )
        },
    },
    {
        'id': 'decimal128-38-0-wide',
        'setup': _alter('HUGEINT', f'X::HUGEINT + {WIDE}::HUGEINT'),
        'derived': lambda: {
            'wide.decimal': (
                [Decimal(str(WIDE + 1))],
                expect_decimal(str(WIDE + 1)),
            ),
            'wide.text': ([str(WIDE + 1)], expect_decimal(str(WIDE + 1))),
            'wide.int': ([WIDE + 1], expect_decimal(str(WIDE + 1))),
            'wide.exponent': (
                [Decimal('1.2345678901234567890123456789012345671E+37')],
                expect_decimal(str(WIDE + 1)),
            ),
            'wide.scaled-zeros': (
                [Decimal(f'{WIDE + 1}.000')],
                expect_decimal(str(WIDE + 1)),
            ),
            'wide.76-digits': ([Decimal('1' * 76)], REFUSED),
            'wide.39-digits-text': (['1' * 39], REFUSED),
        },
    },
    {
        'id': 'decimal128-38-10-wide',
        'setup': _alter(
            'DECIMAL(38, 10)',
            'X::DECIMAL(38, 10) + 1234567890123456789012345678.0123456789',
        ),
        'derived': lambda: {
            'wide.decimal': (
                [Decimal('1234567890123456789012345679.0123456789')],
                expect_decimal('1234567890123456789012345679.0123456789'),
            ),
            'wide.text': (
                ['1234567890123456789012345679.0123456789'],
                expect_decimal('1234567890123456789012345679.0123456789'),
            ),
            'wide.trailing-zeros': (
                [Decimal('1234567890123456789012345679.01234567890000')],
                expect_decimal('1234567890123456789012345679.0123456789'),
            ),
            'wide.inexact': (
                [Decimal('1234567890123456789012345679.01234567891')],
                REFUSED,
            ),
        },
    },
    {
        'id': 'decimal128-38-37',
        'setup': _alter(
            'DECIMAL(38, 37)',
            'X::DECIMAL(38, 37) + 0.1234567890123456789012345678901234567',
        ),
        'derived': lambda: {
            'wide.decimal': (
                [Decimal('1.1234567890123456789012345678901234567')],
                expect_decimal('1.1234567890123456789012345678901234567'),
            ),
            'wide.text': (
                ['1.1234567890123456789012345678901234567'],
                expect_decimal('1.1234567890123456789012345678901234567'),
            ),
            'wide.out-of-range': (
                ['10.1234567890123456789012345678901234567'],
                REFUSED,
            ),
            'wide.inexact': (
                ['1.12345678901234567890123456789012345671'],
                REFUSED,
            ),
        },
    },
    {
        # T1.32g (T1.32a's carried P3), T1.36: a text key holding the text
        # a host writes for three floats in a mix, the reference's text of
        # a float entity (pyarrow's: ``1e+16``, ``1.5e-7``,
        # ``1.0594382859262542e+15``; Python's ``str`` gives ``1.5e-07``
        # and ``1059438285926254.2``, Rust's ``{}`` the plain digits), so a
        # float mixed with text reaches the key a float alone reaches: an
        # exponent each way, and a tie of the shortest digits
        # (1059438285926254.25).
        'id': 'utf8-float-text',
        'setup': _alter('VARCHAR', FLOAT_TEXT_KEYS),
        'extra': lambda: {
            'mixed.float-1e16-text': [1e16, '0'],
            'mixed.float-tiny-text': [1.5e-07, '0'],
            'mixed.float-tie-text': [1059438285926254.2, '0'],
            'text.1e16': ['1e+16', '0'],
            # T1.36 (the final review's P2-2): the same floats alone, which
            # the reference writes so.
            'float.1e16': [1e16],
            'float.tiny': [1.5e-07],
            'float.tie': [1059438285926254.2],
            'numpy.float32-1e16': [_numpy().float32(1e16)],
        },
    },
    {
        # T1.36: the same floats against Rust's text of them, alone and in
        # a mix: the reference reaches no key here.
        'id': 'utf8-float-rust-text',
        'setup': _alter('VARCHAR', RUST_FLOAT_TEXT_KEYS),
        'extra': lambda: {
            'mixed.float-1e16-text': [1e16, '0'],
            'mixed.float-tiny-text': [1.5e-07, '0'],
            'mixed.float-tie-text': [1059438285926254.2, '0'],
            'float.1e16': [1e16],
            'float.tiny': [1.5e-07],
            'float.tie': [1059438285926254.2],
            'text.1e16': ['10000000000000000', '0'],
        },
    },
    {
        # T1.36: a float key holding 1e16 and 1.5e-7, and both mixed with
        # text, which a host writes ``1e+16`` and ``1.5e-7``: the core reads
        # a float key's text with an exponent each way.
        'id': 'float32-exponent',
        'setup': _alter(
            'FLOAT',
            "CASE X WHEN 1 THEN '1e16'::FLOAT WHEN 2 THEN '1.5e-7'::FLOAT "
            'ELSE X END',
        ),
        'extra': lambda: {'mixed.float-text': [1e16, 1.5e-07, '0']},
    },
    {
        # T1.36: the same for a decimal key, whose reader takes a float's
        # text too.
        'id': 'decimal128-38-10-exponent',
        'setup': _alter(
            'DECIMAL(38, 10)',
            'CASE X WHEN 1 THEN 10000000000000000 WHEN 2 THEN 0.00000015 '
            'ELSE X END',
        ),
        'extra': lambda: {'mixed.float-text': [1e16, 1.5e-07, '0']},
    },
    {
        # T1.36 (the final review's P2-2): a text key holding the text the
        # reference writes for a float entity alone, besides ``0`` to
        # ``7``. The reference's key is an ``ArrowDtype`` of text, and
        # pandas casts floats to it with pyarrow (``pa.array`` of the
        # floats, then ``cast`` to text: GH50430's fallback in
        # ``ArrowExtensionArray._box_pa_array``), so a float's text is
        # pyarrow's: the shortest digits of its own width (a float32's as a
        # float32's), plain from 1e-6 to below 1e10, otherwise one digit,
        # the rest after a point, ``e``, the exponent's sign and its
        # digits, unpadded; a whole one with no point, ``-0``, ``inf`` and
        # ``-inf``; NaN missing. The same in the floor and dbr17 cells.
        'id': 'utf8-float-pyarrow-text',
        'setup': _alter('VARCHAR', 'X::VARCHAR') + _ids(PYARROW_FLOAT_TEXTS),
        'extra': _pyarrow_float_entities,
    },
    {
        # T1.36 (P2-2) end to end: a text key of 17-digit numeric IDs, and
        # float entities, as a float column (one holding NaN, say) gives
        # them. The reference writes ``1e16`` as ``1e+16``, no ID here.
        'id': 'utf8-numeric-ids',
        'setup': _alter('VARCHAR', 'X::VARCHAR') + _ids(NUMERIC_IDS),
        'extra': _numeric_id_entities,
    },
    {
        # The same with ``1e+16`` among the IDs, which the reference reaches.
        'id': 'utf8-numeric-ids-and-exponent',
        'setup': _alter('VARCHAR', 'X::VARCHAR')
        + _ids((*NUMERIC_IDS, '1e+16')),
        'extra': _numeric_id_entities,
    },
]


def _inputs() -> dict[str, list[Any]]:
    r"""The entities every key type is crossed with, built afresh for each
    call (so no run sees another's objects)."""
    import numpy as np
    import pandas as pd

    nan = float('nan')
    return {
        # integers: present, absent, and at each width's ends
        'int.present': [1, 2],
        'int.absent': [8],
        'int.negative': [-1],
        'int.128': [128],
        'int.256': [256],
        'int.32768': [32768],
        'int.65536': [65536],
        'int.2pow31': [2**31],
        'int.2pow32': [2**32],
        'int.2pow53-plus-1': [2**53 + 1],
        'int.2pow63-minus-1': [2**63 - 1],
        'int.2pow63': [2**63],
        'int.2pow64': [2**64],
        'int.minus-2pow63-minus-1': [-(2**63) - 1],
        # floats: integral and not
        'float.integral': [1.0, 2.0],
        'float.half': [1.5],
        'float.negative-zero': [-0.0],
        'float.1e20': [1e20],
        'float.inf': [math.inf],
        'float.2pow63': [float(2**63)],
        # text
        'text.digits': ['1', '2'],
        'text.decimal-point': ['1.0'],
        'text.half': ['1.5'],
        'text.spaced': [' 1 '],
        'text.leading-zero': ['01'],
        'text.letters': ['x'],
        'text.empty': [''],
        'text.exponent': ['1e0'],
        'text.negative': ['-1'],
        'text.hex': ['0x1'],
        'text.fullwidth-digit': ['１'],
        'text.2pow64': [str(2**64)],
        # mixed lists
        'mixed.int-text': [1, '2'],
        'mixed.int-float': [1, 2.0],
        'mixed.int-half': [1, 2.5],
        'mixed.float-text': [1.0, '2'],
        'mixed.bool-int': [True, 1],
        'mixed.int-none': [1, None],
        'mixed.text-none': ['1', None],
        'mixed.int-nan': [1, nan],
        # booleans
        'bool.both': [True, False],
        # missing values
        'null.none': [None],
        'null.nan': [nan],
        'null.pd-na': [pd.NA],
        # NumPy scalars
        'numpy.int64': [np.int64(1), np.int64(2)],
        'numpy.float64-half': [np.float64(1.5)],
        # decimals
        'decimal.integral': [Decimal('1')],
        'decimal.half': [Decimal('1.5')],
        # times
        'time.timestamp': [pd.Timestamp('2024-01-01')],
        'time.timestamp-1us': [pd.Timestamp(1, unit='us')],
        'time.timestamp-utc': [pd.Timestamp('2024-01-01', tz='UTC')],
        'time.date': [date(2024, 1, 1)],
        # bytes
        'bytes.digit': [b'1'],
    }


def derived(key: dict[str, Any]) -> dict[str, tuple[list[Any], dict[str, Any]]]:
    r"""``key``'s entities whose expectation is derived, with it."""
    made = key.get('derived')
    return made() if made else {}


def inputs(key: dict[str, Any]) -> dict[str, list[Any]]:
    r"""``key``'s entities: the shared inputs (the round-2 key types have
    none), its own, and those whose expectation is derived."""
    shared = _inputs() if key not in ROUND_2 else {}
    extra = key.get('extra', {})
    if callable(extra):  # built afresh, as the shared inputs are
        extra = extra()
    own = {name: list(values) for name, values in extra.items()}
    return {
        **shared,
        **own,
        **{name: values for name, (values, _) in derived(key).items()},
    }


def all_keys() -> list[dict[str, Any]]:
    return KEYS + ROUND_2


# --- values as JSON ----------------------------------------------------------


def encode(value: Any) -> dict[str, Any]:
    r"""A Python entity as tagged JSON that ``decode`` gives back exactly."""
    import numpy as np
    import pandas as pd

    if value is None:
        return {'t': 'None'}
    if value is pd.NA:
        return {'t': 'pd.NA'}
    if value is pd.NaT:
        return {'t': 'pd.NaT'}
    if isinstance(value, np.generic):
        return {'t': f'numpy.{value.dtype}', 'v': repr(value.item())}
    if isinstance(value, bool):
        return {'t': 'bool', 'v': value}
    if isinstance(value, int):
        return {'t': 'int', 'v': str(value)}
    if isinstance(value, float):
        return {'t': 'float', 'v': repr(value)}
    if isinstance(value, str):
        return {'t': 'str', 'v': value}
    if isinstance(value, bytes):
        return {'t': 'bytes', 'v': base64.b64encode(value).decode()}
    if isinstance(value, Decimal):
        return {'t': 'Decimal', 'v': str(value)}
    if isinstance(value, pd.Timestamp):
        return {
            't': 'pd.Timestamp',
            'v': value.isoformat(),
            'unit': value.unit,
            'tz': None if value.tz is None else str(value.tz),
        }
    if isinstance(value, datetime):
        return {'t': 'datetime', 'v': value.isoformat()}
    if isinstance(value, date):
        return {'t': 'date', 'v': value.isoformat()}
    raise TypeError(f'no encoding for {type(value).__name__}')


def decode(tagged: dict[str, Any]) -> Any:
    r"""The Python entity ``encode`` wrote."""
    import numpy as np
    import pandas as pd

    kind = tagged['t']
    if kind == 'None':
        return None
    if kind == 'pd.NA':
        return pd.NA
    if kind == 'pd.NaT':
        return pd.NaT
    if kind.startswith('numpy.'):
        dtype = np.dtype(kind[len('numpy.') :])
        text = tagged['v']
        return dtype.type(float(text) if dtype.kind == 'f' else int(text))
    if kind == 'bool':
        return bool(tagged['v'])
    if kind == 'int':
        return int(tagged['v'])
    if kind == 'float':
        return float(tagged['v'])
    if kind == 'str':
        return tagged['v']
    if kind == 'bytes':
        return base64.b64decode(tagged['v'])
    if kind == 'Decimal':
        return Decimal(tagged['v'])
    if kind == 'pd.Timestamp':
        stamp = pd.Timestamp(tagged['v']).as_unit(tagged['unit'])
        return stamp if tagged['tz'] is None else stamp.tz_convert(tagged['tz'])
    if kind == 'datetime':
        return datetime.fromisoformat(tagged['v'])
    if kind == 'date':
        return date.fromisoformat(tagged['v'])
    raise ValueError(f'unknown tag {kind!r}')


def _bits(value: float) -> str:
    return struct.pack('>d', value).hex()


def canonical(value: Any) -> Any:
    r"""A value as both hosts can spell it: null, ``["b", bool]``,
    ``["i", digits]``, ``["f", the double's IEEE-754 bits in hex]``,
    ``["s", text]``, ``["d", decimal text]``, or ``["?", type name]``."""
    import numpy as np

    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return ['b', value]
    if isinstance(value, int):
        return ['i', str(value)]
    if isinstance(value, float):
        return ['f', _bits(value)]
    if isinstance(value, str):
        return ['s', value]
    if isinstance(value, Decimal):
        return ['d', str(value)]
    return ['?', type(value).__name__]


# --- the host's natural Arrow column ------------------------------------------


def _missing(value: Any) -> bool:
    import pandas as pd

    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and math.isnan(value))
    )


def _scalar(value: Any, arrow_type: Any) -> dict[str, Any]:
    r"""One value of an Arrow column as the contract's scalar
    (``sdm-contract``'s ``Scalar``, its ``t``-tagged JSON)."""
    import pyarrow as pa

    if value is None:
        return {'t': 'null'}
    integers = {
        pa.int8(): 'i8',
        pa.int16(): 'i16',
        pa.int32(): 'i32',
        pa.int64(): 'i64',
        pa.uint8(): 'u8',
        pa.uint16(): 'u16',
        pa.uint32(): 'u32',
        pa.uint64(): 'u64',
    }
    if arrow_type in integers:
        return {'t': integers[arrow_type], 'v': str(value)}
    if pa.types.is_boolean(arrow_type):
        return {'t': 'bool', 'v': value}
    if pa.types.is_floating(arrow_type):
        text = repr(float(value))
        text = {'nan': 'NaN', 'inf': 'inf', '-inf': '-inf'}.get(text, text)
        return {'t': 'f64' if arrow_type == pa.float64() else 'f32', 'v': text}
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return {'t': 'utf8', 'v': value}
    if pa.types.is_binary(arrow_type):
        return {'t': 'binary', 'v': base64.b64encode(value).decode()}
    if pa.types.is_decimal(arrow_type):
        return {
            't': 'd128' if pa.types.is_decimal128(arrow_type) else 'd256',
            'p': arrow_type.precision,
            's': arrow_type.scale,
            'v': str(unscaled(value, arrow_type.scale)),
        }
    if pa.types.is_date32(arrow_type):
        return {'t': 'date32', 'v': str((value - date(1970, 1, 1)).days)}
    raise TypeError(f'no scalar for {arrow_type}')


def unscaled(value: Decimal, scale: int) -> int:
    r"""``value`` times ``10**scale``, exactly, as an integer: from its
    digits and exponent, never through the caller's decimal context (whose
    precision, 28 digits by default, would round a 38-digit value)."""
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(f'not a finite decimal: {value!r}')
    whole = int(''.join(map(str, digits)) or '0')
    shift = exponent + scale
    if shift >= 0:
        whole *= 10**shift
    else:
        whole, rest = divmod(whole, 10**-shift)
        if rest:
            raise ValueError(f'{value!r} has more than {scale} decimals')
    return -whole if sign else whole


def float_text(value: Any) -> str:
    r"""A float as the reference writes a float entity of a text key, the
    core's text of a float entity since T1.36 (the final review's P2-2):
    pyarrow's cast of the float, at its own width (NumPy's ``float32`` as a
    float32), to text, which pandas' cast to the key's ``ArrowDtype`` of
    text makes. The shortest digits that read back as it, plain from 1e-6
    to below 1e10, otherwise one digit, the rest after a point, ``e``, the
    exponent's sign and its digits, unpadded; a whole one with no point:
    ``1e16`` is ``1e+16``, ``1.0`` is ``1``, ``-0.0`` is ``-0``,
    ``1.5e-07`` is ``1.5e-7``, ``1e-06`` is ``0.000001``, and an infinity
    ``inf`` or ``-inf``."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    width = pa.float32() if isinstance(value, np.float32) else pa.float64()
    text = pc.cast(pa.array([value], width), pa.string())[0].as_py()
    assert isinstance(text, str)
    return text


def host_text(value: Any) -> str:
    r"""A value as the host's text column holds it where pyarrow infers no
    one type: a float (NumPy's too) as the reference writes a float entity
    (``float_text``; T1.32g's rule, a float in a mix written as a float
    alone is, with T1.36's text), any other value its ``str()``."""
    import numpy as np

    if isinstance(value, (float, np.floating)):
        return float_text(value)
    return str(value)


def host_arrow(values: list[Any]) -> dict[str, Any]:
    r"""The entities as one Arrow column by the host's natural inference,
    or as text when it fails: the column's type and each value as the
    contract's scalar. A timestamp is carried as its count of its column's
    unit (``us`` or ``ns``, the contract's units)."""
    import pyarrow as pa

    try:
        column = pa.array(values)
        rule = 'inferred'
    except (pa.ArrowException, OverflowError, TypeError, ValueError):
        if all(_missing(v) for v in values):
            column = pa.nulls(len(values))
        else:
            column = pa.array(
                [None if _missing(v) else host_text(v) for v in values],
                pa.string(),
            )
        rule = 'text'
    kind = column.type
    if pa.types.is_timestamp(kind):
        unit = kind.unit if kind.unit in ('us', 'ns') else 'us'
        counts = column.cast(pa.timestamp(unit, kind.tz)).cast(pa.int64())
        scalars = [
            {'t': 'null'}
            if v is None
            else {'t': 'ts', 'unit': unit, 'zone': kind.tz, 'v': str(v)}
            for v in counts.to_pylist()
        ]
        return {
            'rule': rule,
            'type': str(pa.timestamp(unit, kind.tz)),
            'values': scalars,
        }
    return {
        'rule': rule,
        'type': str(kind),
        'values': [_scalar(v, kind) for v in column.to_pylist()],
    }


# --- the public call -----------------------------------------------------------


def shop() -> list[str]:
    r"""``contract/bases.yaml``'s ``shop`` statements."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def observe(
    key: dict[str, Any], values: list[Any], fields: tuple[str, ...] = ()
) -> dict[str, Any]:
    r"""``Schema.from_source`` and ``materialize`` over ``key``'s source
    with ``values`` as the entities, on the route the environment selects:
    the outcome, and on success the entities as the payload's predicted
    rows carry them, in order. With ``fields``, a refusal also gives those
    of its details (each as text), what names it."""
    import duckdb
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    for statement in shop() + key['setup']:
        connection.execute(statement)
    stage = 'from_source'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            stage = 'materialize'
            request = PredictiveQuery(QUERY, schema).materialize(
                indices=values, anchor_time=pd.Timestamp(ANCHOR)
            )
    except Exception as error:  # noqa: BLE001 - the outcome is recorded
        refusal: dict[str, Any] = {
            'outcome': 'refusal',
            'stage': stage,
            'error': type(error).__name__,
            'code': getattr(error, 'code', None),
        }
        if fields:
            details = getattr(error, 'details', None) or {}
            refusal['fields'] = {
                name: str(details[name])
                for name in fields
                if details.get(name) is not None
            }
        return refusal
    finally:
        connection.close()
    table = request._payloads[0]['predict']['instance_table']
    at = next(
        i
        for i, name in enumerate(table['columns'])
        if name.startswith('__kumo_entity_ref')
    )
    return {
        'outcome': 'success',
        'entities': [canonical(row[at]) for row in table['rows']],
    }


# --- the reference's own cast, observed in place ------------------------------


@contextmanager
def _watching_the_cast(values: list[Any]) -> Iterator[dict[str, Any]]:
    r"""Observes the reference engine's ``pd.Series(indices, dtype=...)``
    of these entities (the same objects, element by element) as it runs:
    the dtype, and the values cast or the exception raised."""
    import pandas as pd
    import pyarrow as pa

    seen: dict[str, Any] = {}
    original = pd.Series.__init__

    def init(self: Any, data: Any = None, *args: Any, **kwargs: Any) -> None:
        ours = (
            kwargs.get('dtype') is not None
            and isinstance(data, (list, tuple))
            and len(data) == len(values)
            and all(a is b for a, b in zip(data, values, strict=True))
        )
        if not ours:
            original(self, data, *args, **kwargs)
            return
        seen['dtype'] = str(kwargs['dtype'])
        try:
            original(self, data, *args, **kwargs)
        except Exception as error:
            seen['raised'] = type(error).__name__
            raise
        seen['values'] = [canonical(v) for v in pa.array(self).to_pylist()]

    pd.Series.__init__ = init  # type: ignore[method-assign]
    try:
        yield seen
    finally:
        pd.Series.__init__ = original  # type: ignore[method-assign]


def _kind(value: Any) -> str:
    r"""What an entity is, for ``register``'s kind rule."""
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, (int, float, Decimal)):
        return 'number'
    if isinstance(value, str):
        return 'text'
    if isinstance(value, (date, datetime)):
        return 'time'
    return type(value).__name__


def _number(cell: Any) -> Decimal | None:
    if not isinstance(cell, list):
        return None
    if cell[0] in ('i', 'd'):
        return Decimal(cell[1])
    if cell[0] == 'f':
        value = struct.unpack('>d', bytes.fromhex(cell[1]))[0]
        return Decimal(value) if math.isfinite(value) else None
    return None


def register(values: list[Any], public: dict[str, Any]) -> list[str]:
    r"""Why a host may differ from the reference on this vector only under a
    register entry: the reference's answer, never a choice. Empty: held to
    the reference exactly."""
    why = []
    if public['outcome'] == 'refusal' and public['code'] == 'INTERNAL_ERROR':
        why.append(
            'the reference fails with INTERNAL_ERROR (E66(b): a named '
            'refusal or a success in its place is a change to register)'
        )
    if public['outcome'] == 'success':
        for value, cell in zip(values, public['entities'], strict=True):
            if _missing(value) or cell is None:
                continue
            kind = _kind(value)
            got = {'i': 'number', 'f': 'number', 'd': 'number'}.get(
                cell[0], 'text' if cell[0] == 's' else cell[0]
            )
            if kind != got:
                why.append(
                    f'the reference accepts a {kind} entity as a {got} key '
                    "(pandas' cast changes its kind)"
                )
                break
            if kind == 'number':
                plain = value.item() if hasattr(value, 'item') else value
                finite = not isinstance(plain, float) or math.isfinite(plain)
                wanted = Decimal(plain) if finite else None
                if wanted is None or wanted != _number(cell):
                    why.append(
                        'the reference accepts a number as a different '
                        "number (pandas' cast changes its value)"
                    )
                    break
    return why


def mixed(values: list[Any]) -> bool:
    r"""Whether the entities are of more than one kind, missing ones aside."""
    return len({_kind(v) for v in values if not _missing(v)}) > 1


def as_text(values: list[Any]) -> list[Any]:
    r"""The entities as the host's text column holds them, as text given."""
    return [None if _missing(v) else host_text(v) for v in values]


def record(
    key: dict[str, Any],
    name: str,
    values: list[Any],
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r"""One vector, observed on the reference: also, when the host hands a
    mix of kinds over as text, the reference's answer to that text given
    as text (``as_text``)."""
    encoded = [encode(v) for v in values]
    arrow = host_arrow(values)
    with _watching_the_cast(values) as seen:
        public = observe(key, values)
    cast: dict[str, Any] | None
    if not seen:
        cast = None  # the call ended before the engine's cast
    elif 'raised' in seen:
        cast = {'dtype': seen['dtype'], 'raised': seen['raised']}
    else:
        cast = {'dtype': seen['dtype'], 'values': seen['values']}
    reference: dict[str, Any] = {'public': public, 'cast': cast}
    if arrow['rule'] == 'text' and mixed(values):
        reference['as_text'] = observe(key, as_text(values))
    vector: dict[str, Any] = {
        'id': f'{key["id"]}.{name}',
        'key': key['id'],
        'input': encoded,
        'host_arrow': arrow,
        'reference': reference,
    }
    if expected is not None:
        vector['expected'] = expected
    return vector


# --- what a host is held to ----------------------------------------------------


def outcome_class(answer: dict[str, Any]) -> str:
    r"""``success`` or ``refusal <CODE>``: a register entry's transition is
    written from one of these to another."""
    if answer['outcome'] == 'success':
        return 'success'
    return f'refusal {answer.get("code")}'


def same(got: dict[str, Any], want: dict[str, Any]) -> bool:
    r"""The same public answer: outcome, code and stage or entities, and the
    class where both observe it (``rust-host`` observes no stage or class,
    and ``fields`` are not the reference's)."""
    keys = set(got) & set(want) - {'fields'}
    return got['outcome'] == want['outcome'] and all(
        got[k] == want[k] for k in keys
    )


def _f32_hex(value: float) -> str | None:
    try:
        return struct.pack('>f', value).hex()
    except OverflowError:
        return None


def _cell_matches(wanted: list[str], cell: Any) -> bool:
    if not isinstance(cell, list):
        return False
    if wanted[0] in ('f32', 'f64'):
        if cell[0] == 'f':
            value = struct.unpack('>d', bytes.fromhex(cell[1]))[0]
        elif cell[0] == 'i':
            value = float(int(cell[1]))
        else:
            return False
        if wanted[0] == 'f32':
            return _f32_hex(value) == wanted[1]
        return struct.pack('>d', value).hex() == wanted[1]
    if wanted[0] == 'dec':
        if cell[0] in ('i', 's'):
            try:
                return decimal_text(cell[1]) == wanted[1]
            except ValueError:
                return False
        if cell[0] == 'f':
            # A payload that writes the key as a double can only be held to
            # the double nearest the value.
            near = float(Fraction(Decimal(wanted[1])))
            return struct.pack('>d', near).hex() == cell[1]
        return False
    return wanted == cell


def matches(expected: dict[str, Any], got: dict[str, Any]) -> bool:
    r"""Whether ``got`` is the derived expectation: a success whose entities
    are the expected values (a float at the expected width's bits, a
    decimal by its exact value), or a refusal of the expected code, which
    where it must be named also names the field and why (``fields``)."""
    if got.get('outcome') != expected['outcome']:
        return False
    if expected['outcome'] == 'refusal':
        if got.get('code') != expected['code']:
            return False
        if expected.get('named') and 'fields' in got:
            named = got['fields']
            return named.get('field') == 'indices' and bool(named.get('reason'))
        return True
    have = got.get('entities') or []
    return len(have) == len(expected['entities']) and all(
        _cell_matches(w, c)
        for w, c in zip(expected['entities'], have, strict=True)
    )


def derive_marks(vectors: list[dict[str, Any]]) -> None:
    r"""Each vector's ``register``: whether a host may differ from the
    reference only under an entry naming the vector and its transition,
    why, and the disposition it is held to; derived from the recorded
    answers alone, never chosen."""
    for vector in vectors:
        values = [decode(tagged) for tagged in vector['input']]
        reference = vector['reference']
        public = reference['public']
        why = register(values, public)
        disposition = None
        if 'expected' in vector and not matches(vector['expected'], public):
            why.append(
                'the expectation derived from the exact value is not the '
                "reference's answer"
            )
            disposition = 'expected'
        text = reference.get('as_text')
        if text is not None and text != public:
            column = [v.get('v') for v in vector['host_arrow']['values']]
            why.append(
                f'the host hands the core the text column {column}, the same '
                'as that text given as text, which the reference answers '
                f'otherwise ({outcome_class(text)})'
            )
            disposition = 'as_text'
        vector['register'] = {
            'required': bool(why),
            'why': why,
            'disposition': disposition,
        }
    alike(vectors)


def alike(vectors: list[dict[str, Any]]) -> None:
    r"""Marks the vectors the core cannot tell apart: on one key, entities
    the host hands over as the same Arrow column (``host_arrow``'s type and
    values), which the reference answers differently (``[1, '2']`` is
    refused where ``['1', '2']`` is taken). The core sees one column, so it
    gives them one answer, and each that differs needs its entry."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for vector in vectors:
        column = vector['host_arrow']
        same = json.dumps(
            [vector['key'], column['type'], column['values']], sort_keys=True
        )
        groups.setdefault(same, []).append(vector)
    for group in groups.values():
        answers = {
            json.dumps(v['reference']['public'], sort_keys=True) for v in group
        }
        if len(answers) < 2:
            continue
        for vector in group:
            others = ', '.join(v['id'] for v in group if v is not vector)
            vector['register']['why'].append(
                f'the host gives the core the same column as {others}, '
                'which the reference answers differently'
            )
            vector['register']['required'] = True


def main() -> int:
    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import sdm_relational_engine  # noqa: F401 - the reference's engine

    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    import sdm_client

    if not Path(sdm_client.__file__).resolve().is_relative_to(reference):
        print("sdm_client is not the reference checkout's", file=sys.stderr)
        return 1
    vectors = []
    for key in all_keys():
        expectations = derived(key)
        for name, values in inputs(key).items():
            expected = expectations[name][1] if name in expectations else None
            vector = record(key, name, values, expected)
            vectors.append(vector)
            public = vector['reference']['public']
            print(
                vector['id'],
                public['outcome'],
                public.get('code') or '',
                file=sys.stderr,
            )
    derive_marks(vectors)
    document = {
        'format': FORMAT,
        'recorded_by': 'contract/tools/key_cast_record.py',
        'reference': commit,
        'cell': 'ref-pandas3',
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'pyarrow': pa.__version__,
            'numpy': np.__version__,
            'duckdb': duckdb.__version__,
        },
        'query': QUERY,
        'anchor_time': ANCHOR,
        'register_ref': REGISTER_REF,
        'keys': [
            {'id': key['id'], 'setup': key['setup']} for key in all_keys()
        ],
        'vectors': vectors,
    }
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    required = sum(v['register']['required'] for v in vectors)
    print(
        f'{len(vectors)} vectors written to {OUT.relative_to(ROOT)}, '
        f'{required} marked register entry required',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
