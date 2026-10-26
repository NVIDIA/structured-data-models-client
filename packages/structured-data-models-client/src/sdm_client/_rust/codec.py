# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The control-message codec of the Python host.

A control message is the contract's JSON, as bytes, with the attachments it
names in a ``dict`` by id. Two rules come from the contract:

- A message the core writes whose encoding would pass the 64 KB control cap
  travels whole as a bytes attachment of its JSON, named by
  ``{"message": "attached", "attachment": n}``; :func:`read` gives the
  message that attachment carries (decision H7).
- Every scalar is a tagged object whose value is a JSON string, so nothing
  passes through a double (decision E4). :func:`decode_scalar` reads every
  tag exactly; the encoders, for the scalars this host writes, write only
  what the core's decoder takes, and refuse anything else rather than
  write it.

Input-sized fields of a ``start`` travel as bytes attachments of their JSON
(decision H5); :func:`attachment` writes one.
"""

from __future__ import annotations

import base64
import decimal
import fractions
import json
import math
import numbers
import re
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

#: The largest control message, in bytes, in either direction.
CONTROL_CAP = 64 * 1024

_INTEGERS = {
    'i8': (-(2**7), 2**7 - 1),
    'i16': (-(2**15), 2**15 - 1),
    'i32': (-(2**31), 2**31 - 1),
    'i64': (-(2**63), 2**63 - 1),
    'u8': (0, 2**8 - 1),
    'u16': (0, 2**16 - 1),
    'u32': (0, 2**32 - 1),
    'u64': (0, 2**64 - 1),
}
_DECIMALS = {'d128': 38, 'd256': 76}
_TEMPORAL = ('ts', 'duration', 'time64')
_UNITS = ('ns', 'us')
_TIMESTAMP_UNITS = ('s', 'ms', 'us', 'ns')
_DIGITS = re.compile(r'(0|-?[1-9][0-9]*)\Z')
_FLOAT = re.compile(r'(NaN|inf|-inf|-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?)\Z')


class CodecError(ValueError):
    r"""A control message or scalar the contract does not hold.

    It names the position or the tag, never the text it read, which may be
    an input value (decision H17).
    """


@dataclass(frozen=True)
class Temporal:
    r"""A temporal scalar as the contract counts it: ``value`` in ``unit``.

    ``tag`` is ``date32`` (days since the epoch, no unit), ``ts`` (since the
    epoch, shown in ``zone``), ``duration`` or ``time64`` (since midnight).
    """

    tag: str
    value: int
    unit: str | None = None
    zone: str | None = None


def _digits(tag: str, text: object) -> int:
    if not isinstance(text, str) or not _DIGITS.match(text):
        raise CodecError(f'{tag}: not an integer in canonical decimal form')
    return int(text)


def decode_scalar(scalar: Any) -> Any:
    r"""A contract scalar as a Python value, read exactly.

    Integers are ``int``, floats ``float``, decimals :class:`~decimal.Decimal`
    built from their digits, text ``str``, binary ``bytes``, and temporal
    values :class:`Temporal`; ``null`` is ``None``.
    """
    if not isinstance(scalar, dict) or not isinstance(scalar.get('t'), str):
        raise CodecError('a scalar is a tagged object')
    tag = scalar['t']
    value = scalar.get('v')
    if tag == 'null':
        return None
    if tag == 'bool':
        if not isinstance(value, bool):
            raise CodecError('bool: not a boolean')
        return value
    if tag in _INTEGERS:
        number = _digits(tag, value)
        low, high = _INTEGERS[tag]
        if not low <= number <= high:
            raise CodecError(f'{tag}: out of range')
        return number
    if tag in ('f32', 'f64'):
        if not isinstance(value, str) or not _FLOAT.match(value):
            raise CodecError(f'{tag}: not a float')
        return _float32(value) if tag == 'f32' else _float64(value)
    if tag in _DECIMALS:
        precision, scale = scalar.get('p'), scalar.get('s')
        number = _digits(tag, value)
        if (
            not isinstance(precision, int)
            or not isinstance(scale, int)
            or not 1 <= precision <= _DECIMALS[tag]
            or scale > precision
            or len(str(abs(number))) > precision
        ):
            raise CodecError(f'{tag}: outside its precision and scale')
        # From its digits, never by scaleb, which rounds to the context.
        with decimal.localcontext(_exact()):
            unscaled = Decimal(number).as_tuple()
            return Decimal((unscaled.sign, unscaled.digits, -scale))
    if tag == 'utf8':
        if not isinstance(value, str):
            raise CodecError('utf8: not text')
        return value
    if tag == 'binary':
        if not isinstance(value, str):
            raise CodecError('binary: not base64 text')
        try:
            return base64.b64decode(value, validate=True)
        except ValueError:
            raise CodecError('binary: not standard base64') from None
    if tag == 'date32':
        number = _digits(tag, value)
        low, high = _INTEGERS['i32']
        if not low <= number <= high:
            raise CodecError('date32: out of range')
        return Temporal('date32', number)
    if tag in _TEMPORAL:
        unit, zone = scalar.get('unit'), scalar.get('zone')
        number = _digits(tag, value)
        low, high = _INTEGERS['i64']
        if (
            unit not in (_TIMESTAMP_UNITS if tag == 'ts' else _UNITS)
            or not low <= number <= high
        ):
            raise CodecError(f'{tag}: unit or value out of range')
        if tag == 'ts':
            if zone is not None and (not isinstance(zone, str) or not zone):
                raise CodecError('ts: the zone is empty')
            return Temporal(tag, number, unit, zone)
        return Temporal(tag, number, unit)
    raise CodecError('a scalar of an unknown tag')


def _float64(text: str) -> float:
    r"""``text`` read as a double; finite text that overflows is refused."""
    value = float(text)
    if math.isinf(value) and text not in ('inf', '-inf'):
        raise CodecError('f64: finite text past the largest double')
    return value


def _bits32(value: float) -> int:
    return int.from_bytes(struct.pack('>f', value), 'big')


def _of_bits32(bits: int) -> float:
    return float(struct.unpack('>f', bits.to_bytes(4, 'big'))[0])


#: The largest float32.
_FLOAT32_MAX = (2 - 2**-23) * 2.0**127
#: Halfway from the largest float32 to 2**128: text whose magnitude reaches
#: it rounds to infinity (a tie goes to the even neighbour, 2**128).
_FLOAT32_OVERFLOW = Decimal(2**128 - 2**103)


def _exact() -> decimal.Context:
    r"""A context of this module's own for exact comparisons, whatever the
    caller's context traps (``FloatOperation`` included)."""
    return decimal.Context()


def _float32(text: str) -> float:
    r"""``text`` rounded to the nearest float32, as a Python float.

    Rounding the text to a double first and the double to a float32 can
    round twice: when the double lands exactly halfway between two
    float32s, or exactly on the overflow threshold, while the text does
    not, the text itself decides, read exactly as a
    :class:`~decimal.Decimal`. Finite text that rounds past the largest
    float32 is refused, as the core refuses it.
    """
    wide = float(text)
    if math.isnan(wide):
        return wide
    if math.isinf(wide) and text in ('inf', '-inf'):
        return wide
    try:
        narrow = _of_bits32(_bits32(wide))
    except OverflowError:
        narrow = math.inf
    if math.isinf(narrow):
        with decimal.localcontext(_exact()):
            below = Decimal(text).copy_abs() < _FLOAT32_OVERFLOW
        if not below:
            raise CodecError('f32: finite text past the largest float32')
        return math.copysign(_FLOAT32_MAX, wide)
    if narrow == wide:
        return narrow
    # The float32 on the double's side of ``narrow`` (bits count magnitude
    # within a sign), and the point halfway to it, exact as a double.
    step = 1 if abs(wide) > abs(narrow) else -1
    other = _of_bits32(_bits32(narrow) + step)
    halfway = (narrow + other) / 2
    if wide != halfway:
        return narrow
    with decimal.localcontext(_exact()):
        exact, middle = Decimal(text), Decimal.from_float(halfway)
        if exact == middle:
            return narrow  # A true tie: struct rounded it to even.
        nearer_other = (exact > middle) == (other > narrow)
    return other if nearer_other else narrow


def integer(tag: str, value: int) -> dict[str, Any]:
    r"""An integer scalar of width ``tag`` (``i8`` to ``u64``)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodecError(f'{tag}: not an integer')
    low, high = _INTEGERS[tag]
    if not low <= value <= high:
        raise CodecError(f'{tag}: out of range')
    return {'t': tag, 'v': str(value)}


def timestamp(value: int, unit: str, zone: str | None) -> dict[str, Any]:
    r"""A ``ts`` scalar: ``value`` in ``unit`` since the epoch, in ``zone``,
    which is none or non-empty text."""
    if unit not in _TIMESTAMP_UNITS or (
        zone is not None and (not isinstance(zone, str) or not zone)
    ):
        raise CodecError('ts: unit or zone out of range')
    return {**integer('i64', value), 't': 'ts', 'unit': unit, 'zone': zone}


def seed(value: Any) -> str | None:
    r"""A seed as the contract writes one: an integer of any size (a NumPy
    integer too) as signed hexadecimal text with no leading zeros, so no
    integer reader's digit limit is met (decision H7). None for none, or for
    a value of another type, which the call's values carry to the core as
    given, and the core refuses (T1.28a)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    value = int(value)
    sign = '-' if value < 0 else ''
    return f'{sign}0x{abs(value):x}'


def _integral(value: Any) -> int:
    r"""A host integer the JSON encoder does not know, such as a NumPy
    integer, which the client's checks take as a whole number, written as
    the integer it is; any other value refused, as the encoder refuses it.
    A bool is not one (NumPy's is not ``numbers.Integral``)."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    raise TypeError(
        f'Object of type {type(value).__name__} is not JSON serializable'
    )


def _scalar(value: Any) -> Any:
    r"""A host scalar the JSON encoder does not know, written as the Python
    value it holds, losslessly: an integer (:func:`_integral`), and from
    T1.28a a NumPy boolean or float, its type kept (a float32 is exactly the
    double it widens to), so the core reads the call's option values as the
    caller gave them; a sequence the encoder does not know (a ``range``, and
    from T1.28b a NumPy array) as its items, as the client reads a sequence
    of hop counts and the engine's config classes a sequence of transforms;
    any other value refused, as the encoder refuses it. From T1.28b a NumPy
    float wider than a double (``np.longdouble``) is written as the double
    it holds where that double is exactly its value, as the engine's config
    classes read it, else refused."""
    import numpy as np

    if isinstance(value, np.floating):
        item = value.item()
        if isinstance(item, float):
            return item
        if float(value) == value:
            return float(value)
        raise TypeError(
            f'Object of type {type(value).__name__} is not JSON serializable'
        )
    if isinstance(value, np.bool_):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return list(value)
    if isinstance(value, (decimal.Decimal, fractions.Fraction)):
        # From T2.9, a finite decimal or fraction as its nearest double,
        # its sign and whether it is zero kept, which is all an option the
        # core reads as a number asks of it: one a double cannot reach is
        # held at the double nearest it of the same sign (the least
        # positive one, or the largest finite), so lag_timesteps=Decimal(2),
        # Decimal('1e-400') or Decimal('1e400') on a static query is taken
        # and ignored, and on a temporal one refused as not whole, as the
        # reference meets each; a NaN or an infinity refused, as the
        # encoder refuses it.
        if not isinstance(value, decimal.Decimal) or value.is_finite():
            sign = 1.0 if value > 0 else -1.0
            try:
                double = float(value)
            except OverflowError:
                double = math.inf
            if math.isinf(double):
                return math.copysign(sys.float_info.max, sign)
            if double == 0 and value != 0:
                return math.copysign(5e-324, sign)
            return double
        raise TypeError(
            f'Object of type {type(value).__name__} is not JSON serializable'
        )
    return _integral(value)


def encode(message: dict[str, Any]) -> bytes:
    r"""A control message the host sends, within the control cap.

    A message past the cap is refused: whatever grows with the input travels
    as an attachment (:func:`attachment`), so a message past it is a defect
    of this host.
    """
    data = json.dumps(
        message, ensure_ascii=False, allow_nan=False, default=_integral
    ).encode()
    if len(data) > CONTROL_CAP:
        raise CodecError(
            f'a control message of {len(data)} bytes is past the control cap'
        )
    return data


def attachment(value: Any, *, named: bool = False) -> bytes:
    r"""An input-sized field as a bytes attachment of its JSON. A value JSON
    cannot hold (one of a type the encoder does not know, a number that is
    not finite, one nested past what the encoder writes) is refused as the
    client's transport refuses it, ``INVALID_REQUEST``: from T1.28a the
    call's option values travel here as the caller gave them, and from
    T2.9 (``named``, for them alone) the refusal names the first option
    whose value JSON cannot hold, in ``details['field']``."""

    def written(item: Any) -> bytes:
        return json.dumps(
            item, ensure_ascii=False, allow_nan=False, default=_scalar
        ).encode()

    def writes(item: Any) -> bool:
        try:
            written(item)
        except (TypeError, ValueError, RecursionError):
            return False
        return True

    try:
        return written(value)
    except (TypeError, ValueError, RecursionError):
        pass
    from sdm_client.errors import StructuredDataError

    # A mapping's first member JSON cannot hold is named (T2.9: the call's
    # option values, a NaN or an infinity among them, say which option).
    option = next(
        (
            str(name)
            for name, item in (
                value.items() if named and isinstance(value, dict) else ()
            )
            if not writes(item)
        ),
        None,
    )
    raise StructuredDataError(
        'this request carries a value that cannot be sent as JSON'
        if option is None
        else f'{option} carries a value that cannot be sent as JSON (a NaN, '
        f'an infinity, or a type JSON has no form for)',
        code='INVALID_REQUEST',
        details=None if option is None else {'field': option},
    )


# --- a caller's value as the contract's Given ----------------------------------
#
# The conformance rule of the contract's ``Given`` (``sdm-contract``'s ops.rs):
# a value the host can write faithfully as the contract's type is written
# ``{"value": ...}``; any other is written ``"unreadable"``, and the core
# refuses it. The host asks nothing else of the value: what a null, an empty
# key or a name the table lacks means is the core's. In Python, text is a
# ``str`` (a subclass written as the plain ``str``); a collection is any
# other iterable but ``bytes``, a mapping and a set (no order), read once
# with ``tuple``. Nothing is caught: an exception raised while reading the
# caller's iterable, ``KeyboardInterrupt`` included, is the caller's, as the
# reference's ``list(key)`` lets it out.

#: A value no contract type holds, as the contract writes it.
UNREADABLE = 'unreadable'
#: The longest text :func:`spelled` gives the core; a value whose ``repr``
#: is longer is given as :data:`UNREADABLE`, so a control message stays
#: within its cap.
SPELLED_MOST = 2_000


def spelled(value: Any) -> Any:
    r"""A value no contract type holds, as ``Given`` writes it with the
    text Python shows it as (``repr``), which the core names in its refusal
    as the reference names the value (T2.14b); :data:`UNREADABLE` where its
    ``repr`` fails or is past :data:`SPELLED_MOST` characters."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a caller's value's own repr
        return UNREADABLE
    if not isinstance(text, str) or len(text) > SPELLED_MOST:
        return UNREADABLE
    return {'spelled': str.__str__(text)}


def given(value: Any) -> Any:
    r"""``value``, of the contract's type, as ``Given`` writes it."""
    return {'value': value}


def given_text(value: Any) -> Any:
    r"""Text as the contract writes it, or any other value
    :func:`spelled`."""
    if isinstance(value, str):
        return given(str.__str__(value))
    return spelled(value)


def given_name_or_none(value: Any) -> Any:
    r"""A name or ``None`` (a time column), as the contract writes it, or
    any other value :func:`spelled`."""
    if value is None:
        return given(None)
    return given_text(value)


def given_key(value: Any) -> tuple[Any, Any]:
    r"""A key as the contract writes it, and the form the reference keeps
    it in (``_as_key_tuple``: a name, or a tuple of the names, read once)."""
    if value is None:
        return given(None), None
    if isinstance(value, str):
        return given([str.__str__(value)]), value
    if isinstance(value, (bytes, Mapping, AbstractSet)) or not isinstance(
        value, Iterable
    ):
        return spelled(value), None
    names = tuple(value)
    if not all(isinstance(name, str) for name in names):
        return spelled(value), None
    return given([str.__str__(name) for name in names]), names


def relation(name: Any) -> dict[str, Any]:
    r"""``name``, a table's name in the source, as the contract's relation
    of one part (contract v26, design choice 12): never split on a dot, so
    ``'a.b'`` reads the one table of that name."""
    return {'source': [name]}


def relation_name(value: Any) -> str:
    r"""The name of a relation of one part the core described: this host
    gives one part, so the core answers one; any other is refused."""
    parts = value.get('source') if isinstance(value, Mapping) else None
    if (
        not isinstance(parts, list)
        or len(parts) != 1
        or not isinstance(parts[0], str)
    ):
        raise CodecError('relation: not a relation of one part')
    return parts[0]


def _json(data: bytes) -> dict[str, Any]:
    try:
        message = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        position = getattr(error, 'pos', getattr(error, 'start', 0))
        raise CodecError(
            f'not a control message (at byte {position})'
        ) from None
    if not isinstance(message, dict) or not isinstance(
        message.get('message'), str
    ):
        raise CodecError('not a control message')
    return message


def read(control: bytes, attachments: dict[int, Any]) -> dict[str, Any]:
    r"""A control message the core wrote, whole.

    An ``attached`` message is replaced by the message its bytes attachment
    carries, which is never itself ``attached``.
    """
    message = _json(control)
    if message['message'] != 'attached':
        return message
    carried = attachments.get(message.get('attachment'))  # type: ignore[arg-type]
    if not isinstance(carried, bytes):
        raise CodecError(
            'an attached message names a bytes attachment that is not with it'
        )
    whole = _json(carried)
    if whole['message'] == 'attached':
        raise CodecError('an attached message carries another')
    return whole
