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
import json
import re
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
        return float(value)
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
        if unit not in _UNITS or not low <= number <= high:
            raise CodecError(f'{tag}: unit or value out of range')
        if tag == 'ts':
            if zone is not None and (not isinstance(zone, str) or not zone):
                raise CodecError('ts: the zone is empty')
            return Temporal(tag, number, unit, zone)
        return Temporal(tag, number, unit)
    raise CodecError('a scalar of an unknown tag')


def integer(tag: str, value: int) -> dict[str, Any]:
    r"""An integer scalar of width ``tag`` (``i8`` to ``u64``)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodecError(f'{tag}: not an integer')
    low, high = _INTEGERS[tag]
    if not low <= value <= high:
        raise CodecError(f'{tag}: out of range')
    return {'t': tag, 'v': str(value)}


def timestamp(value: int, unit: str, zone: str | None) -> dict[str, Any]:
    r"""A ``ts`` scalar: ``value`` in ``unit`` since the epoch, in ``zone``."""
    if unit not in _UNITS or (zone is not None and not zone):
        raise CodecError('ts: unit or zone out of range')
    return {**integer('i64', value), 't': 'ts', 'unit': unit, 'zone': zone}


def encode(message: dict[str, Any]) -> bytes:
    r"""A control message the host sends, within the control cap.

    A message past the cap is refused: whatever grows with the input travels
    as an attachment (:func:`attachment`), so a message past it is a defect
    of this host.
    """
    data = json.dumps(message, ensure_ascii=False, allow_nan=False).encode()
    if len(data) > CONTROL_CAP:
        raise CodecError(
            f'a control message of {len(data)} bytes is past the control cap'
        )
    return data


def attachment(value: Any) -> bytes:
    r"""An input-sized field as a bytes attachment of its JSON."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode()


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
