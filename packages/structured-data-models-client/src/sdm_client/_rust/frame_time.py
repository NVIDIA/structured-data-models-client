# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual native first-value conversions for a fixed core-selected plan."""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._rust import codec


def _scalar(value: Any) -> dict[str, Any]:
    if type(value) is Decimal:
        parts = value.as_tuple()
        exponent = parts.exponent
        encoded = (
            {'kind': 'finite', 'value': exponent}
            if isinstance(exponent, int)
            else {
                'kind': {'F': 'infinity', 'n': 'nan', 'N': 'signaling_nan'}[
                    exponent
                ]
            }
        )
        return {
            'kind': 'decimal',
            'negative': bool(parts.sign),
            'digits': ''.join(str(digit) for digit in parts.digits),
            'exponent': encoded,
        }
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime.datetime)):
        stamp = pd.Timestamp(value)
        scalar = codec.timestamp(
            int(stamp.asm8.view('i8')),
            stamp.unit,
            str(stamp.tz) if stamp.tz is not None else None,
        )
    elif isinstance(value, (bool, np.bool_)):
        scalar = {'t': 'bool', 'v': bool(value)}
    elif isinstance(value, (int, np.integer)):
        scalar = codec.integer('i64' if value < 0 else 'u64', int(value))
    elif isinstance(value, (float, np.floating)):
        # repr is the value codec for an admitted built-in numeric scalar,
        # not the eventual native diagnostic presentation.
        scalar = {'t': 'f64', 'v': repr(float(value))}
    elif type(value) is str:
        scalar = {'t': 'utf8', 'v': value}
    else:
        raise codec.CodecError(
            'a native time first value has no typed scalar representation'
        )
    return {'kind': 'value', 'value': scalar}


def first_values(
    frame: pd.DataFrame, plan: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read actual dropna/iloc/equality operations in the selected order.

    No native exception is translated or delayed. In particular Arrow's
    DecimalScalar.as_py runs here, after earlier constructor warnings.
    Native first objects remain call-local presentation values; the core
    receives inspectable typed facts, never callbacks or object handles.
    """
    facts = []
    native = {}
    for name in plan['columns']:
        present = frame[name].dropna()
        rows, held = len(frame), len(present)
        first = present.iloc[0] if held else None
        same = held > 0 and bool((present == first).all())
        facts.append(
            {
                'name': name,
                'rows': rows,
                'held': held,
                'same': same,
                'first': _scalar(first) if held else None,
            }
        )
        if held:
            native[name] = first
    return {'plan': plan, 'firsts': facts}, native
