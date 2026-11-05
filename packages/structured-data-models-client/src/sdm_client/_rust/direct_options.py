# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native inference-option traversal, after both tables have been encoded."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from numbers import Integral
from typing import Any

import numpy as np


def observe(value: Any, call: Callable[[str, dict[str, Any]], Any]) -> Any:
    def visit(item: Any, depth: int, seen: frozenset[int]) -> Any:
        if isinstance(item, np.generic):
            item = item.item()
        integer = (
            str(int(item))
            if isinstance(item, Integral) and not isinstance(item, bool)
            else None
        )
        call(
            'check_direct_option',
            {
                'integer': integer,
                'repeated': id(item) in seen,
                'depth': depth,
            },
        )
        if isinstance(item, Mapping):
            nested = seen | {id(item)}
            return {
                key: visit(member, depth + 1, nested)
                for key, member in item.items()
            }
        if isinstance(item, (list, tuple)):
            nested = seen | {id(item)}
            return [visit(member, depth + 1, nested) for member in item]
        return item

    return visit(value, 0, frozenset())


def encode(observed: Mapping[Any, Any]) -> dict[str, Any]:
    def key_text(key: Any) -> str:
        # Built-in JSON key spelling, including numeric/bool/null keys.
        # All reached native value traversal has already completed.
        return next(iter(json.loads(json.dumps({key: None}))))

    def encode_value(item: Any) -> dict[str, Any]:
        if item is None:
            return {'value': {'t': 'null'}}
        if isinstance(item, bool):
            return {'value': {'t': 'bool', 'v': item}}
        if isinstance(item, int):
            # JSON uses an integer subclass's stored value, not its __str__
            # or __int__ override. The latter was reached separately above.
            return {'integer': json.dumps(item)}
        if isinstance(item, str):
            return {'value': {'t': 'utf8', 'v': item}}
        if isinstance(item, float) and math.isfinite(item):
            return {'value': {'t': 'f64', 'v': json.dumps(item)}}
        if isinstance(item, list):
            return {'array': [encode_value(member) for member in item]}
        if isinstance(item, dict):
            return {
                'object': [
                    (key_text(key), encode_value(member))
                    for key, member in item.items()
                ]
            }
        raise RuntimeError(
            'native inference options reached the primitive encoder'
        )

    return {
        'entries': [
            (key_text(key), encode_value(item))
            for key, item in observed.items()
        ]
    }


def can_encode(value: Any) -> bool:
    """Whether core JSON can hold this completed native observation."""
    if value is None or type(value) in (bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is str:
        # JSON can carry lone surrogate escapes; native strings can retain
        # those while the core's Unicode text representation cannot.
        return not any(0xD800 <= ord(char) <= 0xDFFF for char in value)
    if type(value) is list:
        return all(can_encode(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str and can_encode(key) and can_encode(item)
            for key, item in value.items()
        )
    return False
