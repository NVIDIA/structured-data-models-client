# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from collections import deque
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel, SecretStr

from predictive_query.schema.typing import WITH_PYDANTIC_V2

# Immutable types that are safe to return as-is (no copying needed)
_IMMUTABLE_TYPES = (
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
    datetime,
    date,
    time,
    timedelta,
    Decimal,
    UUID,
    Enum,
    PurePath,
    SecretStr,
    # Note: frozenset is immutable but may contain items needing conversion
)


def _convert_value(value: Any) -> Any:
    r"""Convert supported values into independent JSON-friendly objects.

    Dataclasses, Pydantic models, named tuples and containers are recursively
    copied. Immutable primitive values are returned unchanged, and an
    unsupported type raises ``TypeError``.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_dict(value)
    if isinstance(value, dict):
        return type(value)(
            (_convert_value(k), _convert_value(v)) for k, v in value.items()
        )
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*[_convert_value(v) for v in value])
    if isinstance(value, (list, tuple)):
        return type(value)(_convert_value(v) for v in value)
    if isinstance(value, (set, frozenset, deque)):
        return [_convert_value(v) for v in value]
    if isinstance(value, BaseModel):
        if WITH_PYDANTIC_V2:
            return _convert_value(value.model_dump())  # type: ignore[attr-defined]
        return _convert_value(value.dict())
    if isinstance(value, _IMMUTABLE_TYPES):
        return value
    if isinstance(value, bytearray):
        raise TypeError(
            f'dataclass_to_dict does not support {type(value).__name__}. '
            f'Convert to a supported type before serialization.'
        )
    raise TypeError(
        f'dataclass_to_dict encountered unexpected type '
        f'{type(value).__name__}. Add it to _IMMUTABLE_TYPES if '
        f'immutable, or handle it explicitly.'
    )


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    r"""Convert a dataclass to a dictionary.

    Defensive alternative to asdict() that works in distributed contexts
    (e.g., Databricks Serverless) where asdict() fails with nested Pydantic.

    Unlike asdict(), this function does NOT use deepcopy, but it does
    recursively process nested dicts, lists, and dataclasses to avoid
    sharing references with the original object.
    """
    result = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        result[field.name] = _convert_value(value)

    # Handle pydantic models with extra='allow' that store additional
    # fields in __pydantic_extra__ (e.g., HopConfig with edge overrides)
    if hasattr(obj, '__pydantic_extra__'):
        extra = obj.__pydantic_extra__
        if extra:
            for key, value in extra.items():
                result[key] = _convert_value(value)

    return result


if WITH_PYDANTIC_V2:
    from pydantic_core import to_jsonable_python as lib_encoder
else:
    from pydantic.json import (  # type: ignore[no-redef]
        pydantic_encoder as lib_encoder,
    )

T = TypeVar('T')


def trusted_encoder(obj: Any) -> Any:
    if isinstance(obj, SecretStr):
        return obj.get_secret_value()

    if WITH_PYDANTIC_V2:
        return lib_encoder(obj, context={'insecure': True})

    return lib_encoder(obj)


def to_json(pydantic_obj: Any, insecure: bool = False) -> str:
    r"""Encodes a pydantic object into JSON.

    The `insecure` flag should only be used by trusted internal code where the
    output of the JSON is not accessible to any users and `SecretStr`s are
    hidden in some other fashion.
    """
    encoder = trusted_encoder if insecure else lib_encoder

    return json.dumps(
        pydantic_obj,
        default=encoder,
        allow_nan=True,
        indent=2,
    )


def to_json_dict(pydantic_obj: Any, insecure: bool = False) -> dict[str, Any]:
    return json.loads(to_json(pydantic_obj, insecure=insecure))


def from_json(obj: Any, cls: type[T]) -> T:
    if isinstance(obj, str):
        obj = json.loads(obj)
    if WITH_PYDANTIC_V2:
        from pydantic import TypeAdapter  # type: ignore[attr-defined]

        adapter = TypeAdapter(cls)
        return adapter.validate_python(obj)
    from pydantic import parse_obj_as  # type: ignore

    return parse_obj_as(cls, obj)  # type: ignore
