# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A request's inference options, written in the types the contract gives
them.

A NIM reads these leniently, so a boolean spelled ``'no'`` asks for nothing,
and refuses a value it cannot read as its type. Writing them here, before
any backend is asked, sends a NIM only what its contract allows, and hands a
model run in this process the same values a NIM would have read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from sdm_client.errors import StructuredDataError

#: The longest ``run_mode`` the contract allows.
_MAX_TEXT = 1024
_TRUE = frozenset({'1', 'on', 't', 'true', 'y', 'yes'})
_FALSE = frozenset({'0', 'off', 'f', 'false', 'n', 'no'})


def typed_inference(inference: Mapping[str, Any]) -> dict[str, Any]:
    r"""``inference`` with each option the contract types in its type.

    Raises:
        StructuredDataError: With code ``INVALID_REQUEST``, for a value that
            cannot be read as its type.
    """
    typed = dict(inference)
    run_mode = typed.get('run_mode')
    if run_mode is not None and (
        not isinstance(run_mode, str) or len(run_mode) > _MAX_TEXT
    ):
        raise StructuredDataError(
            f'inference run_mode is {run_mode!r}; it is text of at most '
            f'{_MAX_TEXT} characters',
            code='INVALID_REQUEST',
        )
    if 'use_prediction_time' in typed:
        typed['use_prediction_time'] = _boolean(typed['use_prediction_time'])
    if 'inference_config' in typed and not isinstance(
        typed['inference_config'], Mapping
    ):
        raise StructuredDataError(
            f'inference_config is {typed["inference_config"]!r}; it is a '
            f'mapping of option names to values',
            code='INVALID_REQUEST',
        )
    return typed


def _boolean(value: Any) -> bool:
    r"""``value`` as the boolean a NIM reads it as.

    A NumPy scalar is read as the Python value it holds: a frame hands those
    back, and a payload built by hand keeps them, but ``np.False_`` is not a
    ``bool`` and ``np.int64(0)`` is not an ``int``. One that holds no Python
    value, such as ``np.longdouble``, whose ``item()`` is itself, is refused
    like any other value that is not a boolean.
    """
    if isinstance(value, np.generic):
        plain = value.item()
        if not isinstance(plain, np.generic):
            return _boolean(plain)
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in _TRUE | _FALSE:
        return value.lower() in _TRUE
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise StructuredDataError(
        f'inference use_prediction_time is {value!r}, which is not a boolean',
        code='INVALID_REQUEST',
    )
