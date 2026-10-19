# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Reading a request's inference options the way a NIM reads them.

A NIM types the block before any model judges it: a misspelled kind of value
is refused as invalid, and a boolean written another way is read as the
boolean it spells, so ``'no'`` asks for nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

_MAX_TEXT = 1024
_TRUE = frozenset({'1', 'on', 't', 'true', 'y', 'yes'})
_FALSE = frozenset({'0', 'off', 'f', 'false', 'n', 'no'})


def inference_problems(inference: Any) -> Iterator[tuple[str, str]]:
    r"""Each way the block fails the NIM's typing, in the NIM's order."""
    if inference is None:
        return
    if not isinstance(inference, dict):
        yield (
            'inference',
            'Input should be a valid dictionary or object to extract fields '
            'from',
        )
        return
    run_mode = inference.get('run_mode')
    if run_mode is not None and not isinstance(run_mode, str):
        yield 'inference.run_mode', 'Input should be a valid string'
    if isinstance(run_mode, str) and len(run_mode) > _MAX_TEXT:
        yield (
            'inference.run_mode',
            f'String should have at most {_MAX_TEXT} characters',
        )
    if 'use_prediction_time' in inference:
        value = inference['use_prediction_time']
        if _boolean(value) is None:
            unread = isinstance(value, (int, str))
            yield (
                'inference.use_prediction_time',
                'Input should be a valid boolean'
                + (', unable to interpret input' if unread else ''),
            )
    if not isinstance(inference.get('inference_config', {}), dict):
        yield (
            'inference.inference_config',
            'Input should be a valid dictionary',
        )


def read_inference(inference: Any) -> dict[str, Any]:
    r"""The inference options, typed, with each field a NIM fills in by
    default present.

    Read once :func:`inference_problems` finds none.
    """
    options = {
        'run_mode': None,
        'use_prediction_time': False,
        'inference_config': {},
        **(inference or {}),
    }
    options['use_prediction_time'] = bool(
        _boolean(options['use_prediction_time'])
    )
    return options


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in _TRUE | _FALSE:
        return value.lower() in _TRUE
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None
