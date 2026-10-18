# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Reading a request's inference options the way a NIM reads them.

A NIM types the block before any model judges it: a misspelled kind of value
is refused as invalid, and a boolean written another way is read as the
boolean it spells, so ``'no'`` asks for nothing.
"""

from __future__ import annotations

from typing import Any, NoReturn

from sdm_client.in_process.refusal import refusal

_MAX_TEXT = 1024
_TRUE = frozenset({'1', 'on', 't', 'true', 'y', 'yes'})
_FALSE = frozenset({'0', 'off', 'f', 'false', 'n', 'no'})


def read_inference(inference: Any) -> dict[str, Any]:
    r"""The inference options, typed, with each field a NIM fills in by
    default present."""
    if inference is None:
        inference = {}
    if not isinstance(inference, dict):
        _invalid(
            'inference',
            'Input should be a valid dictionary or object to extract fields '
            'from',
        )
    options = {
        'run_mode': None,
        'use_prediction_time': False,
        'inference_config': {},
        **inference,
    }
    run_mode = options['run_mode']
    if run_mode is not None and not isinstance(run_mode, str):
        _invalid('inference.run_mode', 'Input should be a valid string')
    if isinstance(run_mode, str) and len(run_mode) > _MAX_TEXT:
        _invalid(
            'inference.run_mode',
            f'String should have at most {_MAX_TEXT} characters',
        )
    options['use_prediction_time'] = _boolean(options['use_prediction_time'])
    if not isinstance(options['inference_config'], dict):
        _invalid(
            'inference.inference_config', 'Input should be a valid dictionary'
        )
    return options


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in _TRUE | _FALSE:
        return value.lower() in _TRUE
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    unread = isinstance(value, (int, str))
    _invalid(
        'inference.use_prediction_time',
        'Input should be a valid boolean'
        + (', unable to interpret input' if unread else ''),
    )


def _invalid(name: str, reason: str) -> NoReturn:
    raise refusal(
        'VALIDATION_FAILED', f'Request validation failed. {name}: {reason}'
    )
