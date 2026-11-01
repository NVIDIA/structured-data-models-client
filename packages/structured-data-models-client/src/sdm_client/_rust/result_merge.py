# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe merge descriptors; preserve original result items and arbitrary owners."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sdm_client._rust.direct import _call


def _candidate(
    value: Any, ancestors: frozenset[int] = frozenset(), depth: int = 0
) -> dict[str, Any]:
    owner = type(value)
    if value is pd.NaT:
        return {'kind': 'json', 'value': 'NaN'}
    if owner in (list, tuple, dict):
        # Keep descriptor nesting below JSON's own limit and observe cycles as
        # an unsupported value so Rust still checks model/output errors first.
        if depth >= 16 or id(value) in ancestors:
            return {
                'kind': 'unsupported',
                'type_name': 'recursive or deeply nested container',
            }
        ancestors = ancestors | {id(value)}
    if owner in (type(None), bool, int, float, str):
        return {'kind': 'json', 'value': json.dumps(value)}
    if owner is bytes:
        return {'kind': 'bytes', 'value': list(value)}
    if isinstance(value, (np.bool_, np.integer, np.floating, np.str_)):
        if isinstance(value, np.floating) and value.dtype.itemsize > 8:
            return {'kind': 'unsupported', 'type_name': owner.__name__}
        return _candidate(value.item())
    if owner in (list, tuple):
        return {
            'kind': 'list' if owner is list else 'tuple',
            'values': [_candidate(v, ancestors, depth + 1) for v in value],
        }
    if owner is dict:
        return {
            'kind': 'mapping',
            'entries': [
                (
                    _candidate(k, ancestors, depth + 1),
                    _candidate(v, ancestors, depth + 1),
                )
                for k, v in value.items()
            ],
        }
    if owner is date:
        return {'kind': 'date', 'ordinal': value.toordinal()}
    if owner is datetime:
        aware = value.tzinfo is not None
        if aware and type(value.tzinfo) not in (timezone, ZoneInfo):
            return {
                'kind': 'unsupported',
                'type_name': type(value.tzinfo).__name__,
            }
        # Subtract the offset as integer ticks rather than constructing a UTC
        # datetime, which would overflow at the built-in year boundaries.
        offset = value.utcoffset() if aware else None
        delta = value.replace(tzinfo=None) - datetime(1970, 1, 1)
        ticks = (
            delta.days * 86400 + delta.seconds
        ) * 1_000_000 + delta.microseconds
        if offset is not None:
            ticks -= (
                offset.days * 86400 + offset.seconds
            ) * 1_000_000 + offset.microseconds
        return {
            'kind': 'date_time',
            'ticks': str(ticks),
            'unit': 'us',
            'aware': aware,
        }
    if owner in (pd.Timestamp, np.datetime64):
        aware = owner is pd.Timestamp and value.tzinfo is not None
        native = value.asm8 if owner is pd.Timestamp else value
        if np.isnat(native):
            return {'kind': 'json', 'value': 'NaN'}
        unit, count = np.datetime_data(native.dtype)
        ticks = int(native.view('i8')) * count
        if unit in ('s', 'ms', 'us', 'ns', 'ps', 'fs', 'as'):
            return {
                'kind': 'date_time',
                'ticks': str(ticks),
                'unit': unit,
                'aware': aware,
            }
        if unit in ('h', 'm'):
            return {
                'kind': 'date_time',
                'ticks': str(ticks * (3600 if unit == 'h' else 60)),
                'unit': 's',
                'aware': aware,
            }
        try:
            if unit == 'Y':
                return _candidate(date(1970 + ticks, 1, 1))
            if unit == 'M':
                year, month = divmod(1970 * 12 + ticks, 12)
                return _candidate(date(year, month + 1, 1))
            if unit in ('D', 'W'):
                return _candidate(
                    date.fromordinal(719163 + ticks * (7 if unit == 'W' else 1))
                )
        except (ValueError, OverflowError):
            pass
    return {'kind': 'unsupported', 'type_name': owner.__name__}


def merge(cls: Any, parts: Any) -> Any:
    # Only small descriptors cross the ABI. No item, metadata or explanation
    # is encoded, converted to Arrow, copied into Rust, or reconstructed.
    descriptors = [
        {
            'model': part.model,
            'request_id': part.request_id,
            'outputs': list(part.outputs),
            'has_explanation': False
            if len(parts) == 1
            else bool(part.explanation),
            'names_rows': part._names_its_rows,
            'names_step': part._names_its_step,
            'candidates': []
            if len(parts) == 1
            else [_candidate(v) for v in part.unscored_candidates],
        }
        for part in parts
    ]
    plan = _call('plan_result_merge', {'parts': descriptors})
    if plan['single']:
        return parts[0]
    first = parts[0]
    metadata = dict(first.metadata)
    metadata.update(
        batches=plan['batches'], batch_request_ids=plan['batch_request_ids']
    )
    candidates = [value for part in parts for value in part.unscored_candidates]
    explanation = plan['explanation_part']
    result = cls(
        model=first.model,
        request_id=first.request_id,
        items=[item for part in parts for item in part.items],
        metadata=metadata,
        explanation=None
        if explanation is None
        else parts[explanation].explanation,
        outputs=first.outputs,
        unscored_candidates=[
            candidates[i] for i in plan['candidate_positions']
        ],
    )
    result._names_its_rows = plan['names_rows']
    result._names_its_step = plan['names_step']
    return result
