# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operation-local native numeric owners; comparison policy stays in Rust."""

from __future__ import annotations

import json
import sys
from decimal import InvalidOperation
from typing import Any

import numpy as np
import pandas as pd

from .._routing import RouteError
from . import codec, results
from .run import run
from .tabular_index import _numeric_source_rules, _scalar

_FIXED = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ulonglong,
    np.float16,
    np.float32,
    np.float64,
)
_NUMERIC = (bool, int, float, *_FIXED)
_PROBE_POLICIES = {
    '2.2.2': 'type_and_value',
    '2.2.3': 'type_and_value',
    '2.3.3': 'type_and_value',
    '3.0.6': 'type_value_and_decimal_invalid',
    '3.1.0': 'type_value_and_decimal_invalid',
}


def eligible(index: pd.Index) -> bool:
    """Recognize exact numeric owners without reading missingness or comparing."""
    if type(index) is not pd.Index or type(index._values) is not np.ndarray:
        return False
    values = index._values
    if values.dtype != np.dtype(object):
        return False
    fixed = False
    for value in values:
        if value is None or value is pd.NA or value is pd.NaT:
            continue
        owner = type(value)
        if owner not in _NUMERIC:
            return False
        fixed |= owner in _FIXED
    return fixed


def _fact(value: Any) -> dict[str, Any]:
    if value is None or value is pd.NA or value is pd.NaT:
        return {
            'kind': 'native_missing',
            'owner': 'none'
            if value is None
            else 'ambiguous'
            if value is pd.NA
            else 'temporal',
        }
    if type(value) not in _NUMERIC:
        raise RouteError(
            'this mutated numeric index owner is not represented by the core'
        )
    return _scalar(value)


class NumericOwners:
    """Retain live probe, captured sort and exception owners for one run."""

    def __init__(self, index: pd.Index, source_rules: str) -> None:
        self.index: Any = index
        self.probe: Any = index._values
        self.copied: Any = None
        self.source_rules = source_rules
        self.owners: dict[int, Any] = {}
        self.identities: dict[int, int] = {}
        self.exceptions: dict[int, Exception] = {}

    def close(self) -> None:
        self.index = self.probe = self.copied = None
        self.owners.clear()
        self.identities.clear()
        self.exceptions.clear()

    def _retain(self, value: Any) -> dict[str, Any]:
        identity = id(value)
        owner = self.identities.get(identity)
        if owner is None:
            owner = len(self.owners) + 1
            self.owners[owner] = value
            self.identities[identity] = owner
        return {'owner': owner, 'fact': _fact(value)}

    def _environment(self) -> dict[str, Any]:
        if self.source_rules == 'common_type_only':
            policy = 'common_type'
        elif self.source_rules == 'fixed_weak_scalars':
            policy = 'weak_scalars'
        else:
            observe = getattr(np, '_get_promotion_state', None)
            mode = None if observe is None else observe()
            policies = {
                'mutable_fixed_integer_pairs': {
                    'legacy': 'exact_integer_pairs',
                    'weak': 'weak_fixed_integer_pairs',
                },
                'mutable_exact_unbounded_integers': {
                    'legacy': 'legacy_exact_unbounded_integers',
                    'weak': 'weak_scalars',
                },
            }
            policy = policies[self.source_rules].get(mode)
            if policy is None:
                raise RouteError(
                    'this live numeric promotion mode is not yet represented by the core'
                )
        return {'kind': 'comparison_environment', 'policy': policy}

    def _capture(self) -> tuple[dict[str, Any], dict[int, bytes]]:
        # Follow the retained index, never a replacement frame.index. The
        # probe's ndarray remains separately retained across handler mutation.
        values = self.index._values
        mask = np.asarray(pd.isna(values))
        self.copied = values[~mask]
        positions = np.arange(len(values), dtype=np.intp)
        source_positions = positions[~mask].tolist()
        missing_positions = positions[mask].tolist()
        facts = [self._retain(value) for value in self.copied]
        return (
            {
                'kind': 'sort_owners',
                'owners_and_facts': 1,
                'source_positions': 2,
                'missing_positions': 3,
            },
            {
                1: codec.attachment(facts),
                2: codec.attachment(source_positions),
                3: codec.attachment(missing_positions),
            },
        )

    def _convert(self, request: dict[str, Any]) -> dict[str, Any]:
        owner = request['owner']
        destination = request['destination']
        kind, width = destination['kind'], destination['width']
        widths = (2, 4, 8) if kind == 'binary_float' else (1, 2, 4, 8)
        if owner not in self.owners or width not in widths:
            raise codec.CodecError('invalid numeric owner conversion request')
        prefix = {
            'signed_integer': 'i',
            'unsigned_integer': 'u',
            'binary_float': 'f',
        }.get(kind)
        if prefix is None:
            raise codec.CodecError('invalid numeric conversion destination')
        storage = np.empty((), dtype=np.dtype(prefix + str(width)))
        storage[()] = self.owners[owner]
        return {
            'kind': 'converted',
            'bits': int.from_bytes(storage.tobytes(), sys.byteorder),
        }

    def __call__(
        self, request: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[int, bytes]]:
        try:
            kind = request['kind']
            if kind == 'load_probe_owner':
                position = request['position']
                if type(position) is not int or not 0 <= position < len(
                    self.probe
                ):
                    raise codec.CodecError('invalid numeric probe position')
                return {
                    'kind': 'probe_owner',
                    **self._retain(self.probe[position]),
                }, {}
            if kind == 'observe_comparison_environment':
                return self._environment(), {}
            if kind == 'capture_sort_owners':
                return self._capture()
            if kind == 'convert_owner':
                return self._convert(request), {}
            raise codec.CodecError('unknown numeric owner request')
        except (RouteError, codec.CodecError):
            raise
        except Exception as error:  # noqa: BLE001 - retain the original native exception
            identity = len(self.exceptions) + 1
            self.exceptions[identity] = error
            classification = (
                'type_error'
                if isinstance(error, TypeError)
                else 'value_error'
                if isinstance(error, ValueError)
                else 'decimal_invalid_operation'
                if isinstance(error, InvalidOperation)
                else 'other'
            )
            return {
                'kind': 'raised',
                'exception': identity,
                'class': classification,
            }, {}


def order(index: pd.Index) -> dict[str, Any]:
    policy = _PROBE_POLICIES.get(pd.__version__)
    if policy is None:
        raise RouteError(
            'this numeric index probe producer is not yet represented by the core'
        )
    source = _numeric_source_rules()
    owners = NumericOwners(index, source)
    raised: Exception | None = None
    try:
        finished = run(
            {'op': 'order_numeric_owners', 'input': 1},
            {
                1: codec.attachment(
                    {
                        'length': len(index),
                        'sort_profile': 'scalar',
                        'source_rules': source,
                        'probe_exception_policy': policy,
                    }
                )
            },
            numeric_owner=owners,
        )
        result = results.expect(finished.result, results.DirectValue)
        answer = json.loads(finished.take(result.value))
        if answer['kind'] == 'propagate':
            raised = owners.exceptions[answer['exception']]
        else:
            return answer
    finally:
        owners.close()
    assert raised is not None
    raise raised
