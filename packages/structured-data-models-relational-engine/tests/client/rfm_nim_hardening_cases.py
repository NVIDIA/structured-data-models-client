# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rfm_nim_payloads import (
    NIM_V1_PREDICTION_PATH,
    NIM_V1_SESSIONS_PATH,
    nim_v1_multiclass_payload,
    nim_v1_regression_payload,
    nim_v1_session_predict_minimal_payload,
    nim_v1_smoke_payload,
)


@dataclass(frozen=True)
class RejectionCase:
    case_id: str
    expected_statuses: tuple[int, ...]
    payload_factory: Callable[[], dict[str, Any]] | None = None
    raw_body_factory: Callable[[], bytes] | None = None
    path: str = NIM_V1_PREDICTION_PATH

    def __post_init__(self) -> None:
        factory_count = sum(
            factory is not None
            for factory in (self.payload_factory, self.raw_body_factory)
        )
        if factory_count != 1:
            raise ValueError(
                'rejection cases require exactly one request-body factory'
            )

    def request_kwargs(self) -> dict[str, Any]:
        if self.payload_factory is not None:
            return {'json': self.payload_factory()}
        assert self.raw_body_factory is not None
        return {
            'data': self.raw_body_factory(),
            'headers': {
                'Content-Type': 'application/json',
            },
        }


@dataclass(frozen=True)
class ExpectedRejectionCase:
    case_id: str
    method: str
    path: str
    expected_status: int
    request_factory: Callable[[], dict[str, Any]]
    problem_details: bool = True

    def request_kwargs(self) -> dict[str, Any]:
        return self.request_factory()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(',', ':'),
    ).encode('utf-8')


def _replace_marker(
    payload: dict[str, Any],
    marker: str,
    replacement: bytes,
) -> bytes:
    body = _json_bytes(payload)
    encoded_marker = json.dumps(marker).encode('ascii')
    assert body.count(encoded_marker) == 1
    return body.replace(encoded_marker, replacement, 1)


def _as_session_create(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop('predict', None)
    payload.pop('output', None)
    payload.pop('inference', None)
    return payload


def _null_classification_target() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['context']['instance_table']['rows'][0][1] = None
    return payload


def _null_regression_target() -> dict[str, Any]:
    payload = nim_v1_regression_payload()
    payload['context']['instance_table']['rows'][0][1] = None
    return payload


def _null_regression_session() -> dict[str, Any]:
    return _as_session_create(_null_regression_target())


def _null_multiclass_target() -> dict[str, Any]:
    payload = nim_v1_multiclass_payload()
    payload['context']['instance_table']['rows'][0][1] = None
    return payload


def _nonfinite_regression_target(token: str) -> bytes:
    payload = nim_v1_regression_payload()
    marker = '__NONFINITE_TARGET__'
    payload['context']['instance_table']['rows'][0][1] = marker
    return _replace_marker(payload, marker, token.encode('ascii'))


def _nonfinite_metadata() -> bytes:
    payload = nim_v1_smoke_payload()
    marker = '__NONFINITE_METADATA__'
    payload['metadata'] = {'probe': marker}
    return _replace_marker(payload, marker, b'NaN')


def _nonfinite_regression_session() -> bytes:
    payload = _as_session_create(nim_v1_regression_payload())
    marker = '__NONFINITE_TARGET__'
    payload['context']['instance_table']['rows'][0][1] = marker
    return _replace_marker(payload, marker, b'NaN')


def _invalid_utf8_feature() -> bytes:
    payload = nim_v1_smoke_payload()
    marker = '__INVALID_UTF8__'
    payload['context']['related_tables']['accounts']['rows'][0][1] = marker
    return _replace_marker(payload, marker, b'"\xff"')


def _lone_surrogate_feature() -> bytes:
    payload = nim_v1_smoke_payload()
    marker = '__LONE_SURROGATE__'
    payload['context']['related_tables']['accounts']['rows'][0][1] = marker
    return _replace_marker(payload, marker, b'"\\ud800"')


def _empty_context() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['context']['instance_table']['rows'] = []
    payload['context']['related_tables']['accounts']['rows'] = []
    return payload


def _empty_context_session() -> dict[str, Any]:
    return _as_session_create(_empty_context())


def _multiclass_target_outside_classes() -> dict[str, Any]:
    payload = nim_v1_multiclass_payload()
    payload['context']['instance_table']['rows'][0][1] = 'gold'
    return payload


def _multiclass_duplicate_classes() -> dict[str, Any]:
    payload = nim_v1_multiclass_payload()
    payload['task']['target']['classes'] = ['bronze', 'bronze']
    return payload


def _whitespace_integer_ids() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    context_ids = (' 501', ' 502')
    for table in (
        payload['context']['instance_table'],
        payload['context']['related_tables']['accounts'],
    ):
        for row, value in zip(table['rows'], context_ids, strict=True):
            row[0] = value
    for table in (
        payload['predict']['instance_table'],
        payload['predict']['related_tables']['accounts'],
    ):
        table['rows'][0][0] = ' 601'
    return payload


def _duplicate_context_column() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    table = payload['context']['instance_table']
    table['columns'].insert(2, 'status')
    for row in table['rows']:
        row.insert(2, row[1])
    return payload


def _duplicate_predict_column() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    table = payload['predict']['instance_table']
    table['columns'].append('anchor_time')
    for row in table['rows']:
        row.append(row[1])
    return payload


def _duplicate_context_column_session() -> dict[str, Any]:
    return _as_session_create(_duplicate_context_column())


def _malformed_json_request() -> dict[str, Any]:
    return {
        'data': b'{"model":',
        'headers': {
            'Content-Type': 'application/json',
        },
    }


def _unknown_session_prediction_request() -> dict[str, Any]:
    return {'json': nim_v1_session_predict_minimal_payload()}


def _empty_request() -> dict[str, Any]:
    return {}


def _oversized_string_request() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['context']['related_tables']['accounts']['rows'][0][1] = 'x' * 1025
    return {'json': payload}


def _unsupported_media_type_request() -> dict[str, Any]:
    return {
        'data': _json_bytes(nim_v1_smoke_payload()),
        'headers': {
            'Content-Type': 'text/plain',
        },
    }


def _short_row_request() -> dict[str, Any]:
    payload = nim_v1_smoke_payload()
    payload['context']['instance_table']['rows'][0] = [501, True]
    return {'json': payload}


EXPECTED_REJECTION_CASES = (
    ExpectedRejectionCase(
        'malformed-json-400',
        'POST',
        NIM_V1_PREDICTION_PATH,
        400,
        _malformed_json_request,
    ),
    ExpectedRejectionCase(
        'unknown-session-404',
        'POST',
        f'{NIM_V1_SESSIONS_PATH}/sess_missing_rfm_sdk_hardening/predictions',
        404,
        _unknown_session_prediction_request,
    ),
    ExpectedRejectionCase(
        'unsupported-method-405',
        'GET',
        NIM_V1_PREDICTION_PATH,
        405,
        _empty_request,
        problem_details=False,
    ),
    ExpectedRejectionCase(
        'oversized-string-413',
        'POST',
        NIM_V1_PREDICTION_PATH,
        413,
        _oversized_string_request,
    ),
    ExpectedRejectionCase(
        'unsupported-media-type-415',
        'POST',
        NIM_V1_PREDICTION_PATH,
        415,
        _unsupported_media_type_request,
    ),
    ExpectedRejectionCase(
        'short-row-422',
        'POST',
        NIM_V1_PREDICTION_PATH,
        422,
        _short_row_request,
    ),
)


REGRESSION_REJECTION_CASES = (
    RejectionCase(
        'null-classification-target',
        (422,),
        payload_factory=_null_classification_target,
    ),
    RejectionCase(
        'null-regression-target',
        (422,),
        payload_factory=_null_regression_target,
    ),
    RejectionCase(
        'null-regression-target-session',
        (422,),
        payload_factory=_null_regression_session,
        path=NIM_V1_SESSIONS_PATH,
    ),
    RejectionCase(
        'negative-infinity-regression-target',
        (400,),
        raw_body_factory=lambda: _nonfinite_regression_target('-Infinity'),
    ),
    RejectionCase(
        'nan-metadata',
        (400,),
        raw_body_factory=_nonfinite_metadata,
    ),
    RejectionCase(
        'nan-regression-target-session',
        (400,),
        raw_body_factory=_nonfinite_regression_session,
        path=NIM_V1_SESSIONS_PATH,
    ),
    RejectionCase(
        'invalid-utf8-feature',
        (400,),
        raw_body_factory=_invalid_utf8_feature,
    ),
    RejectionCase(
        'lone-surrogate-feature',
        (400, 422),
        raw_body_factory=_lone_surrogate_feature,
    ),
    RejectionCase(
        'empty-context',
        (422,),
        payload_factory=_empty_context,
    ),
    RejectionCase(
        'empty-context-session',
        (422,),
        payload_factory=_empty_context_session,
        path=NIM_V1_SESSIONS_PATH,
    ),
    RejectionCase(
        'multiclass-target-outside-classes',
        (422,),
        payload_factory=_multiclass_target_outside_classes,
    ),
    RejectionCase(
        'multiclass-duplicate-classes',
        (422,),
        payload_factory=_multiclass_duplicate_classes,
    ),
    RejectionCase(
        'whitespace-integer-ids',
        (422,),
        payload_factory=_whitespace_integer_ids,
    ),
    RejectionCase(
        'null-multiclass-target',
        (422,),
        payload_factory=_null_multiclass_target,
    ),
    RejectionCase(
        'duplicate-context-column',
        (422,),
        payload_factory=_duplicate_context_column,
    ),
    RejectionCase(
        'duplicate-predict-column',
        (422,),
        payload_factory=_duplicate_predict_column,
    ),
    RejectionCase(
        'duplicate-context-column-session',
        (422,),
        payload_factory=_duplicate_context_column_session,
        path=NIM_V1_SESSIONS_PATH,
    ),
)
