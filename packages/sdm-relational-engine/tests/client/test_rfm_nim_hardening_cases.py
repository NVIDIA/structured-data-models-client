# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from rfm_nim_hardening_cases import (
    EXPECTED_REJECTION_CASES,
    REGRESSION_REJECTION_CASES,
    ExpectedRejectionCase,
    RejectionCase,
)
from rfm_nim_payloads import NIM_V1_PREDICTION_PATH, NIM_V1_SESSIONS_PATH

ALL_CASES = REGRESSION_REJECTION_CASES


def test_rejection_catalog_is_unique_and_ticket_free() -> None:
    case_ids = [case.case_id for case in ALL_CASES]

    assert len(case_ids) == len(set(case_ids))
    assert all(
        case.path in {NIM_V1_PREDICTION_PATH, NIM_V1_SESSIONS_PATH}
        for case in ALL_CASES
    )


def test_expected_rejection_catalog_covers_audited_statuses() -> None:
    assert {case.expected_status for case in EXPECTED_REJECTION_CASES} == {
        400,
        404,
        405,
        413,
        415,
        422,
    }


@pytest.mark.parametrize(
    'case',
    EXPECTED_REJECTION_CASES,
    ids=lambda case: case.case_id,
)
def test_expected_rejection_factories_return_fresh_request_data(
    case: ExpectedRejectionCase,
) -> None:
    first = case.request_kwargs()
    second = case.request_kwargs()

    assert first == second
    assert first is not second
    if first:
        value_key = 'json' if 'json' in first else 'headers'
        assert first[value_key] is not second[value_key]


@pytest.mark.parametrize('case', ALL_CASES, ids=lambda case: case.case_id)
def test_rejection_factories_return_fresh_request_data(
    case: RejectionCase,
) -> None:
    first = case.request_kwargs()
    second = case.request_kwargs()

    assert first == second
    assert first is not second
    if 'json' in first:
        assert first['json'] is not second['json']
    else:
        assert first['headers'] is not second['headers']
        assert first['headers']['Content-Type'] == 'application/json'


def test_raw_rejection_cases_preserve_wire_level_reproductions() -> None:
    by_id = {
        case.case_id: case.request_kwargs()['data']
        for case in ALL_CASES
        if case.raw_body_factory is not None
    }

    assert b'-Infinity' in by_id['negative-infinity-regression-target']
    assert b'NaN' in by_id['nan-metadata']
    assert b'NaN' in by_id['nan-regression-target-session']
    with pytest.raises(UnicodeDecodeError):
        by_id['invalid-utf8-feature'].decode('utf-8')
    assert b'"\\ud800"' in by_id['lone-surrogate-feature']
    with pytest.raises(ValueError):
        json.loads(
            by_id['negative-infinity-regression-target'],
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)
            ),
        )


def test_structured_rejection_cases_preserve_semantic_reproductions() -> None:
    by_id = {
        case.case_id: case.request_kwargs()['json']
        for case in ALL_CASES
        if case.payload_factory is not None
    }

    assert (
        by_id['null-classification-target']['context']['instance_table'][
            'rows'
        ][0][1]
        is None
    )
    assert (
        by_id['null-regression-target']['context']['instance_table']['rows'][0][
            1
        ]
        is None
    )
    assert (
        by_id['null-multiclass-target']['context']['instance_table']['rows'][0][
            1
        ]
        is None
    )
    assert by_id['empty-context']['context']['instance_table']['rows'] == []
    assert (
        by_id['multiclass-target-outside-classes']['context']['instance_table'][
            'rows'
        ][0][1]
        == 'gold'
    )
    assert by_id['multiclass-duplicate-classes']['task']['target'][
        'classes'
    ] == ['bronze', 'bronze']
    assert (
        by_id['whitespace-integer-ids']['predict']['instance_table']['rows'][0][
            0
        ]
        == ' 601'
    )
    assert (
        by_id['duplicate-context-column']['context']['instance_table'][
            'columns'
        ].count('status')
        == 2
    )
    assert (
        by_id['duplicate-predict-column']['predict']['instance_table'][
            'columns'
        ].count('anchor_time')
        == 2
    )
    assert 'predict' not in by_id['empty-context-session']
    assert 'predict' not in by_id['duplicate-context-column-session']
