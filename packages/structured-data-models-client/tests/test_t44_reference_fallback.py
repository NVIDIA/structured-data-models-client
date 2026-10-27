# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual public whole-reference and per-batch native cast fallbacks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import warnings

import pandas as pd
import pytest

from sdm_client import StructuredDataError
from sdm_client.materializers.relational import RelationalMaterializer

from test_t44_text_frames import (
    core as core,
    dt3_no_time_body,
    p3e2_warning_codes,
    question,
    shop,
    text,
)

REFERENCE = json.loads(
    Path(__file__).with_name('t44_reference_fallback.json').read_text()
)
CELL = 'pandas2' if int(pd.__version__.split('.')[0]) == 2 else 'pandas3'


def chain(error):
    result = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        result.append(
            {
                'module': type(error).__module__,
                'type': type(error).__name__,
                'message': str(error),
                'code': getattr(error, 'code', None),
            }
        )
        error = (
            error.__cause__
            if error.__cause__ is not None
            else error.__context__
        )
    return result


@pytest.mark.parametrize(
    'case',
    REFERENCE['cells'][CELL],
    ids=lambda record: f'{record["storage"]}-{record["case"]}',
)
def test_public_native_reference_fallback_preserves_payload_and_cause(
    core, case, monkeypatch
):
    expected = copy.deepcopy(case)
    frames = shop(case['storage'])
    keys = [f'u{i}' for i in range(1, 9)]
    frames['users']['user_id'] = text(keys, case['storage'])
    frames['orders']['user_id'] = text(
        [key for key in keys for _ in range(8)], case['storage']
    )
    asked = question(frames, 'age')
    if core:
        for payload in expected['outcome'].get('payloads', []):
            dt3_no_time_body(payload)
        expected['warnings'] = p3e2_warning_codes(expected['warnings'])

        def legacy_model(*args, **kwargs):
            raise AssertionError(
                'core reference conversion used the Python engine'
            )

        monkeypatch.setattr(RelationalMaterializer, '_get_model', legacy_model)
    original = pd.Series.astype
    counts = {'whole': 0, 'batch': 0}
    faults = []

    def astype(series, dtype, *args, **kwargs):
        # Actual frozen values distinguish these two useful native domains.
        # The eight-row label lookup remains separately covered by its receipt.
        size = len(series)
        if size not in (4, 5):
            return original(series, dtype, *args, **kwargs)
        values = series.tolist()
        if not all(type(value) is str and value in keys for value in values):
            return original(series, dtype, *args, **kwargs)
        site = 'whole' if size == 5 else 'batch'
        counts[site] += 1
        wanted = case['case']
        inject = (
            (
                site == 'whole'
                and wanted.startswith('whole-')
                and counts[site] == 1
            )
            or (
                site == 'batch'
                and wanted == 'batch-type-once'
                and counts[site] == 1
            )
            or (
                site == 'batch'
                and wanted == 'batch-type-both'
                and counts[site] <= 2
            )
        )
        if inject:
            failure = (
                OverflowError if wanted == 'whole-overflow-once' else TypeError
            )
            faults.append((site, counts[site], failure.__name__))
            raise failure('explicit native reference cast witness')
        return original(series, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.Series, 'astype', astype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if expected['outcome']['kind'] == 'error':
            with pytest.raises(StructuredDataError) as failure:
                asked.materialize(
                    indices=['u7', 'u8'],
                    num_context_samples=3,
                    num_neighbors=[3],
                    num_hops=1,
                    rows_per_call=1,
                    random_seed=7,
                )
            assert chain(failure.value) == expected['outcome']['chain']
        else:
            request = asked.materialize(
                indices=['u7', 'u8'],
                num_context_samples=3,
                num_neighbors=[3],
                num_hops=1,
                rows_per_call=1,
                random_seed=7,
            )
            assert list(request.payloads) == expected['outcome']['payloads']
    assert faults
    assert [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in caught
    ] == expected['warnings']
