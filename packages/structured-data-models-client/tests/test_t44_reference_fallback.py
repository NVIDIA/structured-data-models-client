# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual public whole-reference and per-batch native cast fallbacks."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_t44_text_frames import (
    core as core,
)
from test_t44_text_frames import (
    dt3_no_time_body,
    p3e2_warning_codes,
    question,
    shop,
    text,
)

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    StructuredDataError,
)
from sdm_client.materializers.relational import RelationalMaterializer

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
    counts = {'labels': 0, 'whole': 0, 'batch': 0}
    faults = []

    def astype(series, dtype, *args, **kwargs):
        # Actual frozen values distinguish these two useful native domains.
        # Eight sampled rows are the earlier actual label-source lookup.
        size = len(series)
        if size not in (4, 5, 8):
            return original(series, dtype, *args, **kwargs)
        values = series.tolist()
        if not all(type(value) is str and value in keys for value in values):
            return original(series, dtype, *args, **kwargs)
        site = 'labels' if size == 8 else 'whole' if size == 5 else 'batch'
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
        inject = (
            inject
            or (
                site == 'labels'
                and wanted == 'label-type-once'
                and counts[site] == 1
            )
            or (
                site == 'labels'
                and wanted == 'label-type-both'
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


@pytest.mark.parametrize(
    'case',
    REFERENCE['chunk_cells'][CELL],
    ids=lambda record: (
        f'{record["shape"]}-fault-{record["second_label_fault"]}'
    ),
)
def test_native_label_ranges_stop_at_actual_context_completeness(
    core, case, monkeypatch
):
    expected = copy.deepcopy(case)
    count = 25000
    frame = pd.DataFrame(
        {
            'user_id': pd.Series(
                [f'u{i}' for i in range(count)],
                dtype=pd.StringDtype(storage='python'),
            ),
            'age': np.arange(count, dtype=float),
        }
    )
    if case['shape'] == 'sparse':
        frame.loc[np.arange(count) % 4 != 0, 'age'] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': frame}, primary_keys={'users': 'user_id'}, time_columns={}
        )
        asked = PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        )
    if core:
        for payload in expected['outcome'].get('payloads', []):
            dt3_no_time_body(payload)
        expected['warnings'] = p3e2_warning_codes(expected['warnings'])
    label_resources = []
    final_batches = []
    if core:
        from sdm_client._rust import frame_facade

        prepare = frame_facade._prepare

        def observe_resources(value, bindings, state):
            if value['stage'] == 'label_batch':
                previous = value['previous']
                assert previous is None or type(previous) is dict
                assert (previous is None) == (value['range']['start'] == 0)
                if previous is not None:
                    assert previous['range']['stop'] == value['range']['start']
                # A reached batch binds graph, population, optional converted
                # keys and at most one previous real labelled-row resource.
                label_resources.append(len(bindings.attachments))
                assert len(bindings.attachments) <= 4
                assert 'batches' not in value['population']
                assert value['population']['rows'] == count
            elif value['stage'] == 'labels':
                final_batches.append(len(value['batches']))
            return prepare(value, bindings, state)

        monkeypatch.setattr(frame_facade, '_prepare', observe_resources)
    original = pd.Series.astype
    observed = []
    faults = []

    def astype(series, dtype, *args, **kwargs):
        if len(series) not in (5000, 10000):
            return original(series, dtype, *args, **kwargs)
        values = series.tolist()
        if not all(
            type(value) is str and value.startswith('u') for value in values
        ):
            return original(series, dtype, *args, **kwargs)
        if dtype is str:
            observed.append(len(series))
        if len(observed) == 2 and case['second_label_fault']:
            faults.append(len(series))
            raise TypeError('explicit second label chunk witness')
        return original(series, dtype, *args, **kwargs)

    monkeypatch.setattr(pd.Series, 'astype', astype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if expected['outcome']['kind'] == 'error':
            with pytest.raises(StructuredDataError) as failure:
                asked.materialize(
                    model=KumoRelational(),
                    indices=['u24999'],
                    num_context_samples=1001,
                    num_neighbors=[1],
                    num_hops=1,
                    random_seed=7,
                )
            assert chain(failure.value) == expected['outcome']['chain']
        else:
            request = asked.materialize(
                model=KumoRelational(),
                indices=['u24999'],
                num_context_samples=1001,
                num_neighbors=[1],
                num_hops=1,
                random_seed=7,
            )
            assert list(request.payloads) == expected['outcome']['payloads']
            assert request.context_rows == expected['outcome']['context_rows']
            assert request.predict_rows == expected['outcome']['predict_rows']
    assert observed == expected['label_cast_rows']
    assert len(faults) == expected['native_faults']
    if core:
        assert len(label_resources) == len(expected['label_cast_rows'])
        if expected['outcome']['kind'] == 'error':
            assert final_batches == []
        else:
            # The complete actual prefix is bound exactly once, at assembly.
            assert final_batches == [len(label_resources)]
    assert [
        {'type': type(item.message).__name__, 'message': str(item.message)}
        for item in caught
    ] == expected['warnings']
