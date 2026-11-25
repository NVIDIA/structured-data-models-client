# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public merge retains native owners and delegates policy without encoding rows."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sdm_api.generated.tfm_api import PredictionItem

from sdm_client import PredictionResult
from sdm_client.errors import StructuredDataError


@pytest.fixture(autouse=True)
def native_default(monkeypatch):
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)


def part(candidates=(), **kwargs):
    defaults = dict(
        model='m',
        request_id='one',
        items=[],
        outputs=['prediction'],
        unscored_candidates=candidates,
    )
    defaults.update(kwargs)
    return PredictionResult(**defaults)


def test_merge_preserves_owners_without_row_or_metadata_serialization(
    monkeypatch,
):
    from sdm_client._rust import result_merge

    owner = object()
    item = PredictionItem(row_index=0, prediction=1.5)
    first = part(items=[item], metadata={'native': owner}, explanation={})
    second = part(request_id='two', explanation={'native': owner})
    first._names_its_rows = False
    second._names_its_step = True
    calls = []
    original = result_merge._call

    def observe(operation, payload):
        calls.append((operation, payload))
        return original(operation, payload)

    monkeypatch.setattr(result_merge, '_call', observe)
    result = PredictionResult.merge([first, second])
    assert result.items[0] is item
    assert result.metadata['native'] is owner
    assert result.explanation is second.explanation
    assert result.metadata['batch_request_ids'] == ['one', 'two']
    assert result.metadata['batches'] == 2
    assert result._names_its_rows is False
    assert result._names_its_step is True
    assert len(calls) == 1
    assert calls[0][0] == 'plan_result_merge'
    for descriptor in calls[0][1]['parts']:
        assert not {'items', 'metadata', 'explanation'} & descriptor.keys()


def test_union_keeps_repeats_within_a_part_and_drops_earlier_values():
    first = part([1, 1, np.int64(1), 2**53 + 1, 2**53, np.uint64(2**64 - 1)])
    result = PredictionResult.merge(
        [first, part([True, 1.0, float(2**53), 2**64 - 1])]
    )
    assert result.unscored_candidates == first.unscored_candidates
    assert all(
        a is b
        for a, b in zip(
            result.unscored_candidates, first.unscored_candidates, strict=True
        )
    )


@pytest.mark.parametrize(
    ('earlier', 'later', 'kept'),
    [
        (Decimal('1.50'), 1.5, 1),
        (Decimal('2.0'), Decimal('2'), 1),
        (Decimal(2**64), 2**64, 1),
        (Decimal('-Infinity'), float('-inf'), 1),
        (Decimal('0.1'), 0.1, 2),
        (Decimal('NaN'), Decimal('NaN'), 2),
    ],
)
def test_decimal_candidates_merge_by_value(earlier, later, kept):
    result = PredictionResult.merge([part([earlier]), part([later])])
    assert len(result.unscored_candidates) == kept
    assert result.unscored_candidates[0] is earlier


def test_nan_values_remain_distinct_under_ieee_equality():
    nan = float('nan')
    result = PredictionResult.merge([part([nan, nan]), part([float('nan')])])
    assert len(result.unscored_candidates) == 3


def test_temporal_resolution_and_original_objects_survive():
    early = datetime(1, 1, 1)
    late = pd.Timestamp('2500-01-01')
    zoned = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = part([early, late, zoned, date(2026, 1, 1)])
    second = part(
        [
            np.datetime64('0001-01-01', 'us'),
            np.datetime64('2500-01-01', 's'),
            datetime(2026, 1, 1, 5, tzinfo=timezone(timedelta(hours=5))),
            np.datetime64('2026-01-01', 'D'),
            datetime(2026, 1, 1),
        ]
    )
    result = PredictionResult.merge([first, second])
    assert len(result.unscored_candidates) == 5
    assert all(
        a is b
        for a, b in zip(
            result.unscored_candidates[:4],
            first.unscored_candidates,
            strict=True,
        )
    )
    assert result.unscored_candidates[-1] is second.unscored_candidates[-1]


def test_container_candidates_keep_list_tuple_and_bytes_domains():
    first = part([[1, None], (1, None), {'a': [1], 'b': 2}, b'key', 'key'])
    second = part([[True, None], (1.0, None), {'b': 2.0, 'a': [True]}, b'key'])
    result = PredictionResult.merge([first, second])
    assert len(result.unscored_candidates) == 5
    assert all(
        a is b
        for a, b in zip(
            result.unscored_candidates, first.unscored_candidates, strict=True
        )
    )


def test_error_order_and_single_identity_do_not_invoke_custom_comparison():
    class Custom:
        def __eq__(self, other):
            raise AssertionError('comparison callback ran')

        def __bool__(self):
            raise AssertionError('truth callback ran')

    only = part([Custom()], explanation=Custom())
    assert PredictionResult.merge([only]) is only
    with pytest.raises(StructuredDataError, match='no parts to merge'):
        PredictionResult.merge([])
    first = part([Custom()])
    with pytest.raises(StructuredDataError, match='different models'):
        PredictionResult.merge([first, part(model='other', outputs=['scores'])])
    with pytest.raises(StructuredDataError, match='different outputs'):
        PredictionResult.merge([first, part(outputs=['scores'])])
    with pytest.raises(StructuredDataError, match='candidate type Custom'):
        PredictionResult.merge([first, part()])


def test_missing_temporal_and_recursive_container_observations():
    result = PredictionResult.merge([part([pd.NaT]), part([pd.NaT])])
    assert len(result.unscored_candidates) == 2
    recursive = []
    recursive.append(recursive)
    for value in (recursive,):
        with pytest.raises(StructuredDataError, match='different models'):
            PredictionResult.merge([part([value]), part(model='other')])
        with pytest.raises(
            StructuredDataError, match='recursive or deeply nested'
        ):
            PredictionResult.merge([part([value]), part()])


def test_aware_datetimes_compare_by_instant_across_dst_folds():
    from zoneinfo import ZoneInfo

    zone = ZoneInfo('America/New_York')
    first = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=0)
    second = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1)
    result = PredictionResult.merge(
        [
            part([first, second]),
            part(
                [
                    first.astimezone(timezone.utc),
                    second.astimezone(timezone.utc),
                ]
            ),
        ]
    )
    assert len(result.unscored_candidates) == 2
    assert result.unscored_candidates[0] is first
    assert result.unscored_candidates[1] is second


def test_aware_datetime_conversion_does_not_overflow_at_year_boundary():
    value = datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=14)))
    result = PredictionResult.merge([part([value]), part([value])])
    assert len(result.unscored_candidates) == 1
    assert result.unscored_candidates[0] is value
