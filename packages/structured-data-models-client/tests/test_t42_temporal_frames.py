# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native us/ns temporal constructors through complete owned snapshots."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from test_t42_frames import PATHS, describe, projection

from sdm_client import Schema, StructuredDataError
from sdm_client._routing import RouteError


def stamps(values, unit='ns', zone=None):
    if zone is None:
        return pd.Series(np.array(values, dtype=f'datetime64[{unit}]'))
    return pd.Series(
        pd.array(
            [
                pd.NaT if value == 'NaT' else pd.Timestamp(value, tz=zone)
                for value in values
            ],
            dtype=pd.DatetimeTZDtype(unit=unit, tz=zone),
        )
    )


def paired(monkeypatch, frames, **options):
    old, old_warnings = describe(monkeypatch, frames, core=False, **options)
    new, new_warnings = describe(monkeypatch, frames, core=True, **options)
    assert projection(new) == projection(old)
    assert new_warnings == old_warnings
    return new, new_warnings


@pytest.mark.parametrize('unit', ['us', 'ns'])
@pytest.mark.parametrize('zone', [None, 'UTC', 'America/Los_Angeles'])
@pytest.mark.parametrize(
    'variant', ['ordinary', 'empty', 'null', 'constant', 'sparse']
)
def test_native_temporal_family_metadata_and_snapshot(
    monkeypatch, unit, zone, variant
):
    values = ['2024-01-01', 'NaT', '2024-01-03']
    if variant == 'empty':
        values = []
    elif variant == 'null':
        values = ['NaT'] * 3
    elif variant == 'constant':
        values = ['2024-01-01'] * 3
    elif variant == 'sparse':
        values = ['NaT', 'NaT', '2024-01-01']
    frame = pd.DataFrame({'event_id': stamps(values, unit, zone)})
    dtype = frame['event_id'].dtype
    counts = frame['event_id'].array.asi8.copy()
    schema, _ = paired(monkeypatch, {'events': frame})
    assert schema['events'].data is frame
    assert schema['events'].primary_key is None
    assert schema['events']['event_id']._source is None
    snapshot = schema.snapshot()
    assert snapshot['events'].data is frame
    before = projection(snapshot)
    assert projection(pickle.loads(pickle.dumps(snapshot))) == before
    assert frame['event_id'].dtype == dtype
    np.testing.assert_array_equal(frame['event_id'].array.asi8, counts)
    if len(frame):
        frame.loc[:, 'event_id'] = stamps(
            ['2024-01-01'] * len(frame), unit, zone
        )
        fresh, _ = paired(monkeypatch, {'events': frame})
        assert projection(snapshot) == before
        assert fresh['events'].data is frame


@pytest.mark.parametrize('counts', [[1, 2], [-2, -1], [1, 1], [-1, 0, 1]])
def test_native_nanoseconds_are_not_coarsened_for_time_choice(
    monkeypatch, counts
):
    frame = pd.DataFrame(
        {'observed_at': np.array(counts, dtype='int64').view('datetime64[ns]')}
    )
    schema, _ = paired(monkeypatch, {'events': frame})
    assert schema['events'].time_column == (
        'observed_at' if len(set(counts)) > 1 else None
    )
    np.testing.assert_array_equal(frame['observed_at'].array.asi8, counts)


def test_submicrosecond_values_distinguish_repeated_keys_without_becoming_keys(
    monkeypatch,
):
    frame = pd.DataFrame(
        {
            'user_id': [*range(99), 98],
            'observed_at': (
                np.arange(100, dtype='int64') + 1_700_000_000_000_000_000
            ).view('datetime64[ns]'),
        }
    )
    schema, notices = paired(monkeypatch, {'users': frame})
    assert schema['users'].primary_key == 'user_id'
    assert schema['users'].duplicate_keys == 1
    assert notices


@pytest.mark.parametrize('unit', ['us', 'ns'])
@pytest.mark.parametrize(
    'kind',
    [
        'earliest',
        'tie',
        'created',
        'load',
        'birth',
        'constant',
        'sparse',
        'edit',
        'edit_linked',
        'placeholder',
        'late',
    ],
)
def test_whole_frame_time_choice_and_final_notices(monkeypatch, unit, kind):
    rows = 1005 if kind == 'late' else 3
    events = pd.DataFrame(
        {'event_id': np.arange(rows), 'user_id': np.arange(rows) % 3}
    )
    a = ['2025-01-01', '2025-01-02', '2025-01-03']
    b = ['2024-01-01', '2024-01-02', '2024-01-03']
    if kind == 'late':
        a = ['2025-01-01'] * (rows - 1) + ['2020-01-01']
        b = ['2022-01-01'] * (rows - 1) + ['2023-01-01']
    elif kind == 'tie':
        b = a
    elif kind == 'constant':
        a = ['2025-01-01'] * 3
    elif kind == 'sparse':
        a = ['NaT', 'NaT', '2025-01-01']
    elif kind == 'placeholder':
        a = ['1900-01-01', '1970-01-01', '2021-01-01']
        b = ['2022-01-01', '2022-01-02', '2022-01-03']
    name = {
        'created': 'created_at',
        'load': 'loaded_at',
        'birth': 'date_of_birth',
        'edit': 'updated_at',
        'edit_linked': 'updated_at',
    }.get(kind, 'opened_at')
    events[name] = stamps(a, unit)
    if kind not in (
        'load',
        'birth',
        'constant',
        'sparse',
        'edit',
        'edit_linked',
    ):
        events['closed_at'] = stamps(b, unit)
    parents = pd.DataFrame({'user_id': [0, 1, 2]})
    frames = {'events': events, 'users': parents}
    if kind == 'edit_linked':
        frames['lines'] = pd.DataFrame(
            {'line_id': [0, 1, 2], 'event_id': [0, 1, 2]}
        )
    schema, notices = paired(monkeypatch, frames)
    if kind == 'late':
        assert schema['events'].time_column == 'opened_at'
        assert notices
    if kind == 'edit_linked':
        assert schema['events'].time_column is None


@pytest.mark.parametrize('unit', ['us', 'ns'])
@pytest.mark.parametrize('zone', ['UTC', 'America/Los_Angeles'])
def test_native_zoned_constant_warning_preserves_native_spelling(
    monkeypatch, unit, zone
):
    for value in ['1800-01-01', '2024-07-01']:
        frames = {
            'events': pd.DataFrame(
                {
                    'event_id': [1, 2],
                    'observed_at': stamps([value] * 2, unit, zone),
                }
            ),
            'users': pd.DataFrame({'user_id': [1, 2]}),
        }
        _, notices = paired(monkeypatch, frames)
        assert notices


@pytest.mark.parametrize(
    'options',
    [
        {'time_columns': {'events': 'closed_at'}},
        {'time_columns': {'events': None}},
        {'time_columns': {'events': 'missing'}},
        {'primary_keys': {'events': 'observed_at'}},
        {
            'primary_keys': {'events': 'missing'},
            'time_columns': {'events': 'absent'},
        },
    ],
)
def test_temporal_overrides_preserve_result_warning_and_error_stage(
    monkeypatch, options
):
    frames = {
        'events': pd.DataFrame(
            {
                'event_id': [1, 2],
                'observed_at': stamps(['2024-01-01', '2024-01-02']),
                'closed_at': stamps(['2025-01-01', '2025-01-02']),
            }
        )
    }
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            try:
                result = Schema.from_data(frames, **options)
                answer = ('ok', projection(result))
            except StructuredDataError as error:
                answer = ('error', type(error), error.code, str(error))
        outcomes.append(
            (answer, [(str(n.message), n.category) for n in notices])
        )
    assert outcomes[1] == outcomes[0]


def test_temporal_first_table_warning_survives_later_table_failure(monkeypatch):
    first = pd.DataFrame(
        {
            'user_id': [*range(99), None],
            'observed_at': stamps(['2024-01-01'] * 99 + ['2024-01-02']),
        }
    )
    later = pd.DataFrame([[1, 2]], columns=['same', 'same'])
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter('always')
            with pytest.raises(StructuredDataError) as caught:
                Schema.from_data(
                    {'users': first, 'later': later},
                    time_columns={'users': 'missing'},
                )
        outcomes.append((str(caught.value), [str(n.message) for n in notices]))
    assert outcomes[1] == outcomes[0]
    assert outcomes[0][1]


def test_table_owned_time_choice_does_not_reread_mutated_original(monkeypatch):
    from sdm_client._rust import frames as bridge

    first = pd.DataFrame(
        {
            'event_id': [1, 2],
            'observed_at': stamps(['2024-01-01', '2024-01-02']),
        }
    )
    original = bridge._convert

    def convert(name, frame):
        if name == 'later':
            first.loc[:, 'observed_at'] = stamps(['2024-01-01'] * 2)
        return original(name, frame)

    monkeypatch.setattr(bridge, '_convert', convert)
    schema, _ = describe(
        monkeypatch,
        {'events': first, 'later': pd.DataFrame({'id': [1, 2]})},
        core=True,
    )
    assert schema['events'].time_column == 'observed_at'
    assert schema['events'].data is first
    assert first['observed_at'].nunique() == 1
    monkeypatch.setattr(bridge, '_convert', original)
    fresh, _ = paired(monkeypatch, {'events': first})
    assert fresh['events'].time_column is None


@pytest.mark.parametrize('unit', ['s', 'ms'])
def test_coarse_native_units_keep_old_default_until_lossless_frame_admission(
    monkeypatch, unit
):
    frame = pd.DataFrame(
        {
            'observed_at': np.array([2**63 - 2, 2**63 - 1], dtype='int64').view(
                f'datetime64[{unit}]'
            )
        }
    )
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    legacy = Schema.from_data({'events': frame})
    assert legacy['events'].time_column == 'observed_at'
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS)
    with pytest.raises(RouteError, match='numeric and Boolean'):
        Schema.from_data({'events': frame})
