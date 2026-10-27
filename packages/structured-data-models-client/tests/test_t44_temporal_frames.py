# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public typed-time frame journeys, recorded before their implementation."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    SDMWarning,
    StructuredDataError,
)
from sdm_client.materializers.relational import RelationalMaterializer

REFERENCE = json.loads(
    Path(__file__).with_name('t44_temporal_reference.json').read_text()
)
ROUTES = ','.join(
    (
        'schema.from-data/frames/frames',
        'schema.from-data/mapping/frames',
        'query.materialize/indices-regression/frames',
        'client.predict/regression/frames',
    )
)


@pytest.fixture(params=[False, True], ids=['original', 'core'])
def core(request, monkeypatch):
    selected = request.param
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES if selected else '')
    if selected:

        def old_engine_was_called(*args, **kwargs):
            raise AssertionError(
                'the core temporal journey called the old engine'
            )

        monkeypatch.setattr(
            RelationalMaterializer, '_get_model', old_engine_was_called
        )
    return selected


def question(unit='ns', zone=None, *, precise=False, mixed=False):
    counts = np.array(
        [1, -2, 0, -1, -1, 2, 0, 3]
        if unit == 's'
        else [1200, -1800, 500, -1200, -1500, -700, 0, -500],
        dtype='int64',
    )
    if unit != 's':
        counts *= {'ms': 1, 'us': 1000, 'ns': 1_000_000}[unit]
    if precise:
        counts = np.array(
            [
                1200001,
                -1800003,
                500005,
                -1200007,
                -1500009,
                -700011,
                13,
                -500015,
            ]
            if unit == 'us'
            else [
                1200000001,
                -1800000003,
                500000005,
                -1200000007,
                -1500000009,
                -700000011,
                13,
                -500000015,
            ],
            dtype='int64',
        )
    dates = pd.Series(np.tile(counts, 8).view(f'datetime64[{unit}]'))
    if zone is not None:
        dates = dates.dt.tz_localize('UTC').dt.tz_convert(zone)
    frames = {
        'users': pd.DataFrame(
            {
                'user_id': np.arange(1, 9),
                'age': np.arange(21, 29, dtype=np.int32),
            }
        ),
        'events': pd.DataFrame(
            {
                'event_id': np.arange(64),
                'user_id': np.repeat(np.arange(1, 9), 8),
                'marker': np.arange(64),
                'amount': np.arange(64, dtype=np.float32),
                'at': dates,
            }
        ),
    }
    primary = {'users': 'user_id', 'events': 'event_id'}
    if mixed:
        frames['notes'] = pd.DataFrame(
            {
                'note_id': np.arange(96),
                'user_id': np.repeat(np.arange(1, 9), 12),
                'note_marker': np.arange(1000, 1096),
            }
        )
        primary['notes'] = 'note_id'
    schema = Schema.from_data(
        frames,
        primary_keys=primary,
        time_columns={'events': 'at'},
    )
    return (
        PredictiveQuery('PREDICT users.age FOR EACH users.user_id', schema),
        frames,
    )


def materialize(
    asked, anchor='default', *, neighbors=(2,), hops=1, seed=7, **options
):
    at = {} if anchor == 'default' else {'anchor_time': pd.Timestamp(anchor)}
    return asked.materialize(
        indices=[7, 8],
        num_context_samples=3,
        num_neighbors=list(neighbors),
        num_hops=hops,
        random_seed=seed,
        **at,
        **options,
    )


class Scripted:
    url = 'scripted://phase4-temporal-reference'
    serves_sessions = True

    def __init__(self):
        self.calls = []

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        table = payload['predict']['instance_table']
        at = table['columns'].index('instance_id')
        return {
            'id': 'temporal-reference',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {'row_index': i, 'id': str(row[at]), 'prediction': 100.0 + i}
                for i, row in enumerate(table['rows'])
            ],
        }

    def close(self):
        pass


def predict(request, expected):
    scripted = Scripted()
    ids = [call['metadata']['request_id'] for call in expected]
    # UUID generation is a host fact. Replay its captured values so the
    # whole payload remains strict, including the metadata field itself.
    with patch('sdm_client.adapters.relational._request_id', side_effect=ids):
        with SDMClient('http://localhost:1', max_retries=0) as client:
            client._transport.close()
            client._transport = scripted
            result = client.predict(request, model=KumoRelational())
    return result.to_pandas(), scripted.calls


@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('anchor', ['positive', 'negative', 'default'])
def test_typed_timestamp_journey_keeps_the_frozen_payload(core, unit, anchor):
    observed = REFERENCE['cases'][f'{unit}-{anchor}']
    asked, frames = question(unit)
    request = materialize(asked, observed['anchor'])
    before = copy.deepcopy(request.payloads)
    result, sent = predict(request, observed['sent'])
    assert sent == observed['sent']
    assert request.payloads == before
    assert result['id'].tolist() == [7, 8]
    assert result['prediction'].tolist() == [100.0, 101.0]
    joined = result.merge(frames['users'], left_on='id', right_on='user_id')
    assert joined['age'].tolist() == [27, 28]


@pytest.mark.parametrize('zone', ['UTC', 'America/New_York'])
def test_typed_zones_keep_utc_instants_and_second_ties(core, zone):
    observed = REFERENCE['cases']['ns-positive']
    asked, _ = question('ns', zone)
    _, sent = predict(materialize(asked, observed['anchor']), observed['sent'])
    assert sent == observed['sent']


@pytest.mark.parametrize(
    'mutation',
    ['first_duplicate_missing_time', 'second_duplicate_missing_time'],
)
def test_sanitization_keeps_key_then_time_precedence_and_snapshot(
    core, mutation
):
    observed = REFERENCE['cases'][mutation]
    asked, frames = question()
    events = frames['events']
    events.loc[1, 'event_id'] = events.loc[0, 'event_id']
    events.loc[0 if mutation.startswith('first') else 1, 'at'] = pd.NaT
    events.loc[2, 'at'] = pd.NaT
    wanted = [m for m in observed['warnings'] if 'graph was built' in m]
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            request = materialize(asked, observed['anchor'])
        assert [
            str(w.message)
            for w in caught
            if 'graph was built' in str(w.message)
        ] == wanted
        _, sent = predict(request, observed['sent'])
        assert sent == observed['sent']
        events['at'] = pd.date_range('2026-01-01', periods=64, freq='D')


def test_all_missing_times_refuse_before_notice_and_do_not_cache_failure(core):
    asked, frames = question()
    original = frames['events']['at'].copy()
    frames['events']['at'] = pd.Series(np.full(64, np.datetime64('NaT', 'ns')))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(StructuredDataError) as failed:
            materialize(asked)
    expected = REFERENCE['cases']['all_times_missing']['error']
    assert failed.value.code == expected['code']
    assert str(failed.value) == expected['message']
    assert not any('graph was built' in str(w.message) for w in caught)
    frames['events']['at'] = original
    _, sent = predict(
        materialize(asked), REFERENCE['cases']['ns-default']['sent']
    )
    assert sent == REFERENCE['cases']['ns-default']['sent']


def test_time_drop_warning_as_error_keeps_snapshot_before_selection(core):
    observed = REFERENCE['cases']['first_duplicate_missing_time']
    asked, frames = question()
    frames['events'].loc[0, 'at'] = pd.NaT
    frames['events'].loc[1, 'event_id'] = 0
    frames['events'].loc[2, 'at'] = pd.NaT
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'error', message='the graph was built', category=SDMWarning
        )
        with pytest.raises(SDMWarning, match='2 with no time'):
            materialize(asked, observed['anchor'])
    frames['events']['at'] = pd.date_range('2026-01-01', periods=64, freq='D')
    with pytest.warns(SDMWarning, match='2 with no time'):
        request = materialize(asked, observed['anchor'])
    _, sent = predict(request, observed['sent'])
    assert sent == observed['sent']


@pytest.mark.parametrize(
    'case', ['two-hops', 'zero-neighbors', 'all-neighbors', 'two-calls']
)
def test_temporal_walk_and_call_boundaries_keep_recorded_payloads(core, case):
    observed = REFERENCE['extra_cases'][case]
    asked, _ = question()
    options = (
        {}
        if observed['rows_per_call'] is None
        else {'rows_per_call': observed['rows_per_call']}
    )
    request = materialize(
        asked,
        '1970-01-01T00:00:00.100000000',
        neighbors=observed['neighbors'],
        hops=observed['hops'],
        **options,
    )
    result, sent = predict(request, observed['sent'])
    assert sent == observed['sent']
    assert result['id'].tolist() == [7, 8]


@pytest.mark.parametrize('unit', ['s', 'ms'])
@pytest.mark.parametrize('year', ['1600', '2263'])
@pytest.mark.parametrize('role', ['time', 'feature'])
def test_wide_counts_stay_lossless_and_remain_an_unfinished_materializer_family(
    core, unit, year, role
):
    _, frames = question(unit)
    dates = np.tile(
        np.array(
            [
                '2026-01-01',
                '2026-01-02',
                f'{year}-01-01',
                '2026-01-04',
                '2026-01-05',
                '2026-01-06',
                '2026-01-07',
                '2026-01-08',
            ],
            dtype=f'datetime64[{unit}]',
        ),
        8,
    )
    frames['events']['at'] = pd.Series(dates)
    raw = frames['events']['at'].astype('int64').tolist()
    schema = Schema.from_data(
        frames,
        primary_keys={'users': 'user_id', 'events': 'event_id'},
        time_columns={'events': 'at'} if role == 'time' else {},
    )
    assert str(frames['events']['at'].dtype) == f'datetime64[{unit}]'
    assert frames['events']['at'].astype('int64').tolist() == raw
    asked = PredictiveQuery('PREDICT users.age FOR EACH users.user_id', schema)
    with pytest.raises(StructuredDataError) as failed:
        materialize(asked, '2026-01-08')
    if core:
        # An unfinished seam path is not a new final range-refusal policy.
        assert failed.value.code == 'UNSUPPORTED_REQUEST'
        assert failed.value.details['rule'] == (
            'frame_wide_timestamps'
            if role == 'time'
            else 'frame_untimed_timestamp_features'
        )
    else:
        expected = REFERENCE['wide_cases'][f'{unit}-{year}-{role}']['error']
        assert failed.value.code == expected['code']
        assert str(failed.value) == expected['message']
    assert frames['events']['at'].astype('int64').tolist() == raw


@pytest.mark.parametrize('unit', ['us', 'ns'])
@pytest.mark.parametrize('anchor', ['positive', 'negative'])
def test_submillisecond_and_submicrosecond_feature_counts_reach_the_frozen_wire(
    core, unit, anchor
):
    observed = REFERENCE['submicro_cases'][f'{unit}-{anchor}']
    asked, frames = question(unit, precise=True)
    assert (
        frames['events']['at'].astype('int64').tolist()
        == observed['input_counts']
    )
    request = materialize(asked, observed['anchor'])
    _, sent = predict(request, observed['sent'])
    assert sent == observed['sent']
    assert (
        frames['events']['at'].astype('int64').tolist()
        == observed['input_counts']
    )


def test_mixed_temporal_cuts_leave_seeded_and_continuing_static_draws_intact(
    core,
):
    asked, frames = question(mixed=True)
    assert frames['notes'].groupby('user_id').size().tolist() == [12] * 8
    predictions = []
    for observed in REFERENCE['mixed_calls']:
        request = materialize(asked, observed['anchor'], seed=observed['seed'])
        _, sent = predict(request, observed['sent'])
        # Unseeded context identity uses native entropy; every scored row,
        # feature and temporal/static neighbor remains an exact witness.
        assert sent[0]['predict'] == observed['sent'][0]['predict']
        if observed['seed'] is not None:
            assert sent == observed['sent']
        predictions.append(sent[0]['predict'])
    assert predictions[0] != predictions[1]
    assert predictions[2] == predictions[3]
    fresh, _ = question(mixed=True)
    observed = REFERENCE['mixed_fresh']
    _, sent = predict(
        materialize(fresh, observed['anchor'], seed=None), observed['sent']
    )
    assert sent[0]['predict'] == predictions[0]


def test_all_invalid_static_keys_keep_the_original_exact_requirements_message(
    core,
):
    frames = {
        'users': pd.DataFrame({'user_id': [1, 2, 3], 'age': [20, 30, 40]}),
        'notes': pd.DataFrame(
            {
                'note_id': [1, 2, 3, 4, 5, 6],
                'user_id': [1, 1, 2, 2, 3, 3],
                'amount': [1, 2, 3, 4, 5, 6],
            }
        ),
    }
    schema = Schema.from_data(
        frames,
        primary_keys={'users': 'user_id', 'notes': 'note_id'},
        time_columns={},
    )
    asked = PredictiveQuery('PREDICT users.age FOR EACH users.user_id', schema)
    frames['users']['user_id'] = pd.Series([pd.NA] * 3, dtype='Int64')
    with pytest.raises(StructuredDataError) as failed:
        asked.materialize(
            indices=[3],
            num_context_samples=1,
            num_neighbors=[2],
            num_hops=1,
            random_seed=7,
        )
    expected = REFERENCE['all_invalid_static']
    assert failed.value.code == expected['code']
    assert str(failed.value) == expected['message']
