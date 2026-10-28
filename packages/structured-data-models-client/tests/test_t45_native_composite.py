# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete public twins of the frozen native composite-history boundary.

These compare complete original/core observations, including payloads and
causes. The existing immutable full LINK fixture remains an independent oracle.
"""

from __future__ import annotations

import copy
import json
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from t38_forced_routes_record import bare
from t45_frame_routes import FRAME_ROUTES
from test_t45_frames import Scripted, configure

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
from sdm_client.materializers.relational import RelationalMaterializer

CASES = (
    'ordinary',
    'reversed',
    'three',
    'escaped',
    'empty-text',
    'unicode',
    'surrogate',
    'signed',
    'uint64',
    'duplicate',
    'duplicate-null-time',
    'all-null-time',
    'empty',
    'null-first',
    'null-later',
    'float',
    'float-logical-edit',
    'long-name',
    'prefix',
    'collision-int',
    'collision-int-id',
    'collision-text',
    'collision-text-id',
    'collision-float',
    'key-moved',
    'later-table',
    'same-time',
    'relationships',
    'render-error',
    'replace',
    'cached',
    'warning-retry',
)


def _observe(label):
    trace = []
    active = []
    mutations = []
    schema = None

    class Rendered(str):
        def replace(self, old, new, *args):
            if active:
                trace.append(('replace', old, new))
            return Rendered(str.replace(self, old, new, *args))

    class Key(str):
        def __str__(self):
            if active:
                trace.append(('render', str.__str__(self)))
                if label == 'render-error':
                    raise TypeError('native composite render failed')
                if not mutations:
                    if label == 'key-moved':
                        schema['buys'].primary_key = 'buy_id'
                        mutations.append('key-moved')
                    elif label == 'later-table':
                        schema['users']['age'].stype = 'categorical'
                        mutations.append('users-age')
                    elif label == 'same-time':
                        schema['buys'].time_column = None
                        mutations.append('no-time')
                    elif label == 'relationships':
                        schema.relationships.clear()
                        mutations.append('no-links')
            value = str.__str__(self)
            return Rendered(value) if label == 'replace' else value

    data, primary, times, links, query = configure('int64', 'composite-history')
    if label == 'reversed':
        primary['buys'].reverse()
    if label == 'three':
        data['buys']['other'] = ['x', 'y'] * 60
        primary['buys'].append('other')
    variants = {
        'escaped': ['a\x1fb', 'a\\u', '\\', '\x1f'] * 30,
        'empty-text': [''] * 120,
        'unicode': ['é', '東京', '😺'] * 40,
        'signed': [-(2**63), 2**63 - 1] * 60,
    }
    if label in variants:
        data['buys']['part'] = variants[label]
    if label == 'surrogate':
        data['buys']['part'] = pd.Series(
            ['\ud800', '\ud83d\ude3a', '😺'] * 40,
            dtype=pd.StringDtype(storage='python'),
        )
    if label == 'uint64':
        data['buys']['part'] = pd.Series(
            [2**63, 2**64 - 1] * 60, dtype='uint64'
        )
    if label in ('float', 'float-logical-edit'):
        data['buys']['part'] = 0.25
    if label == 'long-name':
        data['buys'] = data['buys'].rename(columns={'part': 'p' * 110})
        primary['buys'][1] = 'p' * 110
    derived = '__kumo_key_' + '_'.join(
        name.encode().hex() for name in primary['buys']
    )
    if label.startswith('collision-'):
        data['buys'][derived] = (
            ['key-' + str(i) for i in range(120)]
            if 'text' in label
            else np.arange(
                120, dtype='float64' if label.endswith('float') else 'int64'
            )
            + 9000
        )
    if label == 'prefix':
        data['buys']['__kumo_existing'] = np.arange(120)
    if label in (
        'key-moved',
        'later-table',
        'same-time',
        'relationships',
        'render-error',
        'replace',
    ):
        data['buys']['part'] = pd.Series(
            [Key('part') for _ in range(120)], dtype=object
        )
    if label == 'later-table':
        data = {name: data[name] for name in ('buys', 'users', 'items')}
    original_map = pd.Series.map

    def mapped(values, *args, **kwargs):
        reached = values.name == 'part' and stage == 'materialize'
        if reached:
            active.append(True)
        try:
            return original_map(values, *args, **kwargs)
        finally:
            if reached:
                active.pop()

    observed = {}
    script = Scripted()
    stage = 'schema'
    with (
        warnings.catch_warnings(record=True) as notices,
        pd.option_context('future.infer_string', False),
        patch.object(pd.Series, 'map', mapped),
    ):
        warnings.simplefilter('always')
        try:
            schema = Schema.from_data(
                data,
                primary_keys=primary,
                time_columns=times,
                relationships=links,
            )
            if label == 'float-logical-edit':
                schema['buys']['part'].dtype = 'string'
            if label.endswith('-id'):
                schema['buys'][derived].stype = 'ID'
            stage = 'query'
            question = PredictiveQuery(query, schema)
            if label in ('duplicate', 'duplicate-null-time', 'warning-retry'):
                data['buys'].loc[1, ['buy_id', 'part']] = (
                    data['buys'].loc[0, ['buy_id', 'part']].tolist()
                )
            if label == 'duplicate-null-time':
                data['buys'].loc[0, 'ts'] = pd.NaT
            if label == 'all-null-time':
                data['buys']['ts'] = pd.NaT
            if label == 'empty':
                data['buys'].drop(index=data['buys'].index, inplace=True)
            if label in ('null-first', 'null-later'):
                name = 'buy_id' if label == 'null-first' else 'part'
                data['buys'][name] = data['buys'][name].astype('Int64')
                data['buys'].loc[0, name] = pd.NA

            def gather():
                return question.materialize(
                    indices=[0, 1],
                    anchor_time=pd.Timestamp('2023-03-01'),
                    random_seed=7,
                    num_context_samples=3,
                    num_hops=1,
                    num_neighbors=[3, 3],
                )

            stage = 'materialize'
            if label == 'warning-retry':
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'error', message='the graph was built without'
                        )
                        gather()
                except Warning as error:
                    observed['warning_error'] = bare(str(error))
                data['buys']['part'] = pd.NA
            request = gather()
            observed['payloads'] = copy.deepcopy(request.payloads)
            observed['rows'] = (request.context_rows, request.predict_rows)
            if label == 'cached':
                data['buys']['part'] = pd.NA
                schema['users']['age'].stype = 'categorical'
                observed['again'] = copy.deepcopy(gather().payloads)
            stage = 'predict'
            with patch(
                'sdm_client.adapters.relational._request_id',
                return_value='composite-' + label,
            ):
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = script
                    answer = client.predict(
                        request, model=KumoRelational()
                    ).to_pandas()
            observed['result'] = json.loads(answer.to_json(orient='split'))
            observed['outcome'] = 'success'
        except Exception as error:
            chain = []
            while error is not None:
                message = bare(str(error))
                if label == 'uint64':
                    message = message.replace(
                        'value 9223372036854775808', 'a value'
                    )
                chain.append(
                    (
                        type(error).__name__,
                        getattr(error, 'code', None),
                        message,
                    )
                )
                error = error.__cause__ or (
                    None if error.__suppress_context__ else error.__context__
                )
            observed.update(outcome='error', stage=stage, chain=chain)
        observed['warnings'] = [
            (type(n.message).__name__, bare(str(n.message))) for n in notices
        ]
        observed['sent'] = script.calls
    observed['trace'] = trace
    observed['mutations'] = mutations
    return json.loads(json.dumps(observed))


@pytest.mark.parametrize('label', CASES)
def test_complete_native_composite_history_matches_original(monkeypatch, label):
    monkeypatch.setenv('SDM_CORE_ROUTES', '')
    expected = _observe(label)
    if label in (
        'ordinary',
        'collision-int-id',
        'collision-text-id',
        'replace',
        'cached',
        'warning-retry',
    ):
        assert expected['outcome'] == 'success'
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_ROUTES)

    def old_engine_was_called(*args, **kwargs):
        raise AssertionError(
            'core composite materialization called the old engine'
        )

    monkeypatch.setattr(
        RelationalMaterializer, '_get_model', old_engine_was_called
    )
    assert _observe(label) == expected
