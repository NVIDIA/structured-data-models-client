# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved requests preserve model payloads and results across client upgrades."""

import pickle
import warnings
from pathlib import Path

import pandas as pd
import pytest
from saved_request_fixtures import ANCHOR, REQUESTS, TABLES, answers, source

from sdm_client import PredictiveQuery, Schema

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/saved-requests'


def assert_scripted_answers(observed, name, rows=None):
    """Expected model answers derive only from the scripted call/row rule."""
    indices = REQUESTS[name][1]['indices'][:rows]
    calls = {'batched': len(indices) // 2, 'forecast': 2, 'ranking': 3}.get(
        name, 1
    )
    for variant, result in enumerate(observed):
        assert len(result['bodies']) == calls
        answer = result['answer']
        items = []
        explanations = {}
        for i, entity in enumerate(indices):
            for step in [1, 2] if name == 'forecast' else [None]:
                call = (
                    step - 1 if step else (i // 2 if name == 'batched' else 0)
                )
                local = i % 2 if name == 'batched' else i
                d = (3 * call + 5 * local) % 9 + 1
                item = {'id': entity, 'row_index': i}
                if step:
                    item['forecast_step'] = step
                if name == 'ranking':
                    scores = [
                        (candidate, ((3 * c + 5 * i) % 9 + 1) / 10)
                        for c, candidate in enumerate([2, 1, 0])
                    ]
                    ranked = sorted(scores, key=lambda pair: -pair[1])[:2]
                    item['rankings'] = [
                        {'candidate': c, 'score': score} for c, score in ranked
                    ]
                    if variant == 0:
                        item['prediction'] = [c for c, _ in ranked]
                else:
                    item['metadata'] = {
                        'call': call,
                        'row_index': local,
                        'source': 'scripted',
                    }
                    item['prediction'] = (d > 5) if name == 'binary' else 1.0
                    if name == 'binary':
                        item['probabilities'] = {
                            'False': (10 - d) / 10,
                            'True': d / 10,
                        }
                    if variant and name in {'regression', 'forecast'}:
                        explanation = {
                            'format': 'scripted',
                            'summary': f'row {local} of call {call}',
                            'score': d / 10,
                            'rank': d,
                            'path': [call, local],
                            'detail': {
                                'call': call,
                                'row_index': local,
                                'note': None,
                            },
                        }
                        item['explanation'] = explanation
                        explanations[str(step)] = explanation
                items.append(item)
        assert answer['items'] == items
        if name != 'ranking':
            prediction_type = bool if name == 'binary' else float
            assert all(
                type(item['prediction']) is prediction_type
                for item in answer['items']
            )
        outputs = (
            (['rankings'] if variant else ['prediction', 'rankings'])
            if name == 'ranking'
            else (
                ['prediction', 'probabilities']
                if name == 'binary'
                else ['prediction', 'explanation', 'quantiles']
                if variant and name in {'regression', 'forecast'}
                else ['prediction', 'quantiles']
                if variant
                else ['prediction']
            )
        )
        assert answer['outputs'] == outputs
        columns = ['row_index', 'id'] + (
            ['forecast_step'] if name == 'forecast' else []
        )
        if 'prediction' in outputs:
            columns.append('prediction')
        if name == 'binary':
            columns += ['probability_False', 'probability_True']
        if name == 'ranking':
            columns.append('rankings')
        frame_rows = []
        for item in items:
            row = [
                item[c]
                if c in item
                else item['probabilities'][c.removeprefix('probability_')]
                for c in columns
            ]
            frame_rows.append(row)
        dtypes = [
            'int64'
            if c in {'row_index', 'id', 'forecast_step'}
            else 'object'
            if name == 'ranking'
            else 'bool'
            if name == 'binary' and c == 'prediction'
            else 'float64'
            for c in columns
        ]
        assert answer['frame'] == {
            'columns': columns,
            'rows': frame_rows,
            'dtypes': dtypes,
        }
        assert answer['unscored'] == ([9] if name == 'ranking' else [])
        assert answer['metadata'] == (
            {'batch_request_ids': ['r1'] * calls, 'batches': calls}
            if name == 'batched'
            else {}
        )
        expected_explanation = None
        if explanations:
            expected_explanation = (
                {'by_forecast_step': explanations}
                if name == 'forecast'
                else explanations['None']
            )
        assert answer['explanation'] == expected_explanation
        if name == 'ranking':
            assert len(result['notices']) == 1
            for fragment in [
                'candidate 9',
                'unscored',
                'No call was sent',
                'num_context_samples',
            ]:
                assert fragment in result['notices'][0]
        elif variant and name != 'binary':
            assert len(result['notices']) == (3 if name == 'forecast' else 1)
            for notice in result['notices']:
                assert "without ['quantiles']" in notice
                assert 'rest of the request is unaffected' in notice
        else:
            assert result['notices'] == []


@pytest.mark.parametrize('version', ['reference', 'phase-2'])
@pytest.mark.parametrize('name', sorted(REQUESTS))
def test_saved_request_predicts_without_its_original_runtime(version, name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = pickle.loads(
            (FIXTURES / f'{version}-{name}.pkl').read_bytes()
        )
    # Saved payloads are sufficient: prediction needs neither the original
    # runtime nor a source connection used when those rows were materialized.
    assert request.schema.source is None
    observed = answers(request, name)
    assert_scripted_answers(observed, name)


def test_saved_fixture_inventory():
    for version in ['reference', 'phase-2']:
        files = list(FIXTURES.glob(f'{version}-*.pkl'))
        names = {p.stem.removeprefix(f'{version}-') for p in files}
        assert names == set(REQUESTS)
        # Verify the preserved serialization format itself, not a claim
        # recorded beside it. Protocol 4 is identified by its pickle header.
        assert all(p.read_bytes().startswith(b'\x80\x04') for p in files)


def materialized(name):
    text, arguments = REQUESTS[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(source(), TABLES)
        return PredictiveQuery(text, schema).materialize(
            anchor_time=pd.Timestamp(ANCHOR), **arguments
        )


@pytest.mark.parametrize('name', sorted(REQUESTS))
def test_payloads_restore_missing_native_state(name):
    request = materialized(name)
    for held in (request, *(request.parts or ())):
        held._core = None
    observed = answers(request, name)
    assert_scripted_answers(observed, name)


def test_changed_payloads_take_precedence_over_cached_native_state():
    request = materialized('batched')
    assert request._core is not None and request._core.batches
    request._payloads = request._payloads[:2]
    observed = answers(request, 'batched')
    assert len(observed[0]['bodies']) == 2
    assert_scripted_answers(observed, 'batched', rows=4)
    request._core = None
    restored = answers(request, 'batched')
    assert_scripted_answers(restored, 'batched', rows=4)
    assert restored == observed
