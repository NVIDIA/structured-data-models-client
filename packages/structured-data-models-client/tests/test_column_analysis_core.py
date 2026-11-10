# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent association oracles through public APIs and the native engine."""

import copy
import pickle
import warnings

import pandas as pd
import pytest

from sdm_client import (
    ColumnExplanations,
    Explanations,
    GradientExplanations,
    KumoRelational,
    PredictionResult,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
)
from sdm_client.errors import StructuredDataError


def payload(rows, kind='regression'):
    return {
        'task': {'kind': kind, 'target': {'column_name': 'y'}},
        'schema': {
            'instance_table': {
                'columns': {
                    'x': {'stype': 'categorical'},
                    'y': {'stype': 'numerical'},
                }
            }
        },
        'context': {'instance_table': {'columns': ['x', 'y'], 'rows': rows}},
    }


def test_public_constructor_has_independent_cohort_oracle():
    result = ColumnExplanations(
        payload([['a', 0], ['a', 2], ['b', 8], ['b', 10]])
    )
    frame = result.to_pandas()
    assert list(frame['cohort']) == ['a', 'b']
    assert list(frame['mean']) == [1.0, 9.0]
    assert list(frame['share']) == [0.5, 0.5]
    frame.loc[0, 'mean'] = -1
    assert result.to_pandas().loc[0, 'mean'] == 1


def test_empty_labels_and_exact_mixed_classes():
    empty = ColumnExplanations(
        payload([['a', None]], 'classification')
    ).to_pandas()
    assert empty.empty
    assert list(empty.columns) == [
        'table',
        'hop',
        'column',
        'stype',
        'cohort',
        'class',
        'share',
        'mean',
    ]
    labels = [2**80 + 1, 0.5, '1']
    frame = ColumnExplanations(
        payload([['a', label] for label in labels], 'classification')
    ).to_pandas()
    assert set(frame['class']) == set(labels)
    assert type(next(v for v in frame['class'] if v == labels[0])) is int


def test_unlabelled_related_rows_do_not_break_analysis():
    p = payload([], 'classification')
    p['schema'] = {
        'instance_table': {
            'columns': {
                'instance_id': {'stype': 'ID'},
                'ref': {'stype': 'ID'},
                'y': {'stype': 'categorical'},
            }
        },
        'related_tables': {
            'users': {
                'primary_key': ['instance_id', 'id'],
                'columns': {
                    'instance_id': {'stype': 'ID'},
                    'id': {'stype': 'ID'},
                    'x': {'stype': 'numerical'},
                },
            }
        },
        'relationships': [
            {
                'source_columns': ['ref'],
                'target_table': 'users',
                'target_columns': ['id'],
            }
        ],
    }
    p['context'] = {
        'instance_table': {
            'columns': ['instance_id', 'ref', 'y'],
            'rows': [[0, 10, 'yes'], [1, 20, None]],
        },
        'related_tables': {
            'users': {
                'columns': ['instance_id', 'id', 'x'],
                'rows': [[0, 10, 3], [1, 20, 99]],
            }
        },
    }
    frame = ColumnExplanations(p).to_pandas()
    assert not frame.empty
    assert set(frame['mean']) == {1.0}
    assert not frame['cohort'].str.contains('99').any()


def test_public_merge_retains_equal_analysis_and_refuses_different_analysis():
    a = ColumnExplanations(payload([['a', 0], ['b', 2]]))
    b = ColumnExplanations(payload([['a', 0], ['b', 4]]))

    def result(columns):
        return PredictionResult(
            model='kumo-tabular',
            request_id='r',
            items=[],
            explanations=Explanations(GradientExplanations(), columns),
        )

    merged = PredictionResult.merge([result(a), result(a)])
    pd.testing.assert_frame_equal(
        merged.explanations.columns.to_pandas(), a.to_pandas()
    )
    with pytest.raises(StructuredDataError, match='different column analyses'):
        PredictionResult.merge([result(a), result(b)])
    merged = PredictionResult.merge([result(a), result(b)], columns=b)
    pd.testing.assert_frame_equal(
        merged.explanations.columns.to_pandas(), b.to_pandas()
    )


class Endpoint:
    url = 'http://localhost'
    serves_sessions = True

    def __init__(self):
        self.sent = []

    def predict(self, p):
        self.sent.append(copy.deepcopy(p))
        assert 'client_analysis_delta' not in p
        assert 'client_column_analysis' not in p
        return {
            'id': 'r',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {
                    'row_index': i,
                    'id': str(
                        row[
                            p['predict']['instance_table']['columns'].index(
                                'instance_id'
                            )
                        ]
                    ),
                    'prediction': 1.0,
                }
                for i, row in enumerate(p['predict']['instance_table']['rows'])
            ],
        }

    def create_session(self, p):
        assert 'client_analysis_delta' not in p
        self.pinned = copy.deepcopy(p)
        return {'session_id': 'session'}

    def session_predict(self, session, p):
        return self.predict(p)

    def delete_session(self, session):
        pass

    def close(self):
        pass


def gathered():
    schema = Schema.from_data(
        {
            'users': pd.DataFrame(
                {'id': range(24), 'value': [float(i) for i in range(24)]}
            )
        },
        primary_keys={'users': 'id'},
    )
    schema['users']['value'].stype = 'numerical'
    return PredictiveQuery(
        'PREDICT users.value FOR EACH users.id', schema
    ).materialize(
        indices=[1, 2],
        random_seed=7,
        num_context_samples=24,
        model=KumoRelational(),
    )


def test_materialize_pickle_predict_analyses_the_rows_it_sends():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = gathered()
        restored = pickle.loads(pickle.dumps(request))
        assert restored._envelopes == request._envelopes
        endpoint = Endpoint()
        client = SDMClient.__new__(SDMClient)
        client._start(endpoint)
        result = client.predict(
            restored,
            model=KumoRelational(),
            outputs=['prediction', 'explanation'],
        )
    frame = result.explanations.columns.to_pandas()
    counts = frame[frame['column'] == 'COUNT(*)']
    assert len(endpoint.sent) == 1
    sent = endpoint.sent[0]['context']['instance_table']
    assert len(sent['rows']) == 22
    at = sent['columns'].index('TARGET')
    labels = [row[at] for row in sent['rows']]
    assert list(counts['mean']) == pytest.approx([sum(labels) / len(labels)])


def test_public_payloads_hide_client_keys_and_write_through():
    import json

    envelope = {
        'model': 'kumo-relational',
        'metadata': {},
        'client_generated_keys': {'orders': '__node_id'},
    }
    request = Request.__new__(Request)
    object.__setattr__(request, '_payloads', (envelope,))
    [payload] = request.payloads
    assert payload == {'model': 'kumo-relational', 'metadata': {}}
    json.dumps(payload)
    payload['metadata'] = {'note': 'edited'}
    assert envelope['metadata'] == {'note': 'edited'}
    request.payloads[0]['metadata'] = {'note': 'again'}
    payload['metadata']['seen'] = True
    assert envelope['metadata'] == {'note': 'again', 'seen': True}
    assert 'client_generated_keys' in envelope
    shared = Request.__new__(Request)
    object.__setattr__(shared, '_payloads', (envelope,))
    assert shared.payloads[0] is payload
    payload.clear()
    with pytest.raises(KeyError):
        payload.popitem()
    with pytest.raises(KeyError):
        del payload['client_generated_keys']
    assert envelope == {'client_generated_keys': {'orders': '__node_id'}}


def test_reading_payloads_never_empties_a_view_another_thread_holds():
    import sys
    import threading

    envelope = {
        'model': 'kumo-relational',
        'metadata': {},
        'task': {'kind': 'regression'},
        'client_generated_keys': {},
    }
    request = Request.__new__(Request)
    object.__setattr__(request, '_payloads', (envelope,))
    [held] = request.payloads
    partial = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            if len(held) != 3:
                partial.append(len(held))

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    watcher = threading.Thread(target=watch)
    try:
        watcher.start()
        for _ in range(5000):
            assert request.payloads is not None
    finally:
        stop.set()
        watcher.join()
        sys.setswitchinterval(previous)
    assert partial == []


def test_unavailable_statistics_warn_without_losing_predictions():
    from test_tabular_direct_core import Endpoint as TabularEndpoint
    from test_tabular_direct_core import client

    from sdm_client import KumoTabular, Request

    request = Request(
        pd.DataFrame(
            {'x': ['a', 'a', 'b', 'b'], 'y': [1e308, 1e308, -1e308, -1e308]}
        ),
        pd.DataFrame({'x': ['a', 'b']}, index=[90, 91]),
        target='y',
        task='regression',
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = client(TabularEndpoint()).predict(
            request, model=KumoTabular(), outputs=['prediction', 'explanation']
        )
    assert len(result) == 2
    assert result.explanations.columns is None
    assert any('column analysis' in str(w.message).lower() for w in caught)


class _SessionEndpoint:
    url = 'http://localhost'
    serves_sessions = True

    def predict(self, payload):
        rows = payload['predict']['instance_table']['rows']
        return {
            'id': 'r',
            'model': 'kumo-tabular',
            'metadata': {},
            'predictions': [
                {'row_index': at, 'prediction': 0.5} for at in range(len(rows))
            ],
        }

    def create_session(self, payload):
        return {'session_id': 's1'}

    def session_predict(self, session_id, payload):
        return self.predict(payload)

    def delete_session(self, session_id):
        return None

    def close(self):
        pass


@pytest.mark.parametrize(
    ('outputs', 'rows', 'warned'),
    [
        (None, 2, False),
        (['prediction', 'explanation'], 2, True),
        (['prediction', 'explanation'], 0, True),
    ],
    ids=['not explained', 'explained', 'explained, no rows'],
)
def test_a_failed_analysis_does_not_block_a_session(outputs, rows, warned):
    from test_tabular_direct_core import client

    from sdm_client import KumoTabular, Request

    def request(rows):
        return Request(
            pd.DataFrame(
                {'x': ['a', 'a', 'b', 'b'], 'y': [1e308, 1e308, -1e308, -1e308]}
            ),
            pd.DataFrame({'x': ['a', 'b'][:rows]}, index=[90, 91][:rows]),
            target='y',
            task='regression',
        )

    def analysis_warnings(caught):
        return [w for w in caught if 'column analysis' in str(w.message)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted = client(_SessionEndpoint()).fit(
            request(2), model=KumoTabular(), outputs=outputs
        )
    assert bool(analysis_warnings(caught)) is warned
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fitted.predict(request(rows))
    assert bool(analysis_warnings(caught)) is warned
    assert fitted.session_id == 's1'
    assert len(result) == rows
    assert result.explanations.columns is None


@pytest.mark.parametrize(
    ('values', 'labels'),
    [
        (
            [float(at) for at in range(11)],
            ['[0 - 1]', *(f'({at} - {at + 1}]' for at in range(1, 10))],
        ),
        ([0.0, 1660 / 7], ['[0 - 23.71]', '(23.71 - 237.14]']),
        (
            [10000.0 + at for at in range(10)],
            [
                '[10000 - 10000.90]',
                *(
                    f'({10000 + 0.9 * at:.2f} - {10000 + 0.9 * (at + 1):.2f}]'
                    for at in range(1, 9)
                ),
                '(10008.10 - 10009]',
            ],
        ),
    ],
    ids=['whole numbers', 'fractions', 'large and close'],
)
def test_numeric_cohort_labels(values, labels):
    p = payload([[value, at] for at, value in enumerate(values)])
    p['schema']['instance_table']['columns']['x']['stype'] = 'numerical'

    frame = ColumnExplanations(p).to_pandas()

    assert list(frame[frame['column'] == 'x']['cohort']) == labels


def test_public_policy_constants_remain_available():
    assert {
        name: getattr(ColumnExplanations, name)
        for name in (
            'NUM_COHORTS',
            'MAX_CATEGORIES',
            'MAX_CLASSES',
            'COUNT',
            'OTHER',
            'NA',
            'TABULAR_TABLE',
        )
    } == {
        'NUM_COHORTS': 10,
        'MAX_CATEGORIES': 500,
        'MAX_CLASSES': 30,
        'COUNT': 'COUNT(*)',
        'OTHER': 'OTHER(*)',
        'NA': 'N/A',
        'TABULAR_TABLE': 'instance_table',
    }


def test_sqlite_materialize_explain_uses_sampled_context_without_new_queries():
    from adbc_driver_sqlite import dbapi as sqlite
    from sdm_connectors import SQLiteSource

    connection = sqlite.connect()
    cursor = connection.cursor()
    cursor.execute(
        'CREATE TABLE users (id INTEGER PRIMARY KEY, value DOUBLE, cohort TEXT)'
    )
    cursor.executemany(
        'INSERT INTO users VALUES (?, ?, ?)',
        [(i, float(i), 'a' if i < 12 else 'b') for i in range(24)],
    )
    connection.commit()
    cursor.close()
    try:
        schema = Schema.from_source(SQLiteSource(connection))
        schema['users']['value'].stype = 'numerical'
        schema['users']['cohort'].stype = 'categorical'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            request = PredictiveQuery(
                'PREDICT users.value FOR EACH users.id', schema
            ).materialize(
                indices=[1, 2],
                num_context_samples=24,
                random_seed=7,
                model=KumoRelational(),
            )
    finally:
        connection.close()
    # A closed in-memory database makes any new SQL fail: analysis must use
    # the materialized evidence, the rows sent once 1 and 2 are scored.
    endpoint = Endpoint()
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    result = client.predict(
        request, model=KumoRelational(), outputs=['prediction', 'explanation']
    )
    frame = result.explanations.columns.to_pandas()
    groups = frame[frame['column'] == 'cohort'].set_index('cohort')
    assert groups.loc['a', 'mean'] == pytest.approx(6.3)
    assert groups.loc['b', 'mean'] == pytest.approx(17.5)
    assert groups.loc['a', 'share'] == pytest.approx(10 / 22)
    assert groups.loc['b', 'share'] == pytest.approx(12 / 22)
    assert len(endpoint.sent) == 1


def test_category_and_class_caps_keep_independent_population_means():
    frame = ColumnExplanations(
        payload([[f'c{i:03}', i] for i in range(501)])
    ).to_pandas()
    assert len(frame) == 10
    other = frame.set_index('cohort').loc['OTHER(*)']
    assert other['share'] == pytest.approx(492 / 501)
    assert other['mean'] == 254.5  # Arithmetic mean of labels 9 through 500.
    for size in (500, 501):
        p = payload([[f'c{i:03}', i] for i in range(size)])
        p['schema']['instance_table']['columns']['x']['stype'] = (
            'multicategorical'
        )
        result = ColumnExplanations(p).to_pandas()
        assert result.empty == (size == 501)
    classes = ColumnExplanations(
        payload([['a', i] for i in range(31)], 'classification')
    ).to_pandas()
    assert list(classes['class']) == list(range(30))
    assert list(classes['mean']) == pytest.approx([1 / 31] * 30)
    assert set(classes['share']) == {1.0}


def test_missing_categories_are_explicit_but_missing_numbers_are_not_bins():
    rows = [[None, 8], ['a', 2], ['a', 4]]
    category = ColumnExplanations(payload(rows)).to_pandas().set_index('cohort')
    assert category.loc['N/A', 'mean'] == 8
    assert category.loc['N/A', 'share'] == pytest.approx(1 / 3)
    assert category.loc['a', 'mean'] == 3
    numerical = payload([[None, 100], [0, 2], [10, 4]])
    numerical['schema']['instance_table']['columns']['x']['stype'] = 'numerical'
    numeric = ColumnExplanations(numerical).to_pandas()
    assert set(numeric['mean']) == {2.0, 4.0}
    # Numeric shares are conditional on the two observed numeric values,
    # matching the documented quantile population rather than inventing a bin.
    assert set(numeric['share']) == {0.5}
    assert 'N/A' not in set(numeric['cohort'])


def test_relational_timestamp_cohorts_measure_time_before_each_anchor():
    p = {
        'task': {
            'kind': 'regression',
            'target': {'column_name': 'y'},
            'anchor_time_column': 'anchor',
        },
        'schema': {
            'instance_table': {
                'columns': {
                    'instance_id': {'stype': 'ID'},
                    'ref': {'stype': 'ID'},
                    'y': {'stype': 'numerical'},
                    'anchor': {'stype': 'timestamp'},
                }
            },
            'related_tables': {
                'events': {
                    'primary_key': ['instance_id', 'id'],
                    'columns': {
                        'instance_id': {'stype': 'ID'},
                        'id': {'stype': 'ID'},
                        'when': {'stype': 'timestamp'},
                    },
                }
            },
            'relationships': [
                {
                    'source_columns': ['ref'],
                    'target_table': 'events',
                    'target_columns': ['id'],
                }
            ],
        },
        'context': {
            'instance_table': {
                'columns': ['instance_id', 'ref', 'y', 'anchor'],
                'rows': [
                    [0, 10, 2, '2026-01-02T00:00:00'],
                    [1, 20, 10, '2026-01-02T00:00:00'],
                ],
            },
            'related_tables': {
                'events': {
                    'columns': ['instance_id', 'id', 'when'],
                    'rows': [
                        [0, 10, '2026-01-01T23:30:00'],
                        [0, 10, '2026-01-02T01:00:00'],
                        [0, 10, None],
                        [1, 20, '2026-01-01T22:30:00'],
                    ],
                }
            },
        },
    }
    frame = ColumnExplanations(p).to_pandas()
    times = frame[frame['column'] == 'when'].set_index('cohort')
    assert set(times.index) == {'[0 hours, 1 hours)', '[1 hours, 2 hours)'}
    assert times.loc['[0 hours, 1 hours)', 'mean'] == 2
    assert times.loc['[1 hours, 2 hours)', 'mean'] == 10
    assert set(times['share']) == {0.5}
