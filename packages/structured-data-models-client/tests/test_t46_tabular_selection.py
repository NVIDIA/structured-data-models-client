# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public PQL controls for the connected integer-key/RangeIndex source path."""

import warnings

import numpy as np
import pandas as pd
import pytest
from test_t46_direct_numeric import Scripted

from sdm_client import PredictiveQuery, SDMClient, _routing
from sdm_client._routing import RouteError
from sdm_client.errors import StructuredDataError


def _table(rows=43):
    frame = pd.DataFrame(
        {
            'entity_id': np.arange(rows, dtype='int64'),
            'feature': np.arange(rows, dtype='float64') + 0.5,
            'answer': np.arange(rows, dtype='float64') + 2.5,
        }
    )
    frame.loc[frame.index[::7], 'answer'] = np.nan
    return frame


def _materialize(monkeypatch, frame, core, **arguments):
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(
            _routing.MATERIALIZE['frames']
            + _routing.TABULAR_PREDICT_FRAME
            + _routing.DIRECT_CONSUMERS
        )
        if core
        else '',
    )
    query = PredictiveQuery(
        'PREDICT things.answer FOR EACH things.entity_id', {'things': frame}
    )
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        request = query.materialize(task='regression', **arguments)
    return request, [str(item.message) for item in notices]


@pytest.mark.parametrize('count', [None, 5, 100])
def test_public_indices_keep_asked_order_and_filter_before_the_sample(
    monkeypatch, count
):
    frame = _table()
    arguments = {'indices': [11, 2, 11, 7], 'num_context_samples': count}
    old, old_notices = _materialize(
        monkeypatch, frame.copy(), False, **arguments
    )
    core, core_notices = _materialize(
        monkeypatch, frame.copy(), True, **arguments
    )
    pd.testing.assert_frame_equal(
        core.context.instance_table, old.context.instance_table
    )
    pd.testing.assert_frame_equal(
        core.predict.instance_table, old.predict.instance_table
    )
    assert list(core.predict.instance_table.entity_id) == [11, 2, 11, 7]
    assert core.context.instance_table.answer.notna().all()
    assert not set(core.context.instance_table.entity_id) & {11, 2, 7}
    assert core.task == old.task
    assert core.query == old.query
    assert core_notices == old_notices


def test_public_supplied_rows_exclude_holdout_and_keep_external_rows(
    monkeypatch,
):
    frame = _table()
    scoring = pd.DataFrame({'entity_id': [11, 500], 'feature': [11.5, 500.5]})
    old, _ = _materialize(monkeypatch, frame.copy(), False, predict=scoring)
    core, _ = _materialize(monkeypatch, frame.copy(), True, predict=scoring)
    pd.testing.assert_frame_equal(
        core.context.instance_table, old.context.instance_table
    )
    pd.testing.assert_frame_equal(
        core.predict.instance_table, old.predict.instance_table
    )
    assert 11 not in set(core.context.instance_table.entity_id)
    assert list(core.predict.instance_table.entity_id) == [11, 500]


def test_public_automatic_cap_keeps_the_native_seed_and_range_index_order(
    monkeypatch,
):
    frame = _table(12_019)
    frame.index = pd.RangeIndex(10, 10 + 3 * len(frame), 3)
    old, old_notices = _materialize(
        monkeypatch, frame.copy(), False, indices=[11]
    )
    core, core_notices = _materialize(
        monkeypatch, frame.copy(), True, indices=[11]
    )
    assert core.context_rows == 10_000
    pd.testing.assert_frame_equal(
        core.context.instance_table, old.context.instance_table
    )
    assert core_notices == old_notices


@pytest.mark.parametrize('core', [False, True])
def test_public_missing_indices_keep_the_original_count_and_values(
    monkeypatch, core
):
    frame = _table()
    with pytest.raises(StructuredDataError) as refused:
        _materialize(monkeypatch, frame, core, indices=[800, 800, 801])
    assert refused.value.code == 'INVALID_REQUEST'
    assert '3 of the 3 indices' in refused.value.message
    assert '[800, 800, 801]' in refused.value.message


def test_selected_core_context_does_not_call_native_sampling(monkeypatch):
    query = PredictiveQuery(
        'PREDICT things.answer FOR EACH things.entity_id', {'things': _table()}
    )

    def unexpected(*args, **kwargs):
        raise AssertionError(
            'tabular context selection reached pandas sampling'
        )

    monkeypatch.setattr(pd.DataFrame, 'sample', unexpected)
    monkeypatch.setenv(
        _routing.ENVIRONMENT, ','.join(_routing.MATERIALIZE['frames'])
    )
    request = query.materialize(
        indices=[2, 11], num_context_samples=5, task='regression'
    )
    assert request.context_rows == 5
    assert request.context.instance_table.index.is_monotonic_increasing


def test_unsampled_context_preserves_source_order_without_sorting(monkeypatch):
    frame = _table()
    frame.entity_id = frame.entity_id.iloc[::-1].to_numpy()
    old, _ = _materialize(monkeypatch, frame.copy(), False, indices=[11])
    core, _ = _materialize(monkeypatch, frame.copy(), True, indices=[11])
    pd.testing.assert_frame_equal(
        core.context.instance_table, old.context.instance_table
    )
    assert not core.context.instance_table.entity_id.is_monotonic_increasing


def test_public_selected_context_reaches_the_existing_prediction_wire(
    monkeypatch,
):
    answers = []
    for core in (False, True):
        request, _ = _materialize(
            monkeypatch, _table(), core, indices=[11, 2], num_context_samples=5
        )
        scripted = Scripted()
        with SDMClient('http://localhost:1', max_retries=0) as client:
            client._transport.close()
            client._transport = scripted
            result = client.predict(request, model='kumo-tabular')
            frame = result.to_pandas()
        assert len(scripted.calls) == 1
        [body] = scripted.calls
        body['metadata']['request_id'] = '<drawn>'
        answers.append((body, frame))
    assert answers[0][0] == answers[1][0]
    pd.testing.assert_frame_equal(answers[0][1], answers[1][1])


@pytest.mark.parametrize('scoring_ids', [[11], [500], []])
def test_supplied_holdout_precedes_the_actual_native_target_read(
    monkeypatch, scoring_ids
):
    observations = []
    for core in (False, True):
        source = _table()
        query = PredictiveQuery(
            'PREDICT things.answer FOR EACH things.entity_id',
            {'things': source},
        )
        scoring = pd.DataFrame(
            {
                'entity_id': pd.Series(scoring_ids, dtype='int64'),
                'feature': pd.Series([0.5] * len(scoring_ids), dtype='float64'),
            }
        )
        target_reads = []
        original_get = pd.DataFrame.__getitem__

        def observed_get(frame, key):
            if isinstance(key, str) and key == 'answer':
                target_reads.append((frame is source, tuple(frame.index)))
            return original_get(frame, key)

        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(
                    _routing.MATERIALIZE['frames']
                    + _routing.TABULAR_PREDICT_FRAME
                )
                if core
                else '',
            )
            patch.setattr(pd.DataFrame, '__getitem__', observed_get)
            request = query.materialize(predict=scoring, task='regression')
        observations.append((target_reads[0], request.context.instance_table))
    assert observations[0][0] == observations[1][0]
    assert observations[1][0][0] == (not scoring_ids)
    expected = tuple(row for row in range(43) if row not in scoring_ids)
    assert observations[1][0][1] == expected
    pd.testing.assert_frame_equal(observations[0][1], observations[1][1])


def test_supplied_entity_read_effects_precede_source_membership(monkeypatch):
    results = []
    for core in (False, True):
        source = _table()
        scoring = pd.DataFrame({'entity_id': [11], 'feature': [0.5]})
        query = PredictiveQuery(
            'PREDICT things.answer FOR EACH things.entity_id',
            {'things': source},
        )
        events = []
        original_get = pd.DataFrame.__getitem__

        def observed_get(frame, key):
            if isinstance(key, str) and key == 'entity_id':
                if frame is scoring:
                    events.append('scoring')
                    source.loc[11, 'entity_id'] = 99
                elif frame is source:
                    events.append('source')
            return original_get(frame, key)

        with monkeypatch.context() as patch:
            patch.setenv(
                _routing.ENVIRONMENT,
                ','.join(
                    _routing.MATERIALIZE['frames']
                    + _routing.TABULAR_PREDICT_FRAME
                )
                if core
                else '',
            )
            patch.setattr(pd.DataFrame, '__getitem__', observed_get)
            request = query.materialize(predict=scoring, task='regression')
        results.append((events[:2], request.context.instance_table))
    assert results[0][0] == results[1][0] == ['scoring', 'source']
    pd.testing.assert_frame_equal(results[0][1], results[1][1])
    assert 99 in set(results[1][1].entity_id)


@pytest.mark.parametrize('changed', ['empty-scoring-entity', 'held-out-target'])
def test_sampled_context_uses_actual_candidate_index_after_native_getters(
    monkeypatch, changed
):
    observed = []
    for core in (False, True):
        with monkeypatch.context() as route:
            route.setenv(
                _routing.ENVIRONMENT,
                ','.join(
                    _routing.MATERIALIZE['frames']
                    + _routing.TABULAR_PREDICT_FRAME
                    + _routing.DIRECT_CONSUMERS
                )
                if core
                else '',
            )
            source = _table(12_019)
            scoring_ids = [] if changed == 'empty-scoring-entity' else [11]
            scoring = pd.DataFrame(
                {
                    'entity_id': pd.Series(scoring_ids, dtype='int64'),
                    'feature': pd.Series(
                        [0.5] * len(scoring_ids), dtype='float64'
                    ),
                }
            )
            query = PredictiveQuery(
                'PREDICT things.answer FOR EACH things.entity_id',
                {'things': source},
            )
            events = []
            original_get = pd.DataFrame.__getitem__

            def observed_get(frame, key):
                if isinstance(key, str) and not events:
                    if changed == 'empty-scoring-entity' and (
                        frame is scoring and key == 'entity_id'
                    ):
                        source.index = pd.RangeIndex(len(source) - 1, -1, -1)
                        events.append(('scoring', len(frame)))
                    elif changed == 'held-out-target' and (
                        key == 'answer' and frame is not source
                    ):
                        frame.index = pd.Index(
                            7 - 3 * np.arange(len(frame), dtype='int64')
                        )
                        events.append(
                            ('target', len(frame), tuple(frame.index))
                        )
                return original_get(frame, key)

            route.setattr(pd.DataFrame, '__getitem__', observed_get)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                request = query.materialize(predict=scoring, task='regression')
            assert len(events) == 1
            assert request.context_rows == 10_000
            context = request.context.instance_table.copy()
            predict = request.predict.instance_table.copy()
            scripted = Scripted()
            with SDMClient('http://localhost:1', max_retries=0) as client:
                client._transport.close()
                client._transport = scripted
                result = client.predict(request, model='kumo-tabular')
                answered = result.to_pandas()
            for body in scripted.calls:
                body['metadata']['request_id'] = '<drawn>'
            observed.append(
                (
                    events,
                    context,
                    predict,
                    request.task,
                    request.query,
                    scripted.calls,
                    answered,
                )
            )
    old, current = observed
    assert old[0] == current[0]
    pd.testing.assert_frame_equal(old[1], current[1])
    pd.testing.assert_frame_equal(old[2], current[2])
    assert old[3:6] == current[3:6]
    pd.testing.assert_frame_equal(old[6], current[6])
    assert current[1].entity_id.is_monotonic_decreasing
    assert (len(current[5]) == 0) == (changed == 'empty-scoring-entity')


def test_unsampled_context_does_not_observe_candidate_index_labels(monkeypatch):
    from sdm_client._rust import tabular_materialize

    def unexpected(frame):
        raise AssertionError('uncapped context observed native index labels')

    monkeypatch.setattr(tabular_materialize, '_index_labels', unexpected)
    old, _ = _materialize(monkeypatch, _table(), False, indices=[11])
    current, _ = _materialize(monkeypatch, _table(), True, indices=[11])
    pd.testing.assert_frame_equal(
        old.context.instance_table, current.context.instance_table
    )


@pytest.mark.parametrize('core', [False, True])
def test_reduced_duplicate_candidate_index_is_an_unfinished_family(
    monkeypatch, core
):
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(
            _routing.MATERIALIZE['frames'] + _routing.TABULAR_PREDICT_FRAME
        )
        if core
        else '',
    )
    source = _table(12_019)
    scoring = pd.DataFrame({'entity_id': [11], 'feature': [0.5]})
    query = PredictiveQuery(
        'PREDICT things.answer FOR EACH things.entity_id', {'things': source}
    )
    original_get = pd.DataFrame.__getitem__
    changed = []

    def observed_get(frame, key):
        if (
            not changed
            and isinstance(key, str)
            and key == 'answer'
            and frame is not source
        ):
            frame.index = pd.Index(np.arange(len(frame), dtype='int64') // 2)
            changed.append(len(frame))
        return original_get(frame, key)

    monkeypatch.setattr(pd.DataFrame, '__getitem__', observed_get)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if core:
            with pytest.raises(RouteError) as refused:
                query.materialize(predict=scoring, task='regression')
            assert type(refused.value) is RouteError
        else:
            request = query.materialize(predict=scoring, task='regression')
            assert request.context_rows == 10_000
            assert not request.context.instance_table.index.is_unique
            assert request.context.instance_table.index.is_monotonic_increasing
    assert changed == [len(source) - 1]
