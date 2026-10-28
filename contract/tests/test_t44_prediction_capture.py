# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent prediction-entry lifetime and truthful producer observations."""

from __future__ import annotations

import json

import pytest
from test_t136_forecast_capture import PLANNED_IDS, RUNNER, _ask


def entries(capture):
    return capture.core().get('predictions', [])


def assert_entry(
    entry, *, ordinal=0, parent=None, argument=None, completion='returned'
):
    assert entry == {
        'ordinal': ordinal,
        'parent': parent,
        'argument': {'form': 'unobserved'} if argument is None else argument,
        'completion': completion,
        'runs': [],
        'calls': [],
        'results': [],
        'final': None,
    }


class Unobserved:
    """Only owner identity may be observed by the entry wrapper."""

    def __str__(self):
        raise AssertionError('capture observed str')

    def __repr__(self):
        raise AssertionError('capture observed repr')

    def __bool__(self):
        raise AssertionError('capture observed truth')

    def __iter__(self):
        raise AssertionError('capture observed iteration')

    def __hash__(self):
        raise AssertionError('capture observed hash')

    def __eq__(self, other):
        raise AssertionError('capture observed equality')

    @property
    def parts(self):
        raise AssertionError('capture observed Request.parts')

    @property
    def model_id(self):
        raise AssertionError('capture observed model_id')

    @property
    def items(self):
        raise AssertionError('capture observed public items')

    @property
    def metadata(self):
        raise AssertionError('capture observed public metadata')

    def to_pandas(self):
        raise AssertionError('capture observed public frame')


class UnobservedText(str):
    __str__ = Unobserved.__str__
    __repr__ = Unobserved.__repr__
    __iter__ = Unobserved.__iter__
    __hash__ = Unobserved.__hash__
    __eq__ = Unobserved.__eq__

    def encode(self, *args, **kwargs):
        raise AssertionError('capture observed subclass encode')


@pytest.mark.parametrize('model_kind', ['exact-string', 'subclass', 'opaque'])
@pytest.mark.parametrize('raised', [False, True], ids=['returned', 'raised'])
def test_entry_forwards_opaque_owners_without_observation(
    monkeypatch, model_kind, raised
):
    from sdm_client._rust import relational_predict, run

    model = (
        'kumo-relational'
        if model_kind == 'exact-string'
        else UnobservedText('kumo-relational')
        if model_kind == 'subclass'
        else Unobserved()
    )
    client, request, outputs, returned = [Unobserved() for _ in range(4)]
    retained = ValueError('entry failed before any native operation')
    calls = []

    def predict(given_client, given_request, given_model, given_outputs):
        assert given_client is client
        assert given_request is request
        assert given_model is model
        assert given_outputs is outputs
        calls.append('entered')
        if raised:
            raise retained
        return returned

    monkeypatch.setattr(relational_predict, 'predict', predict)
    original_native = run.native
    boundary = RUNNER._Boundary()
    capture = boundary.reset()
    try:
        if raised:
            with pytest.raises(ValueError) as caught:
                relational_predict.predict(client, request, model, outputs)
            assert caught.value is retained
        else:
            assert (
                relational_predict.predict(client, request, model, outputs)
                is returned
            )
        assert calls == ['entered']
        [entry] = entries(capture)
        assert_entry(
            entry,
            argument={'form': 'exact_string', 'value': model}
            if model_kind == 'exact-string'
            else None,
            completion='raised' if raised else 'returned',
        )
        assert capture.operations == []
        assert capture.inputs == []
        assert capture.predicted is None
        assert capture.error is None
    finally:
        boundary.close()
    assert run.native is original_native
    assert relational_predict.predict is predict


def test_close_restores_both_hooks_in_lifo_order_and_keeps_external_wrapper(
    monkeypatch,
):
    from sdm_client._rust import relational_predict, run

    original_native = run.native
    original_predict = relational_predict.predict

    def external_native():
        return original_native()

    def external_predict(*args, **kwargs):
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(run, 'native', external_native)
    monkeypatch.setattr(relational_predict, 'predict', external_predict)
    outer = RUNNER._Boundary()
    outer_native, outer_predict = run.native, relational_predict.predict
    inner = RUNNER._Boundary()
    inner_native, inner_predict = run.native, relational_predict.predict
    try:
        with pytest.raises(RUNNER.RunnerFailure, match='LIFO|order|active'):
            outer.close()
        assert run.native is inner_native
        assert relational_predict.predict is inner_predict
    finally:
        inner.close()
        assert run.native is outer_native
        assert relational_predict.predict is outer_predict
        outer.close()
    assert run.native is external_native
    assert relational_predict.predict is external_predict


def test_two_boundary_lifetimes_and_case_resets_do_not_accumulate_entries(
    monkeypatch,
):
    from sdm_client._rust import relational_predict, run

    returned = Unobserved()

    def predict(*args):
        return returned

    monkeypatch.setattr(relational_predict, 'predict', predict)
    original_native = run.native
    for _ in range(2):
        boundary = RUNNER._Boundary()
        try:
            first = boundary.reset()
            assert (
                relational_predict.predict(None, None, 'first', None)
                is returned
            )
            second = boundary.reset()
            assert second is not first
            assert entries(second) == []
            assert (
                relational_predict.predict(None, None, 'second', None)
                is returned
            )
            [first_entry] = entries(first)
            [second_entry] = entries(second)
            assert_entry(
                first_entry, argument={'form': 'exact_string', 'value': 'first'}
            )
            assert_entry(
                second_entry,
                argument={'form': 'exact_string', 'value': 'second'},
            )
        finally:
            boundary.close()
        assert run.native is original_native
        assert relational_predict.predict is predict


def test_reset_during_entry_is_rejected_without_splitting_its_owner(
    monkeypatch,
):
    from sdm_client._rust import relational_predict

    holder = {}
    returned = Unobserved()

    def predict(*args):
        with pytest.raises(
            RUNNER.RunnerFailure, match='active|entry|prediction'
        ):
            holder['boundary'].reset()
        assert holder['boundary'].capture is holder['capture']
        return returned

    monkeypatch.setattr(relational_predict, 'predict', predict)
    boundary = RUNNER._Boundary()
    holder.update(boundary=boundary, capture=boundary.reset())
    try:
        assert (
            relational_predict.predict(None, None, 'active', None) is returned
        )
        [entry] = entries(holder['capture'])
        assert_entry(
            entry, argument={'form': 'exact_string', 'value': 'active'}
        )
    finally:
        boundary.close()


def test_nested_entries_keep_actual_parentage_and_exact_return_owners(
    monkeypatch,
):
    from sdm_client._rust import relational_predict

    outer_result, inner_result = Unobserved(), Unobserved()
    order = []

    def predict(client, request, model, outputs):
        order.append(model)
        if model == 'outer':
            assert (
                relational_predict.predict(client, request, 'inner', outputs)
                is inner_result
            )
            return outer_result
        assert model == 'inner'
        return inner_result

    monkeypatch.setattr(relational_predict, 'predict', predict)
    boundary = RUNNER._Boundary()
    capture = boundary.reset()
    try:
        assert (
            relational_predict.predict(None, None, 'outer', None)
            is outer_result
        )
        assert order == ['outer', 'inner']
        outer, inner = entries(capture)
        assert_entry(outer, argument={'form': 'exact_string', 'value': 'outer'})
        assert_entry(
            inner,
            ordinal=1,
            parent=0,
            argument={'form': 'exact_string', 'value': 'inner'},
        )
    finally:
        boundary.close()


def test_actual_forecast_entry_has_only_real_runs_calls_and_final_evidence(
    monkeypatch,
):
    observed = _ask(monkeypatch, (2,), capture=True)
    assert observed['error'] is None, observed['error']
    capture = observed['capture']
    [entry] = entries(capture)
    assert entry['ordinal'] == 0
    assert entry['parent'] is None
    assert entry['argument'] == {'form': 'unobserved'}
    assert entry['completion'] == 'returned'
    ids = [PLANNED_IDS[0], PLANNED_IDS[2]]
    assert [call['request_id'] for call in entry['calls']] == ids
    starts = [
        item
        for item in capture.inputs
        if item['op'] == 'prepare_request' and item['entry'] == 0
    ]
    assert all(item['run'] in entry['runs'] for item in starts)
    assert len(set(entry['runs'])) == len(entry['runs'])
    assert 'predict' not in capture.operations
    final = entry['results'][entry['final']]
    assert final['metadata'] == capture.predicted['metadata']
    assert final['result'] == capture.predicted['result']
    assert final['data']['predicted']['model'] == capture.predicted['model']
    assert len(final['data']['lineage']) == len(observed['frame'])
    assert {row['call_request_id'] for row in final['data']['lineage']} == set(
        ids
    )
    assert final['input']['stage'] == 'finish_relational_response'
    assert final['input']['input']['aggregation']['kind'] == 'steps'
    assert final['run'] in entry['runs']
    for call in entry['calls']:
        assert call['run'] in entry['runs']
        assert call['input_body_sha256'] == call['body_sha256']
        assert call['body']['metadata']['request_id'] == call['request_id']
        assert call['scope']['call_request_id'] == call['request_id']
        assert call['scope']['model'] == capture.predicted['model']


def test_all_empty_forecast_has_a_real_final_result_and_no_invented_ids(
    monkeypatch,
):
    plain = _ask(monkeypatch, (1, 2, 3), capture=False)
    observed = _ask(monkeypatch, (1, 2, 3), capture=True)
    assert plain['error'] is None, plain['error']
    assert observed['error'] is None, observed['error']
    assert observed['frame'].equals(plain['frame'])
    assert observed['metadata'] == plain['metadata']
    assert observed['items'] == plain['items']
    assert len(observed['frame']) == 0
    assert observed['started'] == observed['received'] == []
    [entry] = entries(observed['capture'])
    assert entry['completion'] == 'returned'
    assert entry['calls'] == []
    assert entry['final'] is not None
    final = entry['results'][entry['final']]
    assert final['data']['lineage'] == []
    assert final['metadata'] == observed['capture'].predicted['metadata']
    assert final['result'] == observed['capture'].predicted['result']
    assert observed['capture'].predicted['body'] is None


def test_actual_invalid_exact_model_records_failed_resolution_without_calls(
    monkeypatch,
):
    model = 'not-a-relational-model'
    observed = _ask(monkeypatch, (), capture=True, model=model)
    assert observed['error'] is not None
    assert not isinstance(observed['error'], RUNNER.RunnerFailure)
    assert observed['received'] == observed['started'] == []
    capture = observed['capture']
    [entry] = entries(capture)
    assert entry['argument'] == {'form': 'exact_string', 'value': model}
    assert entry['completion'] == 'raised'
    assert entry['calls'] == entry['results'] == []
    assert entry['final'] is None
    assert entry['runs']
    lookup = [
        item
        for item in capture.inputs
        if item['op'] == 'prepare_request'
        and item['input']['stage'] == 'model_lookup'
    ]
    assert len(lookup) == 1
    assert lookup[0]['input']['position'] is None
    assert lookup[0]['entry'] == entry['ordinal']
    assert lookup[0]['run'] in entry['runs']
    assert capture.error is not None
    assert capture.predicted is None


def test_actual_output_failure_before_model_lookup_is_entry_only(monkeypatch):
    retained = ValueError('second reached output-name observation failed')

    class Outputs(list):
        def __init__(self):
            super().__init__(['prediction'])
            self.observations = 0

        def __iter__(self):
            self.observations += 1
            if self.observations == 2:
                raise retained
            return super().__iter__()

    outputs = Outputs()
    observed = _ask(monkeypatch, (), capture=True, outputs=outputs)
    assert observed['error'] is retained
    assert outputs.observations == 2
    assert observed['received'] == observed['started'] == []
    capture = observed['capture']
    [entry] = entries(capture)
    assert_entry(entry, completion='raised')
    assert capture.error is None
    assert capture.predicted is None


def test_unrelated_model_preparation_does_not_invent_prediction_intent():
    from sdm_client._rust.prepare import _call

    boundary = RUNNER._Boundary()
    capture = boundary.reset()
    try:
        catalogue = _call({'stage': 'model_catalogue'})
        assert 'kumo-relational' in catalogue
        assert entries(capture) == []
        assert capture.operations == ['prepare_request']
        [started] = capture.inputs
        assert started['op'] == 'prepare_request'
        assert started['input'] == {'stage': 'model_catalogue'}
        assert started['entry'] is None
        assert capture.predicted is None
        assert capture.error is None
    finally:
        boundary.close()


def legacy_capture(empty):
    """Untouched legacy operation shape, with an actual ID for every step."""
    payloads = [
        {
            'predict': {
                'instance_table': {
                    'columns': ['id'],
                    'rows': [] if step in empty else [[step]],
                }
            }
        }
        for step in range(1, 4)
    ]
    state = {
        'version': 1,
        'query': 1,
        'payload': 2,
        'forecast': [
            {'request_id': PLANNED_IDS[1], 'payload': 3},
            {'request_id': PLANNED_IDS[2], 'payload': 4},
        ],
    }
    operation = {
        'op': 'predict',
        'model': 'kumo-relational',
        'request_id': PLANNED_IDS[0],
        'state': state,
    }
    attached = {
        1: b'{}',
        **{
            at + 2: json.dumps(body).encode()
            for at, body in enumerate(payloads)
        },
    }
    capture = RUNNER._Capture()
    capture.started(json.dumps({'operation': operation}).encode(), attached)
    return capture


def test_final_forecast_lineage_cannot_name_an_unobserved_call(monkeypatch):
    control = _ask(monkeypatch, (), capture=True)
    assert control['error'] is None, control['error']
    observed = _ask(monkeypatch, (), capture=True, mode='lineage')
    assert observed['tampered']
    assert {change[0] for change in observed['tampered']} == {'lineage'}
    assert observed['received'] == PLANNED_IDS
    assert isinstance(observed['error'], RUNNER.RunnerFailure), observed[
        'error'
    ]
    assert 'lineage' in str(observed['error']).lower()


@pytest.mark.parametrize('empty', [(1,), (2,), (1, 3), ()])
def test_raw_legacy_three_id_producer_keeps_nonempty_call_order(empty):
    capture = legacy_capture(empty)
    ids = [
        value for step, value in enumerate(PLANNED_IDS, 1) if step not in empty
    ]
    for request_id in ids:
        capture.stepped({'step': 'call_model', 'body': 20})
        capture.taken(
            20, json.dumps({'metadata': {'request_id': request_id}}).encode()
        )
    capture.done(
        {
            'op': 'predict',
            'model': 'kumo-relational',
            'request_id': 'endpoint-final',
            'metadata': 30,
            'result': 31,
        }
    )
    assert capture.operations == ['predict']
    assert capture.inputs[0]['op'] == 'predict'
    assert capture.predicted['body']['metadata']['request_id'] == ids[-1]
    if len(ids) > 1:
        assert [
            call['metadata']['request_id']
            for call in capture.predicted['calls']
        ] == ids


def test_raw_legacy_capture_rejects_an_empty_steps_reserved_id():
    capture = legacy_capture((1, 3))
    capture.stepped({'step': 'call_model', 'body': 20})
    with pytest.raises(
        RUNNER.RunnerFailure, match='does not carry the request id'
    ):
        capture.taken(
            20,
            json.dumps({'metadata': {'request_id': PLANNED_IDS[2]}}).encode(),
        )
