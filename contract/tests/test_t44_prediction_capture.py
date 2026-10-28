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
    from sdm_client._rust import errors, relational_predict, run

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
    original_presentation = errors.error_from_core
    original_carried = run._carried
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
    assert errors.error_from_core is original_presentation
    assert run._carried is original_carried


def test_close_restores_four_hooks_in_lifo_order_and_keeps_external_wrapper(
    monkeypatch,
):
    from sdm_client._rust import errors, relational_predict, run

    original_native = run.native
    original_predict = relational_predict.predict
    original_presentation = errors.error_from_core
    original_carried = run._carried

    def external_native():
        return original_native()

    def external_predict(*args, **kwargs):
        return original_predict(*args, **kwargs)

    def external_presentation(*args, **kwargs):
        return original_presentation(*args, **kwargs)

    def external_carried(*args, **kwargs):
        return original_carried(*args, **kwargs)

    monkeypatch.setattr(run, 'native', external_native)
    monkeypatch.setattr(relational_predict, 'predict', external_predict)
    monkeypatch.setattr(errors, 'error_from_core', external_presentation)
    monkeypatch.setattr(run, '_carried', external_carried)
    outer = RUNNER._Boundary()
    outer_native, outer_predict = run.native, relational_predict.predict
    outer_presentation = errors.error_from_core
    outer_carried = run._carried
    inner = RUNNER._Boundary()
    inner_native, inner_predict = run.native, relational_predict.predict
    inner_presentation = errors.error_from_core
    inner_carried = run._carried
    try:
        with pytest.raises(RUNNER.RunnerFailure, match='LIFO|order|active'):
            outer.close()
        assert run.native is inner_native
        assert relational_predict.predict is inner_predict
        assert errors.error_from_core is inner_presentation
        assert run._carried is inner_carried
    finally:
        inner.close()
        assert run.native is outer_native
        assert relational_predict.predict is outer_predict
        assert errors.error_from_core is outer_presentation
        assert run._carried is outer_carried
        outer.close()
    assert run.native is external_native
    assert relational_predict.predict is external_predict
    assert errors.error_from_core is external_presentation
    assert run._carried is external_carried


def test_two_boundary_lifetimes_and_case_resets_do_not_accumulate_entries(
    monkeypatch,
):
    from sdm_client._rust import errors, relational_predict, run

    returned = Unobserved()

    def predict(*args):
        return returned

    monkeypatch.setattr(relational_predict, 'predict', predict)
    original_native = run.native
    original_presentation = errors.error_from_core
    original_carried = run._carried
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
        assert errors.error_from_core is original_presentation
        assert run._carried is original_carried


def test_reset_during_entry_is_rejected_without_splitting_its_owner(
    monkeypatch,
):
    from sdm_client._rust import errors, relational_predict, run

    holder = {}
    returned = Unobserved()

    def predict(*args):
        with pytest.raises(
            RUNNER.RunnerFailure, match='active|entry|prediction'
        ):
            holder['boundary'].reset()
        assert holder['boundary'].capture is holder['capture']
        assert run.native is holder['native']
        assert relational_predict.predict is holder['predict']
        assert errors.error_from_core is holder['presentation']
        assert run._carried is holder['carried']
        return returned

    monkeypatch.setattr(relational_predict, 'predict', predict)
    boundary = RUNNER._Boundary()
    holder.update(boundary=boundary, capture=boundary.reset())
    holder.update(
        native=run.native,
        predict=relational_predict.predict,
        presentation=errors.error_from_core,
        carried=run._carried,
    )
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


class UnformattedFailure(ValueError):
    __str__ = Unobserved.__str__
    __repr__ = Unobserved.__repr__
    __hash__ = Unobserved.__hash__
    __eq__ = Unobserved.__eq__


def error_entry(monkeypatch, predict, *, converter=None):
    """Reach real preparation inside the transparently observed entry."""
    from sdm_client._rust import errors, relational_predict, run

    monkeypatch.setattr(relational_predict, 'predict', predict)
    if converter is not None:
        monkeypatch.setattr(errors, 'error_from_core', converter)
    original_native = run.native
    original_presentation = errors.error_from_core
    original_carried = run._carried
    boundary = RUNNER._Boundary()
    captured = boundary.reset()
    error = None
    value = None
    try:
        try:
            value = relational_predict.predict(
                None, None, 'kumo-relational', None
            )
        except Exception as raised:  # noqa: BLE001 - preserve the actual owner
            error = raised
    finally:
        boundary.close()
    assert run.native is original_native
    assert relational_predict.predict is predict
    assert errors.error_from_core is original_presentation
    assert run._carried is original_carried
    [entry] = entries(captured)
    assert entry['completion'] == (
        'raised' if error is not None else 'returned'
    )
    return captured, entry, error, value


def lookup_errors(capture):
    found = [
        record
        for record in capture.inputs
        if record['op'] == 'prepare_request'
        and record['input'] == {'stage': 'model_lookup', 'position': None}
    ]
    assert found
    for record in found:
        assert record['error'] is not None
        assert record['result'] is None
        assert record['entry'] == 0
    return found


def test_recovered_refusal_as_context_does_not_become_terminal_core_error(
    monkeypatch,
):
    from sdm_client import StructuredDataError
    from sdm_client._rust.prepare import _call

    retained = UnformattedFailure('host failure after recovery')
    selected = []

    def predict(*args):
        try:
            _call({'stage': 'model_lookup', 'position': None})
        except StructuredDataError as refused:
            selected.append(refused)
            raise retained

    capture, entry, error, _ = error_entry(monkeypatch, predict)
    assert error is retained
    [refused] = selected
    assert BaseException.__getattribute__(error, '__context__') is refused
    assert BaseException.__getattribute__(error, '__cause__') is None
    assert (
        BaseException.__getattribute__(error, '__suppress_context__') is False
    )
    [lookup] = lookup_errors(capture)
    assert lookup['run'] in entry['runs']
    assert capture.error is None
    assert entry.get('failure') is None


def test_exact_presented_refusal_survives_a_later_successful_run(monkeypatch):
    from sdm_client import StructuredDataError
    from sdm_client._rust import errors
    from sdm_client._rust.prepare import _call

    actual_converter = errors.error_from_core
    presentations = []
    selected = []

    def converter(*args, **kwargs):
        made = actual_converter(*args, **kwargs)
        presentations.append((made, kwargs['internal_context']))
        return made

    def predict(*args):
        try:
            _call({'stage': 'model_lookup', 'position': None})
        except StructuredDataError as refused:
            selected.append(refused)
        assert 'kumo-relational' in _call({'stage': 'model_catalogue'})
        raise selected[0]

    capture, entry, error, _ = error_entry(
        monkeypatch, predict, converter=converter
    )
    [refused] = selected
    [(made, native)] = presentations
    assert error is refused is made
    assert native is not refused
    assert BaseException.__getattribute__(error, '__cause__') is None
    assert BaseException.__getattribute__(error, '__context__') is None
    [lookup] = lookup_errors(capture)
    catalogue = capture.inputs[-1]
    assert catalogue['input'] == {'stage': 'model_catalogue'}
    assert catalogue['error'] is None
    assert 'kumo-relational' in catalogue['result']
    assert catalogue['run'] != lookup['run']
    assert capture.error == lookup['error']
    assert entry['failure'] == {'run': lookup['run'], 'relation': 'selected'}


def test_failed_reconstruction_has_no_successful_presentation_pair(monkeypatch):
    from sdm_client._rust.prepare import _call

    retained = UnformattedFailure('native reconstruction failed')
    contexts = []

    def converter(*args, **kwargs):
        contexts.append(kwargs['internal_context'])
        raise retained

    def predict(*args):
        return _call({'stage': 'model_lookup', 'position': None})

    capture, entry, error, _ = error_entry(
        monkeypatch, predict, converter=converter
    )
    assert error is retained
    [native] = contexts
    assert BaseException.__getattribute__(error, '__context__') is native
    assert BaseException.__getattribute__(error, '__cause__') is None
    assert (
        BaseException.__getattribute__(error, '__suppress_context__') is False
    )
    [lookup] = lookup_errors(capture)
    assert lookup['run'] in entry['runs']
    assert capture.error is None
    assert entry.get('failure') is None


def test_explicit_outer_cause_keeps_public_owner_and_native_association(
    monkeypatch,
):
    from sdm_client import StructuredDataError
    from sdm_client._rust.prepare import _call

    retained = UnformattedFailure('distinct host wrapper')
    selected = []

    def predict(*args):
        try:
            _call({'stage': 'model_lookup', 'position': None})
        except StructuredDataError as refused:
            selected.append(refused)
            assert 'kumo-relational' in _call({'stage': 'model_catalogue'})
            raise retained from refused

    capture, entry, error, _ = error_entry(monkeypatch, predict)
    assert error is retained
    [refused] = selected
    assert BaseException.__getattribute__(error, '__cause__') is refused
    assert BaseException.__getattribute__(error, '__context__') is refused
    assert BaseException.__getattribute__(error, '__suppress_context__') is True
    [lookup] = lookup_errors(capture)
    assert capture.inputs[-1]['error'] is None
    assert capture.error == lookup['error']
    assert entry['failure'] == {'run': lookup['run'], 'relation': 'cause'}


def test_reconstruction_reraising_prior_sdk_owner_does_not_select_new_run(
    monkeypatch,
):
    from sdm_client import StructuredDataError
    from sdm_client._rust import errors
    from sdm_client._rust.prepare import _call

    actual_converter = errors.error_from_core
    presentations = []
    attempts = []

    def converter(*args, **kwargs):
        attempts.append(kwargs['internal_context'])
        if presentations:
            raise presentations[0]
        made = actual_converter(*args, **kwargs)
        presentations.append(made)
        return made

    def predict(*args):
        try:
            _call({'stage': 'model_lookup', 'position': None})
        except StructuredDataError:
            pass
        return _call({'stage': 'model_lookup', 'position': None})

    capture, entry, error, _ = error_entry(
        monkeypatch, predict, converter=converter
    )
    [made] = presentations
    assert error is made
    assert len(attempts) == 2
    assert attempts[0] is not attempts[1]
    assert BaseException.__getattribute__(error, '__context__') is attempts[1]
    first, second = lookup_errors(capture)
    assert first['run'] != second['run']
    assert capture.error == first['error']
    assert entry['failure'] == {'run': first['run'], 'relation': 'selected'}


def test_nested_refusal_keeps_its_actual_child_run_and_outer_error_identity(
    monkeypatch,
):
    from sdm_client import StructuredDataError
    from sdm_client._rust import relational_predict
    from sdm_client._rust.prepare import _call

    def predict(client, request, model, outputs):
        if model == 'outer':
            return relational_predict.predict(client, request, 'inner', outputs)
        return _call({'stage': 'model_lookup', 'position': None})

    monkeypatch.setattr(relational_predict, 'predict', predict)
    boundary = RUNNER._Boundary()
    capture = boundary.reset()
    try:
        with pytest.raises(StructuredDataError) as caught:
            relational_predict.predict(None, None, 'outer', None)
        selected = caught.value
    finally:
        boundary.close()
    outer, child = entries(capture)
    assert outer['parent'] is None and child['parent'] == outer['ordinal']
    [lookup] = [
        record for record in capture.inputs if record['op'] == 'prepare_request'
    ]
    assert lookup['entry'] == child['ordinal']
    assert lookup['run'] in child['runs'] and lookup['run'] not in outer['runs']
    assert outer['completion'] == child['completion'] == 'raised'
    assert (
        outer['failure']
        == child['failure']
        == {'run': lookup['run'], 'relation': 'selected'}
    )
    assert capture.error == lookup['error']
    assert BaseException.__getattribute__(selected, '__cause__') is None
    assert BaseException.__getattribute__(selected, '__context__') is None


class NoHandoverReads(UnformattedFailure):
    @property
    def details(self):
        raise AssertionError(
            'observer repeated the no-op handover details read'
        )


@pytest.mark.parametrize('later_success', [False, True])
def test_noop_refused_handover_selects_exact_native_owner(
    monkeypatch, later_success
):
    from sdm_client import StructuredDataError
    from sdm_client._rust import run
    from sdm_client._rust.prepare import _call

    retained = NoHandoverReads('original transport owner')
    selected = []
    actual_carried = run._carried
    handovers = []

    def carried(error, raised):
        assert raised is retained
        result = actual_carried(error, raised)
        handovers.append((error, raised))
        return result

    monkeypatch.setattr(run, '_carried', carried)

    def predict(*args):
        try:
            _call({'stage': 'model_lookup', 'position': None})
        except StructuredDataError as refused:
            selected.append(refused)
        run._carried(selected[0], retained)
        if later_success:
            assert 'kumo-relational' in _call({'stage': 'model_catalogue'})
        raise retained

    capture, entry, error, _ = error_entry(monkeypatch, predict)
    assert error is retained
    assert len(handovers) == 1
    assert handovers[0][0] is selected[0]
    assert handovers[0][1] is retained
    assert BaseException.__getattribute__(error, '__cause__') is None
    assert BaseException.__getattribute__(error, '__context__') is None
    assert BaseException.__getattribute__(error, '__dict__') == {}
    [lookup] = lookup_errors(capture)
    assert capture.error == lookup['error']
    assert entry['failure'] == {'run': lookup['run'], 'relation': 'refused'}


def test_failed_refused_handover_does_not_pair_its_raised_owner(monkeypatch):
    from sdm_client import StructuredDataError
    from sdm_client._rust import run
    from sdm_client._rust.prepare import _call

    target = NoHandoverReads('unselected replacement')
    retained = UnformattedFailure('handover failed')
    attempted = []

    def carried(error, raised):
        attempted.append((error, raised))
        raise retained

    monkeypatch.setattr(run, '_carried', carried)

    def predict(*args):
        selected = None
        try:
            _call({'stage': 'model_lookup', 'position': None})
        except StructuredDataError as refused:
            selected = refused
        run._carried(selected, target)
        raise AssertionError('failed handover returned')

    capture, entry, error, _ = error_entry(monkeypatch, predict)
    assert error is retained
    assert len(attempted) == 1 and attempted[0][1] is target
    assert lookup_errors(capture)
    assert capture.error is None
    assert entry.get('failure') is None


def test_later_part_presentation_failure_is_not_the_refused_target(monkeypatch):
    from sdm_client import StructuredDataError
    from sdm_client._rust import run
    from sdm_client._rust.prepare import _call

    target = NoHandoverReads('successfully handed over')
    retained = UnformattedFailure('later part presentation failed')
    observed = []

    def of_the_part(error, raised):
        observed.append((error, raised))
        raise retained

    monkeypatch.setattr(run, '_of_the_part', of_the_part)

    def predict(*args):
        selected = None
        try:
            _call({'stage': 'model_lookup', 'position': None})
        except StructuredDataError as refused:
            selected = refused
        run._carried(selected, target)
        try:
            raise target
        except ValueError:
            run._of_the_part(selected, target)

    capture, entry, error, _ = error_entry(monkeypatch, predict)
    assert error is retained
    assert len(observed) == 1 and observed[0][1] is target
    assert BaseException.__getattribute__(error, '__context__') is target
    assert BaseException.__getattribute__(error, '__cause__') is None
    assert lookup_errors(capture)
    assert capture.error is None
    assert entry.get('failure') is None


def test_handover_without_a_captured_native_error_cannot_infer_core_failure(
    monkeypatch,
):
    from sdm_client import StructuredDataError
    from sdm_client._rust import run
    from sdm_client._rust.prepare import _call

    retained = NoHandoverReads('no selected native refusal')
    unassociated = StructuredDataError(
        'not reconstructed here', code='UNKNOWN_MODEL'
    )

    def predict(*args):
        assert 'kumo-relational' in _call({'stage': 'model_catalogue'})
        run._carried(unassociated, retained)
        raise retained

    capture, entry, error, _ = error_entry(monkeypatch, predict)
    assert error is retained
    assert all(record['error'] is None for record in capture.inputs)
    assert capture.error is None
    assert entry.get('failure') is None


def test_stale_handover_pair_does_not_cross_prediction_entries(monkeypatch):
    from sdm_client import StructuredDataError
    from sdm_client._rust import relational_predict, run
    from sdm_client._rust.prepare import _call

    retained = NoHandoverReads('same owner in two entries')
    selected = []

    def predict(client, request, model, outputs):
        if model == 'first':
            try:
                _call({'stage': 'model_lookup', 'position': None})
            except StructuredDataError as refused:
                selected.append(refused)
        run._carried(selected[0], retained)
        raise retained

    monkeypatch.setattr(relational_predict, 'predict', predict)
    boundary = RUNNER._Boundary()
    capture = boundary.reset()
    try:
        with pytest.raises(ValueError) as first:
            relational_predict.predict(None, None, 'first', None)
        assert first.value is retained
        first_failure = dict(entries(capture)[0]['failure'])
        assert first_failure['relation'] == 'refused'
        assert capture.error is not None
        with pytest.raises(ValueError) as second:
            relational_predict.predict(None, None, 'second', None)
        assert second.value is retained
        first_entry, second_entry = entries(capture)
        assert first_entry['failure'] == first_failure
        assert second_entry['runs'] == []
        assert second_entry.get('failure') is None
        assert capture.error is None
    finally:
        boundary.close()


@pytest.mark.parametrize(
    'fail_call', [0, 1], ids=['first-call', 'after-paid-step']
)
def test_forecast_endpoint_refusal_preserves_public_failure_and_paid_answers(
    monkeypatch, fail_call
):
    from sdm_client import NimRequestError

    plain = _ask(monkeypatch, (), capture=False, fail_call=fail_call)
    captured = _ask(monkeypatch, (), capture=True, fail_call=fail_call)
    public = []
    partials = []
    for observed in (plain, captured):
        error = observed['error']
        assert isinstance(error, NimRequestError)
        assert len(observed['handovers']) == 1
        reconstructed, replacement = observed['handovers'][0]
        assert error is not reconstructed
        owners = []
        current = error
        while current is not None and not any(
            current is owner for owner in owners
        ):
            owners.append(current)
            current = BaseException.__getattribute__(current, '__cause__')
        assert any(owner is replacement for owner in owners)
        details = dict(error.details or {})
        partial = details.pop('answered', None)
        partials.append(partial)
        public.append(
            {
                'type': type(error),
                'status': error.status_code,
                'code': error.code,
                'message': str(error),
                'details': details,
            }
        )
        assert observed['received'] == PLANNED_IDS[: fail_call + 1]
        assert [
            value['input']['request_id'] for value in observed['started']
        ] == observed['received']
    assert public[0] == public[1]
    if fail_call:
        assert public[0]['details']['parts_answered'] == 1
        assert partials[0] is not None and partials[1] is not None
        assert len(partials[0].items) == len(partials[1].items) == 2
        assert partials[0].model == partials[1].model
        assert partials[0].request_id == partials[1].request_id
        assert partials[0].outputs == partials[1].outputs
        assert partials[0].items == partials[1].items
        assert partials[0].metadata == partials[1].metadata
        assert partials[0].explanation == partials[1].explanation
        assert partials[0].to_pandas().equals(partials[1].to_pandas())
    else:
        assert partials == [None, None]
    capture = captured['capture']
    [entry] = entries(capture)
    assert entry['completion'] == 'raised'
    assert entry['failure']['relation'] == 'refused'
    failed = [
        record
        for record in capture.inputs
        if record['op'] == 'prepare_request'
        and record['input']['stage'] == 'call_relational'
        and record['error'] is not None
    ]
    assert len(failed) == 1
    assert entry['failure']['run'] == failed[0]['run']
    assert capture.error == failed[0]['error']
    assert [call['request_id'] for call in entry['calls']] == captured[
        'received'
    ]


@pytest.mark.parametrize(
    'operation', ['model', 'describe_source', 'materialize']
)
def test_failure_outside_prediction_retains_only_serialized_diagnostic(
    operation,
):
    import gc
    import weakref

    class Payload:
        pass

    class NativeFailure(Exception):
        kind = 'protocol'
        code = 'INVALID_MESSAGE'

    capture = RUNNER._Capture()
    owner = RUNNER._CapturedRun(0, operation, None)
    capture._runs[0] = owner
    capture._current_run = owner
    payload = Payload()
    error = NativeFailure()
    error.attachments = payload
    error_ref = weakref.ref(error)
    payload_ref = weakref.ref(payload)
    capture.failed(error, run=owner)
    assert capture.error == {'kind': 'protocol', 'code': 'INVALID_MESSAGE'}
    assert owner.error == capture.error
    del error, payload
    gc.collect()
    assert error_ref() is None
    assert payload_ref() is None
    assert owner.error_owner is None
    assert capture._runs[0] is owner


def test_prediction_failure_identity_is_released_only_at_outer_completion():
    import gc
    import weakref

    class NativeFailure(Exception):
        kind = 'protocol'
        code = 'INVALID_MESSAGE'

    capture = RUNNER._Capture()
    parent = capture.enter_prediction('kumo-relational')
    child = capture.enter_prediction('kumo-relational')
    owner = RUNNER._CapturedRun(0, 'prepare_request', child)
    capture._runs[0] = owner
    capture.predictions[child]['runs'].append(0)
    error = NativeFailure()
    ref = weakref.ref(error)
    capture.failed(error, run=owner)
    capture.leave_prediction(child, error)
    assert owner.error_owner is error
    capture.leave_prediction(parent, error)
    assert capture.predictions[parent]['failure'] == {
        'run': 0,
        'relation': 'selected',
    }
    assert owner.error_owner is None
    del error
    gc.collect()
    assert ref() is None


def test_actual_outside_entry_native_failure_does_not_keep_its_payload(
    monkeypatch,
):
    import gc
    import weakref

    from sdm_client import StructuredDataError
    from sdm_client._rust import errors
    from sdm_client._rust.prepare import _call

    class Payload:
        pass

    actual = errors.error_from_core
    observed = []

    def converter(*args, **kwargs):
        native = kwargs['internal_context']
        payload = Payload()
        native.lifetime_probe = payload
        observed.append(weakref.ref(payload))
        return actual(*args, **kwargs)

    monkeypatch.setattr(errors, 'error_from_core', converter)
    boundary = RUNNER._Boundary()
    capture = boundary.reset()
    try:
        with pytest.raises(StructuredDataError) as raised:
            _call({'stage': 'model_lookup', 'position': None})
        assert raised.value.code == 'UNKNOWN_MODEL'
        assert capture.predictions == []
        assert capture.error['code'] == 'UNKNOWN_MODEL'
        assert capture._runs[0].error_owner is None
        gc.collect()
        assert len(observed) == 1
        assert observed[0]() is None
    finally:
        boundary.close()
