# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A completed native result becomes carried only at its actual owner boundary."""

from __future__ import annotations

import gc
import importlib
import json
import warnings
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_t136_forecast_capture import RUNNER, _ask
from test_t44_prediction_capture import UnformattedFailure, Unobserved, entries

CLIENT_TESTS = (
    Path(__file__).parents[2] / 'packages/structured-data-models-client/tests'
)


def public_tests(monkeypatch):
    monkeypatch.syspath_prepend(str(CLIENT_TESTS))
    return importlib.import_module('test_t44_mutable_prediction')


@pytest.fixture
def journey(monkeypatch):
    """Real preparation/call/finish reads; an opaque presentation result owner."""
    import requests

    from sdm_client import KumoRelational, SDMClient
    from sdm_client._rust import relational_predict as rp

    module = public_tests(monkeypatch)
    monkeypatch.setenv('SDM_CORE_ROUTES', '')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = module._request()
    monkeypatch.setenv('SDM_CORE_ROUTES', module.FRAME_PREDICT_ROUTES)
    state = SimpleNamespace(
        child=None,
        finished=[],
        references=[],
        posts=[],
        depth=0,
        fake_present=False,
        present_error=None,
        return_list=False,
        tuple_subclass=False,
        rp=rp,
        raw_present=rp._present,
        raw_carry=rp._carry,
    )

    def present(finished, owners, **kwargs):
        details = {}
        if not state.fake_present:
            _, details = state.raw_present(finished, owners, **kwargs)
            state.finished.append(finished)
        result = Unobserved()
        state.references.append(weakref.ref(result))
        if state.present_error is not None:
            raise state.present_error
        if state.tuple_subclass:

            class Pair(tuple):
                pass

            return Pair((result, details))
        return [result, details] if state.return_list else (result, details)

    monkeypatch.setattr(rp, '_present', present)

    def post(url, *, data, **kwargs):
        body = json.loads(data)
        state.posts.append(body)
        table = body['predict']['instance_table']
        at = table['columns'].index('instance_id')
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                'id': 'carry-response',
                'model': 'kumo-relational',
                'metadata': {},
                'predictions': [
                    {'row_index': at_row, 'id': str(row[at]), 'prediction': 'a'}
                    for at_row, row in enumerate(table['rows'])
                ],
            }
        ).encode()
        response._content_consumed = True
        return response

    with SDMClient('http://localhost:1', max_retries=0) as client:
        monkeypatch.setattr(client._transport._session, 'post', post)

        def finish():
            if state.child is None:
                state.child = rp._one(
                    client, request, KumoRelational(), ['prediction']
                )
            value, attached = state.child.attached(2)
            value.pop('bound')
            return rp._finish(
                state.child.bound,
                {'kind': 'calls', **value},
                attached,
                rp._owners([state.child]),
                report_absent=False,
            )

        def invoke(action, *, keep_open=False, nested_boundaries=False):
            state.child = None
            state.finished = []

            def predict(*args, **kwargs):
                return action(state)

            monkeypatch.setattr(rp, 'predict', predict)
            boundaries = [RUNNER._Boundary()]
            if nested_boundaries:
                boundaries.append(RUNNER._Boundary())
            captures = [boundary.reset() for boundary in boundaries]
            error = None
            value = None
            try:
                try:
                    value = rp.predict(
                        client, request, KumoRelational(), ['prediction']
                    )
                except Exception as raised:  # noqa: BLE001 - exact terminal owner
                    error = raised
            finally:
                if not keep_open:
                    for boundary in reversed(boundaries):
                        boundary.close()
            return {
                'capture': captures[-1],
                'captures': captures,
                'boundaries': boundaries,
                'error': error,
                'value': value,
            }

        state.finish = finish
        state.invoke = invoke
        yield state


def carried_run(outcome, *, relation='selected', result_index=-1):
    capture = outcome['capture']
    [entry] = entries(capture)
    result = entry['results'][result_index]
    assert result['metadata'] is not None and result['result'] is not None
    assert result['input']['stage'] == 'finish_relational_response'
    assert entry['carried'] == {'run': result['run'], 'relation': relation}
    return result['run']


def test_successful_carry_binds_exact_presented_owner_without_public_reads(
    journey,
):
    retained = UnformattedFailure('selected carry')

    def action(state):
        result, details = state.finish()
        state.rp._carry(retained, result, details)
        raise retained

    outcome = journey.invoke(action)
    assert outcome['error'] is retained
    carried_run(outcome)
    assert len(journey.posts) == 1


@pytest.mark.parametrize('reentrant', [False, True])
def test_successful_carry_does_not_read_back_a_nonretaining_details_setter(
    journey, reentrant
):
    observations = []

    class Discarding(ValueError):
        @property
        def details(self):
            observations.append('get')
            return {'native': 'held'}

        @details.setter
        def details(self, value):
            observations.append('set')
            assert value['native'] == 'held'
            assert value['answered'] is expected[0]
            if reentrant:
                other, details = journey.finish()
                journey.rp._carry(ValueError('nested carry'), other, details)

    retained = Discarding('does not retain the assigned answer')
    expected = []

    def action(state):
        result, details = state.finish()
        expected.append(result)
        state.rp._carry(retained, result, details)
        raise retained

    outcome = journey.invoke(action)
    assert outcome['error'] is retained
    assert observations == ['get', 'set']
    carried_run(outcome, result_index=0)


@pytest.mark.parametrize(
    'case',
    [
        'different-result',
        'failed-present',
        'failed-carry',
        'non-tuple-return',
        'tuple-subclass-return',
    ],
)
def test_incomplete_or_unmatched_operations_do_not_infer_carry(
    journey, monkeypatch, case
):
    retained = UnformattedFailure('not successfully carried')
    if case == 'failed-present':
        journey.present_error = retained
    elif case == 'failed-carry':

        def carry(error, result, details):
            journey.raw_carry(error, result, details)
            raise error

        monkeypatch.setattr(journey.rp, '_carry', carry)
    elif case == 'non-tuple-return':
        journey.return_list = True
    elif case == 'tuple-subclass-return':
        journey.tuple_subclass = True

    def action(state):
        result, details = state.finish()
        if case == 'different-result':
            result = Unobserved()
        state.rp._carry(retained, result, details)
        raise retained

    outcome = journey.invoke(action)
    assert outcome['error'] is retained
    [entry] = entries(outcome['capture'])
    assert entry['results']
    assert entry['carried'] is None


@pytest.mark.parametrize(
    'foreign',
    [
        'finished-subclass',
        'wrong-finished-control',
        'handle-subclass',
        'unrelated-wrapper',
        'foreign-capture',
        'known-wrapper-cycle',
    ],
)
def test_present_provenance_stops_at_unknown_or_foreign_owners(
    journey, foreign
):
    from sdm_client._rust.run import Finished

    retained = UnformattedFailure('foreign result')

    class FinishedSubclass(Finished):
        pass

    class Unrelated:
        def __getattribute__(self, name):
            raise AssertionError('observer traversed an unrelated wrapper')

    class HandleSubclass(RUNNER._Handle):
        pass

    def action(state):
        state.finish()
        original = state.finished[-1]
        state.fake_present = True
        if foreign == 'finished-subclass':
            finished = FinishedSubclass(original.result, original._run)
        elif foreign == 'wrong-finished-control':
            finished = Finished({}, original._run)
        elif foreign == 'handle-subclass':
            handle = original._run
            wrapper = HandleSubclass(
                handle, handle._native, handle._capture, handle._owner
            )
            finished = Finished({}, wrapper)
        elif foreign == 'unrelated-wrapper':
            finished = Finished({}, Unrelated())
        else:
            other = RUNNER._Capture()
            owner = RUNNER._CapturedRun(
                original._run._owner.ordinal, 'prepare_request', 0
            )
            handle = RUNNER._Handle(object(), object(), other, owner)
            if foreign == 'known-wrapper-cycle':
                handle._handle = handle
            finished = Finished({}, handle)
        result, details = state.rp._present(finished, {})
        state.rp._carry(retained, result, details)
        raise retained

    outcome = journey.invoke(action)
    assert outcome['error'] is retained
    [entry] = entries(outcome['capture'])
    assert entry['results']
    assert entry['carried'] is None


def test_latest_matching_carry_within_same_error_wins(journey):
    retained = UnformattedFailure('two successful carries')

    def action(state):
        first, first_details = state.finish()
        state.rp._carry(retained, first, first_details)
        second, second_details = state.finish()
        state.rp._carry(retained, second, second_details)
        raise retained

    outcome = journey.invoke(action)
    assert outcome['error'] is retained
    carried_run(outcome, result_index=1)


@pytest.mark.parametrize(
    'exact_terminal', [False, True], ids=['nearest-cause', 'terminal-first']
)
def test_terminal_then_nearest_cause_precedes_global_carry_chronology(
    journey, exact_terminal
):
    terminal = UnformattedFailure('outer')
    near = UnformattedFailure('near cause')
    far = UnformattedFailure('far cause')

    def action(state):
        first, first_details = state.finish()
        state.rp._carry(
            terminal if exact_terminal else near, first, first_details
        )
        second, second_details = state.finish()
        state.rp._carry(near if exact_terminal else far, second, second_details)
        near.__cause__ = far
        raise terminal from near

    outcome = journey.invoke(action)
    assert outcome['error'] is terminal
    carried_run(
        outcome,
        result_index=0,
        relation='selected' if exact_terminal else 'cause',
    )


def test_context_only_carry_target_does_not_select_an_answer(journey):
    carried = UnformattedFailure('incidental context')
    terminal = UnformattedFailure('new host failure')

    def action(state):
        result, details = state.finish()
        state.rp._carry(carried, result, details)
        try:
            raise carried
        except ValueError:
            raise terminal

    outcome = journey.invoke(action)
    assert outcome['error'] is terminal
    assert BaseException.__getattribute__(terminal, '__context__') is carried
    [entry] = entries(outcome['capture'])
    assert entry['carried'] is None


def test_stale_presented_owner_cannot_cross_capture_entries(journey):
    first = journey.invoke(lambda state: state.finish()[0])
    stale = first['value']
    retained = UnformattedFailure('stale result owner')

    def action(state):
        _, details = state.finish()
        state.rp._carry(retained, stale, details)
        raise retained

    second = journey.invoke(action)
    assert second['error'] is retained
    [entry] = entries(second['capture'])
    assert entry['results']
    assert entry['carried'] is None


def test_known_nested_capture_handles_bind_each_own_complete_result(journey):
    retained = UnformattedFailure('nested boundaries')

    def action(state):
        result, details = state.finish()
        state.rp._carry(retained, result, details)
        raise retained

    outcome = journey.invoke(action, nested_boundaries=True)
    assert outcome['error'] is retained
    assert len(outcome['captures']) == 2
    for capture in outcome['captures']:
        carried_run({**outcome, 'capture': capture})


def test_nested_exact_rethrow_keeps_child_result_run(journey):
    retained = UnformattedFailure('child carry')

    def action(state):
        if state.depth == 0:
            state.depth = 1
            return state.rp.predict(None, None, 'child', None)
        result, details = state.finish()
        state.rp._carry(retained, result, details)
        raise retained

    outcome = journey.invoke(action)
    assert outcome['error'] is retained
    outer, child = entries(outcome['capture'])
    assert child['parent'] == outer['ordinal']
    result = child['results'][-1]
    assert result['run'] not in outer['runs']
    assert (
        outer['carried']
        == child['carried']
        == {'run': result['run'], 'relation': 'selected'}
    )


@pytest.mark.parametrize('raised', [False, True], ids=['normal', 'exceptional'])
def test_public_result_and_error_owners_are_released_at_outer_completion(
    journey, raised
):
    references = []

    def action(state):
        result, details = state.finish()
        error = UnformattedFailure('collectable carry owner')
        references.extend([weakref.ref(result), weakref.ref(error)])
        state.rp._carry(error, result, details)
        if raised:
            raise error
        return result

    outcome = journey.invoke(action, keep_open=True)
    capture = outcome['capture']
    boundaries = outcome['boundaries']
    try:
        del outcome
        gc.collect()
        assert all(reference() is None for reference in references)
        assert entries(capture)
    finally:
        for boundary in reversed(boundaries):
            boundary.close()


@pytest.mark.parametrize(
    'forecast',
    [False, True],
    ids=['two-paid-shape-hash', 'late-forecast-explanation'],
)
def test_unchanged_public_late_failure_retains_internal_results_without_carry(
    monkeypatch, forecast
):
    module = public_tests(monkeypatch)
    module._core_preparation_owner_guard.__wrapped__(
        monkeypatch,
        SimpleNamespace(
            node=SimpleNamespace(
                callspec=SimpleNamespace(params={'core': True})
            )
        ),
    )
    boundary = RUNNER._Boundary()
    capture = boundary.reset()
    try:
        if forecast:
            module.test_native_decomposed_output_owners_and_paid_boundaries(
                monkeypatch,
                core=True,
                case='forecast-explanation-contains-fails',
            )
        else:
            module.test_native_two_paid_call_output_shape_hash_boundary(
                monkeypatch,
                core=True,
                case='shape-hash-raises',
            )
    finally:
        boundary.close()
    [entry] = entries(capture)
    assert entry['completion'] == 'raised'
    assert len(entry['calls']) == 2
    assert all(call['body'] is not None for call in entry['calls'])
    assert entry['results']
    assert all(
        result['metadata'] is not None and result['result'] is not None
        for result in entry['results']
    )
    assert entry['carried'] is None


def test_true_paid_forecast_carry_names_actual_partial_steps_run(monkeypatch):
    observed = _ask(monkeypatch, (), capture=True, fail_call=1)
    assert observed['error'] is not None
    capture = observed['capture']
    [entry] = entries(capture)
    assert entry['carried']['relation'] == 'cause'
    [result] = [
        result
        for result in entry['results']
        if result['run'] == entry['carried']['run']
    ]
    assert result['input']['input']['aggregation']['kind'] == 'steps'
    assert result['metadata'] is not None and result['result'] is not None
    assert result['data']['answer_details']['parts_answered'] == 1
    assert len(result['data']['lineage']) == 2
    assert len(observed['received']) == 2
    assert len(observed['error'].details['answered'].items) == 2
