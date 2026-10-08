# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumed host containers keep their original protocols and ownership."""

import copy
import warnings
from collections import OrderedDict, UserDict

import pytest

from tests.rust_core import test_request_beyond_the_corpus as helpers
from tests.rust_core import test_request_held_values as held_tests


@pytest.fixture(scope='module', name='baseline')
def consumed_baseline():
    return held_tests.baseline.__wrapped__()


class PlainDict(dict):
    pass


class PlainList(list):
    pass


class PlainTuple(tuple):
    pass


MAPPINGS = [dict, PlainDict, OrderedDict, UserDict]
CONTAINERS = [*MAPPINGS, list, PlainList, tuple, PlainTuple]
PLACEMENTS = [
    (field, family)
    for field in (
        'envelope',
        'task',
        'target',
        'options',
        'inference',
        'requested_inference',
    )
    for family in (
        MAPPINGS if field in ('envelope', 'task', 'target') else CONTAINERS
    )
]


def construct(original, host, core, field, factory, *, infer_entity=False):
    from sdm_client import Request

    payloads = copy.deepcopy(list(original.payloads))
    kwargs = {}
    if field == 'envelope':
        supplied = factory(payloads[0])
        payloads[0] = supplied
    elif field in ('task', 'target'):
        parent = payloads[0] if field == 'task' else payloads[0]['task']
        supplied = factory(parent[field])
        parent[field] = supplied
    else:
        supplied = factory([('extra', 1)])
        if field == 'requested_inference':
            kwargs[field] = supplied
        else:
            parent = payloads[0]['task'] if field == 'options' else payloads[0]
            parent[field] = supplied
    if not infer_entity:
        kwargs['entity_table'] = original.entity_table
    request = (host.request_materialized if core else Request.materialized)(
        payloads=payloads,
        query=original.query,
        indices=list(original.indices),
        windowed=False,
        schema=host.schema_from_data(helpers.shop())
        if core
        else original.schema,
        **kwargs,
    )
    held = request.state['payloads'] if core else request.payloads
    if field == 'envelope':
        assert held[0] is supplied
    elif field in ('task', 'target', 'options', 'inference'):
        parent = held[0]['task'] if field in ('target', 'options') else held[0]
        assert parent[field] is supplied
    return request


def describe(request, core):
    if core:
        state = request.state
        task = state['task']
        return (
            task['kind'],
            task['target']['column_name'],
            task['target']['dtype'],
            task['options'],
            state['inference'],
            state['requested_inference'],
            state['entity_table'],
        )
    return (
        request.task.kind,
        request.task.target.column_name,
        request.task.target.dtype,
        request.task.options,
        request.inference,
        request.requested_inference,
        request.entity_table,
    )


@pytest.mark.parametrize('family', MAPPINGS, ids=lambda family: family.__name__)
def test_fitted_session_scoring_envelope(family):
    host = helpers._host()
    fit_original = helpers._today(helpers._as_given, helpers.GATHER)
    score_original = helpers._today(
        helpers._as_given, {**helpers.GATHER, 'indices': [3, 4]}
    )
    baseline_session = helpers._Watched()
    baseline_fit = helpers._client(baseline_session).fit(
        construct(fit_original, host, False, 'envelope', dict),
        model=helpers.REL,
    )
    helpers._outcome(
        lambda: baseline_fit.predict(
            construct(score_original, host, False, 'envelope', dict)
        )
    )
    response = {
        'fail': [
            {
                'at': 1,
                'status': 200,
                'body': copy.deepcopy(baseline_session.responses[-1]['body']),
            }
        ]
    }
    outcomes = []
    for core in (False, True):
        session = helpers._Watched(copy.deepcopy(response))

        def run(core=core, session=session):
            client = helpers._client(session)
            if core:
                client = host.Client(client._transport)
            fitted = client.fit(
                construct(fit_original, host, core, 'envelope', dict),
                model=helpers.REL,
            )
            answer = fitted.predict(
                construct(score_original, host, core, 'envelope', family)
            )
            frame = (
                host.frame(host.flatten(answer)) if core else answer.to_pandas()
            )
            return frame.to_json(orient='split')

        outcome, warnings = helpers._outcome(run)
        assert outcome[0] == 'value', outcome
        assert session.failed is None
        assert (
            sum(entry['method'] == 'POST' for entry in session.responses) == 2
        )
        outcomes.append((outcome, warnings, helpers._sent(session)))
    assert outcomes[0] == outcomes[1]
    overlap = [
        warning
        for warning in outcomes[0][1]
        if warning[1].startswith('2 of the 2 rows being scored')
    ]
    assert len(overlap) == (0 if family is UserDict else 1)


@pytest.mark.parametrize('family', MAPPINGS, ids=lambda family: family.__name__)
def test_session_warning_comparison_rejects_one_sided_changes(
    monkeypatch, family
):
    host = helpers._host()
    original = host._give

    def changed(given):
        if family is UserDict:
            warnings.warn(
                'one-sided session warning', UserWarning, stacklevel=2
            )
        original(
            [
                warning
                for warning in given or ()
                if not warning['message'].startswith(
                    '2 of the 2 rows being scored'
                )
            ]
        )

    monkeypatch.setattr(host, '_give', changed)
    with pytest.raises(AssertionError):
        test_fitted_session_scoring_envelope(family)


def compare(baseline, field, factory, *, infer_entity=False, events=None):
    original, response = baseline
    host = helpers._host()
    outcomes = []
    traces = []
    for core in (False, True):
        if events is not None:
            events.clear()
        requests = []

        def make(core=core, requests=requests):
            request = construct(
                original, host, core, field, factory, infer_entity=infer_entity
            )
            requests.append(request)
            return describe(request, core)

        construction = helpers._outcome(make)
        session = helpers._Watched(copy.deepcopy(response))
        prediction = None
        if requests:
            client = helpers._client(session)

            def predict(core=core, client=client, request=requests[0]):
                result = (
                    host.frame(
                        host.flatten(
                            host.Client(client._transport).predict(
                                request, model=helpers.REL
                            )
                        )
                    )
                    if core
                    else client.predict(request, model=helpers.REL).to_pandas()
                )
                return result.to_json(orient='split')

            prediction = helpers._outcome(predict)
        outcomes.append(
            (
                construction,
                prediction,
                helpers._sent(session),
                [entry['method'] for entry in session.responses],
            )
        )
        if events is not None:
            traces.append(list(events))
        assert session.failed is None
    assert outcomes[1] == outcomes[0]
    if events is not None:
        assert traces[1] == traces[0]


@pytest.mark.parametrize(('field', 'family'), PLACEMENTS)
def test_consumed_container_inventory(baseline, field, family):
    compare(baseline, field, family)


@pytest.mark.parametrize('field', ['envelope', 'task', 'target'])
@pytest.mark.parametrize('family', MAPPINGS)
def test_consumed_container_entity_default(baseline, field, family):
    compare(baseline, field, family, infer_entity=True)


@pytest.mark.parametrize('family', [PlainList, PlainTuple])
def test_consumed_entity_name_sequence(baseline, family):
    def wrapped(task):
        return PlainDict(
            task, entity_table_names=family(task['entity_table_names'])
        )

    compare(baseline, 'task', wrapped, infer_entity=True)


@pytest.mark.parametrize('field', ['envelope', 'task', 'target'])
@pytest.mark.parametrize('fail_at', [None, 1, 2, 3])
def test_consumed_get_order_and_short_circuit(baseline, field, fail_at):
    events = []

    class Watched(dict):
        def get(self, key, *defaults):
            events.append((key, len(defaults)))
            if len(events) == fail_at:
                raise ValueError('selected get failed')
            return super().get(key, *defaults)

    compare(baseline, field, Watched, events=events)


@pytest.mark.parametrize(
    'field', ['options', 'inference', 'requested_inference']
)
@pytest.mark.parametrize(
    'mode', ['mapping', 'iterable', 'bad-pair', 'iter-error', 'false']
)
def test_consumed_dict_protocol_order(baseline, field, mode):
    events = []

    class Mapping:
        def keys(self):
            events.append('keys')
            return ['extra']

        def __getitem__(self, key):
            events.append(('getitem', key))
            return 1

        def __bool__(self):
            events.append('bool')
            return mode != 'false'

    class Iterable:
        def __iter__(self):
            events.append('iter')
            if mode == 'iter-error':
                raise ValueError('selected iteration failed')
            yield ['extra', 1]
            if mode == 'bad-pair':
                yield ['broken']

        def __bool__(self):
            events.append('bool')
            return True

    compare(
        baseline,
        field,
        lambda unused: (
            Mapping() if mode in ('mapping', 'false') else Iterable()
        ),
        events=events,
    )


@pytest.mark.parametrize('operation', ['get', 'dict', 'instance', 'getitem'])
def test_consumed_exception_identity_and_reentry(baseline, operation):
    original, unused = baseline
    host = helpers._host()
    failure = ValueError('original consumed protocol exception')
    calls = []

    def fail():
        calls.append(operation)
        assert host._native.query_text('nested') == 'nested'
        raise failure

    class Get(dict):
        def get(self, *args):
            return fail()

    class Pairs(list):
        def __iter__(self):
            return fail()

    class Names(list):
        def __getitem__(self, key):
            return fail()

    class Instance:
        def keys(self):
            return ['extra']

        def __getitem__(self, key):
            return 1

        @property
        def __class__(self):
            return fail()

    with pytest.raises(ValueError) as raised:
        request = construct(
            original,
            host,
            True,
            'task' if operation in ('get', 'getitem') else 'inference',
            Get
            if operation == 'get'
            else Pairs
            if operation == 'dict'
            else lambda contents: (
                PlainDict(
                    contents,
                    entity_table_names=Names(contents['entity_table_names']),
                )
                if operation == 'getitem'
                else Instance()
            ),
            infer_entity=operation == 'getitem',
        )
        session = helpers._Watched()
        host.Client(helpers._client(session)._transport).predict(
            request, model=helpers.REL
        )
        assert session.responses == []
    assert raised.value is failure
    assert calls == [operation]


@pytest.mark.parametrize(
    'field', ['options', 'inference', 'requested_inference']
)
@pytest.mark.parametrize('family', [PlainDict, UserDict, PlainList, PlainTuple])
def test_consumed_copy_retains_shallow_aliases(baseline, field, family):
    original, unused = baseline
    host = helpers._host()
    leaf = []
    source = family([('first', leaf), ('second', leaf)])
    request = construct(original, host, True, field, lambda unused: source)
    mapping = (
        request.state['task']['options']
        if field == 'options'
        else request.state[field]
    )
    assert type(mapping) is dict
    assert mapping is not source
    assert mapping['first'] is mapping['second'] is leaf


@pytest.mark.parametrize(
    'field', ['options', 'inference', 'requested_inference']
)
@pytest.mark.parametrize(
    'mode', ['collision', 'hash-error', 'equality-error', 'later-bad-pair']
)
def test_consumed_pair_hash_equality_counts(baseline, field, mode):
    events = []

    class Key:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            events.append(('hash', self.value))
            if mode == 'hash-error':
                raise ValueError('hash failure')
            return 7

        def __eq__(self, other):
            events.append(('equal', self.value, other.value))
            if mode == 'equality-error':
                raise ValueError('equality failure')
            return self.value == other.value

    def factory(unused):
        pairs = PlainList([[Key(1), 1], [Key(2), 2], [Key(1), 3]])
        if mode == 'later-bad-pair':
            pairs.append(['broken'])
        return pairs

    original, unused = baseline
    host = helpers._host()
    outcomes = []
    traces = []
    for core in (False, True):
        events.clear()

        def make(core=core):
            request = construct(original, host, core, field, factory)
            mapping = describe(request, core)[
                3 if field == 'options' else 4 if field == 'inference' else 5
            ]
            return [(key.value, value) for key, value in mapping.items()]

        outcomes.append(helpers._outcome(make))
        traces.append(list(events))
    assert outcomes[1] == outcomes[0]
    assert traces[1] == traces[0]
