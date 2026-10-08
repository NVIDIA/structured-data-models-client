# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original request values survive construction until transport reads them."""

import copy
import datetime as dt
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from tests.rust_core import test_request_beyond_the_corpus as helpers


class Opaque:
    def __repr__(self):
        raise AssertionError('an opaque result value must not be serialized')


VALUES = {
    'none': lambda: None,
    'bool': lambda: False,
    'int': lambda: 2**70,
    'float': lambda: 1.25,
    'complex': lambda: 1 + 2j,
    'str': lambda: 'scalar',
    'bytes': lambda: b'a\x00b',
    'bytearray': lambda: bytearray(b'a\x00b'),
    'memoryview': lambda: memoryview(b'a\x00b'),
    'decimal': lambda: Decimal('1.234567890123456789'),
    'fraction': lambda: Fraction(1, 3),
    'date': lambda: dt.date(2026, 1, 2),
    'timestamp': lambda: pd.Timestamp('2026-01-02T03:04:05.123456789'),
    'timedelta': lambda: pd.Timedelta(123456789, unit='ns'),
    'period': lambda: pd.Period('2026-01', freq='M'),
    'interval': lambda: pd.Interval(1, 2),
    'na': lambda: pd.NA,
    'nat': lambda: pd.NaT,
    'numpy-bool': lambda: np.bool_(False),
    'numpy-int': lambda: np.int8(-3),
    'numpy-uint': lambda: np.uint64(2**64 - 1),
    'numpy-float': lambda: np.float32(1.25),
    'longdouble': lambda: np.longdouble('1.234567890123456789'),
    'complex64': lambda: np.complex64(1 + 2j),
    'complex128': lambda: np.complex128(1 + 2j),
    'clongdouble': lambda: np.clongdouble(
        '1.234567890123456789+2.34567890123456789j'
    ),
    'numpy-str': lambda: np.str_('scalar'),
    'numpy-bytes': lambda: np.bytes_(b'a\x00b'),
    'numpy-void': lambda: np.void(b'a\x00b'),
    'structured-void': lambda: np.array(
        [(3, 1.25)], dtype=[('count', 'i4'), ('score', 'f8')]
    )[0],
    'numpy-datetime': lambda: np.datetime64('2026-01-02T03:04:05.123456789'),
    'numpy-timedelta': lambda: np.timedelta64(123456789, 'ns'),
    'opaque': Opaque,
    'set': lambda: {1},
    'frozenset': lambda: frozenset({1}),
}


@pytest.fixture(scope='module')
def baseline():
    original = helpers._today(helpers._as_given, helpers.GATHER)
    session = helpers._Watched()
    helpers._client(session).predict(original, model=helpers.REL)
    return original, {
        'fail': [
            {'at': 0, 'status': 200, 'body': session.responses[-1]['body']}
        ]
    }


def compare(baseline, field, factory, how='predict'):
    from sdm_client import Request

    original, response = baseline
    host = helpers._host()
    sessions = []
    outcomes = []
    for core in (False, True):
        value = factory()
        payloads = copy.deepcopy(list(original.payloads))
        parent = (
            payloads[0]
            if field == 'extra'
            else payloads[0]['task'].setdefault('options', {})
            if field == 'options'
            else payloads[0]['inference']['inference_config']
            if field == 'inference_config'
            else payloads[0]['inference']
        )
        key = 'num_estimators' if field == 'inference_config' else 'extra'
        parent[key] = value
        arguments = dict(
            payloads=payloads,
            query=original.query,
            indices=list(original.indices),
            entity_table=original.entity_table,
            windowed=False,
        )
        constructed, said = helpers._outcome(
            lambda core=core, arguments=arguments: (
                host.request_materialized(
                    schema=host.schema_from_data(helpers.shop()), **arguments
                )
                if core
                else Request.materialized(schema=original.schema, **arguments)
            )
        )
        assert constructed[0] == 'value', constructed
        assert said == []
        request = constructed[1]
        held = request.state['payloads'][0] if core else request.payloads[0]
        held = (
            held
            if field == 'extra'
            else held['task']['options']
            if field == 'options'
            else held['inference']['inference_config']
            if field == 'inference_config'
            else held['inference']
        )
        assert held[key] is value
        if field in ('options', 'inference'):
            coerced = (
                request.state['task']['options']
                if core and field == 'options'
                else request.state['inference']
                if core
                else request.task.options
                if field == 'options'
                else request.inference
            )
            assert type(coerced[key]) is type(value)
            if type(value) not in (
                type(None),
                bool,
                int,
                float,
                str,
                list,
                tuple,
            ):
                assert coerced[key] is value
        session = helpers._Watched(
            copy.deepcopy(response) if how == 'predict' else None
        )
        sessions.append(session)
        client = helpers._client(session)
        client = host.Client(client._transport) if core else client

        def ask(client=client, request=request, core=core):
            answer = getattr(client, how)(request, model=helpers.REL)
            if how == 'fit':
                answer.close()
                return 'fitted'
            frame = (
                host.frame(host.flatten(answer)) if core else answer.to_pandas()
            )
            return frame.to_json(orient='split')

        outcomes.append(helpers._outcome(ask))
    assert outcomes[1] == outcomes[0]
    assert helpers._sent(sessions[1]) == helpers._sent(sessions[0])
    assert [entry['method'] for entry in sessions[1].responses] == [
        entry['method'] for entry in sessions[0].responses
    ]
    assert all(session.failed is None for session in sessions)


@pytest.mark.parametrize('family', VALUES)
@pytest.mark.parametrize('field', ['extra', 'options', 'inference'])
@pytest.mark.parametrize('how', ['predict', 'fit'])
def test_held_scalar_families(baseline, family, field, how):
    compare(baseline, field, VALUES[family], how)


@pytest.mark.parametrize(
    'factory',
    [
        lambda: [1],
        lambda: (1,),
        lambda: {1},
        lambda: frozenset({1}),
        lambda: b'a',
        lambda: bytearray(b'a'),
    ],
)
@pytest.mark.parametrize(
    'field', ['extra', 'options', 'inference', 'inference_config']
)
def test_held_container_families(baseline, factory, field):
    compare(baseline, field, factory)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('output_statistic', set()),
        ('anchor_time_column', set()),
        ('kind', set()),
        ('kind', np.str_('regression')),
        ('kind', np.str_('unknown')),
        ('context_cell', bytearray(b'a')),
        ('context_cell', np.longdouble('1.25')),
        ('run_mode', np.str_('fast')),
    ],
)
def test_held_semantic_consumers(field, value):
    helpers.test_materialized_constructor_semantic_boundary(field, value)


@pytest.mark.parametrize(
    'family',
    [
        'distinct',
        'equal',
        'nan',
        'missing',
        'collision',
        'hash-error',
        'equality-error',
    ],
)
def test_held_context_cardinality(family):
    events = []

    class Cell:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            events.append(('hash', self.value))
            if family == 'hash-error':
                raise TypeError('held hash failure')
            return 7

        def __eq__(self, other):
            events.append(('equal', self.value, other.value))
            if family == 'equality-error':
                raise ValueError('held equality failure')
            return self.value == other.value

    def edit(payloads):
        payload = payloads[0]
        table = payload['context']['instance_table']
        position = table['columns'].index(
            payload['task']['target']['column_name']
        )
        for index, row in enumerate(table['rows']):
            row[position] = (
                Cell(index % 2)
                if family in ('collision', 'hash-error', 'equality-error')
                else pd.NA
                if family == 'missing'
                else np.longdouble('nan')
                if family == 'nan'
                else np.longdouble('1.234567890123456789')
                + np.longdouble(index % 2 if family == 'distinct' else 0)
            )
        return payloads

    outcomes = []
    traces = []
    for core in (False, True):
        request = (helpers._core if core else helpers._today)(
            edit, helpers.GATHER
        )
        session = helpers._Watched()
        client = helpers._client(session)
        if core:
            client = helpers._host().Client(client._transport)
        events.clear()
        outcomes.append(
            helpers._outcome(
                lambda client=client, request=request: client.predict(
                    request, model=helpers.REL
                )
            )
        )
        traces.append(list(events))
        assert session.responses == []
        assert session.failed is None
    assert outcomes[1] == outcomes[0]
    assert traces[1] == traces[0]


@pytest.mark.parametrize(
    'literal',
    [
        {'$host_ref': '0'},
        {'$held': {'reference': {'$host_ref': '0'}, 'value': 1}},
        {
            '$held': {
                'reference': {'$host_ref': '0'},
                'value': {'$host_ref': '1'},
                'owner': 'literal',
            }
        },
        {'nested': [{'$host_ref': '0'}, {'$held': {'value': 1}}]},
    ],
)
def test_held_literal_markers_are_user_values(baseline, literal):
    compare(baseline, 'extra', lambda: literal)


def test_held_owner_aliases_scope_and_lifetime():
    import gc
    import weakref

    from sdm_core.answers import HeldValues, decode, encode

    value = Opaque()
    weak = weakref.ref(value)
    owner = HeldValues()
    encoded = encode([value, value], references=owner)
    assert len(owner) == 1
    decoded = decode(encoded, owner)
    assert decoded[0] is decoded[1] is value
    other_owner = HeldValues()
    encode(Opaque(), references=other_owner)
    with pytest.raises(ValueError, match='outside its operation owner'):
        decode(encoded, other_owner)
    with pytest.raises(ValueError, match='require their operation owner'):
        decode(encoded)
    corrupt = copy.deepcopy(encoded)
    corrupt[0]['$held']['reference']['$host_ref'] = '99'
    with pytest.raises(ValueError, match='outside its operation owner'):
        decode(corrupt, owner)
    del value, decoded
    gc.collect()
    assert weak() is not None
    del owner
    gc.collect()
    assert weak() is None


def test_held_unknown_leaf_does_not_run_user_protocols(baseline):
    class Unreadable:
        def __getattribute__(self, name):
            raise AssertionError('passthrough attribute read')

        def __bool__(self):
            raise AssertionError('passthrough truth read')

        def __repr__(self):
            raise AssertionError('passthrough repr read')

    original, unused = baseline
    payloads = copy.deepcopy(list(original.payloads))
    value = Unreadable()
    payloads[0]['extra'] = value
    host = helpers._host()
    rebuilt = host.request_materialized(
        payloads=payloads,
        schema=host.schema_from_data(helpers.shop()),
        entity_table=original.entity_table,
    )
    assert rebuilt.state['payloads'][0]['extra'] is value


@pytest.mark.parametrize('operation', ['truth', 'hashable', 'repr'])
def test_held_protocol_exception_identity_and_reentry(baseline, operation):
    from sdm_core.answers import HeldValues, decode, encode

    failure = ValueError('original held protocol failure')
    calls = []

    class Failing:
        def invoke(self):
            calls.append(operation)
            nested = HeldValues()
            assert helpers._host()._native.query_text('nested') == 'nested'
            assert (
                decode(encode(VALUES['numpy-str'](), references=nested), nested)
                == 'scalar'
            )
            raise failure

        def __bool__(self):
            return self.invoke()

        def __hash__(self):
            return self.invoke()

        def __repr__(self):
            return self.invoke()

    host = helpers._host()
    original, unused = baseline
    payloads = copy.deepcopy(list(original.payloads))
    value = Failing()
    if operation == 'truth':
        payloads[0]['output'] = {'prediction_statistic': value}
    else:
        payloads[0]['task']['kind'] = value
        if operation == 'repr':
            Failing.__hash__ = lambda self: 3
    request = host.request_materialized(
        payloads=payloads,
        schema=host.schema_from_data(helpers.shop()),
        entity_table=original.entity_table,
    )
    assert calls == []
    session = helpers._Watched()
    with pytest.raises(ValueError) as raised:
        host.Client(helpers._client(session)._transport).predict(
            request, model=helpers.REL
        )
    assert raised.value is failure
    assert calls == [operation]
    assert session.responses == []


def test_held_values_reused_by_predict_fit_parts_and_session(baseline):
    import gc
    import weakref

    class Text(str):
        pass

    original, response = baseline
    host = helpers._host()
    value = Text('original object')
    weak = weakref.ref(value)
    payloads = copy.deepcopy(list(original.payloads))
    payloads[0]['extra'] = value
    payloads[0]['task'].setdefault('options', {})['extra'] = value
    payloads[0]['inference']['extra'] = value
    request = host.request_materialized(
        payloads=payloads,
        schema=host.schema_from_data(helpers.shop()),
        query=original.query,
        indices=original.indices,
        entity_table=original.entity_table,
    )
    session = helpers._Watched(copy.deepcopy(response))
    transport = helpers._client(session)._transport
    captured = []
    primitive = transport.predict

    def predict(body):
        assert body['extra'] is weak()
        assert body['task']['options']['extra'] is weak()
        captured.append('predict')
        return primitive(body)

    transport.predict = predict
    client = host.Client(transport)
    client.predict(request, model=helpers.REL)
    client.predict(request, model=helpers.REL)
    assert captured == ['predict', 'predict']
    assert request.state['task']['options']['extra'] is value
    part = dict(request.state)
    request.state['parts'] = [part]
    request.state['decomposition'] = {
        'kind': 'forecasting',
        'parts': [[original.query, 1]],
        'top': None,
    }
    client.predict(request, model=helpers.REL)
    from sdm_core.answers import HeldValues, decode, encode

    owner = HeldValues()
    encoded = host._request_mappings(
        request.state, lambda item, owner=owner: encode(item, references=owner)
    )
    restored = host._request_mappings(
        encoded, lambda item, owner=owner: decode(item, owner)
    )
    assert restored['parts'][0]['payloads'][0]['extra'] is value
    request.state['parts'] = None
    request.state['decomposition'] = None

    fit_session = helpers._Watched()
    fit_transport = helpers._client(fit_session)._transport
    create = fit_transport.create_session
    session_predict = fit_transport.session_predict

    def create_session(body):
        assert body['task']['options']['extra'] is weak()
        return create(body)

    def score(session_id, body):
        assert body['inference']['extra'] is weak()
        return session_predict(session_id, body)

    fit_transport.create_session = create_session
    fit_transport.session_predict = score
    fitted_client = host.Client(fit_transport)
    fitted = fitted_client.fit(request, model=helpers.REL)
    fitted.predict(request)
    fitted.predict(request)
    assert all(entry is not None for entry in fitted.references.values())
    assert any(entry is value for entry in fitted.references.values())
    del value, payloads, request, part, restored, owner
    gc.collect()
    assert weak() is not None
    fitted.close()
    assert fitted.references == {}
    gc.collect()
    assert weak() is None
    assert session.failed is None
    assert fit_session.failed is None


@pytest.mark.parametrize(
    'field', ['options', 'inference', 'requested_inference']
)
@pytest.mark.parametrize(
    'family',
    [
        'distinct',
        'mixed',
        'collision',
        'hash-error',
        'equality-error',
        'later-invalid',
    ],
)
def test_held_constructor_dictionary_assignments(baseline, field, family):
    from sdm_client import Request

    events = []

    class Key:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            events.append(('hash', self.value))
            if family == 'hash-error':
                raise TypeError('constructor hash failure')
            return 7

        def __eq__(self, other):
            events.append(('equal', self.value, getattr(other, 'value', other)))
            if family == 'equality-error':
                raise ValueError('constructor equality failure')
            return isinstance(other, Key) and self.value == other.value

    original, unused = baseline
    host = helpers._host()
    outcomes = []
    traces = []
    for core in (False, True):
        keys = (
            [np.longdouble('1.25'), np.longdouble('2.25')]
            if family == 'distinct'
            else [1, np.int64(1), np.int64(2), 2]
            if family == 'mixed'
            else [Key(1), Key(2), Key(1)]
        )
        pairs = [[key, index] for index, key in enumerate(keys)]
        if family == 'later-invalid':
            pairs.append(['invalid'])
        payloads = copy.deepcopy(list(original.payloads))
        kwargs = {}
        if field == 'requested_inference':
            kwargs[field] = pairs
        elif field == 'options':
            payloads[0]['task']['options'] = pairs
        else:
            payloads[0][field] = pairs
        events.clear()
        outcome, warnings = helpers._outcome(
            lambda core=core, payloads=payloads, kwargs=kwargs: (
                host.request_materialized if core else Request.materialized
            )(
                payloads=payloads,
                query=original.query,
                indices=original.indices,
                schema=host.schema_from_data(helpers.shop())
                if core
                else original.schema,
                entity_table=original.entity_table,
                **kwargs,
            )
        )
        assert warnings == []
        if outcome[0] == 'value':
            request = outcome[1]
            mapping = (
                request.state['task']['options']
                if core and field == 'options'
                else request.state[field]
                if core
                else request.task.options
                if field == 'options'
                else getattr(request, field)
            )
            positions = [
                next(index for index, key in enumerate(keys) if key is retained)
                for retained in mapping
            ]
            outcomes.append(('value', positions, list(mapping.values())))
        else:
            outcomes.append(outcome)
        traces.append(list(events))
    assert outcomes[1] == outcomes[0]
    assert traces[1] == traces[0]


@pytest.mark.parametrize(
    'field', ['options', 'inference', 'requested_inference', 'metadata']
)
def test_held_mapping_copy_does_not_rehash_original_keys(baseline, field):
    from sdm_client import Request

    class Key:
        def __init__(self):
            self.calls = 0

        def __hash__(self):
            self.calls += 1
            if self.calls > 1:
                raise TypeError('unexpected key rehash')
            return 7

    original, unused = baseline
    host = helpers._host()
    outcomes = []
    for core in (False, True):
        key = Key()
        value = [Opaque()]
        mapping = {key: value}
        payloads = copy.deepcopy(list(original.payloads))
        kwargs = {}
        if field == 'requested_inference':
            kwargs[field] = mapping
        elif field == 'options':
            payloads[0]['task']['options'] = mapping
        else:
            payloads[0][field] = mapping
        request = (host.request_materialized if core else Request.materialized)(
            payloads=payloads,
            query=original.query,
            indices=original.indices,
            schema=host.schema_from_data(helpers.shop())
            if core
            else original.schema,
            entity_table=original.entity_table,
            **kwargs,
        )
        if field != 'metadata':
            copied = (
                request.state['task']['options']
                if core and field == 'options'
                else request.state[field]
                if core
                else request.task.options
                if field == 'options'
                else getattr(request, field)
            )
            assert copied is not mapping
            assert list(copied)[0] is key
            assert list(copied.values())[0] is value
        assert key.calls == 1
        if field == 'metadata':
            session = helpers._Watched()
            client = helpers._client(session)
            if core:
                client = host.Client(client._transport)
            outcomes.append(
                helpers._outcome(
                    lambda client=client, request=request: client.predict(
                        request, model=helpers.REL
                    )
                )
            )
            assert key.calls == 1
            assert session.responses == []
    if outcomes:
        assert outcomes[1] == outcomes[0]
