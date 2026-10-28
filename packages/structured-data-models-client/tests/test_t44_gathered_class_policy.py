# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public gathered class grammar after the reached native cast operations."""

from __future__ import annotations

import copy
import json
import math
import warnings

import pandas as pd
import pytest
from t45_frame_routes import FRAME_PREDICT_ROUTES
from test_t44_mutable_prediction import _request

from sdm_client import KumoRelational, SDMClient

OUTPUTS = ['prediction', 'probabilities']
METADATA = {'native': {'ordered': ['second', 'first'], 'nullable': None}}


class Scripted:
    def __init__(self, request, predictions, probabilities, after_send=None):
        self.request = request
        self.predictions = predictions
        self.probabilities = probabilities
        self.after_send = after_send
        self.calls = []
        self.expected = copy.deepcopy(request.payloads[0])
        self.expected['output'] = {'fields': OUTPUTS}
        self.expected['metadata']['request_id'] = 'sdm-gathered-class-policy'
        # The public materialize call requested no inference configuration.
        # Its engine defaults do not travel in the model request.
        self.expected['inference'].pop('inference_config')

    def close(self):
        pass

    def predict_body(self, body):
        sent = json.loads(body)
        assert sent == self.expected
        self.calls.append(sent)
        table = sent['predict']['instance_table']
        instance = table['columns'].index('instance_id')
        assert len(table['rows']) == 2
        if self.after_send is not None:
            self.after_send(self.request)
        return json.dumps(
            {
                'id': 'gathered-class-reply',
                'model': 'kumo-relational',
                'metadata': METADATA,
                'predictions': [
                    {
                        'row_index': at,
                        'id': str(row[instance]),
                        'prediction': self.predictions[at],
                        'probabilities': dict(self.probabilities[at]),
                        'metadata': {'ordinal': at, 'native': ['kept', None]},
                    }
                    for at, row in enumerate(table['rows'])
                ],
            }
        ).encode()


def prepared(monkeypatch, dtype, predictions, probabilities, *, late=None):
    monkeypatch.setenv('SDM_CORE_ROUTES', '')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = _request()
    if late is None:
        request.payloads[0]['task']['target']['dtype'] = dtype
    script = Scripted(request, predictions, probabilities, after_send=late)
    monkeypatch.setenv('SDM_CORE_ROUTES', FRAME_PREDICT_ROUTES)

    def legacy(*args, **kwargs):
        raise AssertionError('gathered class prediction called old _prepare')

    monkeypatch.setattr(SDMClient, '_prepare', legacy)
    monkeypatch.setattr(
        'sdm_client.adapters.relational._request_id',
        lambda: 'sdm-gathered-class-policy',
    )
    return request, script


def predict(request, script):
    with SDMClient('http://localhost:1', max_retries=0) as client:
        client._transport.close()
        client._transport = script
        return client.predict(request, model=KumoRelational(), outputs=OUTPUTS)


def assert_result(result, script, predictions, probabilities):
    assert script.calls == [script.expected]
    assert result.model == 'kumo-relational'
    assert result.request_id == 'gathered-class-reply'
    assert result.outputs == tuple(OUTPUTS)
    assert result.metadata == METADATA
    assert result.explanation is None
    assert result.unscored_candidates == ()
    assert len(result.items) == 2
    for at, (item, expected, keys) in enumerate(
        zip(result.items, predictions, probabilities, strict=True)
    ):
        assert item.row_index == at
        assert type(item.id) is int and item.id == at
        assert type(item.prediction) is type(expected)
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(item.prediction)
        else:
            assert item.prediction == expected
        assert list(item.probabilities.items()) == keys
        assert item.metadata == {'ordinal': at, 'native': ['kept', None]}
        assert item.forecast_step is None
    # Build the complete expected public frame from explicit policy answers,
    # independently of the core's typed columns and fallback representation.
    expected_frame = pd.DataFrame(
        {'row_index': [0, 1], 'id': [0, 1], 'prediction': predictions}
    )
    names = dict.fromkeys(key for row in probabilities for key, _ in row)
    for name in names:
        expected_frame['probability_' + name] = [
            dict(row).get(name, float('nan')) for row in probabilities
        ]
    pd.testing.assert_frame_equal(result.to_pandas(), expected_frame)


# Expected answers are the established Gathered grammar, deliberately distinct
# from native int/float/strip admission in the registered compatibility cases.
@pytest.mark.parametrize(
    ('dtype', 'raw', 'expected'),
    [
        pytest.param('bool', ['TRUE', ' f '], [True, False], id='bool-text'),
        pytest.param('bool', [0, 1], [False, True], id='bool-integers'),
        pytest.param('bool', [True, False], [True, False], id='bool-values'),
        pytest.param('bool', [2, 1.0], [2, 1.0], id='bool-non-bool-numbers'),
        pytest.param('bool', ['maybe', True], ['maybe', True], id='bool-mixed'),
        pytest.param('bool', [None, True], [None, True], id='bool-null'),
        pytest.param('int64', [' +10 ', '-2'], [10, -2], id='int-text'),
        pytest.param('int64', [1.5, '2'], [1.5, 2], id='int-fractional-mixed'),
        pytest.param('int64', ['1_0', '10'], ['1_0', 10], id='int-underscore'),
        pytest.param('int64', ['١٠', '-2'], ['١٠', -2], id='int-unicode'),
        pytest.param(
            'int64',
            [9.19e18, -9.19e18],
            [9190000000000000000, -9190000000000000000],
            id='int-whole-inside',
        ),
        pytest.param(
            'int64',
            [9.21e18, -9.21e18],
            [9.21e18, -9.21e18],
            id='int-whole-outside',
        ),
        pytest.param(
            'int64',
            ['9223372036854775808', '10'],
            ['9223372036854775808', 10],
            id='int-text-overflow',
        ),
        pytest.param('int64', [True, False], [True, False], id='int-bool-raw'),
        pytest.param('int64', [None, '2'], [None, 2], id='int-null'),
        pytest.param(
            'int64',
            [float('inf'), -float('inf')],
            [float('inf'), -float('inf')],
            id='int-nonfinite',
        ),
        pytest.param(
            'float64', [' +2.5 ', '-1e1'], [2.5, -10.0], id='float-text'
        ),
        pytest.param('float64', [1, 2.5], [1.0, 2.5], id='float-numbers'),
        pytest.param(
            'float64',
            ['1_0.5', '2.5'],
            ['1_0.5', 2.5],
            id='float-underscore',
        ),
        pytest.param(
            'float64', ['١٠.5', '2'], ['١٠.5', 2.0], id='float-unicode'
        ),
        pytest.param(
            'float64', [True, 2.5], [True, 2.5], id='float-bool-mixed'
        ),
        pytest.param(
            'float64',
            ['NaN', 'inf'],
            [float('nan'), float('inf')],
            id='float-nonfinite',
        ),
        pytest.param(
            'string', ['true', '01'], ['true', '01'], id='text-unchanged'
        ),
        pytest.param('string', [1, 'two'], [1, 'two'], id='text-mixed-raw'),
        pytest.param(
            'int64',
            [[1, 'a'], {'native': True}],
            [[1, 'a'], {'native': True}],
            id='containers-raw',
        ),
    ],
)
def test_complete_gathered_class_values_keep_their_declared_policy(
    monkeypatch, dtype, raw, expected
):
    probabilities = [[('class-b', 0.25), ('class-a', 0.75)]] * 2
    request, script = prepared(monkeypatch, dtype, raw, probabilities)
    result = predict(request, script)
    assert_result(result, script, expected, probabilities)


@pytest.mark.parametrize(
    'separator', [' \t\n', '\u00a0', '\x1c', '\x1d', '\x1e', '\x1f']
)
def test_gathered_boolean_whitespace_keeps_raw_and_restored_populations(
    monkeypatch, separator
):
    raw = [separator + 'true' + separator, separator + 'false' + separator]
    probabilities = [[(raw[0], 0.25), (raw[1], 0.75)]] * 2
    if separator in ('\x1c', '\x1d', '\x1e', '\x1f'):
        expected, keyed = raw, probabilities
    else:
        expected = [True, False]
        keyed = [[('True', 0.25), ('False', 0.75)]] * 2
    request, script = prepared(monkeypatch, 'bool', raw, probabilities)
    assert_result(predict(request, script), script, expected, keyed)


@pytest.mark.parametrize(
    ('dtype', 'given', 'expected'),
    [
        (
            'bool',
            [('false', 0.2), ('true', 0.8)],
            [('False', 0.2), ('True', 0.8)],
        ),
        (
            'bool',
            [('true', 0.2), ('1', 0.3), ('false', 0.5)],
            [('true', 0.2), ('1', 0.3), ('false', 0.5)],
        ),
        (
            'bool',
            [('true', 0.4), ('\x1ctrue\x1c', 0.6)],
            [('True', 0.4), ('\x1ctrue\x1c', 0.6)],
        ),
        ('int64', [(' +2 ', 0.6), ('01', 0.4)], [('2', 0.6), ('1', 0.4)]),
        (
            'int64',
            [('01', 0.2), ('1', 0.3), ('+2', 0.5)],
            [('01', 0.2), ('1', 0.3), ('+2', 0.5)],
        ),
        ('int64', [('1_0', 0.4), ('+10', 0.6)], [('1_0', 0.4), ('10', 0.6)]),
        ('int64', [('١٠', 0.4), ('+2', 0.6)], [('١٠', 0.4), ('2', 0.6)]),
        (
            'float64',
            [(' +2.5 ', 0.6), ('1e1', 0.4)],
            [('2.5', 0.6), ('10.0', 0.4)],
        ),
        (
            'float64',
            [('1', 0.2), ('1.0', 0.3), ('+2.5', 0.5)],
            [('1', 0.2), ('1.0', 0.3), ('+2.5', 0.5)],
        ),
        (
            'float64',
            [('1_0.5', 0.4), ('+10.5', 0.6)],
            [('1_0.5', 0.4), ('10.5', 0.6)],
        ),
        ('string', [('true', 0.4), ('01', 0.6)], [('true', 0.4), ('01', 0.6)]),
    ],
)
def test_probability_names_preserve_entry_order_and_final_collision_policy(
    monkeypatch, dtype, given, expected
):
    request, script = prepared(monkeypatch, dtype, ['raw', 'raw'], [given] * 2)
    assert_result(
        predict(request, script), script, ['raw', 'raw'], [expected] * 2
    )


def test_probability_order_is_kept_separately_for_each_answer(monkeypatch):
    given = [[('+2', 0.3), ('01', 0.7)], [('01', 0.6), ('+2', 0.4)]]
    keyed = [[('2', 0.3), ('1', 0.7)], [('1', 0.6), ('2', 0.4)]]
    request, script = prepared(monkeypatch, 'int64', ['2', '1'], given)
    assert_result(predict(request, script), script, [2, 1], keyed)


class Declared(str):
    def __new__(
        cls,
        value,
        events,
        *,
        prefix=True,
        floating=False,
        fail_at=None,
        failure=None,
    ):
        held = str.__new__(cls, value)
        held.events = events
        held.prefix = prefix
        held.floating = floating
        held.fail_at = fail_at
        held.failure = failure
        held.reads = 0
        return held

    def __bool__(self):
        self.events.append(('bool',))
        return True

    def startswith(self, prefix, *args):
        self.reads += 1
        self.events.append(('startswith', prefix))
        if self.reads == self.fail_at:
            raise self.failure
        if isinstance(prefix, tuple):
            return self.prefix
        if self.floating:
            return prefix == 'float'
        return str.startswith(self, prefix, *args)

    def __str__(self):
        raise AssertionError('dtype fact called overridden __str__')

    def __iter__(self):
        raise AssertionError('dtype fact called overridden __iter__')

    def encode(self, *args, **kwargs):
        raise AssertionError('dtype fact called overridden encode')


@pytest.mark.parametrize(
    'dtype,expected,keys',
    [
        ('bool', [True, False], [('True', 0.25), ('False', 0.75)]),
        ('string', ['true', 'false'], [('true', 0.25), ('false', 0.75)]),
    ],
)
def test_live_declaration_distinguishes_identical_unchanged_native_values(
    monkeypatch, dtype, expected, keys
):
    events = []
    held = Declared(dtype, events, prefix=False)

    def late(request):
        request.payloads[0]['task']['target']['dtype'] = held

    probabilities = [[('true', 0.25), ('false', 0.75)]] * 2
    request, script = prepared(
        monkeypatch, None, ['true', 'false'], probabilities, late=late
    )
    assert_result(predict(request, script), script, expected, [keys] * 2)
    assert events == [('bool',), ('startswith', ('int', 'float', 'bool'))]


def test_ordinary_native_boolean_casts_keep_their_exact_read_order(monkeypatch):
    events = []
    held = Declared('bool', events)

    def late(request):
        request.payloads[0]['task']['target']['dtype'] = held

    given = [[('true', 0.25), ('false', 0.75)]] * 2
    request, script = prepared(
        monkeypatch, None, ['true', 'false'], given, late=late
    )
    assert_result(
        predict(request, script),
        script,
        [True, False],
        [[('True', 0.25), ('False', 0.75)]] * 2,
    )
    assert events == [
        ('bool',),
        ('startswith', ('int', 'float', 'bool')),
        *([('startswith', 'bool')] * 6),
    ]


def test_late_classification_skip_never_observes_the_dtype_fact(monkeypatch):
    events = []
    held = Declared('bool', events)

    class Kind(str):
        def __str__(self):
            events.append(('kind.str',))
            return self

        def __contains__(self, value):
            events.append(('kind.contains', value))
            return False

    def late(request):
        request.payloads[0]['task']['kind'] = Kind('multiclass_classification')
        request.payloads[0]['task']['target']['dtype'] = held

    given = [[('true', 0.25), ('false', 0.75)]] * 2
    request, script = prepared(
        monkeypatch, None, ['true', 'false'], given, late=late
    )
    assert_result(predict(request, script), script, ['true', 'false'], given)
    assert events == [('kind.str',), ('kind.contains', 'classification')]


def test_stateful_native_dtype_effects_precede_strict_gathered_selection(
    monkeypatch,
):
    events = []
    held = Declared('int64', events, floating=True)

    def late(request):
        request.payloads[0]['task']['target']['dtype'] = held

    given = [[('1_0.5', 0.25), ('+2.5', 0.75)]] * 2
    request, script = prepared(
        monkeypatch, None, ['1.5', '1_0.5'], given, late=late
    )
    assert_result(
        predict(request, script),
        script,
        [1.5, '1_0.5'],
        [[('1_0.5', 0.25), ('2.5', 0.75)]] * 2,
    )
    assert events == [
        ('bool',),
        ('startswith', ('int', 'float', 'bool')),
        *(
            [
                ('startswith', 'bool'),
                ('startswith', 'int'),
                ('startswith', 'float'),
            ]
            * 6
        ),
    ]


def test_native_dtype_failure_is_not_skipped_for_a_strictly_raw_class(
    monkeypatch,
):
    events = []
    failure = TypeError('actual second-row native dtype failure')
    caller = LookupError('active caller')
    held = Declared('int64', events, fail_at=6, failure=failure)

    def late(request):
        request.payloads[0]['task']['target']['dtype'] = held

    request, script = prepared(
        monkeypatch, None, ['1_0', '2'], [[('1', 1.0)]] * 2, late=late
    )
    with pytest.raises(TypeError) as caught:
        try:
            raise caller
        except LookupError:
            predict(request, script)
    assert caught.value is failure
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is caller
    assert script.calls == [script.expected]
    assert events == [
        ('bool',),
        ('startswith', ('int', 'float', 'bool')),
        ('startswith', 'bool'),
        ('startswith', 'int'),
        ('startswith', 'bool'),
        ('startswith', 'int'),
        ('startswith', 'bool'),
    ]
