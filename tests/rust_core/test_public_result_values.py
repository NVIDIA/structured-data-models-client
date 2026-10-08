# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result operations retain Python values across the native layout boundary."""

from __future__ import annotations

import datetime as dt
import gc
import inspect
import weakref
from dataclasses import fields
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from sdm_api.generated.tfm_api import PredictionItem
from sdm_client import PredictionResult, _impl


class Opaque:
    def __repr__(self):
        raise AssertionError('an opaque result value must not be serialized')


BOUNDARY_NAMES = (
    'plain',
    'numpy-string',
    'numpy-integer',
    'opaque',
    'equal-count',
    'equal-error',
    'equal-alias',
)


def constructor_protocol_observation(operation, case, field='outputs'):
    calls = []
    error = ValueError('output equality refused')

    class Output(str):
        def __eq__(self, other):
            calls.append(str(other))
            if case == 'equal-error':
                raise error
            if case == 'equal-alias':
                return other == 'prediction'
            return super().__eq__(other)

        __hash__ = str.__hash__

    value = {
        'plain': 'prediction',
        'numpy-string': np.str_('prediction'),
        'numpy-integer': np.int64(1),
        'opaque': object(),
        'equal-count': Output('prediction'),
        'equal-error': Output('prediction'),
        'equal-alias': Output('alias'),
        'none': None,
        'integer': 7,
    }[case]

    def view(found):
        if found is None or type(found) in (str, int, float, bool):
            return type(found).__name__, found
        if found is value:
            return type(found).__name__, 'original'
        if isinstance(found, (list, tuple)):
            return type(found).__name__, [view(item) for item in found]
        if isinstance(found, dict):
            return 'dict', [
                (view(key), view(item)) for key, item in found.items()
            ]
        if isinstance(found, PredictionItem):
            return 'item', view(found.prediction)
        return type(found).__name__, 'other'

    try:
        if operation.startswith('absent-'):
            result = PredictionResult(
                model='model',
                request_id='request',
                items=[]
                if operation == 'absent-empty'
                else [PredictionItem(row_index=0)],
            )
            outcome = view(result.absent([value]))
        else:
            arguments = dict(model='model', request_id='request', items=[])
            arguments[field] = (
                [value]
                if field in ('items', 'outputs', 'unscored_candidates')
                else {'value': value}
                if field == 'metadata'
                else value
            )
            if operation == 'nonempty-frame':
                arguments['items'] = [
                    PredictionItem(row_index=0, prediction=value)
                ]
            result = PredictionResult(**arguments)
            if operation.startswith('merge'):
                result = PredictionResult.merge(
                    [result, PredictionResult(**arguments)]
                )
            if operation.endswith('frame'):
                frame = result.to_pandas()
                outcome = (
                    frame.columns.tolist(),
                    frame.dtypes.astype(str).tolist(),
                    frame.shape,
                )
            elif operation == 'absence':
                outcome = view(result.absent(['prediction', 'explanation']))
            else:
                outcome = tuple(
                    view(getattr(result, name))
                    for name in (
                        'model',
                        'request_id',
                        'items',
                        'outputs',
                        'metadata',
                        'explanation',
                        'unscored_candidates',
                    )
                )
        return calls, ('success', outcome)
    except Exception as found:
        return calls, (
            'error',
            type(found).__name__,
            str(found),
            found is error,
        )


@pytest.mark.parametrize('case', BOUNDARY_NAMES)
@pytest.mark.parametrize(
    'operation',
    [
        'empty-frame',
        'merge',
        'merge-empty-frame',
        'absent-empty',
        'absent-nonempty',
    ],
)
def test_constructor_requested_name_protocol(monkeypatch, case, operation):
    monkeypatch.setattr(_impl, 'OLD', True)
    old = constructor_protocol_observation(operation, case)
    monkeypatch.setattr(_impl, 'OLD', False)
    assert constructor_protocol_observation(operation, case) == old


@pytest.mark.parametrize('case', ['none', 'integer', 'opaque'])
@pytest.mark.parametrize('field', ['model', 'request_id'])
def test_constructor_merge_original_identity(monkeypatch, case, field):
    monkeypatch.setattr(_impl, 'OLD', True)
    old = constructor_protocol_observation('merge', case, field)
    monkeypatch.setattr(_impl, 'OLD', False)
    assert constructor_protocol_observation('merge', case, field) == old


@pytest.mark.parametrize('case', (*BOUNDARY_NAMES, 'none', 'integer'))
@pytest.mark.parametrize(
    'field',
    [
        'model',
        'request_id',
        'items',
        'outputs',
        'metadata',
        'explanation',
        'unscored_candidates',
    ],
)
@pytest.mark.parametrize(
    'operation',
    [
        'construct',
        'absence',
        'empty-frame',
        'nonempty-frame',
        'merge',
        'merge-empty-frame',
    ],
)
def test_complete_constructor_field_protocols(
    monkeypatch, case, field, operation
):
    monkeypatch.setattr(_impl, 'OLD', True)
    old = constructor_protocol_observation(operation, case, field)
    monkeypatch.setattr(_impl, 'OLD', False)
    assert constructor_protocol_observation(operation, case, field) == old


@pytest.mark.parametrize('field', ['model', 'outputs'])
@pytest.mark.parametrize(
    'mode',
    [
        'equal',
        'different',
        'collision',
        'hash-error',
        'eq-error',
        'sort-error',
        'repr-error',
    ],
)
def test_constructor_set_dispatch_and_validation_order(
    monkeypatch, field, mode
):
    def run(old):
        monkeypatch.setattr(_impl, 'OLD', old)
        calls = []
        failure = ValueError('set protocol refused')

        class Value:
            def __init__(self, index):
                self.index = index

            def __hash__(self):
                calls.append(('hash', self.index))
                if mode == 'hash-error':
                    raise failure
                return (
                    19
                    if mode in ('equal', 'collision', 'eq-error')
                    else self.index
                )

            def __eq__(self, other):
                calls.append(('equal', self.index, other.index))
                if mode == 'eq-error':
                    raise failure
                return mode == 'equal'

            def __lt__(self, other):
                calls.append(('less', self.index, other.index))
                if mode == 'sort-error':
                    raise failure
                return self.index < other.index

            def __repr__(self):
                calls.append(('repr', self.index))
                if mode == 'repr-error':
                    raise failure
                return f'value{self.index}'

        parts = []
        for index in (1, 2):
            value = Value(index)
            arguments = dict(model='model', request_id='request', items=[])
            arguments[field] = (value,) if field == 'outputs' else value
            parts.append(PredictionResult(**arguments))
        try:
            merged = PredictionResult.merge(parts)
            assert getattr(merged, field) is getattr(parts[0], field)
            outcome = 'success'
        except Exception as found:
            outcome = type(found).__name__, str(found), found is failure
        return calls, outcome

    assert run(False) == run(True)


@pytest.mark.parametrize(
    'case', ['normal', 'hash-error', 'equal-error', 'getattr-error', 'empty']
)
def test_requested_names_deduplicate_before_presence(monkeypatch, case):
    def run(old):
        monkeypatch.setattr(_impl, 'OLD', old)
        calls = []
        failure = ValueError('requested protocol refused')

        class Name(str):
            def __hash__(self):
                calls.append(('hash', str(self)))
                if case == 'hash-error':
                    raise failure
                return str.__hash__(self)

            def __eq__(self, other):
                calls.append(('equal', str(self), str(other)))
                if case == 'equal-error':
                    raise failure
                return str.__eq__(self, other)

        class Item:
            def __getattr__(self, name):
                calls.append(('getattr', str(name), type(name).__name__))
                if case == 'getattr-error':
                    raise failure
                return None

        first, duplicate = Name('prediction'), Name('prediction')
        result = PredictionResult(
            model='model',
            request_id='request',
            items=[] if case == 'empty' else [Item()],
        )
        try:
            absent = result.absent([first, duplicate, first])
            outcome = len(absent), not absent or absent[0] is first
        except Exception as found:
            outcome = type(found).__name__, str(found), found is failure
        return calls, outcome

    assert run(False) == run(True)


@pytest.mark.parametrize('operation', ['merge', 'frame', 'absent'])
def test_result_generic_protocol_gil_reentry_lifetime(monkeypatch, operation):
    import ctypes

    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)
    original = ResultValues.protocol
    registries = []
    calls = []

    def protocol(self, name, operands):
        assert ctypes.pythonapi.PyGILState_Check() == 1
        registries.append(weakref.ref(self))
        calls.append(name)
        assert (
            PredictionResult(
                model='nested', request_id='nested', items=[]
            ).absent(object())
            == ()
        )
        return original(self, name, operands)

    monkeypatch.setattr(ResultValues, 'protocol', protocol)
    result = PredictionResult(
        model='model',
        request_id='request',
        items=[] if operation == 'frame' else [PredictionItem(row_index=0)],
        outputs=['prediction'],
    )
    if operation == 'merge':
        PredictionResult.merge([result, result])
        assert calls == ['set', 'set']
    elif operation == 'frame':
        result.to_pandas()
        assert calls == ['contains'] * 4
    else:
        result.absent(['prediction'])
        assert calls == [
            'dict_keys',
            'equal',
            'contains',
            'getattr_default_none',
            'contains',
        ]
    gc.collect()
    assert all(reference() is None for reference in registries)


@pytest.mark.parametrize('operation', ['merge', 'frame', 'absent'])
def test_result_generic_protocol_panic_containment(monkeypatch, operation):
    from sdm_client.errors import StructuredDataError
    from sdm_core import _native
    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)

    def panic(*arguments):
        raise _native.CorePanic('result generic protocol panic')

    monkeypatch.setattr(ResultValues, 'protocol', panic)
    result = PredictionResult(
        model='model',
        request_id='request',
        items=[] if operation == 'frame' else [PredictionItem(row_index=0)],
    )
    with pytest.raises(
        StructuredDataError, match='result generic protocol panic'
    ):
        if operation == 'merge':
            PredictionResult.merge([result, result])
        elif operation == 'frame':
            result.to_pandas()
        else:
            result.absent(['prediction'])


@pytest.mark.parametrize(
    'operation,bad',
    [
        ('set', 'null'),
        ('set', '[{"$host_ref":"0"}, -1]'),
        ('set', '[{"$host_ref":"0"}, 3]'),
        ('set', '[0, 1]'),
        ('contains', '1'),
        ('dict_keys', '{}'),
        ('dict_keys', '[1]'),
        ('equal', '[]'),
        ('sorted_repr', 'false'),
        ('getattr_default_none', '1'),
    ],
)
def test_result_generic_protocol_rejects_malformed_facts(
    monkeypatch, operation, bad
):
    from sdm_client.errors import StructuredDataError
    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)
    original = ResultValues.protocol

    def protocol(self, name, operands):
        return bad if name == operation else original(self, name, operands)

    monkeypatch.setattr(ResultValues, 'protocol', protocol)
    result = PredictionResult(
        model='model',
        request_id='request',
        items=[] if operation == 'contains' else [PredictionItem(row_index=0)],
    )
    with pytest.raises((ValueError, StructuredDataError), match='host'):
        if operation in ('set', 'sorted_repr'):
            other = PredictionResult(
                model='other' if operation == 'sorted_repr' else 'model',
                request_id='request',
                items=[],
            )
            PredictionResult.merge([result, other])
        elif operation == 'contains':
            result.to_pandas()
        else:
            result.absent(['prediction'])


def test_result_generic_protocol_native_controls_and_missing_facts():
    import json

    from sdm_core import _native
    from sdm_core.answers import ResultValues, absence_json, result_json

    wire = json.dumps(
        {
            'model': 'model',
            'request_id': 'request',
            'items': [],
            'outputs': ['prediction'],
        }
    )
    assert _native.absent(wire, ['prediction']) == []
    assert json.loads(_native.merge([wire, wire]))['model'] == 'model'
    assert _native.flatten(wire).column_names == [
        'row_index',
        'id',
        'prediction',
    ]
    references = ResultValues()
    result = PredictionResult(model='model', request_id='request', items=[])
    held = result_json(result, references=references)
    with pytest.raises(_native.CoreError, match='requires host'):
        _native.merge([held, held])
    with pytest.raises(ValueError, match='missing result host protocol'):
        _native.flatten(
            result_json(result, references=references, layout=True),
            references.infer_column,
            references.column_order,
        )
    result = PredictionResult(
        model='model', request_id='request', items=[PredictionItem(row_index=0)]
    )
    with pytest.raises(ValueError, match='missing result host protocol'):
        _native.absent(absence_json(result, ['prediction'], references), [])


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
}
FIELDS = (
    'prediction',
    'probabilities',
    'quantiles',
    'scores',
    'rankings',
    'embeddings',
)


class FormattedMappingKey:
    def __init__(self, string='string-name', formatted='format-name'):
        self.string = string
        self.formatted = formatted

    def __str__(self):
        if isinstance(self.string, Exception):
            raise self.string
        return self.string

    def __format__(self, specification):
        assert specification == ''
        if isinstance(self.formatted, Exception):
            raise self.formatted
        return self.formatted


MAPPING_KEYS = {
    'string': {'label': 0.5},
    'integer': {1: 0.5},
    'float': {0.1: 0.5},
    'float16': {np.float16(0.1): 0.5},
    'float32': {np.float32(0.1): 0.5},
    'float64': {np.float64(0.1): 0.5},
    'longdouble': {np.longdouble('0.1'): 0.5},
    'numpy-int': {np.int32(1): 0.5},
    'complex64': {np.complex64(0.1 + 0.2j): 0.5},
    'complex128': {np.complex128(0.1 + 0.2j): 0.5},
    'decimal': {Decimal('0.1'): 0.5},
    'fraction': {Fraction(1, 3): 0.5},
    'bytes': {b'label': 0.5},
    'none': {None: 0.5},
    'tuple': {(1, 'label'): 0.5},
    'custom-format': {FormattedMappingKey(): 0.5},
    'refuses-format': {
        FormattedMappingKey(formatted=ValueError('format refused')): 0.5
    },
    'refuses-string': {
        FormattedMappingKey(string=ValueError('string refused')): 0.5
    },
    'collision-float16': {np.float16(0.1): 0.25, 0.1: 0.75},
    'collision-float32': {np.float32(0.1): 0.25, 0.1: 0.75},
    'collision-reversed': {0.1: 0.75, np.float32(0.1): 0.25},
    'collision-custom': {
        FormattedMappingKey('same', 'first'): 0.25,
        FormattedMappingKey('same', 'second'): 0.75,
    },
    'collision-formatted': {
        FormattedMappingKey('first', 'same'): 0.25,
        FormattedMappingKey('second', 'same'): 0.75,
    },
    'original-collision': {1: 0.25, '1': 0.75},
    **{
        'scalar-' + family: {make(): 0.5}
        for family, make in VALUES.items()
        if family not in ('bytearray', 'numpy-void', 'structured-void')
    },
}


@pytest.mark.parametrize('family', MAPPING_KEYS)
@pytest.mark.parametrize('field', ['probabilities', 'quantiles'])
@pytest.mark.parametrize('merged', [False, True])
def test_mapping_key_format_matrix(monkeypatch, family, field, merged):
    assert len(VALUES) == 33
    assert len(MAPPING_KEYS) == 54

    def run():
        result = PredictionResult(
            model='kumo-tabular',
            request_id='mapping',
            items=[
                PredictionItem(row_index=0, **{field: MAPPING_KEYS[family]})
            ],
        )
        if merged:
            result = PredictionResult.merge([result, result])
        return result.to_pandas()

    compare_boundary(monkeypatch, run)


@pytest.mark.parametrize('field', ['probabilities', 'quantiles'])
@pytest.mark.parametrize(
    'case',
    [
        'count',
        'format-error',
        'earlier-field',
        'earlier-cell',
        'format-before-cell',
        'cell-error',
    ],
)
def test_mapping_format_protocol(monkeypatch, field, case):
    def run(old):
        monkeypatch.setattr(_impl, 'OLD', old)
        calls = []
        format_error = ValueError('format refused')
        earlier_error = ValueError('earlier field refused')
        cell_error = ValueError('cell inference refused')

        class Key:
            def __init__(self, name):
                self.name = name

            def __str__(self):
                calls.append(('str', self.name))
                return self.name

            def __format__(self, specification):
                calls.append(('format', self.name, specification))
                if case in ('format-error', 'format-before-cell'):
                    raise format_error
                return self.name

        class Earlier(dict):
            def __bool__(self):
                calls.append(('earlier-truth',))
                raise earlier_error

        class Cell:
            def __lt__(self, other):
                calls.append(('cell-sort',))
                raise cell_error

            def __gt__(self, other):
                calls.append(('cell-sort',))
                raise cell_error

        class UnreadableItem:
            @property
            def row_index(self):
                calls.append(('earlier-cell',))
                raise earlier_error

        mapping = {Key('first'): 0.25, Key('second'): 0.75}
        item = PredictionItem(row_index=0, **{field: mapping})
        if case == 'earlier-field':
            item = (
                PredictionItem(
                    row_index=0,
                    probabilities=Earlier(value=1),
                    quantiles=mapping,
                )
                if field == 'quantiles'
                else PredictionItem(row_index=0, probabilities=Earlier(mapping))
            )
        if case == 'earlier-cell':
            item = UnreadableItem()
        items = (
            [
                PredictionItem(row_index=Cell(), **{field: mapping}),
                PredictionItem(row_index=Cell(), **{field: mapping}),
            ]
            if case in ('format-before-cell', 'cell-error')
            else [item]
        )
        result = PredictionResult(
            model='kumo-tabular', request_id='protocol', items=items
        )
        try:
            frame = result.to_pandas()
        except Exception as error:
            assert any(
                error is expected
                for expected in (format_error, earlier_error, cell_error)
            )
            return calls, (
                type(error),
                str(error),
                error is format_error,
                error is earlier_error,
                error is cell_error,
            )
        return calls, frame.columns.tolist()

    assert run(False) == run(True)


@pytest.mark.parametrize('count', [0, 1, 2])
@pytest.mark.parametrize('outputs', [(), ('prediction',)])
def test_empty_result_column_metadata(monkeypatch, count, outputs):
    frames = []
    for old in (True, False):
        monkeypatch.setattr(_impl, 'OLD', old)
        result = PredictionResult(
            model='kumo-tabular',
            request_id='empty',
            items=[PredictionItem(row_index=None) for position in range(count)],
            outputs=outputs,
        )
        frames.append(result.to_pandas())
    pd.testing.assert_frame_equal(
        frames[1], frames[0], check_index_type=True, check_column_type=True
    )


@pytest.mark.parametrize('field', ['scores', 'embeddings'])
@pytest.mark.parametrize(
    'case', ['length-count', 'length-error', 'hint-count', 'hint-error']
)
def test_result_sequence_list_protocol(monkeypatch, field, case):
    def run(old):
        monkeypatch.setattr(_impl, 'OLD', old)
        calls = []
        length_error = ValueError('length refused')

        class Sequence:
            def __iter__(self):
                calls.append('iter')
                return iter([1, 2])

            def __len__(self):
                calls.append('len')
                if case == 'length-error':
                    raise length_error
                return 2

        class Iterator:
            def __init__(self):
                self.values = iter([1, 2])

            def __iter__(self):
                calls.append('iter')
                return self

            def __next__(self):
                calls.append('next')
                return next(self.values)

            def __length_hint__(self):
                calls.append('hint')
                if case == 'hint-error':
                    raise length_error
                return 2

        value = Iterator() if case.startswith('hint') else Sequence()
        result = PredictionResult(
            model='kumo-tabular',
            request_id='sequence',
            items=[PredictionItem(row_index=0, **{field: value})],
        )
        try:
            frame = result.to_pandas()
        except Exception as error:
            assert error is length_error
            return calls, (type(error), str(error))
        return calls, (frame.columns.tolist(), frame.to_numpy().tolist())

    assert run(False) == run(True)


@pytest.mark.parametrize('field', ['model', 'request_id', 'outputs'])
@pytest.mark.parametrize('value', [None, 7, object()])
def test_nonempty_layout_ignores_unrelated_fields(monkeypatch, field, value):
    result = PredictionResult(
        model='kumo-tabular',
        request_id='ignored',
        items=[PredictionItem(row_index=0, prediction=1)],
    )
    setattr(result, field, (value,) if field == 'outputs' else value)
    compare_boundary(monkeypatch, result.to_pandas)


class OrderedStructural:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return (
            isinstance(other, OrderedStructural) and self.value == other.value
        )


STRUCTURAL_VALUES = {
    'empty': [],
    'none-only': [None, None],
    'ordinary-integers': [2, 1, 2],
    'ordinary-floats': [2.5, 1.5, 2.5],
    'float-ties': [2, 2.0, 1, 1.0],
    'integer-float': [2, 1.5, 1],
    'wide-integer-float': [2**70, 1.5, -(2**70)],
    'integer-infinity': [2, float('inf'), 1, float('-inf')],
    'integer-nan': [2, float('nan'), 1],
    'integer-missing': [2, None, 1],
    'integer-na': [2, pd.NA, 1],
    'integer-mixed-missing': [2, None, pd.NA, float('nan'), 1],
    'numpy-integer-float': [np.int8(2), 1.5, np.int8(1)],
    'numpy-integer-numpy-float': [
        np.uint64(2**63),
        np.float32(1.5),
        np.int8(-1),
    ],
    'numpy-integers': [np.int64(2), np.uint64(1), np.int8(2)],
    'integer-bool': [2, True, 0, False],
    'numpy-integer-bool': [np.int8(2), np.bool_(True), np.int8(0)],
    'bool-only': [True, False, True],
    'numpy-bool-only': [np.bool_(True), np.bool_(False)],
    'decimal-only': [Decimal('2.5'), Decimal('1.5'), Decimal('2.5')],
    'decimal-integer': [2, Decimal('1.5'), 1],
    'decimal-nan': [Decimal('2'), Decimal('NaN'), Decimal('1')],
    'fraction-only': [Fraction(5, 2), Fraction(3, 2), Fraction(5, 2)],
    'fraction-integer': [2, Fraction(3, 2), 1],
    'string-only': ['b', 'a', 'b'],
    'integer-string': [2, '1', 1],
    'bytes-only': [b'b', b'a', b'b'],
    'date-only': [dt.date(2020, 1, 2), dt.date(2020, 1, 1)],
    'datetime-only': [dt.datetime(2020, 1, 2), dt.datetime(2020, 1, 1)],
    'timestamp-missing': [
        pd.Timestamp('2020-01-02'),
        pd.NaT,
        pd.Timestamp('2020-01-01'),
    ],
    'numpy-datetime': [
        np.datetime64('2020-01-02'),
        np.datetime64('2020-01-01'),
    ],
    'timedelta-only': [pd.Timedelta(days=2), pd.Timedelta(days=1)],
    'period-only': [
        pd.Period('2020-02', freq='M'),
        pd.Period('2020-01', freq='M'),
    ],
    'interval-only': [pd.Interval(2, 3), pd.Interval(1, 2)],
    'ordered-object': [
        OrderedStructural(2),
        OrderedStructural(1),
        OrderedStructural(2),
    ],
    'opaque-object': [object(), object()],
    'list-only': [[2], [1], [2]],
    'complex-only': [2 + 1j, 1 + 1j],
}


@pytest.mark.parametrize('family', STRUCTURAL_VALUES)
@pytest.mark.parametrize('field', ['row_index', 'forecast_step'])
@pytest.mark.parametrize(
    'operation', ['construct', 'absent', 'frame', 'merge', 'merge-frame']
)
def test_complete_structural_family_matrix(
    monkeypatch, family, field, operation
):
    assert len(STRUCTURAL_VALUES) == 38
    values = STRUCTURAL_VALUES[family]

    def run():
        original = PredictionResult(
            model='kumo-tabular',
            request_id='structural',
            items=[
                PredictionItem(
                    **{
                        'row_index': position,
                        field: value,
                        'prediction': position,
                    }
                )
                for position, value in enumerate(values)
            ],
        )
        result = (
            PredictionResult.merge([original, original])
            if operation.startswith('merge')
            else original
        )
        originals = original.items * (2 if operation.startswith('merge') else 1)
        assert len(result.items) == len(originals)
        assert all(
            actual is expected
            for actual, expected in zip(result.items, originals, strict=True)
        )
        assert all(
            getattr(item, field) is value
            for item, value in zip(original.items, values, strict=True)
        )
        if operation == 'absent':
            return result.absent(['prediction', 'explanation'])
        if operation.endswith('frame'):
            return result.to_pandas()
        return None

    monkeypatch.setattr(_impl, 'OLD', True)
    try:
        expected = run()
    except Exception as error:
        expected_error = (type(error), str(error))
        monkeypatch.setattr(_impl, 'OLD', False)
        with pytest.raises(expected_error[0]) as actual:
            run()
        assert str(actual.value) == expected_error[1]
    else:
        monkeypatch.setattr(_impl, 'OLD', False)
        actual = run()
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        else:
            assert actual == expected


@pytest.mark.parametrize('positions', [[], [0], [0, 0], [0, 2], [1, 0, 2]])
def test_host_column_rejects_invalid_permutation(monkeypatch, positions):
    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)
    monkeypatch.setattr(ResultValues, 'column_order', lambda *args: positions)
    result = PredictionResult(
        model='kumo-tabular',
        request_id='order',
        items=[PredictionItem(row_index=2), PredictionItem(row_index=1)],
    )
    with pytest.raises(ValueError, match='not a complete permutation'):
        result.to_pandas()


def test_host_column_rejects_wrong_length(monkeypatch):
    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)
    monkeypatch.setattr(ResultValues, 'infer_column', lambda *args: ('0', 2))
    result = boundary_result('item', 'row_index', 1)
    with pytest.raises(ValueError, match='column length differs'):
        result.to_pandas()


@pytest.mark.parametrize('missing', ['both', 'inference', 'ordering'])
def test_host_column_requires_facts(missing):
    from sdm_core import _native
    from sdm_core.answers import ResultValues, result_json

    references = ResultValues()
    encoded = result_json(
        boundary_result('item', 'row_index', 1),
        references=references,
        layout=True,
    )
    if missing == 'both':
        with pytest.raises(
            _native.CoreError, match='requires host column facts'
        ):
            _native.flatten(encoded)
    else:
        with pytest.raises(ValueError, match=f'missing host column {missing}'):
            _native.flatten(
                encoded,
                None if missing == 'inference' else references.infer_column,
                None if missing == 'ordering' else references.column_order,
            )


@pytest.mark.parametrize('stage', ['infer_column', 'column_order'])
def test_host_column_exception_identity_and_gil(monkeypatch, stage):
    import ctypes

    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)
    error = ValueError('host column refuses')

    def refuse(*args):
        assert ctypes.pythonapi.PyGILState_Check() == 1
        raise error

    monkeypatch.setattr(ResultValues, stage, refuse)
    with pytest.raises(ValueError) as actual:
        boundary_result('item', 'row_index', 1).to_pandas()
    assert actual.value is error


@pytest.mark.parametrize('old', [True, False])
def test_structural_sort_comparison_sequence_reentrant_and_lifetime(
    monkeypatch, old
):
    import ctypes

    monkeypatch.setattr(_impl, 'OLD', old)
    nested = boundary_result('item', 'prediction', Opaque())
    calls = []

    class Ordered:
        def __init__(self, value):
            self.value = value

        def __lt__(self, other):
            assert ctypes.pythonapi.PyGILState_Check() == 1
            assert (
                nested.to_pandas()['prediction'][0]
                is nested.items[0].prediction
            )
            calls.append(('lt', self.value, other.value))
            return self.value < other.value

        def __gt__(self, other):
            calls.append(('gt', self.value, other.value))
            return self.value > other.value

    items = [
        PredictionItem(row_index=Ordered(value), prediction=position)
        for position, value in enumerate([2, 1, 2])
    ]
    references = [weakref.ref(item.row_index) for item in items]
    result = PredictionResult(
        model='kumo-tabular', request_id='reentry', items=items
    )
    frame = result.to_pandas()
    assert frame['prediction'].tolist() == [1, 0, 2]
    observed = list(calls)
    calls.clear()
    monkeypatch.setattr(_impl, 'OLD', True)
    expected = result.to_pandas()
    assert observed == calls
    assert all(
        actual is original
        for actual, original in zip(
            frame['row_index'], expected['row_index'], strict=True
        )
    )
    del frame, expected, result, items
    gc.collect()
    assert all(reference() is None for reference in references)


def test_host_columns_inferred_before_sort_and_only_index_sorted(monkeypatch):
    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)
    infer, order = ResultValues.infer_column, ResultValues.column_order
    calls = []

    def infer_record(self, encoded):
        found = infer(self, encoded)
        calls.append(('infer', found[0]))
        return found

    def order_record(self, reference, stable, ascending, missing_last):
        calls.append(('order', reference, stable, ascending, missing_last))
        return order(self, reference, stable, ascending, missing_last)

    monkeypatch.setattr(ResultValues, 'infer_column', infer_record)
    monkeypatch.setattr(ResultValues, 'column_order', order_record)
    result = PredictionResult(
        model='kumo-tabular',
        request_id='columns',
        items=[
            PredictionItem(row_index=2, forecast_step=Opaque(), prediction=2),
            PredictionItem(row_index=1, forecast_step=Opaque(), prediction=1),
        ],
    )
    assert result.to_pandas()['prediction'].tolist() == [1, 2]
    assert [call[0] for call in calls] == ['infer', 'infer', 'infer', 'order']
    assert calls[-1] == ('order', calls[0][1], True, True, True)
    calls.clear()
    result = PredictionResult(
        model='kumo-tabular',
        request_id='no-index',
        items=[
            PredictionItem(
                row_index=None,
                forecast_step=item.forecast_step,
                prediction=item.prediction,
            )
            for item in result.items
        ],
    )
    result.to_pandas()
    assert [call[0] for call in calls] == ['infer', 'infer']


@pytest.mark.parametrize('stage', ['infer_column', 'column_order'])
def test_host_column_panic_containment(monkeypatch, stage):
    from sdm_client.errors import StructuredDataError
    from sdm_core import _native
    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)

    def panic(*args):
        raise _native.CorePanic('column protocol panic control')

    monkeypatch.setattr(ResultValues, stage, panic)
    with pytest.raises(
        StructuredDataError, match='column protocol panic control'
    ) as error:
        boundary_result('item', 'row_index', 1).to_pandas()
    assert error.value.code == 'INTERNAL_ERROR'


@pytest.mark.parametrize(
    'stage,value,error_type',
    [
        ('infer_column', None, TypeError),
        ('infer_column', ('column', -1), OverflowError),
        ('column_order', None, TypeError),
        ('column_order', [-1], OverflowError),
    ],
)
def test_host_column_rejects_malformed_callback_types(
    monkeypatch, stage, value, error_type
):
    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)
    monkeypatch.setattr(ResultValues, stage, lambda *args: value)
    with pytest.raises(error_type):
        boundary_result('item', 'row_index', 1).to_pandas()


def test_host_inference_failure_precedes_sort_and_releases_columns(monkeypatch):
    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)
    original = ResultValues.infer_column
    references = []
    error = ValueError('later column inference refused')

    def infer(self, encoded):
        found = original(self, encoded)
        references.append(weakref.ref(self.values[int(found[0])]))
        if len(references) == 2:
            raise error
        return found

    def order(*args):
        pytest.fail('sorting must follow all column inference')

    monkeypatch.setattr(ResultValues, 'infer_column', infer)
    monkeypatch.setattr(ResultValues, 'column_order', order)
    result = PredictionResult(
        model='kumo-tabular',
        request_id='error-order',
        items=[
            PredictionItem(row_index=object(), forecast_step=Opaque()),
            PredictionItem(row_index=object(), forecast_step=Opaque()),
        ],
    )
    with pytest.raises(ValueError) as actual:
        result.to_pandas()
    assert actual.value is error
    error.__traceback__ = None
    del actual
    gc.collect()
    assert len(references) == 2
    assert all(reference() is None for reference in references)


def field_value(field, value):
    if field in ('probabilities', 'quantiles'):
        return {'value': value}
    if field in ('scores', 'embeddings'):
        return [value]
    if field == 'rankings':
        return [{'value': value}]
    return value


@pytest.mark.parametrize('family', VALUES)
@pytest.mark.parametrize('field', FIELDS)
@pytest.mark.parametrize(
    'operation', ['requested', 'unrelated', 'frame', 'merge']
)
def test_scalar_family_across_result_operations(
    monkeypatch, family, field, operation
):
    value = VALUES[family]()
    payload = field_value(field, value)
    item = PredictionItem(row_index=0, **{field: payload})
    result = PredictionResult(
        model='kumo-tabular', request_id='values', items=[item]
    )

    def run():
        if operation == 'requested':
            return result.absent([field, field, 'unknown'])
        if operation == 'unrelated':
            return result.absent(
                ['scores' if field != 'scores' else 'prediction']
            )
        if operation == 'merge':
            merged = PredictionResult.merge([result, result])
            assert merged.items[0] is merged.items[1] is item
            assert getattr(merged.items[0], field) is payload
            return merged.to_pandas()
        return result.to_pandas()

    monkeypatch.setattr(_impl, 'OLD', True)
    try:
        expected = run()
    except Exception as error:
        expected_error = (type(error), str(error))
        monkeypatch.setattr(_impl, 'OLD', False)
        with pytest.raises(expected_error[0]) as actual:
            run()
        assert str(actual.value) == expected_error[1]
    else:
        monkeypatch.setattr(_impl, 'OLD', False)
        actual = run()
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        else:
            assert actual == expected
    assert result.items[0] is item
    assert getattr(item, field) is payload


@pytest.mark.parametrize('old', [True, False])
def test_absence_short_circuits_and_does_not_touch_other_fields(
    monkeypatch, old
):
    class Unreadable:
        def __eq__(self, other):
            raise AssertionError('unrequested or short-circuited value read')

    monkeypatch.setattr(_impl, 'OLD', old)
    result = PredictionResult(
        model='kumo-tabular',
        request_id='short-circuit',
        items=[
            PredictionItem(row_index=0, prediction=1, scores=Unreadable()),
            PredictionItem(row_index=1, prediction=Unreadable()),
        ],
        metadata={'opaque': Opaque()},
    )
    assert result.absent(['prediction', 'unknown', 'prediction']) == ()


@pytest.mark.parametrize('old', [True, False])
def test_frame_ignores_metadata_and_retains_nested_identity(monkeypatch, old):
    monkeypatch.setattr(_impl, 'OLD', old)
    shared = {'value': Opaque()}
    shared['self'] = shared
    item = PredictionItem(
        row_index=0, prediction=shared, metadata={'value': Opaque()}
    )
    result = PredictionResult(
        model='kumo-tabular',
        request_id='nested',
        items=[item],
        metadata={'value': Opaque()},
        explanation={'value': Opaque()},
        unscored_candidates=[Opaque()],
    )
    frame = result.to_pandas()
    assert frame.at[0, 'prediction'] is shared
    assert shared['self'] is shared
    assert PredictionResult.merge([result]) is result


@pytest.mark.parametrize('old', [True, False])
def test_result_references_are_local_and_release_their_values(monkeypatch, old):
    monkeypatch.setattr(_impl, 'OLD', old)
    first = Opaque()
    second = Opaque()
    weak = weakref.ref(first)
    result = PredictionResult(
        model='kumo-tabular',
        request_id='lifetime',
        items=[
            PredictionItem(row_index=1, prediction=first, scores=[first]),
            PredictionItem(row_index=0, prediction=second, scores=[first]),
        ],
    )
    frame = result.to_pandas()
    assert frame.at[0, 'prediction'] is second
    assert frame.at[1, 'prediction'] is first
    assert frame.at[0, 'scores'][0] is frame.at[1, 'scores'][0] is first
    next_result = PredictionResult(
        model='kumo-tabular',
        request_id='next',
        items=[
            PredictionItem(row_index=0, prediction=second),
            PredictionItem(row_index=1, prediction=second),
        ],
    )
    next_frame = next_result.to_pandas()
    assert next_frame.at[1, 'prediction'] is second
    assert frame.at[1, 'prediction'] is first
    del first, frame, next_frame, result, next_result
    gc.collect()
    assert weak() is None


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize('key', ['value', '$host_ref', '$np'])
def test_merge_keeps_opaque_metadata_and_original_item_fields(
    monkeypatch, old, key
):
    monkeypatch.setattr(_impl, 'OLD', old)
    value = Opaque()
    item = PredictionItem(
        row_index=0,
        prediction=value,
        metadata={key: value},
        explanation={'value': value},
    )
    result = PredictionResult(
        model='kumo-tabular',
        request_id='merge',
        items=[item],
        metadata={key: value},
        explanation={'value': value},
    )
    merged = PredictionResult.merge([result, result])
    assert merged.items[0] is merged.items[1] is item
    assert merged.metadata[key] is value
    assert merged.explanation is result.explanation
    assert merged.items[0].metadata is item.metadata
    assert merged.items[0].explanation is item.explanation


@pytest.mark.parametrize('old', [True, False])
def test_rankings_keep_nested_values_and_nontext_or_reserved_keys(
    monkeypatch, old
):
    monkeypatch.setattr(_impl, 'OLD', old)
    value = Opaque()
    entries = [
        {1: value, ('key', 2): value},
        {'$host_ref': value},
        {'$np': value},
    ]
    result = PredictionResult(
        model='kumo-tabular',
        request_id='rankings',
        items=[PredictionItem(row_index=0, rankings=entries)],
    )
    found = result.to_pandas().at[0, 'rankings']
    assert found == entries
    assert found is not entries
    for actual, original in zip(found, entries, strict=True):
        assert actual is not original
        assert list(actual) == list(original)
        assert all(actual[key] is value for key in original)


@pytest.mark.parametrize('old', [True, False])
def test_absence_errors_follow_requested_order(monkeypatch, old):
    class Unreadable:
        def __init__(self, name):
            self.name = name

        def __eq__(self, other):
            raise ValueError(self.name)

    monkeypatch.setattr(_impl, 'OLD', old)
    result = PredictionResult(
        model='kumo-tabular',
        request_id='order',
        items=[
            PredictionItem(
                row_index=0,
                prediction=Unreadable('prediction'),
                scores=Unreadable('scores'),
            )
        ],
    )
    for requested in (['scores', 'prediction'], ['prediction', 'scores']):
        with pytest.raises(ValueError, match=f'^{requested[0]}$'):
            result.absent(requested)


ITEM_EXAMPLES = {
    'row_index': 2,
    'id': 'entity',
    'forecast_step': 3,
    'prediction': 1 + 2j,
    'probabilities': {'class': 1 + 2j},
    'scores': [1 + 2j],
    'rankings': [{'candidate': 1 + 2j}],
    'embeddings': [1 + 2j],
    'quantiles': {'0.5': 1 + 2j},
    'explanation': {'value': 1 + 2j},
    'metadata': {'value': 1 + 2j},
}
RESULT_EXAMPLES = {
    'model': 'kumo-tabular',
    'request_id': 'other-request',
    'items': [PredictionItem(row_index=2, prediction=1 + 2j)],
    'metadata': {'value': 1 + 2j},
    'explanation': {'value': 1 + 2j},
    'outputs': ['prediction', 'explanation'],
    'unscored_candidates': [1 + 2j],
}
OPERATIONS = ('construct', 'absent', 'frame', 'merge', 'merge-frame')
COLLECTIONS = (
    'item.id',
    'item.metadata',
    'item.explanation',
    'result.metadata',
    'result.explanation',
    'result.unscored_candidates',
)


def test_complete_result_field_and_family_inventory():
    assert set(ITEM_EXAMPLES) == {
        field.name for field in fields(PredictionItem)
    }
    assert set(RESULT_EXAMPLES) == set(
        inspect.signature(PredictionResult).parameters
    )
    assert len(VALUES) == 33
    assert set(FIELDS) | {'row_index', 'forecast_step'} | {
        name.removeprefix('item.')
        for name in COLLECTIONS
        if name.startswith('item.')
    } == set(ITEM_EXAMPLES)
    assert {'model', 'request_id', 'items', 'outputs'} | {
        name.removeprefix('result.')
        for name in COLLECTIONS
        if name.startswith('result.')
    } == set(RESULT_EXAMPLES)


def boundary_result(level, field, payload):
    item_args = {'row_index': 0, 'prediction': 1}
    if level == 'item':
        item_args[field] = payload
    result_args = {
        'model': 'kumo-tabular',
        'request_id': 'boundary',
        'items': [PredictionItem(**item_args)],
    }
    if level == 'result':
        result_args[field] = payload
    return PredictionResult(**result_args)


def boundary_operation(result, operation):
    if operation.startswith('merge'):
        merged = PredictionResult.merge([result, result])
        assert all(
            actual is original
            for actual, original in zip(
                merged.items, (*result.items, *result.items), strict=True
            )
        )
        assert list(merged.metadata) == list(
            dict(
                result.metadata,
                batches=2,
                batch_request_ids=['boundary', 'boundary'],
            )
        )
        assert all(
            merged.metadata[key] is value
            for key, value in result.metadata.items()
            if key not in ('batches', 'batch_request_ids')
        )
        assert merged.explanation is result.explanation
        if result.unscored_candidates:
            assert (
                merged.unscored_candidates[0] is result.unscored_candidates[0]
            )
        result = merged
    if operation.endswith('frame'):
        return result.to_pandas()
    if operation == 'absent':
        return result.absent(['prediction', 'explanation'])
    return None


def compare_boundary(monkeypatch, operation):
    monkeypatch.setattr(_impl, 'OLD', True)
    try:
        expected = operation()
    except Exception as error:
        expected_error = (type(error), str(error))
        monkeypatch.setattr(_impl, 'OLD', False)
        with pytest.raises(expected_error[0]) as actual:
            operation()
        assert str(actual.value) == expected_error[1]
    else:
        monkeypatch.setattr(_impl, 'OLD', False)
        actual = operation()
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected)
        else:
            assert actual == expected


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize(
    'level,field',
    [
        *[('item', field) for field in ITEM_EXAMPLES],
        *[('result', field) for field in RESULT_EXAMPLES],
    ],
)
def test_complete_field_operations(monkeypatch, level, field, operation):
    payload = (ITEM_EXAMPLES if level == 'item' else RESULT_EXAMPLES)[field]
    compare_boundary(
        monkeypatch,
        lambda: boundary_operation(
            boundary_result(level, field, payload), operation
        ),
    )


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('collection', COLLECTIONS)
@pytest.mark.parametrize('family', VALUES)
def test_remaining_collection_families(
    monkeypatch, collection, family, operation
):
    value = VALUES[family]()
    level, field = collection.split('.')
    payload = (
        {'value': value}
        if field in ('metadata', 'explanation')
        else [value]
        if field == 'unscored_candidates'
        else value
    )
    compare_boundary(
        monkeypatch,
        lambda: boundary_operation(
            boundary_result(level, field, payload), operation
        ),
    )


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize(
    'candidates,indices',
    [
        ([[7, 7], [7, 7]], [0, 1]),
        ([[1, True, 1.0], [1]], [0, 1, 2]),
        ([[3, 1], [1, 2]], [0, 1, 3]),
        ([[3], [1, 1, 2], [2, 4, 4]], [0, 1, 2, 3, 5, 6]),
        ([[[1]], [(1,)]], [0, 1]),
        ([[{'value': 1}], [{'value': 1}]], [0]),
    ],
)
def test_merge_part_start_multiplicity(monkeypatch, old, candidates, indices):
    monkeypatch.setattr(_impl, 'OLD', old)
    parts = [
        boundary_result('result', 'unscored_candidates', part)
        for part in candidates
    ]
    originals = [value for part in parts for value in part.unscored_candidates]
    found = PredictionResult.merge(parts).unscored_candidates
    assert len(found) == len(indices)
    assert all(
        value is originals[index]
        for value, index in zip(found, indices, strict=True)
    )


@pytest.mark.parametrize('old', [True, False])
def test_candidate_protocol_order_and_short_circuit(monkeypatch, old):
    monkeypatch.setattr(_impl, 'OLD', old)
    calls = []

    class Equal:
        def __init__(self, name):
            self.name = name

        def __eq__(self, other):
            calls.append((self.name, other.name))
            return self.name[0] == other.name[0]

    first, same, different = Equal('a-first'), Equal('a-same'), Equal('b')
    parts = [
        boundary_result('result', 'unscored_candidates', values)
        for values in ([first, first], [same, different, different])
    ]
    found = PredictionResult.merge(parts).unscored_candidates
    assert len(found) == 4
    assert found[0] is found[1] is first
    assert found[2] is found[3] is different
    assert calls == [
        ('a-first', 'a-same'),
        ('a-first', 'b'),
        ('a-first', 'b'),
        ('a-first', 'b'),
        ('a-first', 'b'),
    ]


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize('failure', ['array', 'custom'])
@pytest.mark.parametrize('validation', ['none', 'model', 'outputs'])
def test_candidate_errors_follow_validation(
    monkeypatch, old, failure, validation
):
    from sdm_client.errors import StructuredDataError

    class RefusesEquality:
        def __eq__(self, other):
            raise ValueError('candidate equality refused')

    monkeypatch.setattr(_impl, 'OLD', old)
    make = (lambda: np.array([1, 2])) if failure == 'array' else RefusesEquality
    parts = [
        boundary_result('result', 'unscored_candidates', [make()])
        for _ in range(2)
    ]
    if validation == 'model':
        parts[1].model = 'other-model'
    if validation == 'outputs':
        parts[1].outputs = ('scores',)
    if validation == 'none':
        message = (
            'truth value of an array'
            if failure == 'array'
            else 'candidate equality refused'
        )
        with pytest.raises(ValueError, match=message):
            PredictionResult.merge(parts)
    else:
        with pytest.raises(
            StructuredDataError,
            match=f'different {"models" if validation == "model" else "outputs"}',
        ):
            PredictionResult.merge(parts)


@pytest.mark.parametrize('old', [True, False])
def test_merge_typed_metadata_and_mutable_candidate_aliases(monkeypatch, old):
    monkeypatch.setattr(_impl, 'OLD', old)
    mapping, sequence = {'value': []}, [1, 2]
    result = boundary_result(
        'result', 'unscored_candidates', [mapping, sequence]
    )
    result.metadata = {
        1: mapping,
        '1': sequence,
        ('key', 2): mapping,
        '$host_ref': sequence,
    }
    merged = PredictionResult.merge([result, result])
    assert list(merged.metadata) == [
        1,
        '1',
        ('key', 2),
        '$host_ref',
        'batches',
        'batch_request_ids',
    ]
    assert merged.metadata[1] is merged.unscored_candidates[0] is mapping
    assert merged.metadata['1'] is merged.unscored_candidates[1] is sequence
    mapping['value'].append(3)
    sequence.append(4)
    assert merged.unscored_candidates[0]['value'] == [3]
    assert merged.unscored_candidates[1] == [1, 2, 4]


@pytest.mark.parametrize('old', [True, False])
def test_merge_explanation_truthiness_order(monkeypatch, old):
    from sdm_client.errors import StructuredDataError

    class RefusesTruth:
        def __bool__(self):
            raise ValueError('explanation truth refused')

    monkeypatch.setattr(_impl, 'OLD', old)
    first = boundary_result('result', 'explanation', RefusesTruth())
    second = boundary_result('result', 'outputs', ['scores'])
    second.model = 'other-model'
    with pytest.raises(StructuredDataError, match='different models'):
        PredictionResult.merge([first, second])
    second.model = first.model
    with pytest.raises(ValueError, match='explanation truth refused'):
        PredictionResult.merge([first, second])
    first._explanation = {'value': 1}
    second._explanation = RefusesTruth()
    second.outputs = first.outputs
    assert (
        PredictionResult.merge([first, second]).explanation is first.explanation
    )


@pytest.mark.parametrize('old', [True, False])
def test_merge_candidates_reentrant_none_comparison_and_reference_lifetime(
    monkeypatch, old
):
    monkeypatch.setattr(_impl, 'OLD', old)
    nested_value = Opaque()
    nested = boundary_result('result', 'unscored_candidates', [nested_value])

    class Reentrant:
        def __eq__(self, other):
            merged = PredictionResult.merge([nested, nested])
            assert merged.unscored_candidates[0] is nested_value
            return other is None

    candidate = Reentrant()
    reference = weakref.ref(candidate)
    first = boundary_result('result', 'unscored_candidates', [candidate])
    second = boundary_result('result', 'unscored_candidates', [None])
    merged = PredictionResult.merge([first, second])
    assert len(merged.unscored_candidates) == 1
    assert merged.unscored_candidates[0] is candidate
    later = PredictionResult.merge([nested, nested])
    assert later.unscored_candidates[0] is nested_value
    assert merged.unscored_candidates[0] is candidate
    del candidate, first, merged
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize('family', VALUES)
@pytest.mark.parametrize('operation', OPERATIONS)
def test_metadata_key_families(monkeypatch, family, operation):
    key = VALUES[family]()
    value = Opaque()

    def run():
        result = boundary_result('result', 'metadata', {key: value})
        if operation.startswith('merge'):
            result = PredictionResult.merge([result, result])
        assert next(iter(result.metadata)) is key
        assert result.metadata[key] is value
        if operation.endswith('frame'):
            return result.to_pandas()
        if operation == 'absent':
            return result.absent(['prediction'])
        return None

    compare_boundary(monkeypatch, run)


@pytest.mark.parametrize('old', [True, False])
def test_candidate_identity_short_circuits_array_equality(monkeypatch, old):
    monkeypatch.setattr(_impl, 'OLD', old)
    value = np.array([1, 2])
    result = boundary_result('result', 'unscored_candidates', [value, value])
    found = PredictionResult.merge([result, result]).unscored_candidates
    assert len(found) == 2
    assert found[0] is found[1] is value


@pytest.mark.parametrize('old', [True, False])
@pytest.mark.parametrize(
    'family', ['numpy-bytes', 'numpy-void', 'structured-void']
)
def test_merge_keeps_full_numpy_payloads(monkeypatch, old, family):
    monkeypatch.setattr(_impl, 'OLD', old)
    value = (
        VALUES[family]()
        if family == 'structured-void'
        else (
            np.bytes_(b'a\x00b\x00')
            if family == 'numpy-bytes'
            else np.void(b'a\x00b\x00')
        )
    )
    payload = value.tobytes()
    result = boundary_result('result', 'unscored_candidates', [value])
    found = PredictionResult.merge([result, result]).unscored_candidates[0]
    assert found is value
    assert found.dtype == value.dtype
    assert found.tobytes() == payload


def test_result_references_do_not_change_wire_encoding():
    import json

    from sdm_core.answers import ResultValues, result_json

    value = 1 + 2j
    result = boundary_result('result', 'unscored_candidates', [value])
    references = ResultValues()
    encoded = json.loads(result_json(result, references=references))
    assert '$host_ref' in encoded['unscored_candidates'][0]
    with pytest.raises(TypeError, match='no plain form for complex'):
        result_json(result)


@pytest.mark.parametrize('old', [True, False])
def test_metadata_copy_does_not_rehash_keys(monkeypatch, old):
    monkeypatch.setattr(_impl, 'OLD', old)

    class Key:
        readable = True

        def __hash__(self):
            if not self.readable:
                raise ValueError('metadata key must not be rehashed')
            return 41

    key = Key()
    result = boundary_result('result', 'metadata', {key: Opaque()})
    key.readable = False
    found = PredictionResult.merge([result, result])
    assert next(iter(found.metadata)) is key


@pytest.mark.parametrize('old', [True, False])
def test_metadata_assignment_errors_keep_identity_and_precede_explanation(
    monkeypatch, old
):
    from sdm_client.errors import StructuredDataError

    monkeypatch.setattr(_impl, 'OLD', old)
    failure = ValueError('metadata assignment refused')

    class Key:
        def __hash__(self):
            return hash('batches')

        def __eq__(self, other):
            raise failure

    class Unreadable:
        def __bool__(self):
            raise AssertionError('metadata must precede explanation')

    first = boundary_result('result', 'metadata', {Key(): 1})
    first._explanation = Unreadable()
    second = boundary_result('result', 'model', 'other-model')
    with pytest.raises(StructuredDataError, match='different models'):
        PredictionResult.merge([first, second])
    second.model = first.model
    with pytest.raises(ValueError) as error:
        PredictionResult.merge([first, second])
    assert error.value is failure


@pytest.mark.parametrize('operation', ['metadata', 'equal', 'truth'])
def test_native_merge_refuses_missing_host_protocol(operation):
    import json

    from sdm_core import _native

    result = {'model': 'model', 'request_id': 'request', 'items': []}
    reference = {'$host_ref': '0'}
    if operation == 'metadata':
        result['metadata'] = reference
    elif operation == 'truth':
        result['explanation'] = reference
    else:
        result['unscored_candidates'] = [reference]
    with pytest.raises(_native.CoreError, match='requires host'):
        _native.merge([json.dumps(result), json.dumps(result)])


def test_host_protocol_callbacks_hold_the_gil(monkeypatch):
    import ctypes

    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)
    calls = []

    def guarded(name, original):
        def call(*args):
            assert ctypes.pythonapi.PyGILState_Check() == 1
            calls.append(name)
            return original(*args)

        return call

    for name in ('equal', 'truth', 'metadata_entry'):
        monkeypatch.setattr(
            ResultValues, name, guarded(name, getattr(ResultValues, name))
        )
    result = boundary_result('result', 'unscored_candidates', [Opaque()])
    result._explanation = {'value': Opaque()}
    PredictionResult.merge([result, result])
    assert calls == ['metadata_entry', 'metadata_entry', 'truth', 'equal']


def test_merge_protocol_failure_maps_core_panic(monkeypatch):
    from sdm_client.errors import StructuredDataError
    from sdm_core import _native
    from sdm_core.answers import ResultValues

    monkeypatch.setattr(_impl, 'OLD', False)

    def panic(*args):
        raise _native.CorePanic('merge protocol panic control')

    monkeypatch.setattr(ResultValues, 'equal', panic)
    result = boundary_result('result', 'unscored_candidates', [Opaque()])
    with pytest.raises(
        StructuredDataError, match='merge protocol panic control'
    ) as error:
        PredictionResult.merge([result, result])
    assert error.value.code == 'INTERNAL_ERROR'


INTEGER_VALUES = {
    'none': lambda: None,
    'na': lambda: pd.NA,
    'nan': lambda: float('nan'),
    'zero': lambda: 0,
    'negative': lambda: -3,
    'signed-min': lambda: -(2**63),
    'signed-max': lambda: 2**63 - 1,
    'unsigned-max': lambda: 2**64 - 1,
    'wide': lambda: 2**70,
    'negative-wide': lambda: -(2**70),
    'big': lambda: 2**200,
    'negative-big': lambda: -(2**200),
    **{
        dtype: lambda dtype=dtype: np.dtype(dtype).type(2)
        for dtype in (
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint16',
            'uint32',
            'uint64',
        )
    },
}


@pytest.mark.parametrize('family', INTEGER_VALUES)
@pytest.mark.parametrize('field', ['row_index', 'forecast_step'])
@pytest.mark.parametrize('operation', OPERATIONS)
def test_structural_integer_families(monkeypatch, family, field, operation):
    value = INTEGER_VALUES[family]()
    compare_boundary(
        monkeypatch,
        lambda: boundary_operation(
            boundary_result('item', field, value), operation
        ),
    )


@pytest.mark.parametrize(
    'values',
    [
        [2**53 + 1, 2**53, 2**53 + 1],
        [2**63 + 1, 2**63, 2**63 + 1],
        [2**70 + 1, 2**70, 2**70 + 1],
        [2**200, -(2**200), 0, 2**200],
        [np.int8(2), np.int8(-3), np.int8(2)],
        [np.uint64(2**64 - 1), np.uint64(2**64 - 2), np.uint64(2**64 - 1)],
        [2**53 + 1, None, 2**53, 2**53 + 1],
        [2**200, None, -(2**200), 2**200],
        [np.int8(2), None, np.int8(-3), np.int8(2)],
        [np.uint64(2**64 - 1), None, np.uint64(2**64 - 2)],
        [np.int8(2), pd.NA, np.int8(-3), np.int8(2)],
        [2**200, pd.NA, -(2**200), 2**200],
        [np.int8(2), float('nan'), np.int8(-3), np.int8(2)],
    ],
)
@pytest.mark.parametrize('field', ['row_index', 'forecast_step'])
@pytest.mark.parametrize('merged', [False, True])
def test_structural_integers_keep_dtype_and_stable_order(
    monkeypatch, values, field, merged
):
    result = PredictionResult(
        model='kumo-tabular',
        request_id='integers',
        items=[
            PredictionItem(
                **{'row_index': index, field: value, 'prediction': index}
            )
            for index, value in enumerate(values)
        ],
    )

    def run():
        found = PredictionResult.merge([result, result]) if merged else result
        return found.to_pandas()

    compare_boundary(monkeypatch, run)
