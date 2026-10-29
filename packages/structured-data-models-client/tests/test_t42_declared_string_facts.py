# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native cardinality is independent of exact exported Unicode points."""

from __future__ import annotations

import pickle
import warnings

import pandas as pd
import pytest
from test_t42_frames import PATHS, projection

from sdm_client import Schema
from sdm_client._rust import frame_objects, frames


@pytest.fixture(autouse=True)
def native_string_options():
    # Preserve the successful frozen native string/date-advice options.
    with pd.option_context('future.infer_string', False):
        yield


def observe(monkeypatch, make, core, **options):
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
    frame = make()
    original = frame.copy(deep=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            schema = Schema.from_data({'rows': frame}, **options)
        except Exception as error:
            result = (
                type(error).__module__,
                type(error).__name__,
                getattr(error, 'code', None),
                str(error),
            )
        else:
            result = projection(schema)
            assert schema['rows'].data is frame
            assert (
                projection(pickle.loads(pickle.dumps(schema.snapshot())))
                == result
            )
    pd.testing.assert_frame_equal(frame, original)
    return result, [(type(item.message), str(item.message)) for item in caught]


@pytest.mark.parametrize('size', [0, 1, 2, 63, 64, 65, 66, 100])
@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('points', ['ordinary', 'surrogate', 'pair-scalar'])
@pytest.mark.parametrize('dtype', ['string-python', 'object', 'category'])
def test_string_meaning_boundaries_keep_actual_native_equality(
    monkeypatch, size, missing, points, dtype
):
    def make():
        values = [
            f'{number}'
            if points == 'ordinary'
            else f'{number}\ud800'
            if points == 'surrogate'
            else ('\ud83d\ude3a' if number % 2 else '😺') + str(number)
            for number in range(size)
        ]
        if missing:
            values.append(None)
        native_dtype = (
            pd.StringDtype(storage='python')
            if dtype == 'string-python'
            else dtype
        )
        return pd.DataFrame(
            {
                'before': range(len(values)),
                'value': pd.Series(values, dtype=native_dtype),
                'after': range(len(values)),
            }
        )

    assert observe(monkeypatch, make, True) == observe(monkeypatch, make, False)


@pytest.mark.parametrize('size', [64, 65, 66])
@pytest.mark.parametrize('missing', [False, True])
def test_measured_native_surrogate_population_is_the_exact_constructor_fact(
    monkeypatch, size, missing
):
    values = [f'{number}\ud800' for number in range(size)] + (
        [None] if missing else []
    )

    def make():
        return pd.DataFrame(
            {'value': pd.Series(values, dtype=pd.StringDtype(storage='python'))}
        )

    # Historical profiles counted nonmissing surrogate strings differently.
    # The constructor carries this runtime's actual native observation.
    frame = make()
    expected = int(frame['value'].nunique(dropna=True))
    rows, native = frames._convert('rows', frame)
    assert native['value'].descriptor == {
        'category': False,
        'meaning': {'dtype': 'string', 'untyped': False, 'distinct': expected},
    }
    assert [row['text'] for row in rows.column('value').to_pylist()] == [
        None if value is None else str.encode(value, 'utf-8', 'surrogatepass')
        for value in values
    ]
    assert observe(monkeypatch, make, True) == observe(monkeypatch, make, False)


@pytest.mark.parametrize('key', ['value_id', ['value_id'], ('value_id',)])
@pytest.mark.parametrize('missing', [False, True])
def test_declared_string_key_count_remains_fresh_and_native(
    monkeypatch, key, missing
):
    def make():
        values = [f'{number}\ud800' for number in range(100)]
        if missing:
            values[-1] = None
        return pd.DataFrame(
            {
                'value_id': pd.Series(
                    values, dtype=pd.StringDtype(storage='python')
                )
            }
        )

    options = {'primary_keys': {'rows': key}}
    assert observe(monkeypatch, make, True, **options) == observe(
        monkeypatch, make, False, **options
    )


def test_lossless_meaning_carries_the_actual_reached_native_count_without_replacing_points(
    monkeypatch,
):
    values = [f'{number}\ud800' for number in range(65)] + [
        None,
        '\ud83d\ude3a',
        '😺',
    ]
    series = pd.Series(
        values, dtype=pd.StringDtype(storage='python'), name='value'
    )
    expected = int(series.nunique(dropna=True))
    calls = []
    native_nunique = pd.Series.nunique
    native_storage = frame_objects.storage

    def nunique(source, *args, **kwargs):
        calls.append(('nunique', source.name, kwargs))
        return native_nunique(source, *args, **kwargs)

    def storage(source):
        calls.append(('storage', source.name))
        return native_storage(source)

    monkeypatch.setattr(pd.Series, 'nunique', nunique)
    monkeypatch.setattr(frame_objects, 'storage', storage)
    frame = pd.DataFrame({'value': series})
    rows, native = frames._convert('rows', frame)
    assert calls == [
        ('nunique', 'value', {'dropna': True}),
        ('storage', 'value'),
    ]
    assert native['value'].descriptor == {
        'category': False,
        'meaning': {'dtype': 'string', 'untyped': False, 'distinct': expected},
    }
    assert frames._native_dtype(native['value']) is frame['value'].dtype
    assert [row['text'] for row in rows.column('value').to_pylist()] == [
        None if value is None else str.encode(value, 'utf-8', 'surrogatepass')
        for value in values
    ]
    calls.clear()
    observed, fresh = frames._observe('rows', frame, ['value'])
    assert calls == [('storage', 'value')]
    assert observed.equals(rows)
    assert 'distinct' not in fresh['value'].descriptor['meaning']


def test_native_count_occurs_before_later_declared_string_column_is_resolved(
    monkeypatch,
):
    outcomes = []

    def run(core):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        frame = pd.DataFrame(
            {
                'first': pd.Series(
                    [f'{number}\ud800' for number in range(65)],
                    dtype=pd.StringDtype(storage='python'),
                ),
                'later': pd.Series(
                    [f'old{number}\udfff' for number in range(65)],
                    dtype=pd.StringDtype(storage='python'),
                ),
            }
        )
        trace = []
        original = pd.Series.nunique
        changed = False

        def nunique(source, *args, **kwargs):
            nonlocal changed
            trace.append(
                (
                    source.name,
                    len(source),
                    str(source.iloc[0]) if len(source) else None,
                )
            )
            result = original(source, *args, **kwargs)
            if source.name == 'first' and not changed:
                changed = True
                frame['later'] = pd.Series(
                    ['new\udfff'] * 65, dtype=pd.StringDtype(storage='python')
                )
            return result

        with (
            monkeypatch.context() as hooks,
            warnings.catch_warnings(record=True) as caught,
        ):
            hooks.setattr(pd.Series, 'nunique', nunique)
            warnings.simplefilter('always')
            schema = Schema.from_data({'rows': frame})
        outcomes.append(
            (
                projection(schema),
                trace,
                [(type(item.message), str(item.message)) for item in caught],
            )
        )

    run(False)
    run(True)
    assert outcomes[1] == outcomes[0]


def test_native_cardinality_failure_stops_before_later_source_observation(
    monkeypatch,
):
    outcomes = []

    def run(core):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        frame = pd.DataFrame(
            {
                'first': pd.Series(
                    ['\ud800'], dtype=pd.StringDtype(storage='python')
                ),
                'later': pd.Series(
                    ['\udfff'], dtype=pd.StringDtype(storage='python')
                ),
            }
        )
        native_error = RuntimeError('native distinct observation failed')
        trace = []
        original = pd.Series.nunique

        def nunique(source, *args, **kwargs):
            trace.append(source.name)
            if source.name == 'first':
                raise native_error
            return original(source, *args, **kwargs)

        with monkeypatch.context() as hooks:
            hooks.setattr(pd.Series, 'nunique', nunique)
            with pytest.raises(RuntimeError) as error:
                Schema.from_data({'rows': frame})
        assert error.value is native_error
        outcomes.append(trace)

    run(False)
    run(True)
    assert outcomes[1] == outcomes[0] == ['first']
