# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen public StringDtype constructor counterparts, without materialization."""

from __future__ import annotations

import pickle
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from test_t42_frames import PATHS, describe, projection

from sdm_client import Relationship, Schema, StructuredDataError
from sdm_client.core.warnings import SDMWarning


@pytest.fixture(
    params=[
        ('python', 'NA'),
        ('pyarrow', 'NA'),
        ('python', 'nan'),
        ('pyarrow', 'nan'),
    ]
)
def string_dtype(request):
    storage, missing = request.param
    if missing == 'nan':
        if int(pd.__version__.split('.')[0]) < 3:
            pytest.skip('StringDtype(na_value=nan) requires pandas 3')
        return pd.StringDtype(storage=storage, na_value=np.nan)
    return pd.StringDtype(storage=storage)


def paired(monkeypatch, frames, **options):
    old, old_notices = describe(monkeypatch, frames, core=False, **options)
    new, new_notices = describe(monkeypatch, frames, core=True, **options)
    assert projection(new) == projection(old)
    assert new_notices == old_notices
    return new, new_notices


@pytest.mark.parametrize(
    'values',
    [
        [],
        [None, pd.NA, np.nan],
        ['a', 'b', None],
        ['', 'a\x00b', 'é', '😺'],
        ['same'] * 3,
        [str(n) for n in range(32)],
        [str(n) for n in range(64)],
        [str(n) for n in range(65)],
    ],
)
def test_string_metadata_snapshots_and_original_data(
    monkeypatch, string_dtype, values
):
    frame = pd.DataFrame({'value': pd.Series(values, dtype=string_dtype)})
    schema, _ = paired(monkeypatch, {'rows': frame})
    assert schema['rows'].data is frame
    assert schema['rows']['value'].dtype == 'string'
    assert not schema['rows']['value'].untyped
    assert schema['rows']['value']._source is None
    snapshot = schema.snapshot()
    before = projection(snapshot)
    assert projection(pickle.loads(pickle.dumps(snapshot))) == before
    assert snapshot['rows'].data is frame
    if len(frame):
        frame['value'] = pd.Series(['changed'] * len(frame), dtype=string_dtype)
        fresh, _ = paired(monkeypatch, {'rows': frame})
        assert fresh['rows']['value'].stype == 'categorical'
        assert projection(snapshot) == before


@pytest.mark.parametrize(
    'key', ['user_id', ['user_id'], ('user_id',), ('user_id', 'group')]
)
@pytest.mark.parametrize('variant', ['unique', 'repeat', 'null'])
def test_string_key_forms_exact_examples_and_errors(
    monkeypatch, string_dtype, key, variant
):
    values = [str(n) for n in range(100)]
    if variant == 'repeat':
        values[-1] = values[-2]
    elif variant == 'null':
        values[-1] = None
    frame = pd.DataFrame(
        {
            'user_id': pd.Series(values, dtype=string_dtype),
            'group': pd.Series(values, dtype=string_dtype),
        }
    )
    if variant == 'null':
        outcomes = []
        for core in (False, True):
            monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with pytest.raises(StructuredDataError) as error:
                    Schema.from_data(
                        {'users': frame}, primary_keys={'users': key}
                    )
            outcomes.append(
                (
                    error.value.code,
                    str(error.value),
                    [(str(n.message), n.category) for n in caught],
                )
            )
        assert outcomes[1] == outcomes[0]
        assert outcomes[0][2][0][1] is SDMWarning
    else:
        paired(monkeypatch, {'users': frame}, primary_keys={'users': key})


def test_string_inferred_and_explicit_relationships(monkeypatch, string_dtype):
    frames = {
        'users': pd.DataFrame(
            {'user_id': pd.Series(['a', 'b', '😺'], dtype=string_dtype)}
        ),
        'orders': pd.DataFrame(
            {
                'order_id': [1, 2, 3],
                'user_id': pd.Series(['a', None, '😺'], dtype=string_dtype),
            }
        ),
    }
    for options in (
        {},
        {
            'relationships': [
                Relationship('orders', 'user_id', 'users', 'user_id')
            ]
        },
        {'relationships': []},
    ):
        paired(monkeypatch, frames, **options)


DATE_CASES = [
    ['2024-01-01', '2024-01-02', None],
    ['15/03/2024', '16/03/2024'],
    ['01/02/2024', '15/03/2024'],
    ['01/02/2024', '03/15/2024'],
    ['01/02/2024', '02/03/2024'],
    ['15/03/2024', '03/15/2024'],
    ['2024-02-31', '2024-01-02'],
    ['1900-01-01', '2100-12-31'],
    ['1800-01-01', '2024-01-02'],
    ['2024-01-01', '2024/01/02'],
    ['2024-01-01', '15/03/2024'],
    ['2024-01-01', '2024-01-02T01:00'],
    [' 2024-01-01 ', '2024-01-02'],
    ['2024-01-01T01:00+01:00', '2024-01-02T01:00+02:00'],
    ['2024-01-01T01:00+01:00', '2024-01-02T01:00'],
    ['2024-01-01T01:00 UTC', '2024-01-02T01:00 UTC'],
    ['٢٠٢٤-٠١-٠١', '٢٠٢٤-٠١-٠٢'],
    ['2024-01-01T01:00+01:99', '2024-01-02T01:00+01:99'],
    ['2024-01-01T01:00:60', '2024-01-02T01:00:60'],
    ['2024-01-01T01:00:00.123456789123', '2024-01-02T01:00:00.123456789123'],
    ['2024/01-01', '2024/01-02'],
    ['15/03/24', '16/03/24'],
    ['01/02/2024'] * 1000 + ['03/15/2024'],
    ['2024-01-01'] * 1000 + ['junk'],
]


@pytest.mark.parametrize('values', DATE_CASES)
def test_string_date_notices_read_whole_column_at_final_stage(
    monkeypatch, string_dtype, values
):
    frame = pd.DataFrame({'value': pd.Series(values, dtype=string_dtype)})
    paired(monkeypatch, {'rows': frame})
    # Naming text as a time column is a constructor assertion, even for
    # invalid dates. Its later consumption belongs to materialization.
    schema, notices = paired(
        monkeypatch, {'rows': frame}, time_columns={'rows': 'value'}
    )
    assert schema['rows'].time_column == 'value'
    assert not notices


@pytest.mark.parametrize(
    'values', [[], [None, pd.NA, np.nan], ['ordinary', 'text']]
)
def test_string_time_override_does_not_parse_cells(
    monkeypatch, string_dtype, values
):
    frame = pd.DataFrame({'value': pd.Series(values, dtype=string_dtype)})
    schema, notices = paired(
        monkeypatch, {'rows': frame}, time_columns={'rows': 'value'}
    )
    assert schema['rows']['value'].stype == 'timestamp'
    assert not notices


def test_string_final_notice_order_and_date_candidate_exclusion(
    monkeypatch, string_dtype
):
    # Date advice precedes the unlinkable-table advice and categorical-size
    # warning. A date column already reported is excluded as a key candidate.
    dates = [f'2024-01-{day:02}' for day in range(1, 29)] + [
        f'2024-02-{day:02}' for day in range(1, 13)
    ]
    frames = {
        'first': pd.DataFrame(
            {
                'date': pd.Series(dates, dtype=string_dtype),
                'label': pd.Series(
                    [str(n) for n in range(40)], dtype=string_dtype
                ),
            }
        ),
        'second': pd.DataFrame(
            {
                'date': pd.Series(dates, dtype=string_dtype),
                'label': pd.Series(
                    [str(n) for n in range(40)], dtype=string_dtype
                ),
            }
        ),
    }
    _, notices = paired(monkeypatch, frames)
    assert 'holds dates but is stored as text' in notices[0]
    assert 'holds dates but is stored as text' in notices[1]


def test_string_table_notice_survives_later_error_but_date_notice_does_not(
    monkeypatch, string_dtype
):
    first = pd.DataFrame(
        {
            'user_id': pd.Series(
                [str(n) for n in range(99)] + [None], dtype=string_dtype
            ),
            'date': pd.Series(['2024-01-01'] * 100, dtype=string_dtype),
        }
    )
    later = pd.DataFrame([[1, 2]], columns=['same', 'same'])
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(StructuredDataError) as error:
                Schema.from_data(
                    {'users': first, 'later': later},
                    primary_keys={'users': 'missing'},
                )
        outcomes.append(
            (
                error.value.code,
                str(error.value),
                [(str(n.message), n.category) for n in caught],
            )
        )
    assert outcomes[1] == outcomes[0]
    assert len(outcomes[0][2]) == 1
    assert 'have no value' in outcomes[0][2][0][0]


@pytest.mark.parametrize(
    'values',
    [
        ['\ud800'],
        ['\udfff'],
        ['valid', '\ud800', None],
        ['é', '😺', 'a\0b'],
    ],
)
def test_string_native_unicode_keeps_exact_constructor_outcome(
    monkeypatch, values
):
    frame = pd.DataFrame(
        {'value': pd.Series(values, dtype=pd.StringDtype(storage='python'))}
    )
    outcomes = []
    for core in (False, True):
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                schema = Schema.from_data({'rows': frame})
            except UnicodeEncodeError as error:
                outcome = ('error', type(error), str(error))
            else:
                snapshot = schema.snapshot()
                assert schema['rows'].data is frame
                assert snapshot['rows'].data is frame
                pd.testing.assert_series_equal(
                    schema['rows'].data['value'], frame['value']
                )
                outcome = ('schema', projection(schema), projection(snapshot))
        outcomes.append(
            (outcome, [(str(n.message), n.category) for n in caught])
        )
    # Frozen CP310/pandas2 accepts these exact native values. Other supported
    # native execution cells can reach an actual later UTF8 conversion failure.
    # Both successes and genuine failures must keep their complete outcome.
    assert outcomes[1] == outcomes[0]


def test_string_date_advice_warning_as_error_stops_later_advice(
    monkeypatch, string_dtype
):
    import sdm_client.schema as schema_module

    original = schema_module._warn_dates_stored_as_text
    frames = {
        name: pd.DataFrame(
            {
                'date': pd.Series(
                    ['2024-01-01', '2024-01-02'], dtype=string_dtype
                )
            }
        )
        for name in ('first', 'second')
    }
    outcomes = []
    for core in (False, True):
        touched = []

        def observe(table, touched=touched):
            touched.append(table.name)
            return original(table)

        monkeypatch.setattr(
            schema_module, '_warn_dates_stored_as_text', observe
        )
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings():
            warnings.simplefilter('error', SDMWarning)
            with pytest.raises(SDMWarning) as error:
                Schema.from_data(frames)
        outcomes.append((str(error.value), touched))
    assert outcomes[1] == outcomes[0]
    assert outcomes[0][1] == ['first']


@pytest.mark.parametrize('label_elsewhere', [False, True])
def test_string_date_advice_prunes_only_diagnostic_key_candidates(
    monkeypatch, string_dtype, label_elsewhere
):
    dates = ['2024-01-01', '2024-01-02']
    frames = {
        'first': pd.DataFrame(
            {
                'date': pd.Series(dates, dtype=string_dtype),
                'label': pd.Series(['a', 'b'], dtype=string_dtype),
            }
        ),
        'second': pd.DataFrame({'date': pd.Series(dates, dtype=string_dtype)}),
    }
    if label_elsewhere:
        frames['third'] = pd.DataFrame(
            {'label': pd.Series(['a', 'b'], dtype=string_dtype)}
        )
    schema, notices = paired(monkeypatch, frames)
    assert all(table.primary_key is None for table in schema.tables.values())
    assert not schema.relationships
    key_advice = [
        notice for notice in notices if 'has no primary key' in notice
    ]
    if label_elsewhere:
        assert len(key_advice) == 2
        assert all("'label' identifies" in notice for notice in key_advice)
        assert all('another table too' in notice for notice in key_advice)
    else:
        assert not key_advice


@pytest.mark.parametrize(
    'warning_policy', ['always', 'all_errors', 'link_error']
)
def test_string_advice_keeps_link_validation_and_grouped_time_notice_order(
    monkeypatch, string_dtype, warning_policy
):
    frames = {
        'users': pd.DataFrame(
            {'user_id': pd.Series(['a', 'b'], dtype=string_dtype)}
        ),
        'orders': pd.DataFrame(
            {
                'order_id': [10, 11],
                'user_id': [1, 2],
                'date': pd.Series(
                    ['2024-01-01', '2024-01-02'], dtype=string_dtype
                ),
            }
        ),
        'earlier': pd.DataFrame(
            {
                'one': pd.to_datetime(['2024-01-01', '2024-01-02']),
                'two': pd.to_datetime(['2025-01-01', '2025-01-02']),
            }
        ),
        'later': pd.DataFrame(
            {
                'one': pd.to_datetime(['2024-01-01', '2024-01-02']),
                'two': pd.to_datetime(['2025-01-01', '2025-01-02']),
            }
        ),
    }
    _, notices = paired(monkeypatch, frames)
    assert 'orders.user_id' in notices[0]
    assert any('earlier' in notice and 'later' in notice for notice in notices)
    if warning_policy == 'always':
        return

    import sdm_client.schema as schema_module

    original = schema_module._warn_dates_stored_as_text
    outcomes = []
    for core in (False, True):
        touched = []

        def observe(table, touched=touched):
            touched.append(table.name)
            return original(table)

        monkeypatch.setattr(
            schema_module, '_warn_dates_stored_as_text', observe
        )
        monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if warning_policy == 'all_errors':
                warnings.simplefilter('error', SDMWarning)
            else:
                warnings.filterwarnings(
                    'error', message=re.escape(notices[0]), category=SDMWarning
                )
            with pytest.raises(SDMWarning) as error:
                Schema.from_data(frames)
        outcomes.append(
            (str(error.value), [str(n.message) for n in caught], touched)
        )
    assert outcomes[0] == (notices[0], [], [])
    assert outcomes[1] == outcomes[0]


def test_string_advice_never_reenters_python_schema_inference(
    monkeypatch, string_dtype
):
    import sdm_client.schema as schema_module

    frame = pd.DataFrame(
        {'date': pd.Series(['2024-01-01', '2024-01-02'], dtype=string_dtype)}
    )
    old, old_notices = describe(monkeypatch, {'rows': frame}, core=False)

    def forbidden(*args, **kwargs):
        raise AssertionError('native advice reentered Python schema inference')

    monkeypatch.setattr(
        schema_module, '_infer_table_and_time_column', forbidden
    )
    monkeypatch.setattr(schema_module, '_infer_primary_key', forbidden)
    monkeypatch.setattr(schema_module, '_infer_relationships', forbidden)
    new, new_notices = describe(monkeypatch, {'rows': frame}, core=True)
    assert projection(new) == projection(old)
    assert new_notices == old_notices
