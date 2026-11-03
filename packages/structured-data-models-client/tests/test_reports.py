# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

from sdm_client import Column, PredictionResult, Schema, Table
from sdm_client.errors import StructuredDataError


def test_metadata_reports_are_detached_and_keep_order():
    import pandas as pd

    schema = Schema(
        {
            'users': Table(
                'users',
                pd.DataFrame({'id': [1]}),
                columns={'id': Column('id', 'int64', 'ID')},
                primary_key='id',
            )
        }
    )
    report = schema.metadata_report()
    assert report.tables[0].primary_key == ('id',)
    assert report.to_pandas().to_dict('records') == [
        dict(
            table='users',
            column='id',
            dtype='int64',
            stype='ID',
            primary_key=True,
            time_column=False,
        )
    ]
    schema.tables['users'].columns['id'].dtype = 'string'
    assert report.tables[0].columns[0].dtype == 'int64'
    with pytest.raises(FrozenInstanceError):
        report.tables[0].name = 'other'
    assert list(schema.link_report().to_pandas().columns) == [
        'source_table',
        'source_column',
        'target_table',
        'target_column',
    ]


def result(payload):
    return PredictionResult(
        model='kumo-relational', request_id='r', items=[], explanation=payload
    )


def test_view_preserves_raw_and_freezes_only_explanation():
    payload = {
        'format': 'kumo_rfm_v2_1',
        'summary': 'Why',
        'unknown': None,
        'details': {'warning': 'Careful', 'cohorts': [{'nested': [1]}]},
    }
    answer = result(payload)
    view = answer.explanation_view
    assert answer.explanation is payload
    assert view.summary == 'Why'
    assert view.warnings == ('Careful',)
    assert view.raw['unknown'] is None
    assert view.details['cohorts'] == view.cohorts
    payload['details']['cohorts'][0]['nested'].append(2)
    assert view.cohorts[0]['nested'] == (1,)
    with pytest.raises(TypeError):
        view.raw['summary'] = 'changed'
    assert view.to_text() == 'Why\nWarning: Careful'
    assert view.feature_importance() is None
    assert result(None).explanation_view is None


def test_malformed_fields_are_lazy_and_unknown_formats_are_not_attribution():
    view = result(
        {'summary': 1, 'cohorts': [], 'format': 'future'}
    ).explanation_view
    assert view.cohorts == ()
    assert view.feature_importance() is None
    with pytest.raises(StructuredDataError) as error:
        _ = view.summary
    assert error.value.code == 'INVALID_REQUEST'
    assert error.value.details['path'] == '$.summary'


def test_non_json_cycle_and_non_string_key_rejected():
    for payload in ({'x': object()}, {1: 'a'}):
        with pytest.raises(StructuredDataError):
            _ = result(payload).explanation_view
    payload = {}
    payload['cycle'] = payload
    with pytest.raises(StructuredDataError):
        _ = result(payload).explanation_view
