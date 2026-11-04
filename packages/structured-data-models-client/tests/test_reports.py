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


def test_shared_attribution_fixtures():
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / 'tests/fixtures/reports/attribution.json'
    )
    for case in json.loads(path.read_text()):
        view = result(case['payload']).explanation_view
        if 'error' in case:
            with pytest.raises(StructuredDataError) as error:
                view.feature_importance(case['normalize'])
            assert error.value.code == 'INVALID_REQUEST', case['name']
            assert error.value.details['path'] == case['error'], case['name']
        else:
            report = view.feature_importance(case['normalize'])
            if case['rows'] is None:
                assert report is None, case['name']
            else:
                assert report.normalized == case['normalize']
                assert [
                    [r.table, r.column, r.score] for r in report.rows
                ] == case['rows'], case['name']
                assert list(report.to_pandas().columns) == [
                    'table',
                    'column',
                    'score',
                ]


@pytest.mark.parametrize(
    'score',
    [float('nan'), float('inf'), -float('inf'), 10**5000],
    ids=['nan', 'inf', 'negative-inf', 'huge-int'],
)
def test_nonfinite_scores_use_shared_error(score):
    payload = {
        'format': 'kumo_rfm_v2_1',
        'details': {
            'subgraphs': [
                {'tables': {'t': {'n': {'cells': {'c': {'score': score}}}}}}
            ]
        },
    }
    with pytest.raises(StructuredDataError) as error:
        result(payload).explanation_view.feature_importance()
    assert error.value.code == 'INVALID_REQUEST'
    assert (
        error.value.details['path']
        == '$.details.subgraphs[0].tables.t.n.cells.c.score'
    )


def test_report_rendering_does_not_call_native(monkeypatch, capsys):
    import sdm_client._rust.direct as direct
    from sdm_client.reports import MetadataReport, TableMetadata

    monkeypatch.setattr(
        direct, '_call', lambda *args, **kwargs: pytest.fail('native call')
    )
    report = MetadataReport((TableMetadata('empty', (), (), None),))
    assert report.to_pandas().empty
    assert list(report.to_pandas().columns) == [
        'table',
        'column',
        'dtype',
        'stype',
        'primary_key',
        'time_column',
    ]
    report.print()
    assert capsys.readouterr().out == 'empty\n'
    result({'summary': 'local'}).explanation_view.print()
    assert capsys.readouterr().out == 'local\n'


def test_shared_metadata_fixture():
    import json
    from pathlib import Path

    from sdm_client import (
        ColumnMetadata,
        LinkMetadata,
        LinkReport,
        MetadataReport,
        TableMetadata,
    )

    path = (
        Path(__file__).resolve().parents[3]
        / 'tests/fixtures/reports/metadata.json'
    )
    fixture = json.loads(path.read_text())
    report = MetadataReport(
        tuple(
            TableMetadata(
                t['name'],
                tuple(ColumnMetadata(**c) for c in t['columns']),
                tuple(t['primary_key']),
                t['time_column'],
            )
            for t in fixture['tables']
        )
    )
    assert report.to_text() == fixture['text']
    links = LinkReport(tuple(LinkMetadata(**r) for r in fixture['links']))
    assert links.to_text() == fixture['link_text']
    assert links.to_pandas().to_dict('records') == fixture['links']


def test_source_table_metadata_does_not_read_rows_or_call_native(monkeypatch):
    import sdm_client._rust.direct as direct
    import sdm_client._rust.run as run_module
    from sdm_client import SourceTable

    table = SourceTable(
        'events',
        'warehouse.events',
        {'id': Column('id', 'int64', 'ID')},
        primary_key='id',
    )
    monkeypatch.setattr(
        direct, '_call', lambda *args, **kwargs: pytest.fail('native call')
    )
    monkeypatch.setattr(
        run_module, 'native', lambda: pytest.fail('native load')
    )
    assert table.metadata_report().primary_key == ('id',)
    assert (
        table.metadata_report().to_text()
        == 'events\n  id: int64, ID (primary key)'
    )


def test_shared_accessor_precedence_and_lazy_errors():
    import json
    from collections.abc import Mapping
    from pathlib import Path

    def plain(value):
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        return value

    path = (
        Path(__file__).resolve().parents[3]
        / 'tests/fixtures/reports/accessors.json'
    )
    for case in json.loads(path.read_text()):
        view = result(case['payload']).explanation_view
        for field in ('summary', 'warnings', 'cohorts', 'subgraphs'):
            if field in case:
                assert plain(getattr(view, field)) == case[field], case['name']
        if 'error_field' in case:
            with pytest.raises(StructuredDataError) as caught:
                getattr(view, case['error_field'])
            assert caught.value.code == 'INVALID_REQUEST', case['name']
            assert caught.value.details['path'] == case['error_path'], case[
                'name'
            ]


def test_explanation_depth_boundary_and_reused_containers():
    shared = {'value': [1]}
    view = result({'left': shared, 'right': shared}).explanation_view
    shared['value'].append(2)
    assert view.raw['left']['value'] == view.raw['right']['value'] == (1,)
    payload = None
    for _ in range(128):
        payload = {'child': payload}
    assert result(payload).explanation_view is not None
    with pytest.raises(StructuredDataError) as caught:
        _ = result({'child': payload}).explanation_view
    assert caught.value.code == 'INVALID_REQUEST'
    assert caught.value.details['path'] == '$' + '.child' * 129


def test_public_report_and_schema_rendering_accessors(capsys):
    import pandas as pd

    from sdm_client import FeatureImportance, FeatureImportanceRow, SourceTable

    expected_text = 'users\n  id: int64, ID (primary key)'
    expected_rows = [
        dict(
            table='users',
            column='id',
            dtype='int64',
            stype='ID',
            primary_key=True,
            time_column=False,
        )
    ]
    table = Table(
        'users',
        pd.DataFrame({'id': [1]}),
        columns={'id': Column('id', 'int64', 'ID')},
        primary_key='id',
    )
    source = SourceTable(
        'users',
        'warehouse.users',
        {'id': Column('id', 'int64', 'ID')},
        primary_key='id',
    )
    for owner in (table, source):
        report = owner.metadata_report()
        assert report.to_text() == expected_text
        assert report.to_pandas().to_dict('records') == expected_rows
        report.print()
        assert capsys.readouterr().out == expected_text + '\n'
        owner.print_metadata()
        assert capsys.readouterr().out == expected_text + '\n'
    schema = Schema({'users': table})
    schema.print_metadata()
    assert capsys.readouterr().out == expected_text + '\n'
    assert schema.link_report().to_text() == ''
    schema.link_report().print()
    assert capsys.readouterr().out == '\n'
    schema.print_links()
    assert capsys.readouterr().out == '\n'
    importance = FeatureImportance(
        (FeatureImportanceRow('users', 'id', -0.25),), False
    )
    assert importance.to_text() == 'users.id: -0.25'
    assert importance.to_pandas().to_dict('records') == [
        dict(table='users', column='id', score=-0.25)
    ]
    importance.print()
    assert capsys.readouterr().out == 'users.id: -0.25\n'
