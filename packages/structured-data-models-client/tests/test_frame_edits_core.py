# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame schema edits use shared policy with fresh values and explicit origin."""

import copy
import pickle

import pandas as pd
import pytest

from sdm_client import Relationship, Schema
from sdm_client.core.warnings import SDMWarning
from sdm_client.errors import StructuredDataError
from sdm_client.schema import Column


def schema():
    return Schema.from_data(
        {
            'users': pd.DataFrame(
                {'id': range(100), 'other': range(100), 'value': [0.5] * 100}
            ),
            'events': pd.DataFrame({'id': range(100), 'user': range(100)}),
        },
        primary_keys={'users': 'id', 'events': 'id'},
        relationships=[],
    )


def forbid_legacy(monkeypatch):
    def old(*args, **kwargs):
        raise AssertionError('legacy schema edit policy ran')

    monkeypatch.setattr(Column, '_check_stype', old)
    monkeypatch.setattr(Column, '_check_supports_dtype', old)
    monkeypatch.setattr(Schema, '_settle', old)
    monkeypatch.setattr(Schema, '_check_relationships', old)


def test_frame_type_edit_uses_core_and_invalid_edit_is_atomic(monkeypatch):
    value = schema()
    forbid_legacy(monkeypatch)
    value['users']['value'].stype = 'categorical'
    assert value['users']['value'].stype == 'categorical'
    with pytest.raises(StructuredDataError):
        value['users']['value'].stype = 'not-a-type'
    assert value['users']['value'].stype == 'categorical'
    assert all(
        column._source is None
        for table in value.tables.values()
        for column in table.columns.values()
    )


def test_frame_key_edits_observe_current_values_and_leave_failed_edits_unchanged(
    monkeypatch,
):
    value = schema()
    forbid_legacy(monkeypatch)
    frame = value['users'].data
    frame.loc[:, 'other'] = 1
    with pytest.raises(StructuredDataError):
        value['users'].primary_key = 'other'
    assert value['users'].primary_key == 'id'
    frame.loc[:, 'other'] = range(100)
    frame.loc[3, 'other'] = None
    with pytest.raises(StructuredDataError):
        value['users'].primary_key = 'other'
    assert value['users'].primary_key == 'id'
    frame.loc[3, 'other'] = 3
    value['users'].primary_key = 'other'
    assert value['users'].primary_key == 'other'
    assert value['users'].duplicate_keys == 0
    value['users'].primary_key = None
    assert value['users'].duplicate_keys == 0


def test_frame_near_unique_key_keeps_observed_duplicate_count():
    value = schema()
    value['users'].data.loc[99, 'other'] = 98
    with pytest.warns(SDMWarning):
        value['users'].primary_key = 'other'
    assert value['users'].duplicate_keys == 1


@pytest.mark.parametrize(
    'clone',
    [
        copy.copy,
        copy.deepcopy,
        lambda s: s.snapshot(),
        lambda s: pickle.loads(pickle.dumps(s)),
    ],
)
def test_frame_copy_and_pickle_preserve_origin_and_edit_isolation(
    monkeypatch, clone
):
    value = schema()
    forbid_legacy(monkeypatch)
    copied = clone(value)
    copied['users']['value'].stype = 'categorical'
    assert copied._core_origin == 'frames'
    assert value['users']['value'].stype == 'numerical'
    assert copied['users']['value'] is not value['users']['value']
    assert all(
        column._source is None
        for table in copied.tables.values()
        for column in table.columns.values()
    )


def test_adopted_frame_tables_retain_origin_and_relationship_rules(monkeypatch):
    value = schema()
    forbid_legacy(monkeypatch)
    adopted = Schema(value.tables)
    link = Relationship('events', 'user', 'users', 'id')
    adopted.relationships.append(link)
    assert adopted._core_origin == 'frames'
    assert adopted['events']['user'].stype == 'ID'
    assert value['events']['user'].stype != 'ID'
    with pytest.raises(StructuredDataError):
        adopted.relationships.append(
            Relationship('events', 'missing', 'users', 'id')
        )
    assert adopted.relationships == [link]
    adopted.relationships.clear()
    assert adopted['events']['user'].stype == value['events']['user'].stype


def test_frame_time_edit_preserves_key_exclusion_and_origin(monkeypatch):
    value = schema()
    forbid_legacy(monkeypatch)
    with pytest.raises(StructuredDataError):
        value['users'].time_column = 'id'
    assert value['users'].time_column is None


def test_manual_and_core_frame_table_adoption_use_shared_edits(monkeypatch):
    from sdm_client import Table

    value = schema()
    frame = pd.DataFrame({'code': range(100), 'amount': [1.5] * 100})
    manual = Table(
        name='manual',
        data=frame,
        columns={
            'code': Column('code', 'int64', 'numerical'),
            'amount': Column('amount', 'float64', 'numerical'),
        },
    )
    forbid_legacy(monkeypatch)
    adopted = Schema({'users': value['users'], 'manual': manual})
    adopted['manual'].primary_key = 'code'
    adopted['manual']['amount'].stype = 'categorical'
    assert adopted._core_origin == 'frames'
    assert adopted['manual'].duplicate_keys == 0
    assert manual.primary_key is None
    assert manual['amount'].stype == 'numerical'
    assert all(
        column._source is None
        for table in adopted.tables.values()
        for column in table.columns.values()
    )
