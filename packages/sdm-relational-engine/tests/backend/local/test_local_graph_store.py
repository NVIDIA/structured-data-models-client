# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from predictive_query.schema.typing import Stype
from sdm_relational_engine.api.rfm.context import REV_REL
from sdm_relational_engine.graph import Edge
from sdm_relational_engine.rfm import Graph, TaskReferenceError
from sdm_relational_engine.rfm.backend.local import LocalGraphStore


def test_local_graph_store() -> None:
    df_dict = {}
    df_dict['USERS'] = pd.DataFrame(
        {
            'USER_ID': pd.Series([0, 1, pd.NA, 2, 1, 3, 4], dtype='Int64'),
            'TIME': ['1990-01-01'] * 6 + [None],
            'TEXT': [
                'this is!',
                'text',
                None,
                '',
                ' abc',
                'hello!world',
                '?. :',
            ],
        }
    )

    df_dict['ORDERS'] = pd.DataFrame(
        {
            'ORDER_ID': [str(i) for i in range(9)],
            'USER_ID': pd.Series(
                [0, 1, 4, 2, 5, pd.NA, 4, 3, 3], dtype='Int64'
            ),
            'TIME': pd.date_range('2023-01-01', periods=8)[::-1].tolist()
            + [pd.NaT],
        }
    )

    df_dict['VIEWS'] = pd.DataFrame(
        {
            'USER_ID': pd.Series(
                [0, 1, 4, 2, 5, pd.NA, 4, 3, 3], dtype='Int64'
            ),
            'TIME': pd.date_range('2023-01-01', periods=9)[::-1],
        }
    )

    df_dict['RETURNS'] = pd.DataFrame(
        {
            'ORDER_ID': ['2', '4', '7'],
        }
    )

    graph = Graph.from_data(df_dict, verbose=False)
    assert graph['USERS'].primary_key is not None
    assert graph['USERS'].primary_key.name == 'USER_ID'
    assert graph['USERS'].time_column is not None
    assert graph['USERS'].time_column.name == 'TIME'
    assert graph['ORDERS'].primary_key is not None
    assert graph['ORDERS'].primary_key.name == 'ORDER_ID'
    assert graph['ORDERS'].time_column is not None
    assert graph['ORDERS'].time_column.name == 'TIME'
    assert graph['VIEWS'].primary_key is None
    assert graph['VIEWS'].time_column is not None
    assert graph['VIEWS'].time_column.name == 'TIME'
    assert graph['RETURNS'].primary_key is None
    assert graph['RETURNS'].time_column is None

    graph['USERS']['TEXT'].stype = Stype.text
    graph['RETURNS']['ORDER_ID'].stype = Stype.text

    assert graph.edges == [
        Edge(src_table='ORDERS', fkey='USER_ID', dst_table='USERS'),
        Edge(src_table='VIEWS', fkey='USER_ID', dst_table='USERS'),
        Edge(src_table='RETURNS', fkey='ORDER_ID', dst_table='ORDERS'),
    ]

    store = LocalGraphStore(graph)
    report = store.sanitization_report
    assert report.status.value == 'available'
    assert set(report.tables) == set(df_dict)
    users_report = report.tables['USERS']
    assert users_report.input_rows == 7
    assert users_report.output_rows == 4
    assert users_report.null_primary_key_rows == 1
    assert users_report.duplicate_primary_key_rows == 1
    assert users_report.null_time_rows == 1
    assert users_report.total_dropped_rows == 3
    orders_report = report.tables['ORDERS']
    assert orders_report.input_rows == 9
    assert orders_report.output_rows == 8
    assert orders_report.null_primary_key_rows == 0
    assert orders_report.duplicate_primary_key_rows == 0
    assert orders_report.null_time_rows == 1
    for table_report in report.tables.values():
        assert table_report.total_dropped_rows == (
            table_report.null_primary_key_rows
            + table_report.duplicate_primary_key_rows
            + table_report.null_time_rows
        )

    for table_name, df in store.df_dict.items():
        # No edges are dropped:
        assert len(df) == len(df_dict[table_name])
        # Timestamps are correctly casted:
        if 'TIME' in df:
            assert pd.api.types.is_datetime64_any_dtype(df['TIME'])
        # Invalid rows are correctly detected:
        mask = store.mask_dict.get(table_name)
        if table_name == 'VIEWS':
            assert mask is None
        elif table_name == 'USERS':
            assert mask is not None
            assert np.array_equal(
                mask, np.array([True, True, False, True, False, True, False])
            )
        elif table_name == 'ORDERS':
            assert mask is not None
            assert np.array_equal(mask, np.array([True] * 8 + [False]))
        else:
            assert table_name == 'RETURNS'
            assert mask is None

    expected_edge_types = [
        ('ORDERS', 'USER_ID', 'USERS'),
        ('USERS', f'{REV_REL}USER_ID', 'ORDERS'),
        ('VIEWS', 'USER_ID', 'USERS'),
        ('USERS', f'{REV_REL}USER_ID', 'VIEWS'),
        ('RETURNS', 'ORDER_ID', 'ORDERS'),
        ('ORDERS', f'{REV_REL}ORDER_ID', 'RETURNS'),
    ]

    assert len(store.colptr_dict) == len(expected_edge_types)
    assert len(store.row_dict) == len(expected_edge_types)
    for edge_type in expected_edge_types:
        assert store.colptr_dict[edge_type].dtype == np.int64
        assert store.row_dict[edge_type].dtype == np.int64

    colptr = store.colptr_dict[('ORDERS', 'USER_ID', 'USERS')]
    row = store.row_dict[('ORDERS', 'USER_ID', 'USERS')]
    #                                        0  1  -  2  -  3, -
    assert np.array_equal(colptr, np.array([0, 1, 2, 2, 3, 3, 4, 4]))
    assert np.array_equal(row, np.array([0, 1, 3, 7]))

    colptr = store.colptr_dict[('USERS', f'{REV_REL}USER_ID', 'ORDERS')]
    row = store.row_dict[('USERS', f'{REV_REL}USER_ID', 'ORDERS')]
    #                                        0  1  -  2  -  -  -  3  -
    assert np.array_equal(colptr, np.array([0, 1, 2, 2, 3, 3, 3, 3, 4, 4]))
    assert np.array_equal(row, np.array([0, 1, 3, 5]))

    colptr = store.colptr_dict[('VIEWS', 'USER_ID', 'USERS')]
    row = store.row_dict[('VIEWS', 'USER_ID', 'USERS')]
    #                                        0  1  -  2  -  3,3-
    assert np.array_equal(colptr, np.array([0, 1, 2, 2, 3, 3, 5, 5]))
    assert np.array_equal(row, np.array([0, 1, 3, 8, 7]))

    colptr = store.colptr_dict[('USERS', f'{REV_REL}USER_ID', 'VIEWS')]
    row = store.row_dict[('USERS', f'{REV_REL}USER_ID', 'VIEWS')]
    #                                        0  1  -  2  -  -  -  3  3
    assert np.array_equal(colptr, np.array([0, 1, 2, 2, 3, 3, 3, 3, 4, 5]))
    assert np.array_equal(row, np.array([0, 1, 3, 5, 5]))

    assert len(store.pkey_map_dict) == 2
    pkey_map = store.pkey_map_dict['USERS']
    pd.testing.assert_frame_equal(
        pkey_map,
        pd.DataFrame(
            dict(arange=[0, 1, 3, 5]),
            index=pd.Index([0, 1, 2, 3], dtype='Int64', name='USER_ID'),
        ),
    )
    pkey_map = store.pkey_map_dict['ORDERS']
    pd.testing.assert_frame_equal(
        pkey_map,
        pd.DataFrame(
            dict(arange=range(8)),
            index=pd.Index([str(i) for i in range(8)], name='ORDER_ID'),
        ),
    )

    assert len(store.time_dict) == 3
    assert len(store.time_dict['USERS']) == len(df_dict['USERS'])
    assert len(store.time_dict['ORDERS']) == len(df_dict['ORDERS'])
    assert len(store.time_dict['VIEWS']) == len(df_dict['VIEWS'])
    assert store.min_max_time_dict == {
        'USERS': (pd.Timestamp('1990-01-01'), pd.Timestamp('1990-01-01')),
        'ORDERS': (pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-08')),
        'VIEWS': (pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-09')),
    }

    with pytest.raises(KeyError, match='does not exist'):
        store.get_node_id('', pd.Series([], dtype='int'))

    with pytest.raises(KeyError, match='No primary keys passed'):
        store.get_node_id('USERS', pd.Series([], dtype='int'))

    with pytest.raises(KeyError, match=r'primary keys \[4\] do not exist'):
        store.get_node_id('USERS', pd.Series([0, 1, 2, 3, 4]))

    assert np.array_equal(
        store.get_node_id('USERS', pd.Series([0, 3])),
        np.array([0, 5]),
    )

    with pytest.raises(KeyError, match=r"primary keys \['8'\] do not exist"):
        store.get_node_id('ORDERS', pd.Series([5, 8]))

    assert np.array_equal(
        store.get_node_id('ORDERS', pd.Series([5, 7])),
        np.array([5, 7]),
    )

    with pytest.raises(ValueError, match="'VIEWS' does not have a primary"):
        store.get_node_id('VIEWS', pd.Series(['x', 'y']))


def test_sanitization_reason_precedence_and_reference_count() -> None:
    graph = Graph.from_data(
        {
            'USERS': pd.DataFrame(
                {
                    'USER_ID': pd.Series([pd.NA, 1, 1, 2], dtype='Int64'),
                    'TIME': [None, None, None, '2025-01-01'],
                }
            ),
        },
        verbose=False,
    )
    store = LocalGraphStore(graph, verbose=False)
    report = store.sanitization_report.tables['USERS']

    assert report.input_rows == 4
    assert report.output_rows == 1
    assert report.null_primary_key_rows == 1
    assert report.duplicate_primary_key_rows == 1
    assert report.null_time_rows == 1

    with pytest.raises(TaskReferenceError) as exc_info:
        store.validate_entity_references(
            'USERS',
            pd.Series([2, 'not-an-integer'], dtype=object),
        )

    assert exc_info.value.unresolved_rows == 1
