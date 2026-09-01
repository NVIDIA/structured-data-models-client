# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from predictive_query.schema.typing import Dtype, Stype
from sdm_relational_engine.rfm import LocalTable


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'user_id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25.0, 30.0, 35.0, 40.0, 45.0],
            'created_at': pd.date_range('2023-01-01', periods=5),
            'is_active': [True, True, False, True, False],
        }
    )


def test_init(sample_df: pd.DataFrame) -> None:
    table = LocalTable(
        df=sample_df,
        name='users',
        primary_key='user_id',
        time_column='created_at',
    )

    assert table.primary_key is not None
    assert table.primary_key.name == 'user_id'
    assert table.primary_key.stype == Stype.ID
    assert table.time_column is not None
    assert table.time_column.name == 'created_at'
    assert table.time_column.stype == Stype.timestamp

    assert len(table.columns) == 5
    assert all(col.name in sample_df.columns for col in table.columns)

    assert all(col.dtype is not None for col in table.columns)
    assert all(col.stype is not None for col in table.columns)


def test_infer_metadata(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users').infer_metadata()

    assert table.primary_key is not None
    assert table.primary_key.name == 'user_id'
    assert table.primary_key.dtype == Dtype.int
    assert table.primary_key.stype == Stype.ID
    assert table.time_column is not None
    assert table.time_column.name == 'created_at'
    assert table.time_column.dtype == Dtype.date
    assert table.time_column.stype == Stype.timestamp


def test_pkey_overwrite(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users').infer_metadata()

    assert table.primary_key is not None
    assert table.primary_key.name == 'user_id'
    assert table['name'].stype == Stype.categorical
    assert table['name'].dtype == Dtype.string

    table.primary_key = 'name'
    assert table.primary_key is not None
    assert table.primary_key.name == 'name'
    assert table.primary_key.stype == Stype.ID
    assert table.primary_key.dtype == Dtype.string


def test_no_primary_key(sample_df: pd.DataFrame) -> None:
    sample_df = sample_df.drop(columns=['user_id'])

    table = LocalTable(df=sample_df, name='users').infer_metadata()
    assert table.primary_key is None


def test_no_time_column(sample_df: pd.DataFrame) -> None:
    sample_df = sample_df.drop(columns=['created_at'])

    table = LocalTable(sample_df, name='users').infer_metadata()
    assert table.time_column is None


def test_time_column() -> None:
    sample_df = pd.DataFrame(
        {
            'TIME': ['1990-01-01'],
            'CREATE': ['1991-01-01'],
        }
    )
    table = LocalTable(sample_df, name='users').infer_metadata()
    assert table['TIME'].stype == Stype.timestamp
    assert table['CREATE'].stype == Stype.timestamp
    assert table.time_column is not None
    assert table.time_column.name == 'CREATE'


def test_no_end_time_column(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users').infer_metadata()
    assert table.end_time_column is None


def test_end_time_column() -> None:
    sample_df = pd.DataFrame(
        {
            'CREATE': ['1991-01-01'],
            'END': ['1992-01-01'],
        }
    )
    table = LocalTable(sample_df, name='users').infer_metadata()
    assert table['CREATE'].stype == Stype.timestamp
    assert table['END'].stype == Stype.timestamp
    assert table.time_column is not None
    assert table.time_column.name == 'CREATE'
    table.end_time_column = 'END'
    assert table.end_time_column is not None
    assert table.end_time_column.name == 'END'

    with pytest.raises(ValueError, match='defined to be an end time column'):
        table.time_column = 'END'

    with pytest.raises(ValueError, match='defined to be a time column'):
        table.end_time_column = 'CREATE'


def test_data_validation() -> None:
    with pytest.raises(ValueError, match='is empty'):
        LocalTable(pd.DataFrame(), name='users')

    df = pd.DataFrame({'id': [1, 2, 3]})[['id', 'id']]
    with pytest.raises(ValueError, match='must have unique column names'):
        LocalTable(df, name='users')

    df = pd.DataFrame({'': [1, 2, 3]})
    with pytest.raises(ValueError, match='must have non-empty column names'):
        LocalTable(df, name='users')

    df = pd.DataFrame({'A  B C': [1, 2, 3]})
    table = LocalTable(df, name='users')
    assert table.has_column('A  B C')


def test_set_name(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users')

    with pytest.raises(AttributeError):
        table['created_at'].name = 'created-at'  # type: ignore


def test_set_dtype(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users')

    with pytest.raises(AttributeError):
        table['created_at'].dtype = Dtype.int  # type: ignore


def test_wrong_time_stype(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users')

    with pytest.raises(ValueError, match='incompatible semantic type'):
        table['created_at'].stype = Stype.text


def test_wrong_pkey_stype(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users').infer_metadata()

    assert table.primary_key is not None
    assert table.primary_key.name == 'user_id'
    with pytest.raises(ValueError, match="must have 'ID' semantic type"):
        table['user_id'].stype = Stype.categorical


def test_change_stype(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users')

    with pytest.raises(ValueError, match='incompatible semantic type'):
        table['age'].stype = Stype.text

    table['age'].stype = Stype.categorical
    assert table['age'].stype == Stype.categorical


def test_add_remove_column(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users')

    assert 'user_id' in table
    del table['user_id']
    assert 'user_id' not in table
    table.add_column('user_id')
    assert 'user_id' in table


def test_metadata(sample_df: pd.DataFrame) -> None:
    table = LocalTable(sample_df, name='users').infer_metadata()
    table['name'].stype = Stype.text
    table['age'].stype = Stype.numerical

    pd.testing.assert_frame_equal(
        table.metadata,
        pd.DataFrame(
            {
                'Name': ['user_id', 'name', 'age', 'created_at', 'is_active'],
                'Data Type': ['int', 'string', 'float', 'date', 'bool'],
                'Semantic Type': [
                    'ID',
                    'text',
                    'numerical',
                    'timestamp',
                    'categorical',
                ],
                'Primary Key': [True, False, False, False, False],
                'Time Column': [False, False, False, True, False],
                'End Time Column': [False, False, False, False, False],
            }
        ),
    )
