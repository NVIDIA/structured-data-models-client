# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reader must not lose a value on the way in.

Every guard in the wire layer runs after the frame exists, so a connector that
rounds while parsing defeats all of them. These pin the routes a large integer
can take, and the two that used to round.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_connectors import connect, read, read_table
from sdm_connectors.sql import ConnectorError

BIG = 9_007_199_254_740_993


def wire(series: pd.Series) -> list:
    from sdm_client.wire.dtypes import infer_tfm_dtype, serialize_column

    return serialize_column(series, infer_tfm_dtype(series))


# --- CSV: a blank must not promote the column -------------------------------


def test_a_csv_integer_column_with_a_blank_stays_exact(tmp_path):
    path = tmp_path / 'ids.csv'
    path.write_text(f'id,v\n{BIG},1.0\n,2.0\n{BIG + 2},3.0\n')
    frame = read('local', path=str(path))
    assert wire(frame['id']) == [str(BIG), None, str(BIG + 2)]


def test_a_csv_without_a_blank_keeps_its_plain_dtypes(tmp_path):
    """Nothing ordinary should start looking unusual."""
    path = tmp_path / 'plain.csv'
    path.write_text('a,b,c\n1,x,1.5\n2,y,2.5\n')
    frame = read('local', path=str(path))
    assert frame.dtypes.equals(pd.read_csv(path).dtypes)


def test_a_csv_float_column_with_a_blank_is_still_a_float(tmp_path):
    path = tmp_path / 'floats.csv'
    # Two columns, so the gap is a blank field rather than a blank line.
    path.write_text('f,g\n1.5,1\n,2\n2.5,3\n')
    frame = read('local', path=str(path))
    assert pd.api.types.is_float_dtype(frame['f'])
    assert wire(frame['f']) == [1.5, None, 2.5]


# --- data=: the DataFrame constructor promotes too ---------------------------


def test_a_python_int_list_with_a_none_stays_exact():
    frame = read(
        'local', data={'id': [BIG, None, BIG + 2], 'v': [1.0, 2.0, 3.0]}
    )
    assert wire(frame['id']) == [str(BIG), None, str(BIG + 2)]


def test_a_python_int_list_without_a_none_is_a_plain_int64():
    frame = read('local', data={'id': [BIG, BIG + 2]})
    assert str(frame['id'].dtype) == 'int64'


def test_a_float_list_with_a_none_is_left_alone():
    frame = read('local', data={'v': [1.5, None, 2.5]})
    assert pd.api.types.is_float_dtype(frame['v'])


def test_a_bool_list_is_not_mistaken_for_integers():
    frame = read('local', data={'flag': [True, None, False]})
    assert not pd.api.types.is_integer_dtype(frame['flag'])


def test_a_dataframe_passed_through_data_is_untouched():
    original = pd.DataFrame({'id': [BIG, BIG + 2]})
    assert read('local', data=original) is original


# --- DuckDB: Arrow before the driver's own pandas conversion ----------------


def test_a_sum_over_a_bigint_column_is_exact():
    """`sum(BIGINT)` widens to HUGEINT, which `fetchdf()` rounds."""
    connection = connect('duckdb')
    connection.execute(
        f'create table t as select {BIG}::BIGINT id from range(3)'
    )
    frame = read_table(connection, query='select sum(id) s from t')
    assert wire(frame['s']) == [str(BIG * 3)]


def test_a_hugeint_column_is_exact():
    connection = connect('duckdb')
    connection.execute(f'create table h as select {BIG}::HUGEINT h')
    assert wire(read_table(connection, table='h')['h']) == [str(BIG)]


def test_a_bigint_column_with_a_null_is_still_exact():
    connection = connect('duckdb')
    connection.execute(
        f'create table n as select * from '
        f'(values ({BIG}::BIGINT, 1.0), (NULL, 2.0)) t(id, v)'
    )
    assert wire(read_table(connection, table='n')['id']) == [str(BIG), None]


def test_a_decimal_with_a_scale_keeps_it():
    connection = connect('duckdb')
    connection.execute('create table d as select 12.3456::DECIMAL(38,4) m')
    frame = read_table(connection, query='select * from d')
    assert float(frame['m'].iloc[0]) == pytest.approx(12.3456)


def test_an_ordinary_query_is_unchanged():
    connection = connect('duckdb')
    connection.execute('create table t as select 1 a, 2.5 b')
    frame = read_table(connection, table='t')
    assert frame['a'].iloc[0] == 1
    assert frame['b'].iloc[0] == 2.5


# --- a remote directory is a directory ---------------------------------------


def test_a_local_directory_is_a_parquet_dataset(tmp_path):
    directory = tmp_path / 'parts'
    directory.mkdir()
    frame = pd.DataFrame({'a': range(4)})
    frame.head(2).to_parquet(directory / 'p0.parquet', index=False)
    frame.tail(2).to_parquet(directory / 'p1.parquet', index=False)
    assert len(read('local', path=str(directory))) == 4


def test_a_remote_prefix_is_not_asked_of_the_local_filesystem():
    """`os.path.isdir` is always False for an s3:// URI.

    The filesystem that owns the path is asked instead, and it reports both a
    verdict and why it could not answer, so a store that cannot be reached is
    told apart from one that answered "not a directory".
    """
    from sdm_connectors.reader import _is_directory

    is_directory, unreachable = _is_directory(
        's3://a-bucket-that-does-not-exist-42/prefix'
    )
    assert is_directory is False
    assert unreachable is None or isinstance(unreachable, Exception)


def test_an_unreachable_store_is_not_reported_as_a_format_problem():
    """A suffix-less path has no read afterwards to surface the real fault.

    Bad credentials on `s3://bucket/parts` used to come back as "unsupported
    file format", which sends the caller to fix the wrong thing.
    """
    import sdm_connectors.reader as reader

    def _unreachable(path, storage_options=None):
        return False, PermissionError('the supplied credentials were refused')

    original = reader._is_directory
    reader._is_directory = _unreachable
    try:
        with pytest.raises(ConnectorError) as error:
            reader._resolve_format('s3://bucket/parts')
    finally:
        reader._is_directory = original
    message = str(error.value)
    assert 'could not determine the format' in message
    assert 'credentials were refused' in message
    assert 'unsupported file format' not in message


def test_a_trailing_slash_still_means_a_directory():
    from sdm_connectors.reader import _resolve_format

    assert _resolve_format('s3://bucket/prefix/') == 'parquet'


# --- data=: what pandas would have done, only exact -------------------------


@pytest.mark.parametrize(
    ('label', 'values'),
    [
        ('past int64', [2**100]),
        ('below int64', [-(2**63) - 1]),
        ('at 2**63', [2**63]),
    ],
)
def test_an_integer_outside_int64_is_left_to_pandas(label, values):
    """`Int64` cannot hold it, and pandas keeps an exact object column.

    Forcing the dtype raised `OverflowError` where nothing used to fail.
    """
    frame = read('local', data={'id': values})
    assert str(frame['id'].iloc[0]) == str(values[0])


def test_a_pandas_na_is_a_null_like_any_other():
    frame = read('local', data={'id': [BIG, pd.NA, BIG + 2]})
    assert wire(frame['id']) == [str(BIG), None, str(BIG + 2)]


def test_a_numpy_integer_with_a_null_stays_exact():
    """A numpy scalar is an integer too, and used to round."""
    import numpy as np

    frame = read('local', data={'id': [np.int64(BIG), None]})
    assert wire(frame['id']) == [str(BIG), None]


def test_a_bool_column_with_a_null_is_not_made_an_integer():
    frame = read('local', data={'flag': [True, None, False]})
    assert not pd.api.types.is_integer_dtype(frame['flag'])


def test_an_empty_column_is_left_alone():
    frame = read('local', data={'id': [], 'v': []})
    assert len(frame) == 0


def test_a_string_column_lands_where_a_plain_read_would_put_it(tmp_path):
    """The null-aware backend must not change what an ordinary read gives."""
    path = tmp_path / 'strings.csv'
    path.write_text('a,b\n1,x\n2,y\n')
    assert (
        read('local', path=str(path))['b'].dtype == pd.read_csv(path)['b'].dtype
    )


def test_an_unreachable_store_error_carries_a_code():
    """Every ConnectorError carries a stable code so a caller can branch
    without matching on message text; the new one was missing it."""
    import sdm_connectors.reader as reader

    def _unreachable(path, storage_options=None):
        return False, PermissionError('refused')

    original = reader._is_directory
    reader._is_directory = _unreachable
    try:
        with pytest.raises(ConnectorError) as error:
            reader._resolve_format('s3://bucket/parts')
    finally:
        reader._is_directory = original
    assert error.value.code == 'CONNECT_FAILED'
