# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from predictive_query.ast import (
    Aggregation,
    Column,
    Condition,
    Constant,
    DateOffsetRange,
    Filter,
    Join,
    LogicalOperation,
)
from predictive_query.schema.typing import (
    AggregationType,
    BoolOp,
    Dtype,
    MemberOp,
    RelOp,
    TimeUnit,
)
from sdm_relational_engine.rfm.pquery import PQueryPandasExecutor


@pytest.mark.parametrize(
    ('column_name', 'expected'),
    [
        ('AGE', pd.Series([10, None, 30], dtype='float32')),
        ('GENDER', pd.Series(['male', None, 'female'])),
        ('CAT', pd.Series([1, pd.NA, 3], dtype='Int64')),
    ],
)
@pytest.mark.parametrize('filter_na', [False, True])
def test_column(
    column_name: str,
    expected: pd.Series,
    filter_na: bool,
) -> None:
    feat_dict = {
        'USERS': pd.DataFrame(
            {
                'AGE': [10, None, 30],
                'GENDER': ['male', None, 'female'],
                'CAT': pd.Series([1, pd.NA, 3], dtype='Int64'),
            }
        )
    }

    column = Column(fqn=f'USERS.{column_name}')
    out, mask = PQueryPandasExecutor().execute_column(
        column, feat_dict, filter_na
    )

    if filter_na:
        expected = expected.dropna().reset_index(drop=True)
        if pd.api.types.is_integer_dtype(expected):
            expected = expected.astype('int')
    pd.testing.assert_series_equal(out, expected)
    assert np.array_equal(mask, np.array([True, False, True]))


@pytest.mark.parametrize(
    ('op, expected'),
    [
        (RelOp.EQ, pd.Series([False, True, False, True, False])),
        (RelOp.NEQ, pd.Series([True, False, False, False, True])),
        (RelOp.LEQ, pd.Series([True, True, False, True, False])),
        (RelOp.GEQ, pd.Series([False, True, False, True, True])),
        (RelOp.LT, pd.Series([True, False, False, False, False])),
        (RelOp.GT, pd.Series([False, False, False, False, True])),
    ],
)
def test_rel_op_int(op: RelOp, expected: pd.Series) -> None:
    left = pd.Series([2, 4, pd.NA, 4, 6], dtype='Int64')

    out = PQueryPandasExecutor().execute_rel_op(
        left, op, right=Constant(value='4', dtype_maybe=Dtype.int)
    )
    pd.testing.assert_series_equal(out, expected)


@pytest.mark.parametrize(
    ('op, expected'),
    [
        (RelOp.EQ, pd.Series([False, True, False, True, False])),
        (RelOp.NEQ, pd.Series([True, False, False, False, True])),
        (RelOp.LEQ, pd.Series([True, True, False, True, False])),
        (RelOp.GEQ, pd.Series([False, True, False, True, True])),
        (RelOp.LT, pd.Series([True, False, False, False, False])),
        (RelOp.GT, pd.Series([False, False, False, False, True])),
    ],
)
def test_rel_op_float(op: RelOp, expected: pd.Series) -> None:
    left = pd.Series([2.5, 4.5, pd.NA, 4.5, 6.5], dtype='Float64')

    out = PQueryPandasExecutor().execute_rel_op(
        left, op, right=Constant(value='4.5', dtype_maybe=Dtype.float)
    )
    pd.testing.assert_series_equal(out, expected)


@pytest.mark.parametrize(
    ('op, expected'),
    [
        (RelOp.EQ, pd.Series([False, True, False, True, False])),
        (RelOp.NEQ, pd.Series([True, False, False, False, True])),
        (RelOp.LEQ, pd.Series([True, True, False, True, False])),
        (RelOp.GEQ, pd.Series([False, True, False, True, True])),
        (RelOp.LT, pd.Series([True, False, False, False, False])),
        (RelOp.GT, pd.Series([False, False, False, False, True])),
    ],
)
def test_rel_op_timestamp(op: RelOp, expected: pd.Series) -> None:
    left = pd.Series(
        ['2019-06-07', '2023-01-01', pd.NaT, '2023-01-01', '2025-12-31'],
        dtype='datetime64[ns]',
    )

    out = PQueryPandasExecutor().execute_rel_op(
        left, op, right=Constant(value='2023-01-01', dtype_maybe=Dtype.date)
    )
    pd.testing.assert_series_equal(out, expected)


def test_rel_op_none() -> None:
    left = pd.Series([2, 4, pd.NA, 4, 6], dtype='Int64')

    out = PQueryPandasExecutor().execute_rel_op(
        left, op=RelOp.EQ, right=Constant(value='NULL', dtype_maybe=None)
    )
    pd.testing.assert_series_equal(
        out,
        pd.Series([False, False, True, False, False]),
    )

    out = PQueryPandasExecutor().execute_rel_op(
        left, op=RelOp.NEQ, right=Constant(value='NULL', dtype_maybe=None)
    )
    pd.testing.assert_series_equal(
        out,
        pd.Series([True, True, False, True, True]),
    )


@pytest.mark.parametrize(
    ('op, expected'),
    [
        (MemberOp.IN, pd.Series([True, False, False, False, True])),
    ],
)
def test_member_op(op: MemberOp, expected: pd.Series) -> None:
    left = pd.Series([2, 4, pd.NA, 4, 6], dtype='Int64')

    out = PQueryPandasExecutor().execute_member_op(
        left=left,
        op=MemberOp.IN,
        right=Constant(
            value=[
                Constant(value='2', dtype_maybe=Dtype.int),
                Constant(value='6', dtype_maybe=Dtype.int),
            ],
            dtype_maybe=Dtype.intlist,
        ),
    )
    pd.testing.assert_series_equal(out, expected)


@pytest.mark.parametrize(
    ('op', 'expected_out', 'expected_mask'),
    [
        pytest.param(
            AggregationType.SUM,
            pd.Series([6, 0, 0, 16, 2, 0], dtype='float32'),
            np.array([True, True, True, True, True, True]),
            id='SUM',
        ),
        pytest.param(
            AggregationType.AVG,
            pd.Series([3, None, None, 16 / 3, 2, None], dtype='float32'),
            np.array([True, False, False, True, True, False]),
            id='AVG',
        ),
        pytest.param(
            AggregationType.MIN,
            pd.Series([2, None, None, 4, 2, None], dtype='float32'),
            np.array([True, False, False, True, True, False]),
            id='MIN',
        ),
        pytest.param(
            AggregationType.MAX,
            pd.Series([4, None, None, 6, 2, None], dtype='float32'),
            np.array([True, False, False, True, True, False]),
            id='MAX',
        ),
        pytest.param(
            AggregationType.COUNT,
            pd.Series([2, 0, 0, 3, 1, 0], dtype='float32'),
            np.array([True, True, True, True, True, True]),
            id='COUNT',
        ),
        pytest.param(
            AggregationType.LIST_DISTINCT,
            pd.Series([[2, 4], pd.NA, pd.NA, [4, 6], [2], pd.NA]),
            np.array([True, False, False, True, True, False]),
            id='LIST_DISTINCT',
        ),
    ],
)
@pytest.mark.parametrize('filter_na', [False, True])
def test_aggregation_type(
    op: AggregationType,
    expected_out: pd.Series,
    expected_mask: np.ndarray,
    filter_na: bool,
) -> None:
    feat = pd.Series([2, 4, pd.NA, 4, 6, 6, 2], dtype='Int64')
    batch = np.array([0, 0, 1, 3, 3, 3, 4])

    out, mask = PQueryPandasExecutor().execute_aggregation_type(
        op=op,
        feat=feat,
        batch=batch,
        batch_size=6,
        filter_na=filter_na,
    )

    if filter_na:
        expected_out = expected_out.dropna().reset_index(drop=True)
    pd.testing.assert_series_equal(out, expected_out)
    assert np.array_equal(mask, expected_mask)


@pytest.mark.parametrize(
    ('op', 'expected_out', 'expected_mask'),
    [
        pytest.param(
            AggregationType.MIN,
            pd.Series(['2025-01-01', None, '2025-02-01', None]),
            np.array([True, False, True, False]),
            id='MIN',
        ),
        pytest.param(
            AggregationType.MAX,
            pd.Series(['2025-01-10', None, '2025-02-01', None]),
            np.array([True, False, True, False]),
            id='MAX',
        ),
    ],
)
@pytest.mark.parametrize('filter_na', [False, True])
def test_aggregation_type_timestamp(
    op: AggregationType,
    expected_out: pd.Series,
    expected_mask: np.ndarray,
    filter_na: bool,
) -> None:
    feat = pd.to_datetime(
        pd.Series(
            [
                '2025-01-01',
                '2025-01-10',
                '2025-02-01',
            ]
        )
    )
    batch = np.array([0, 0, 2])

    out, mask = PQueryPandasExecutor().execute_aggregation_type(
        op=op,
        feat=feat,
        batch=batch,
        batch_size=4,
        filter_na=filter_na,
    )

    expected_out = pd.to_datetime(expected_out)
    if filter_na:
        expected_out = expected_out.dropna().reset_index(drop=True)
    pd.testing.assert_series_equal(out, expected_out)
    assert np.array_equal(mask, expected_mask)


@pytest.mark.parametrize('start', [0, None])
@pytest.mark.parametrize('filter_na', [False, True])
def test_aggregation(
    start: int | None,
    filter_na: bool,
) -> None:
    aggr = Aggregation(
        target=Column(fqn='fact.test'),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(start, 2, TimeUnit.DAYS),
    )

    feat_dict = {
        'fact': pd.DataFrame({'test': range(1, 9)}),
    }
    time_dict = {
        'fact': pd.Series(pd.date_range('2023-01-01', periods=8)),
    }
    batch_dict = {
        'fact': np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    }
    anchor_time = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-06']))

    out, mask = PQueryPandasExecutor().execute_aggregation(
        aggr=aggr,
        feat_dict=feat_dict,
        time_dict=time_dict,
        batch_dict=batch_dict,
        anchor_time=anchor_time,
        filter_na=filter_na,
    )

    if start == 0:
        pd.testing.assert_series_equal(
            out,
            pd.Series([2 + 3, 7 + 8], dtype='float32'),
        )
    else:
        pd.testing.assert_series_equal(
            out,
            pd.Series([1 + 2 + 3, 5 + 6 + 7 + 8], dtype='float32'),
        )
    assert np.array_equal(mask, np.array([True, True]))


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        (BoolOp.AND, pd.Series([True, False, False, False])),
        (BoolOp.OR, pd.Series([True, True, True, False])),
        (BoolOp.NOT, pd.Series([False, True, False, True])),
    ],
)
def test_bool_op(op: BoolOp, expected: pd.Series) -> None:
    left = pd.Series([True, False, True, False])
    right = pd.Series([True, True, False, False])

    out = PQueryPandasExecutor().execute_bool_op(left, op, right)
    pd.testing.assert_series_equal(out, expected)


@pytest.mark.parametrize('filter_na', [False, True])
def test_logical_operation(filter_na: bool) -> None:
    feat_dict = {
        'USERS': pd.DataFrame(
            {
                'AGE': [10, None, 30, 20],
                'GENDER': ['male', 'female', None, 'male'],
            }
        )
    }
    time_dict = {}
    batch_dict = {'fact': np.array([0, 1, 2])}
    anchor_time = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-01']))

    logical_operation = LogicalOperation(
        left=Condition(
            target=Column(fqn='USERS.AGE'),
            op=RelOp.GEQ,
            value=Constant(value='20', dtype_maybe=Dtype.int),
        ),
        bool_op=BoolOp.AND,
        right=Condition(
            target=Column(fqn='USERS.GENDER'),
            op=RelOp.EQ,
            value=Constant(value='"male"', dtype_maybe=Dtype.string),
        ),
    )

    out, mask = PQueryPandasExecutor().execute_logical_operation(
        logical_operation=logical_operation,
        feat_dict=feat_dict,
        time_dict=time_dict,
        batch_dict=batch_dict,
        anchor_time=anchor_time,
        filter_na=filter_na,
    )
    if filter_na:
        pd.testing.assert_series_equal(out, pd.Series([False, True]))
    else:
        pd.testing.assert_series_equal(
            out,
            pd.Series([False, False, False, True]),
        )
    assert np.array_equal(mask, np.array([True, False, False, True]))


@pytest.mark.parametrize('filter_na', [False, True])
def test_condition(filter_na: bool) -> None:
    feat_dict = {'USERS': pd.DataFrame({'AGE': [10, None, 30]})}
    time_dict = {}
    batch_dict = {'fact': np.array([0, 1, 2])}
    anchor_time = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-01']))

    condition = Condition(
        target=Column(fqn='USERS.AGE'),
        op=RelOp.GT,
        value=Constant(value='20', dtype_maybe=Dtype.int),
    )
    out, mask = PQueryPandasExecutor().execute_condition(
        condition=condition,
        feat_dict=feat_dict,
        time_dict=time_dict,
        batch_dict=batch_dict,
        anchor_time=anchor_time,
        filter_na=filter_na,
    )
    if filter_na:
        pd.testing.assert_series_equal(out, pd.Series([False, True]))
    else:
        pd.testing.assert_series_equal(out, pd.Series([False, False, True]))
    assert np.array_equal(mask, np.array([True, False, True]))


def test_join() -> None:
    aggr = Aggregation(
        target=Column(fqn='fact.test'),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(0, 2, TimeUnit.DAYS),
    )
    join = Join(rhs_target=aggr)

    feat_dict = {
        'fact': pd.DataFrame({'test': range(1, 9)}),
    }
    time_dict = {
        'fact': pd.Series(pd.date_range('2023-01-01', periods=8)),
    }
    batch_dict = {
        'fact': np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    }
    anchor_time = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-06']))

    out, mask = PQueryPandasExecutor().execute_join(
        join=join,
        feat_dict=feat_dict,
        time_dict=time_dict,
        batch_dict=batch_dict,
        anchor_time=anchor_time,
        filter_na=True,
    )

    pd.testing.assert_series_equal(
        out,
        pd.Series([2 + 3, 7 + 8], dtype='float32'),
    )
    assert np.array_equal(mask, np.array([True, True]))


@pytest.mark.parametrize('filter_na', [False, True])
def test_filter(filter_na: bool) -> None:
    feat_dict = {
        'USERS': pd.DataFrame({'AGE': [10, 25, 30], 'HEIGHT': [156, None, 186]})
    }
    time_dict = {}
    batch_dict = {'fact': np.array([0, 1, 2])}
    anchor_time = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-01']))

    condition = Condition(
        target=Column(fqn='USERS.AGE'),
        op=RelOp.GT,
        value=Constant(value='20', dtype_maybe=Dtype.int),
    )
    filter = Filter(target=Column(fqn='USERS.HEIGHT'), condition=condition)
    out, mask = PQueryPandasExecutor().execute_filter(
        filter=filter,
        feat_dict=feat_dict,
        time_dict=time_dict,
        batch_dict=batch_dict,
        anchor_time=anchor_time,
        filter_na=filter_na,
    )
    if filter_na:
        pd.testing.assert_series_equal(out, pd.Series([186], dtype=np.float32))
    else:
        pd.testing.assert_series_equal(
            out, pd.Series([None, 186], dtype=np.float32)
        )
    assert np.array_equal(mask, np.array([False, False, True]))


def test_filtered_aggregation() -> None:
    aggr = Aggregation(
        target=Filter(
            target=Column(fqn='fact.test'),
            condition=Condition(
                target=Column(fqn='fact.condition'),
                op='=',
                value=Constant.from_value(0),
            ),
        ),
        aggr=AggregationType.SUM,
        aggr_time_range=DateOffsetRange(0, 2, TimeUnit.DAYS),
    )

    feat_dict = {
        'fact': pd.DataFrame(
            {'test': range(1, 9), 'condition': [0, 0, 1, 0, 0, 1, 0, 0]}
        ),
    }
    time_dict = {
        'fact': pd.Series(pd.date_range('2023-01-01', periods=8)),
    }
    batch_dict = {
        'fact': np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    }
    anchor_time = pd.Series(pd.to_datetime(['2023-01-01', '2023-01-06']))

    out, mask = PQueryPandasExecutor().execute_aggregation(
        aggr=aggr,
        feat_dict=feat_dict,
        time_dict=time_dict,
        batch_dict=batch_dict,
        anchor_time=anchor_time,
        filter_na=False,
    )

    pd.testing.assert_series_equal(
        out,
        pd.Series([2, 7 + 8], dtype='float32'),
    )
    assert np.array_equal(mask, np.array([True, True]))
