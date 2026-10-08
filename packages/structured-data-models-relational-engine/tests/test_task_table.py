# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from predictive_query.schema.task import TaskType
from predictive_query.schema.typing import Dtype, Stype
from sdm_relational_engine.rfm import TaskTable
from sdm_relational_engine.rfm.base import Column
from sdm_relational_engine.rfm.inference.task import _get_target_stype


@pytest.mark.parametrize(
    'task_type',
    [
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.REGRESSION,
        TaskType.TEMPORAL_LINK_PREDICTION,
    ],
)
def test_basic(task_type: TaskType) -> None:
    entity_table_names = ('USERS',)
    if task_type == TaskType.BINARY_CLASSIFICATION:
        target = [True, False, True]
        target_dtype = Dtype.bool
        target_stype = Stype.categorical
    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        target = ['A', 'B', 'C']
        target_dtype = Dtype.string
        target_stype = Stype.categorical
    elif task_type == TaskType.REGRESSION:
        target = [1.0, 2.0, 3.0]
        target_dtype = Dtype.float
        target_stype = Stype.numerical
    else:
        assert task_type == TaskType.TEMPORAL_LINK_PREDICTION
        target = [['A'], ['A', 'B'], ['C']]
        entity_table_names = ('USERS', 'ITEMS')
        target_dtype = Dtype.stringlist
        target_stype = Stype.multicategorical

    context_df = pd.DataFrame(
        {
            'ENTITY': [0, 1, 2],
            'TARGET': target,
            'TIME': ['2020-01-01'] * 3,
        }
    )
    pred_df = pd.DataFrame(
        {
            'ENTITY': [3, 4],
            'TIME': ['2021-01-01'] * 2,
        }
    )

    task_table = TaskTable(
        task_type=task_type,
        context_df=context_df,
        pred_df=pred_df,
        entity_table_name=entity_table_names,
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='TIME',
    )

    assert task_table.task_type == task_type
    assert task_table.entity_table_name == entity_table_names[0]
    assert task_table.entity_table_names == entity_table_names
    assert task_table.entity_column == Column(
        name='ENTITY',
        expr=None,
        dtype=Dtype.int,
        stype=Stype.ID,
    )
    assert task_table.target_column == Column(
        name='TARGET',
        expr=None,
        dtype=target_dtype,
        stype=target_stype,
    )
    assert task_table.has_time_column()
    assert task_table.time_column == Column(
        name='TIME',
        expr=None,
        dtype=Dtype.string,
        stype=Stype.timestamp,
    )

    if task_type.is_link_pred:
        entity_table_repr = f'entity_table_names={entity_table_names}'
    else:
        entity_table_repr = f'entity_table_name={entity_table_names[0]}'
    assert str(task_table) == (
        f'TaskTable(\n'
        f'  task_type={task_type},\n'
        f'  num_context_examples=3,\n'
        f'  num_prediction_examples=2,\n'
        f'  num_columns=3,\n'
        f'  {entity_table_repr},\n'
        f'  entity_column=ENTITY,\n'
        f'  target_column=TARGET,\n'
        f'  time_column=TIME,\n'
        f')'
    )

    out = task_table.narrow_context(start=0, length=2)
    assert out.num_context_examples == 2

    out = task_table.narrow_prediction(start=1, length=1)
    assert out.num_prediction_examples == 1


def test_entity_time() -> None:
    task_table = TaskTable(
        task_type=TaskType.BINARY_CLASSIFICATION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0, 1, 2],
                'TARGET': [True, False, True],
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': [3, 4],
            }
        ),
        entity_table_name='USER',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column=TaskTable.ENTITY_TIME,
    )

    assert task_table.use_entity_time
    assert str(task_table) == (
        'TaskTable(\n'
        '  task_type=binary_classification,\n'
        '  num_context_examples=3,\n'
        '  num_prediction_examples=2,\n'
        '  num_columns=2,\n'
        '  entity_table_name=USER,\n'
        '  entity_column=ENTITY,\n'
        '  target_column=TARGET,\n'
        '  use_entity_time=True,\n'
        ')'
    )


def test_custom_features() -> None:
    task_table = TaskTable(
        task_type=TaskType.BINARY_CLASSIFICATION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0, 1, 2],
                'TARGET': [True, False, True],
                'FEAT_1': [0.0, 1.0, 2.0],
                'FEAT_2': ['A', 'B', 'C'],
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': [3, 4],
                'FEAT_1': [3.0, 4.0],
                'FEAT_2': ['D', 'E'],
            }
        ),
        entity_table_name='USER',
        entity_column='ENTITY',
        target_column='TARGET',
    )

    assert len(task_table.columns) == 4
    assert task_table['ENTITY'].dtype == Dtype.int
    assert task_table['ENTITY'].stype == Stype.ID
    assert task_table['TARGET'].dtype == Dtype.bool
    assert task_table['TARGET'].stype == Stype.categorical
    assert task_table['FEAT_1'].dtype == Dtype.float
    assert task_table['FEAT_1'].stype == Stype.categorical
    assert task_table['FEAT_2'].dtype == Dtype.string
    assert task_table['FEAT_2'].stype == Stype.categorical


def test_add_columns_does_not_reinfer_existing_task_columns() -> None:
    task_table = TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0, 1, 2],
                'TARGET': [0, 1, 0],
                'TIME': pd.date_range('2026-01-01', periods=3),
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': [3, 4],
                'TIME': pd.date_range('2026-01-04', periods=2),
            }
        ),
        entity_table_name='USER',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='TIME',
    )

    task_table._context_df['LAG'] = [0.1, 0.2, 0.3]
    task_table._pred_df['LAG'] = [0.4, 0.5]
    task_table.add_columns(['LAG'])

    assert task_table.target_column.stype == Stype.numerical
    assert task_table.time_column is not None
    assert task_table.time_column.stype == Stype.timestamp
    assert task_table['LAG'].dtype == Dtype.float


def test_add_columns_rejects_unknown_and_duplicate_columns() -> None:
    task_table = TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0, 1],
                'TARGET': [1.0, 2.0],
            }
        ),
        pred_df=pd.DataFrame({'ENTITY': [2]}),
        entity_table_name='USER',
        entity_column='ENTITY',
        target_column='TARGET',
    )

    with pytest.raises(KeyError, match='not found'):
        task_table.add_columns(['MISSING'])

    with pytest.raises(KeyError, match='already exists'):
        task_table.add_columns(['TARGET'])


# Regression tests: these three sites used to raise ``ValueError`` with an
# empty message.
def test_unsupported_task_type_message() -> None:
    with pytest.raises(ValueError) as err:
        TaskTable(
            task_type=TaskType.MULTILABEL_RANKING,
            context_df=pd.DataFrame(
                {
                    'ENTITY': [0, 1],
                    'TARGET': [[0], [1]],
                }
            ),
            pred_df=pd.DataFrame({'ENTITY': [2]}),
            entity_table_name='USER',
            entity_column='ENTITY',
            target_column='TARGET',
        )

    assert 'multilabel_ranking' in str(err.value)
    assert 'not supported' in str(err.value)


@pytest.mark.parametrize('entity_table_name', [[], ['A', 'B', 'C']])
def test_invalid_entity_table_name_arity_message(
    entity_table_name: list[str],
) -> None:
    with pytest.raises(ValueError) as err:
        TaskTable(
            task_type=TaskType.BINARY_CLASSIFICATION,
            context_df=pd.DataFrame(
                {
                    'ENTITY': [0, 1],
                    'TARGET': [True, False],
                }
            ),
            pred_df=pd.DataFrame({'ENTITY': [2]}),
            entity_table_name=entity_table_name,
            entity_column='ENTITY',
            target_column='TARGET',
        )

    assert 'entity_table_name' in str(err.value)
    assert str(len(entity_table_name)) in str(err.value)


def test_get_target_stype_unsupported_task_type_message() -> None:
    with pytest.raises(ValueError) as err:
        _get_target_stype(TaskType.MULTILABEL_CLASSIFICATION)

    assert 'multilabel_classification' in str(err.value)
