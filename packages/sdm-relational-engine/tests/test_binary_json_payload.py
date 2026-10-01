# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import json
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from predictive_query.schema.task import TaskType
from predictive_query.schema.typing import Stype
from sdm_relational_engine.rfm.inference.config import (
    ClassificationInferenceConfig,
    RunMode,
)
from sdm_relational_engine.rfm.inference.context import (
    PredictionContext,
    SampledSubgraph,
    SampledTable,
)
from sdm_relational_engine.rfm.inference.payload import (
    _dtype_name,
    _instance_dataframe,
    payload_size_bytes,
    predict_request_to_json,
)
from sdm_relational_engine.rfm.inference.requests import RFMPredictRequest


def _binary_context(
    y_train: pd.Series,
    y_test: pd.Series | None,
) -> PredictionContext:
    batch_size = len(y_train) + (len(y_test) if y_test is not None else 0)
    table_name = 'ENTITY'
    primary_key = 'ENTITY_ID'
    return PredictionContext(
        task_type=TaskType.BINARY_CLASSIFICATION,
        entity_table_names=(table_name,),
        sampled_subgraph=SampledSubgraph(
            anchor_time=np.full(
                batch_size,
                pd.Timestamp.min.value,
                dtype=np.int64,
            ),
            tables={
                table_name: SampledTable(
                    df=pd.DataFrame(
                        {
                            primary_key: np.arange(batch_size),
                        }
                    ),
                    row=None,
                    batch=np.arange(batch_size),
                    num_sampled_nodes=[batch_size],
                    stype_dict={primary_key: Stype.ID},
                    primary_key=primary_key,
                ),
            },
            links={},
        ),
        y_train=y_train,
        y_test=y_test,
    )


def _binary_payload(context: PredictionContext) -> dict[str, Any]:
    request = RFMPredictRequest(
        context=context,
        run_mode=RunMode.FAST,
        inference_config=ClassificationInferenceConfig(),
    )
    return predict_request_to_json(request)


@pytest.mark.parametrize(
    'labels',
    [
        pd.Series([0, 1], name='TARGET', dtype='int64'),
        pd.Series(
            [np.int32(0), np.int64(1)],
            name='TARGET',
            dtype='object',
        ),
    ],
    ids=['python-integers', 'numpy-integers'],
)
def test_binary_integer_targets_serialize_as_json_booleans(
    labels: pd.Series,
) -> None:
    context = _binary_context(
        y_train=labels,
        y_test=pd.Series(
            [np.int64(1), np.int32(0)],
            name='TARGET',
            dtype='object',
        ),
    )

    instance_df, _, _ = _instance_dataframe(context)
    assert instance_df['TARGET'].tolist() == [False, True, True, False]
    assert all(
        isinstance(value, bool) for value in instance_df['TARGET'].tolist()
    )

    payload = json.loads(json.dumps(_binary_payload(context)))
    table = payload['context']['instance_table']
    target_index = table['columns'].index('TARGET')
    target_values = [row[target_index] for row in table['rows']]
    assert target_values == [False, True]
    assert all(isinstance(value, bool) for value in target_values)
    assert payload['task']['target']['dtype'] == 'bool'
    assert 'TARGET' not in payload['predict']['instance_table']['columns']


def test_binary_boolean_targets_remain_booleans() -> None:
    context = _binary_context(
        y_train=pd.Series([False, True], name='TARGET', dtype='bool'),
        y_test=pd.Series([True], name='TARGET', dtype='bool'),
    )

    instance_df, _, _ = _instance_dataframe(context)

    values = instance_df['TARGET'].tolist()
    assert values == [False, True, True]
    assert all(isinstance(value, bool) for value in values)


@pytest.mark.parametrize('invalid_value', [2, -1, 0.5, 'true'])
def test_binary_target_rejects_values_outside_explicit_mapping(
    invalid_value: Any,
) -> None:
    context = _binary_context(
        y_train=pd.Series([0, invalid_value], name='TARGET', dtype='object'),
        y_test=None,
    )

    with pytest.raises(
        ValueError,
        match='must be booleans or numeric 0/1 values',
    ):
        _binary_payload(context)


def test_binary_prediction_target_rejects_invalid_numeric_value() -> None:
    context = _binary_context(
        y_train=pd.Series([0, 1], name='TARGET', dtype='int64'),
        y_test=pd.Series([2], name='TARGET', dtype='int64'),
    )

    with pytest.raises(
        ValueError,
        match=r'numeric 0/1 values, but got 2\.',
    ):
        _binary_payload(context)


# -- DECIMAL columns ------------------------------------------------------
#
# Databricks returns DECIMAL as Python `Decimal`, which neither
# `is_integer_dtype` nor `is_float_dtype` recognises. Before this was handled,
# such a column was announced as `string` and then killed the request at
# `json.dumps` with "Object of type Decimal is not JSON serializable",
# reproduced against a real warehouse on a `decimal(38,0)` id column, at every
# batch size.
#
# `decimal(38, 0)` permits 38 digits where int64 holds 19, so naming such a
# column `int64` is only honest once the values are known to fit.

_WIDE_DECIMAL_ID = Decimal('12345678901234567890123456789012345678')


@pytest.mark.parametrize(
    'column, expected',
    [
        (
            pd.Series(
                [Decimal('900100015'), Decimal('900200001')], dtype=object
            ),
            'int64',
        ),
        (
            pd.Series([Decimal('1.50'), Decimal('2.25')], dtype=object),
            'float64',
        ),
        (
            pd.Series(
                [Decimal('900100015')],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            'int64',
        ),
        (
            pd.Series(
                [Decimal('1.50')], dtype=pd.ArrowDtype(pa.decimal128(10, 2))
            ),
            'float64',
        ),
        (pd.Series([Decimal('1'), 'x'], dtype=object), 'string'),
        (pd.Series([Decimal('1'), _WIDE_DECIMAL_ID], dtype=object), 'string'),
        (
            pd.Series(
                [_WIDE_DECIMAL_ID], dtype=pd.ArrowDtype(pa.decimal128(38, 0))
            ),
            'string',
        ),
        (
            pd.Series(
                [Decimal('1'), Decimal('NaN'), Decimal('Infinity')],
                dtype=object,
            ),
            'int64',
        ),
    ],
    ids=[
        'integral-object-is-an-id-not-a-string',
        'scaled-object-is-a-float',
        'arrow-backed-is-named-by-its-scale',
        'arrow-backed-scaled-is-a-float',
        'partly-decimal-keeps-the-previous-behaviour',
        'too-wide-for-int64-keeps-its-digits-as-a-string',
        'too-wide-arrow-backed-keeps-its-digits-as-a-string',
        'non-finite-values-do-not-decide-the-name',
    ],
)
def test_decimal_columns_are_named_by_what_they_hold(
    column: pd.Series,
    expected: str,
) -> None:
    assert _dtype_name(column) == expected


def test_a_decimal_column_survives_the_whole_request() -> None:
    context = _binary_context(
        y_train=pd.Series([0, 1, 0], name='TARGET', dtype='int64'),
        y_test=pd.Series([1], name='TARGET', dtype='int64'),
    )
    table = context.sampled_subgraph.tables['ENTITY'].df
    table['ITEM_ID'] = pd.Series(
        [Decimal(f'90010001{i}') for i in range(4)], dtype=object
    )
    table['PRICE'] = pd.Series(
        [Decimal('0.05'), Decimal('NaN'), Decimal('Infinity'), Decimal('2.25')],
        dtype=object,
    )
    table['WIDE_ID'] = pd.Series([_WIDE_DECIMAL_ID] * 4, dtype=object)

    payload = _binary_payload(context)
    # The original failure was here, measuring the payload rather than
    # sending it: TypeError: Object of type Decimal is not JSON serializable.
    assert payload_size_bytes(payload) > 0

    columns = payload['schema']['related_tables']['ENTITY']['columns']
    assert columns['ITEM_ID']['dtype'] == 'int64'
    assert columns['PRICE']['dtype'] == 'float64'
    assert columns['WIDE_ID']['dtype'] == 'string'

    emitted = payload['context']['related_tables']['ENTITY']
    rows = json.loads(json.dumps(emitted, allow_nan=False))['rows']
    values = {
        name: [row[emitted['columns'].index(name)] for row in rows]
        for name in ('ITEM_ID', 'PRICE', 'WIDE_ID')
    }
    assert values['ITEM_ID'] == [900100010, 900100011, 900100012]
    assert all(isinstance(value, int) for value in values['ITEM_ID'])
    assert values['PRICE'] == [0.05, None, None]
    assert values['WIDE_ID'] == [str(_WIDE_DECIMAL_ID)] * 3
