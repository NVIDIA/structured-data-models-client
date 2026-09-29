# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two ways a column's declared type stopped describing its values.

A warehouse `DECIMAL` arrives from pandas as an `object` column of `Decimal`.
The schema types it correctly and the tabular path sends it, but the relational
path handed the engine the raw frame, which re-infers and eventually meets a
`Decimal` where a dtype name belongs. Since every numeric column of a Databricks
or Snowflake table is `DECIMAL`, that was the first question anyone asked of a
warehouse through this SDK's own connector.

And the rows being scored could re-type a column the schema had already typed.
Widening exists for the opposite case -- an all-null context column carries no
type and must not turn the scored half's integers into text -- but nothing
bounded it, so text in a `float64` column widened the column to `string` while
its `stype` stayed `numerical`, and the request went out declaring a column that
means a number and holds a string.
"""

from __future__ import annotations

import decimal
import warnings

import numpy as np
import pandas as pd
import pytest
from sdm_api.generated.tfm_api import PredictionItem

from sdm_client import KumoRelational, PredictiveQuery, Schema
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.errors import StructuredDataError
from sdm_client.wire.answers import as_declared_target
from sdm_client.wire.dtypes import as_declared

D = decimal.Decimal


def wire_column(request, name: str) -> dict:
    envelope = next(
        iter(TabularAdapter().prediction_plan(request, ('prediction',)))
    )
    return envelope['schema']['instance_table']['columns'][name]


def scored(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def context(rows: int = 200) -> Schema:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            'rid': np.arange(rows),
            'score': rng.normal(100, 10, rows).round(1),
            'tier': rng.choice(['a', 'b', 'c'], rows),
        }
    )
    frame['amount'] = (frame['score'] * 3).round(2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})


def query(schema: Schema) -> PredictiveQuery:
    return PredictiveQuery('PREDICT t.amount FOR EACH t.rid', schema)


# --- the rows being scored do not re-type the column -----------------------


def test_text_in_a_numeric_column_is_refused():
    """Answered rather than refused, this came back 300.0, 329.9, 299.7 where
    the truth is 300, 330, 360."""
    schema = context()
    with pytest.raises(StructuredDataError) as raised:
        query(schema).materialize(
            predict=scored(
                rid=[9001, 9002], score=['100.0', '110.0'], tier=['a', 'b']
            )
        )
    message = str(raised.value)
    assert raised.value.code == 'INVALID_REQUEST'
    assert "column 'score'" in message
    assert 'pd.to_numeric' in message


def test_the_refusal_names_both_what_was_declared_and_what_arrived():
    schema = context()
    with pytest.raises(StructuredDataError) as raised:
        query(schema).materialize(
            predict=scored(rid=[9001], score=['x'], tier=['a'])
        )
    message = str(raised.value)
    assert "'float64'" in message
    assert "'numerical'" in message
    assert "'string'" in message


@pytest.mark.parametrize(
    'values, expected',
    [
        ([100.0, 110.0], 'float64'),
        ([100, 110], 'float64'),
        (np.array([100, 110], dtype='int32'), 'float64'),
    ],
    ids=['float', 'int into float', 'int32 into float'],
)
def test_a_widening_that_still_means_numerical_is_allowed(values, expected):
    """Widening was written for a real case and most of it is right."""
    schema = context()
    request = query(schema).materialize(
        predict=scored(rid=[9001, 9002], score=values, tier=['a', 'b'])
    )
    assert wire_column(request, 'score')['dtype'] == expected
    assert wire_column(request, 'score')['stype'] == 'numerical'


def test_unseen_values_in_a_categorical_column_are_allowed():
    """A categorical column widening to string still means categorical."""
    schema = context()
    request = query(schema).materialize(
        predict=scored(rid=[9001], score=[100.0], tier=['z'])
    )
    assert wire_column(request, 'tier')['stype'] == 'categorical'


def test_an_all_null_context_column_still_takes_its_type_from_the_scored_rows():
    """The case widening exists for. The context says nothing, so the scored
    rows decide, and their integers must not be sent as text."""
    frame = pd.DataFrame(
        {
            'rid': range(200),
            'unknown': [None] * 200,
            'amount': [float(i % 9) for i in range(200)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})
    assert schema['t']['unknown'].untyped

    request = PredictiveQuery(
        'PREDICT t.amount FOR EACH t.rid', schema
    ).materialize(predict=scored(rid=[9001, 9002], unknown=[1, 2]))

    assert wire_column(request, 'unknown')['dtype'] == 'int64'


def test_the_wire_never_declares_a_column_that_contradicts_itself():
    """The shape of the defect, stated directly: no column may go out with a
    dtype that cannot mean its stype."""
    schema = context()
    request = query(schema).materialize(
        predict=scored(rid=[9001], score=[100.0], tier=['z'])
    )
    from sdm_client.schema import supports_dtype

    envelope = next(
        iter(TabularAdapter().prediction_plan(request, ('prediction',)))
    )
    for name, column in envelope['schema']['instance_table']['columns'].items():
        if 'stype' in column:
            assert supports_dtype(column['stype'], column['dtype']), name


# --- a warehouse decimal is the type the schema says it is -----------------


def warehouse(rows: int = 40) -> dict[str, pd.DataFrame]:
    users = pd.DataFrame(
        {
            'USERID': [D(i) for i in range(1, rows + 1)],
            'AGE': [D(20 + i % 40) for i in range(rows)],
            'GENDER': ['M', 'F'] * (rows // 2),
        }
    )
    ratings = pd.DataFrame(
        {
            'RID': np.arange(rows * 5),
            'USERID': [D(1 + i % rows) for i in range(rows * 5)],
            'STARS': [D(1 + i % 5) for i in range(rows * 5)],
        }
    )
    return {'users': users, 'ratings': ratings}


def warehouse_schema(**overrides) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            warehouse(),
            primary_keys=overrides.pop(
                'primary_keys', {'users': 'USERID', 'ratings': 'RID'}
            ),
            **overrides,
        )


def test_the_relational_path_takes_a_decimal_frame():
    """It refused with `dtype '<class 'decimal.Decimal'>' not understood`,
    naming no table, no column and no remedy -- on the same frame the tabular
    path had already answered."""
    pytest.importorskip('sdm_relational_engine')

    schema = warehouse_schema()
    request = PredictiveQuery(
        'PREDICT users.AGE FOR EACH users.USERID', schema
    ).materialize(indices=[D(1), D(2)], model=KumoRelational())

    assert request.predict_rows == 2


def test_the_engine_is_told_the_type_the_schema_computed():
    """The two halves disagreed: the client read the values and called a whole
    Decimal column `int64`, the engine mapped decimal content to float
    unconditionally. An exact identifier in a float is the part that bites."""
    pytest.importorskip('sdm_relational_engine')

    schema = warehouse_schema()
    assert schema['users']['USERID'].dtype == 'int64'

    request = PredictiveQuery(
        'PREDICT users.AGE FOR EACH users.USERID', schema
    ).materialize(indices=[D(1), D(2)], model=KumoRelational())

    columns = request.payloads[0]['schema']['related_tables']['users'][
        'columns'
    ]
    assert columns['USERID']['dtype'] == 'int64'


def test_a_composite_key_over_decimals_materializes():
    """The engine refuses a key part it has typed float, so this was blocked
    outright on warehouse data."""
    pytest.importorskip('sdm_relational_engine')

    schema = warehouse_schema(
        primary_keys={'users': 'USERID', 'ratings': ['USERID', 'RID']}
    )
    request = PredictiveQuery(
        'PREDICT users.AGE FOR EACH users.USERID', schema
    ).materialize(indices=[D(1), D(2)], model=KumoRelational())

    assert request.predict_rows == 2


def test_a_fractional_decimal_stays_a_float():
    from sdm_client.materializers.relational import _to_declared

    values = pd.Series([D('1.5'), D('2.25')])
    assert _to_declared(values, 'float64', 't.v').tolist() == [1.5, 2.25]


def test_a_decimal_column_with_a_gap_does_not_become_a_float():
    """Widening to float to carry the gap would round an identifier."""
    from sdm_client.materializers.relational import _to_declared

    converted = _to_declared(pd.Series([D(1), None, D(3)]), 'int64', 't.v')

    assert str(converted.dtype) == 'Int64'
    assert converted.tolist() == [1, pd.NA, 3]


def test_a_wide_decimal_key_stays_exact():
    """Past 2**53 a float loses the low bits, and these are identifiers."""
    from sdm_client.materializers.relational import _to_declared

    wide = D(9007199254740993)
    converted = _to_declared(pd.Series([wide, D(1)]), 'int64', 't.v')

    assert converted.iloc[0] == 9007199254740993


def test_a_column_that_is_not_decimal_is_untouched():
    from sdm_client.materializers.relational import _as_typed
    from sdm_client.schema import Table

    frame = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame})
    table: Table = schema['t']

    assert _as_typed(table) is table.data


def test_an_untyped_column_does_not_contradict_anything():
    """An all-null context column is typed `string`/`categorical` because a
    schema has to say something, and `untyped` records that it is a placeholder
    rather than a finding. Timestamps in the scored rows cannot mean
    `categorical`, so without that record they would be refused -- against a
    column whose declared meaning was never a claim about the values."""
    frame = pd.DataFrame(
        {
            'rid': range(200),
            'unknown': [None] * 200,
            'amount': [float(i % 9) for i in range(200)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})
    assert schema['t']['unknown'].untyped
    assert schema['t']['unknown'].stype == 'categorical'

    request = PredictiveQuery(
        'PREDICT t.amount FOR EACH t.rid', schema
    ).materialize(
        predict=pd.DataFrame(
            {
                'rid': [9001, 9002],
                'unknown': pd.to_datetime(['2024-01-01', '2024-02-01']),
            }
        )
    )

    assert wire_column(request, 'unknown')['dtype'] == 'timestamp[us]'


def test_a_column_that_was_typed_still_refuses_a_contradiction():
    """The control for the test above: the same values, against a column whose
    type was a finding rather than a placeholder."""
    schema = context()
    with pytest.raises(StructuredDataError):
        query(schema).materialize(
            predict=pd.DataFrame(
                {
                    'rid': [9001, 9002],
                    'score': pd.to_datetime(['2024-01-01', '2024-02-01']),
                    'tier': ['a', 'b'],
                }
            )
        )


def test_a_whole_decimal_too_wide_for_an_int_keeps_every_digit():
    """The schema types these `string` precisely so the digits survive.
    Reading "not an int" as "a float" put 9223372036854775808 back as
    9.223372036854776e+18 -- the corruption this exists to prevent,
    reintroduced one branch over."""
    from sdm_client.materializers.relational import _to_declared
    from sdm_client.wire.dtypes import infer_tfm_dtype

    values = pd.Series([D('9223372036854775808'), D(1)])
    assert infer_tfm_dtype(values) == 'string'

    converted = _to_declared(values, 'string', 't.k')

    assert converted.tolist() == ['9223372036854775808', '1']


def test_a_fractional_value_in_a_column_declared_int_is_refused():
    """Truncating 1.9 to 1 before the engine sees it is a wrong answer with
    nothing to say so. A schema the caller edited can ask for this."""
    from sdm_client.materializers.relational import _to_declared

    with pytest.raises(StructuredDataError) as raised:
        _to_declared(pd.Series([D('1.9'), D('2.1')]), 'int64', 'sales.qty')
    message = str(raised.value)
    assert 'sales.qty' in message
    assert 'whole number' in message


def test_a_null_is_a_null_whichever_way_the_column_was_read():
    """An Arrow-backed decimal column's gaps are `pd.NA`, not `None`, and
    `int(pd.NA)` raises."""
    from sdm_client.materializers.relational import _holds_decimal, _to_declared

    pa = pytest.importorskip('pyarrow')
    values = pd.Series(
        pd.array([D(1), None, D(3)], dtype=pd.ArrowDtype(pa.decimal128(10, 0)))
    )
    assert _holds_decimal(values)

    converted = _to_declared(values, 'int64', 't.v')

    assert str(converted.dtype) == 'Int64'
    assert converted.tolist() == [1, pd.NA, 3]


def test_a_request_built_by_hand_is_checked_too():
    """It carries no schema, so one is inferred from the context -- which
    happens after the column checks, so checking there left the request with
    no query to have validated it as the only one uncovered."""
    from sdm_client.request import Request

    context_frame = pd.DataFrame(
        {
            'rid': range(200),
            'score': [float(i) for i in range(200)],
            'amount': [float(i * 3) for i in range(200)],
        }
    )
    with pytest.raises(StructuredDataError) as raised:
        Request(
            context=context_frame,
            predict=pd.DataFrame({'rid': [9001], 'score': ['100.0']}),
            target='amount',
            task='regression',
        )
    assert "column 'score'" in str(raised.value)


def test_a_request_built_by_hand_with_good_values_is_untouched():
    from sdm_client.request import Request

    context_frame = pd.DataFrame(
        {
            'rid': range(200),
            'score': [float(i) for i in range(200)],
            'amount': [float(i * 3) for i in range(200)],
        }
    )
    request = Request(
        context=context_frame,
        predict=pd.DataFrame({'rid': [9001], 'score': [100.0]}),
        target='amount',
        task='regression',
    )
    assert request.predict_rows == 1


def placeholder_schema() -> Schema:
    frame = pd.DataFrame(
        {
            'rid': range(200),
            'unknown': [None] * 200,
            'amount': [float(i % 9) for i in range(200)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})


def test_a_placeholder_meaning_is_not_a_claim():
    schema = placeholder_schema()
    assert schema['t']['unknown'].untyped
    assert not schema['t']['unknown'].stype_declared


def test_assigning_a_meaning_records_that_somebody_chose_it():
    schema = placeholder_schema()
    schema['t']['unknown'].stype = 'text'
    assert schema['t']['unknown'].stype_declared


def test_a_meaning_the_caller_assigned_is_honoured_against_the_rows():
    """Skipping every untyped column was too broad: once a caller says what a
    column means, the scored rows have to mean it too."""
    schema = placeholder_schema()
    schema['t']['unknown'].stype = 'text'

    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery('PREDICT t.amount FOR EACH t.rid', schema).materialize(
            predict=pd.DataFrame({'rid': [10], 'unknown': [123]})
        )
    assert "column 'unknown'" in str(raised.value)


@pytest.mark.parametrize(
    'values, dtype, stype',
    [
        (pd.to_datetime(['2024-01-01']), 'timestamp[us]', 'timestamp'),
        ([1], 'int64', 'categorical'),
        (['a'], 'string', 'categorical'),
        ([1.5], 'float64', 'categorical'),
    ],
    ids=['timestamps', 'ints', 'strings', 'floats'],
)
def test_a_placeholder_never_goes_out_contradicting_itself(
    values, dtype, stype
):
    """The placeholder `categorical` cannot carry a timestamp, so the column
    went out as `timestamp[us]` meaning `categorical` -- the same
    self-contradiction on the wire that a re-typed column used to produce."""
    schema = placeholder_schema()
    request = PredictiveQuery(
        'PREDICT t.amount FOR EACH t.rid', schema
    ).materialize(predict=pd.DataFrame({'rid': [10], 'unknown': values}))

    column = wire_column(request, 'unknown')

    assert column == {'dtype': dtype, 'stype': stype}
    from sdm_client.schema import supports_dtype

    assert supports_dtype(column['stype'], column['dtype'])


def test_a_declared_meaning_is_never_quietly_rewritten(recwarn):
    """`text` is the one meaning this model cannot read at all, so it is
    carried as categorical rather than sent to be refused. That is a rewrite
    of what the caller declared, which is why it is said out loud: the rule is
    not that a declared meaning is never changed, but that it is never changed
    in silence."""
    schema = placeholder_schema()
    schema['t']['unknown'].stype = 'text'
    request = PredictiveQuery(
        'PREDICT t.amount FOR EACH t.rid', schema
    ).materialize(predict=pd.DataFrame({'rid': [10], 'unknown': ['some text']}))

    assert wire_column(request, 'unknown')['stype'] == 'categorical'
    said = ' '.join(str(w.message) for w in recwarn)
    assert 'cannot read a text column' in said
    assert 'unknown' in said


def test_a_copied_column_keeps_what_was_recorded_about_it():
    from sdm_client.schema import _copy_column

    schema = placeholder_schema()
    schema['t']['unknown'].stype = 'text'
    copied = _copy_column(schema['t']['unknown'])

    assert copied.untyped
    assert copied.stype_declared


def test_the_fallback_leaves_a_declared_meaning_alone():
    """Tested on the function rather than through `materialize`, because a
    declared meaning that contradicts its values is refused before a request
    gets this far. The guard is what keeps the rewrite safe to have at all: it
    may correct a placeholder and must never overrule a caller."""
    from sdm_client.adapters.tabular import _coherent_stype
    from sdm_client.schema import Column

    column = Column(name='c', dtype='string', stype='categorical')
    object.__setattr__(column, 'untyped', True)

    assert _coherent_stype('categorical', 'timestamp[us]', column) == (
        'timestamp'
    )

    object.__setattr__(column, 'stype_declared', True)

    assert _coherent_stype('categorical', 'timestamp[us]', column) == (
        'categorical'
    )


def test_a_zero_row_batch_still_has_to_mean_what_it_says():
    """`pd.Series([], dtype='int64')` still says int64. Skipping empty frames
    let a zero-row batch build the same contradictory envelope as a full one."""
    from sdm_client.request import Request

    schema = placeholder_schema()
    schema['t']['unknown'].stype = 'text'

    with pytest.raises(StructuredDataError):
        Request(
            context=schema['t'].data,
            predict=pd.DataFrame(
                {
                    'rid': pd.Series([], dtype='int64'),
                    'unknown': pd.Series([], dtype='int64'),
                }
            ),
            target='amount',
            task='regression',
            schema=schema,
            entity_table='t',
        )


def test_a_zero_row_batch_carrying_no_type_is_not_refused():
    """An empty object column reports no type at all, which is not a claim
    about anything and must not be read as one."""
    from sdm_client.request import Request

    schema = placeholder_schema()
    schema['t']['unknown'].stype = 'text'

    request = Request(
        context=schema['t'].data,
        predict=pd.DataFrame(
            {
                'rid': pd.Series([], dtype='int64'),
                'unknown': pd.Series([], dtype=object),
            }
        ),
        target='amount',
        task='regression',
        schema=schema,
        entity_table='t',
    )
    assert request.predict_rows == 0


# --- a declared bool is read back as one --------------------------------


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('false', False),
        ('true', True),
        ('False', False),
        (0, False),
        (1, True),
        (True, True),
        ('maybe', 'maybe'),
        (2, 2),
    ],
)
def test_a_declared_bool_is_read_from_what_the_wire_spells(value, expected):
    """`bool('false')` is True, which would invert the answer, so the spellings
    are matched rather than cast. Anything else is returned untouched."""
    read = as_declared(value, 'bool')

    assert read == expected
    assert type(read) is type(expected)


def test_a_bool_target_is_restored_like_an_int_one():
    """A `> 0` query declares a bool target and one endpoint spells it
    'false', so `prediction == False` counted none of them."""
    items = [
        PredictionItem(row_index=0, prediction='false'),
        PredictionItem(row_index=1, prediction='true'),
    ]
    payload = {
        'task': {
            'kind': 'binary_classification',
            'target': {'column_name': 'TARGET', 'dtype': 'bool'},
        }
    }

    read = as_declared_target(items, payload)

    assert [item.prediction for item in read] == [False, True]


@pytest.mark.parametrize(
    'spelled',
    [{'false': 0.6, 'true': 0.4}, {'False': 0.6, 'True': 0.4}],
    ids=['as one endpoint spells it', 'as the other does'],
)
def test_a_bool_target_names_its_probability_columns_the_same_either_way(
    spelled,
):
    """`frame['probability_True']` worked on one model and raised on the other
    for the same declared column, which is what restoring `prediction` alone
    left behind."""
    payload = {
        'task': {
            'kind': 'binary_classification',
            'target': {'column_name': 'TARGET', 'dtype': 'bool'},
        }
    }
    items = [
        PredictionItem(row_index=0, prediction='false', probabilities=spelled)
    ]

    read = as_declared_target(items, payload)

    assert read[0].probabilities == {'False': 0.6, 'True': 0.4}


def test_an_integer_target_keeps_the_probability_keys_it_was_given():
    payload = {
        'task': {
            'kind': 'binary_classification',
            'target': {'column_name': 'TARGET', 'dtype': 'int64'},
        }
    }
    items = [
        PredictionItem(
            row_index=0, prediction='1', probabilities={'0': 0.2, '1': 0.8}
        )
    ]

    assert as_declared_target(items, payload)[0].probabilities == {
        '0': 0.2,
        '1': 0.8,
    }


def test_probabilities_that_would_collapse_are_left_as_they_came():
    payload = {
        'task': {
            'kind': 'binary_classification',
            'target': {'column_name': 'TARGET', 'dtype': 'bool'},
        }
    }
    given = {'true': 0.4, 'True': 0.6}
    items = [
        PredictionItem(row_index=0, prediction='true', probabilities=given)
    ]

    assert as_declared_target(items, payload)[0].probabilities == given


@pytest.mark.parametrize(
    ('column', 'expected'),
    [
        (pd.Series([D(2**64 + i) for i in range(20)], dtype=object), 'string'),
        (
            pd.Series(
                [np.uint64(2**64 - 1 - i) for i in range(20)], dtype=object
            ),
            'string',
        ),
        (pd.Series([2**64 + i for i in range(20)], dtype=object), 'string'),
        (pd.Series(np.array([1, 2, 3], dtype='uint64')), 'int64'),
        (pd.Series([1, 2, 3], dtype='int64'), 'int64'),
        (pd.Series([1, 2, 3], dtype='int32'), 'int64'),
        (pd.Series([1, None, 3], dtype='Int64'), 'int64'),
    ],
    ids=[
        'Decimal',
        'uint64 object',
        'python int',
        'uint64 in range',
        'int64',
        'int32',
        'Int64 with a null',
    ],
)
def test_a_whole_number_is_typed_by_whether_it_fits_not_by_how_it_arrives(
    column, expected
):
    """The too-wide rule was applied to a decimal column and not the integer
    column beside it, so a `uint64` past the range was announced as `int64` --
    a dtype the values provably cannot hold."""
    from sdm_client.wire import infer_tfm_dtype

    assert infer_tfm_dtype(column) == expected


def test_a_column_typed_string_for_its_width_serializes():
    """The point of the rule: the dtype the schema names has to be one the
    values can travel under."""
    from sdm_client.wire.dtypes import serialize_column

    wide = pd.Series([np.uint64(2**64 - 1), np.uint64(2**64 - 2)], dtype=object)
    assert serialize_column(wide, 'string') == [
        '18446744073709551615',
        '18446744073709551614',
    ]
