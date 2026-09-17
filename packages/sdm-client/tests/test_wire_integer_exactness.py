# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An integer float64 cannot hold exactly must not travel as a number.

The rule was already enforced for a Python ``int``. It was not enforced for a
``Decimal`` (which is what a warehouse driver hands back for ``DECIMAL`` and
``NUMBER``) or for a numpy integer, because the guard asked what type the value
was rather than what it was worth. A ``uint64`` satisfied the rule while still
exceeding the dtype being declared.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from sdm_client.errors import StructuredDataError
from sdm_client.wire.dtypes import (
    INT64_MAX,
    JSON_SAFE_INT_MAX,
    infer_tfm_dtype,
    json_ready,
    serialize_cell,
    serialize_column,
)

BIG = JSON_SAFE_INT_MAX + 2


def cells(values, dtype=None):
    series = pd.Series(values, dtype=dtype or object)
    return serialize_column(series, infer_tfm_dtype(series))


# --- a whole-number Decimal is an integer, and travels as one ---------------


def test_a_whole_decimal_column_is_typed_as_the_integer_it_is():
    series = pd.Series([Decimal(str(BIG)), Decimal('1')], dtype=object)
    assert infer_tfm_dtype(series) == 'int64'


def test_a_whole_decimal_past_the_safe_range_travels_exactly():
    assert cells([Decimal(str(BIG)), Decimal('1')]) == [str(BIG), 1]


def test_a_null_does_not_change_the_type_of_a_decimal_column():
    assert cells([Decimal(str(BIG)), None]) == [str(BIG), None]


def test_a_decimal_nan_is_a_null_like_any_other():
    assert cells([Decimal('NaN'), Decimal('1')]) == [None, 1]


def test_a_fractional_decimal_is_still_a_float():
    series = pd.Series([Decimal('12.3456')], dtype=object)
    assert infer_tfm_dtype(series) == 'float64'
    assert cells([Decimal('12.3456'), Decimal('1.00')]) == [12.3456, 1.0]


def test_a_decimal_too_wide_for_int64_travels_as_a_string():
    """The digits survive, which is what the relational encoder does too."""
    series = pd.Series(
        [Decimal('123456789012345678901234567890')], dtype=object
    )
    assert infer_tfm_dtype(series) == 'string'
    assert cells([Decimal('123456789012345678901234567890')]) == [
        '123456789012345678901234567890'
    ]


@pytest.mark.parametrize(
    'value', [2**63, 2**64, 10**20], ids=['int64 max + 1', '2**64', '10**20']
)
def test_a_whole_decimal_just_past_int64_is_not_rounded_into_a_float(value):
    """`float(2**63)` is exact, so an exactness check alone would pass it."""
    assert cells([Decimal(str(value))]) == [str(value)]


def test_the_int64_boundary_decimal_is_still_an_integer():
    assert (
        infer_tfm_dtype(pd.Series([Decimal(str(INT64_MAX))], dtype=object))
        == 'int64'
    )


def test_a_decimal_too_large_for_a_float_does_not_raise_overflowerror():
    """`float(Decimal('1e400'))` is `inf`, and `int(inf)` is an OverflowError."""
    assert cells([Decimal('1e400')]) == ['1' + '0' * 400]


def test_a_wide_decimal_travels_as_digits_not_scientific_notation():
    """The whole point of routing these through `string` is keeping the digits.

    `str(Decimal)` switches to an exponent as soon as the exponent is
    positive, so this column used to go out as `'1E+400'` -- which no consumer
    reading it as an integer can parse, and which this test previously
    asserted was correct.
    """
    for value in (Decimal('1e400'), Decimal('1E+30'), Decimal(str(2**70))):
        sent = cells([value])[0]
        assert 'E' not in sent and 'e' not in sent, sent
        assert int(sent) == int(value)


def test_a_column_mixing_a_wide_decimal_with_a_small_one_travels_whole():
    assert cells([Decimal(str(2**63)), Decimal('1')]) == [str(2**63), '1']


def test_a_fractional_decimal_past_the_safe_range_is_refused():
    """Past 2**53 a float64 cannot hold even the integer part."""
    with pytest.raises(StructuredDataError) as error:
        cells([Decimal('12345678901234567.8901')])
    assert 'too large' in str(error.value)


def test_a_decimal_infinity_is_refused_before_the_request_is_built():
    """It used to fail inside the JSON encoder, as a transport error."""
    with pytest.raises(StructuredDataError) as error:
        cells([Decimal('Infinity')])
    assert error.value.code == 'INVALID_REQUEST'
    assert 'JSON-representable' in str(error.value)


# --- the guard is about the value, not the type -----------------------------


@pytest.mark.parametrize(
    'value',
    [BIG, Decimal(str(BIG)), np.int64(BIG)],
    ids=['int', 'Decimal', 'np.int64'],
)
def test_every_kind_of_integer_is_refused_in_a_float64_column(value):
    """The three used to disagree: only the Python int was caught."""
    with pytest.raises(StructuredDataError):
        serialize_cell(value, 'float64')


def test_a_representable_integer_in_a_float64_column_still_passes():
    assert serialize_cell(3, 'float64') == 3.0
    assert serialize_cell(Decimal('3'), 'float64') == 3.0
    assert serialize_cell(np.int64(3), 'float64') == 3.0


# --- a value outside int64 cannot travel under int64 ------------------------


def test_a_uint64_column_inside_the_range_is_unchanged():
    series = pd.Series([5, 6], dtype='uint64')
    assert infer_tfm_dtype(series) == 'int64'
    assert serialize_column(series, 'int64') == [5, 6]


@pytest.mark.parametrize('dtype', ['uint64', 'UInt64'])
def test_a_uint64_past_int64_is_refused_by_name(dtype):
    series = pd.Series([2**64 - 1, 1], dtype=dtype)
    with pytest.raises(StructuredDataError) as error:
        serialize_column(series, infer_tfm_dtype(series))
    assert 'outside the range' in str(error.value)
    assert str(INT64_MAX) in str(error.value)


def test_the_int64_boundary_itself_is_allowed():
    assert serialize_cell(INT64_MAX, 'int64') == str(INT64_MAX)
    with pytest.raises(StructuredDataError):
        serialize_cell(INT64_MAX + 1, 'int64')


# --- nothing that worked before stopped working -----------------------------


def test_a_plain_int_column_is_unaffected():
    series = pd.Series([BIG, 1])
    assert infer_tfm_dtype(series) == 'int64'
    assert serialize_column(series, 'int64') == [str(BIG), 1]


def test_a_float_column_is_unaffected():
    series = pd.Series([1.5, 2.5])
    assert serialize_column(series, infer_tfm_dtype(series)) == [1.5, 2.5]


def test_a_large_float_is_still_sent_as_a_float():
    """Only an exact type is held to exactness; a float was never exact."""
    assert cells([1.2345678901234568e16], dtype='float64') == [
        1.2345678901234568e16
    ]


def test_a_bool_is_not_an_integer_for_this_purpose():
    assert serialize_cell(True, 'float64') == 1.0


# --- a decimal is a decimal however pandas stores it -------------------------


def test_an_arrow_backed_decimal_is_named_like_an_object_one():
    """Read through Arrow, a DECIMAL used to be announced as `string`.

    None of the pandas dtype predicates match an `ArrowDtype`, so the column
    fell through to the final `'string'`. An id column announced as a string
    stops being an id to the model, and the connectors prefer the Arrow path
    precisely because it is the one that keeps a wide integer exact.
    """
    pa = pytest.importorskip('pyarrow')

    whole = pd.Series(
        [Decimal('123')], dtype=pd.ArrowDtype(pa.decimal128(38, 0))
    )
    fractional = pd.Series(
        [Decimal('1.50')], dtype=pd.ArrowDtype(pa.decimal128(38, 2))
    )
    assert infer_tfm_dtype(whole) == 'int64'
    assert infer_tfm_dtype(fractional) == 'float64'
    assert infer_tfm_dtype(whole) == infer_tfm_dtype(
        pd.Series([Decimal('123')], dtype=object)
    )


def test_a_narrow_arrow_decimal_is_answered_from_the_type_alone():
    """18 digits always fit int64, so no cell has to be read to know it."""
    pa = pytest.importorskip('pyarrow')

    series = pd.Series(
        [Decimal('1')], dtype=pd.ArrowDtype(pa.decimal128(18, 0))
    )
    assert infer_tfm_dtype(series) == 'int64'


def test_a_python_int_too_large_for_a_float_is_refused_not_raised_raw():
    """`float(10**400)` raises OverflowError rather than returning `inf`.

    A Python int has no width, so the conversion refuses outright where a
    wide Decimal quietly becomes `inf`. Same fault, different exception, and
    the finite guard below never ran for this one.
    """
    with pytest.raises(StructuredDataError) as error:
        serialize_cell(10**400, 'float64')
    assert 'outside the range' in str(error.value)
    assert error.value.code == 'INVALID_REQUEST'


def test_an_ordinary_float_is_untouched_by_that_guard():
    assert serialize_cell(1.5, 'float64') == 1.5
    assert serialize_cell(2**53, 'float64') == float(2**53)


def test_a_decimal_just_past_the_cutoff_is_not_rounded_into_range():
    """The check used to run on the float, which had already rounded it.

    `float(Decimal('9007199254740991.1'))` is exactly the cutoff, so a value
    above it compared as equal and was let through as the rounded number.
    """
    with pytest.raises(StructuredDataError) as error:
        serialize_cell(Decimal('9007199254740991.1'), 'float64')
    assert 'too large' in str(error.value)


def test_a_decimal_below_the_cutoff_still_rounds_as_a_float_does():
    assert serialize_cell(Decimal('1.5'), 'float64') == 1.5


def test_an_integer_a_float_cannot_hold_is_refused_not_rounded():
    """The other half of the cross-encoder invariant.

    `sdm-relational-engine`'s `test_json_payload.py` asserts the same value is
    refused there. Kept as two tests because neither package's test job
    installs the other, and a skipped test would prove nothing in CI.
    """
    with pytest.raises(StructuredDataError):
        serialize_cell(2**53 + 1, 'float64')


def test_a_mapping_that_nests_too_deeply_is_refused_before_the_stack_runs_out():
    """This walk happens while building the payload, above the transport's
    own recursion guard, so a RecursionError here escaped uncaught."""
    from sdm_client.wire.dtypes import json_ready

    deep: dict = {}
    cursor = deep
    for _ in range(5_000):
        nxt: dict = {}
        cursor['x'] = nxt
        cursor = nxt

    with pytest.raises(StructuredDataError) as error:
        json_ready(deep)
    assert 'nests more than' in str(error.value)


def test_a_mapping_that_refers_to_itself_is_refused():
    from sdm_client.wire.dtypes import json_ready

    cyclic: dict = {}
    cyclic['self'] = cyclic
    with pytest.raises(StructuredDataError):
        json_ready(cyclic)


def test_an_ordinary_nested_value_is_untouched():
    from sdm_client.wire.dtypes import json_ready

    assert json_ready({'a': [1, 2], 'b': {'c': 3}}) == {
        'a': [1, 2],
        'b': {'c': 3},
    }


def test_a_numeric_string_a_float_cannot_hold_is_refused():
    """A string is exact text, and `float()` rounds it without complaint.

    Reachable whenever a caller names `float64` on a column of numeric
    strings, which is the shape a warehouse read of a wide DECIMAL takes.
    """
    with pytest.raises(StructuredDataError) as error:
        serialize_cell('9223372036854775809', 'float64')
    assert 'losing' in str(error.value)


def test_an_ordinary_numeric_string_still_converts():
    assert serialize_cell('1.5', 'float64') == 1.5
    assert serialize_cell('42', 'float64') == 42.0


def test_a_float_wider_than_float64_is_refused_where_one_exists():
    """`longdouble` is 80- or 128-bit on x86 and plain float64 on arm64.

    Skipped rather than asserted away where the platform has nothing wider,
    because there is genuinely no precision to lose there.
    """
    wide = np.longdouble('9007199254740993')
    if wide.dtype.itemsize <= 8:
        pytest.skip('this platform has no float wider than float64')
    with pytest.raises(StructuredDataError):
        serialize_cell(wide, 'float64')


class TestInferenceValuesAreHeldToTheSameRule:
    """`inference=` is hand-written payload, not a typed column.

    The rule that made this branch necessary -- an integer a float64 cannot
    hold exactly must not travel as a JSON number -- was enforced on table
    cells and not here, so `inference={'seed': 2**53 + 1}` reached the wire as
    a number and a reader on a float64 JSON stack got a different seed.
    """

    def test_an_integer_a_float64_cannot_hold_is_refused(self):
        with pytest.raises(StructuredDataError) as error:
            json_ready({'seed': 2**53 + 1})
        assert error.value.code == 'INVALID_REQUEST'

    def test_a_numpy_integer_is_refused_by_its_value_not_its_type(self):
        with pytest.raises(StructuredDataError):
            json_ready({'seed': np.int64(2**53 + 1)})
        assert json_ready({'seed': np.int64(4)}) == {'seed': 4}

    def test_an_unsigned_integer_past_int64_is_refused(self):
        with pytest.raises(StructuredDataError):
            json_ready({'seed': np.uint64(2**64 - 1)})

    def test_the_boundary_is_what_a_float64_actually_holds(self):
        """Not the safe-integer bound: 2**53 is exact, and so is every even
        number above it. Refusing on the bound alone rejects values that
        travel perfectly well."""
        assert json_ready(2**53) == 2**53
        assert json_ready(2**53 + 2) == 2**53 + 2
        assert json_ready(-(2**53)) == -(2**53)
        with pytest.raises(StructuredDataError):
            json_ready(2**53 + 1)
        with pytest.raises(StructuredDataError):
            json_ready(-(2**53) - 1)

    def test_a_bool_is_not_an_integer_for_this_purpose(self):
        assert json_ready({'explain': True}) == {'explain': True}

    def test_a_nested_value_is_reached(self):
        with pytest.raises(StructuredDataError):
            json_ready({'opts': {'seeds': [1, 2**53 + 1]}})


def test_a_null_does_not_send_an_integer_key_through_a_float():
    """Pandas has no integer that holds a null, so a key column of
    `[None, 902]` arrives as float. Widening on that put `float64` on the wire
    for a column the schema calls `int64`, which is how an identifier loses its
    low bit: an id past the float53 boundary left as a different id.
    """
    import warnings

    from sdm_client import PredictiveQuery, Schema
    from sdm_client.adapters.tabular import TabularAdapter

    frame = pd.DataFrame(
        {'user_id': range(1, 21), 'tenure': [1.0] * 20, 'churn': ['a'] * 20}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': frame}, primary_keys={'users': 'user_id'}
        )
        request = PredictiveQuery(
            'PREDICT users.churn FOR EACH users.user_id', schema
        ).materialize(
            predict=pd.DataFrame({'user_id': [None, 902], 'tenure': [3.0, 4.0]})
        )
    payload = list(TabularAdapter().prediction_plan(request, ('prediction',)))[
        0
    ]
    columns = payload['schema']['instance_table']['columns']

    assert columns['user_id']['dtype'] == 'int64'
    assert payload['predict']['instance_table']['rows'][1][0] == 902


def test_a_column_that_is_genuinely_float_stays_float():
    """The values decide, not the presence of a null. A float column of whole
    numbers is still a float, because nothing declared it otherwise."""
    import warnings

    from sdm_client import PredictiveQuery, Schema
    from sdm_client.adapters.tabular import TabularAdapter

    frame = pd.DataFrame(
        {'user_id': range(1, 21), 'tenure': [1.5] * 20, 'churn': ['a'] * 20}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': frame}, primary_keys={'users': 'user_id'}
        )
        request = PredictiveQuery(
            'PREDICT users.churn FOR EACH users.user_id', schema
        ).materialize(
            predict=pd.DataFrame({'user_id': [901, 902], 'tenure': [None, 4.0]})
        )
    payload = list(TabularAdapter().prediction_plan(request, ('prediction',)))[
        0
    ]

    assert (
        payload['schema']['instance_table']['columns']['tenure']['dtype']
        == 'float64'
    )
