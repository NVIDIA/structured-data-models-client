# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A response that does not match the contract is refused, not coerced.

`_require_int` was `int(value)`, which accepts a bool and truncates a float --
so a `row_index` of 1.2 became 1, attaching a prediction meant for no row to
row 1, which a join on `row_index` would then honour. The client is careful in
the other direction: it refuses to *send* a float that cannot be an int64, or a
non-finite number. Accepting those on the way back was the inconsistency.

These live in `sdm-api`, which is dependency-free and raises
`ResponseContractError`; `sdm-client`'s adapters turn that into
`INVALID_RESPONSE`.
"""

from __future__ import annotations

from math import inf, nan

import pytest

from sdm_api.generated.tfm_api import PredictionResponse, ResponseContractError


def parse(**item):
    return PredictionResponse.from_dict(
        {'id': 'r', 'model': 'm', 'predictions': [item], 'metadata': {}}
    ).predictions[0]


# --- row_index is a join key, so it has to be a row index -------------------


@pytest.mark.parametrize('value', [1.2, -0.5, 3.999])
def test_a_fractional_row_index_is_refused(value):
    with pytest.raises(ResponseContractError) as error:
        parse(row_index=value)
    assert 'whole number' in str(error.value)


def test_a_bool_row_index_is_refused():
    """A bool is an int in Python, so `int(True)` used to give row 1."""
    with pytest.raises(ResponseContractError):
        parse(row_index=True)


@pytest.mark.parametrize('value', ['1', None, [1], {'a': 1}])
def test_a_row_index_that_is_not_a_number_is_refused(value):
    with pytest.raises(ResponseContractError):
        parse(row_index=value)


@pytest.mark.parametrize('value', [nan, inf, -inf])
def test_a_non_finite_row_index_is_refused(value):
    with pytest.raises(ResponseContractError):
        parse(row_index=value)


@pytest.mark.parametrize('value', [0, 1, 1.0, 42])
def test_a_whole_row_index_is_accepted(value):
    assert parse(row_index=value).row_index == int(value)


# --- an optional integer is held to the same rule ---------------------------


def test_a_fractional_forecast_step_is_refused():
    with pytest.raises(ResponseContractError):
        parse(row_index=0, forecast_step=2.5)


def test_a_missing_forecast_step_stays_missing():
    assert parse(row_index=0).forecast_step is None


def test_a_whole_forecast_step_is_accepted():
    assert parse(row_index=0, forecast_step=3).forecast_step == 3


# --- a probability that is not a number is not a probability ----------------


@pytest.mark.parametrize('field', ['probabilities', 'quantiles'])
def test_a_non_finite_mapping_value_is_refused(field):
    with pytest.raises(ResponseContractError) as error:
        parse(row_index=0, **{field: {'a': nan}})
    assert 'non-finite' in str(error.value)


@pytest.mark.parametrize('field', ['scores', 'embeddings'])
def test_a_non_finite_sequence_value_is_refused(field):
    with pytest.raises(ResponseContractError):
        parse(row_index=0, **{field: [1.0, inf]})


def test_ordinary_probabilities_are_accepted():
    item = parse(
        row_index=0, prediction='yes', probabilities={'a': 0.5, 'b': 0.5}
    )
    assert item.probabilities == {'a': 0.5, 'b': 0.5}


# --- a required string is a string ------------------------------------------


def test_a_response_id_that_is_not_a_string_is_refused():
    with pytest.raises(ResponseContractError):
        PredictionResponse.from_dict(
            {'id': 7, 'model': 'm', 'predictions': [], 'metadata': {}}
        )


def test_a_well_formed_response_is_unchanged():
    parsed = PredictionResponse.from_dict(
        {
            'id': 'r',
            'model': 'kumo-tabular',
            'predictions': [
                {
                    'row_index': 1,
                    'prediction': 'yes',
                    'probabilities': {'yes': 0.9},
                },
                {
                    'row_index': 0,
                    'prediction': 'no',
                    'probabilities': {'yes': 0.1},
                },
            ],
            'metadata': {'k': 'v'},
        }
    )
    assert [item.row_index for item in parsed.predictions] == [1, 0]
    assert parsed.model == 'kumo-tabular'


def test_an_entity_id_that_is_not_a_string_is_refused():
    """Every other optional field on a prediction validates; this one used to
    coerce.

    A response is correlated back to the caller's rows by this id, so a server
    sending `7` where the contract says a string was quietly turned into `'7'`
    and matched against something the caller never sent.
    """
    with pytest.raises(ResponseContractError) as error:
        parse(row_index=0, id=7)
    assert "'id'" in str(error.value)


def test_an_entity_id_that_is_absent_or_null_is_still_allowed():
    """The field is optional in the contract; refusing a type is not the same
    as requiring a value."""
    assert parse(row_index=0).id is None
    assert parse(row_index=0, id=None).id is None
    assert parse(row_index=0, id='e1').id == 'e1'
