# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Building a request, and refusing one the model cannot answer.

Everything here runs offline. What it exercises is the half of a prediction
that happens before anything is sent, which is where a mistake should be caught.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    Relationship,
    Request,
    Schema,
    SDMClient,
    UnknownModelError,
    UnsupportedRequestError,
)
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.base import check_request_against_model, required_capabilities
from sdm_client.errors import StructuredDataError


def context(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'row_id': range(n),
            'age': [20.0 + i for i in range(n)],
            'churn': [i % 2 for i in range(n)],
        }
    )


def to_score() -> pd.DataFrame:
    return pd.DataFrame({'row_id': [90, 91], 'age': [33.0, 41.0]})


def simple_request(**kwargs) -> Request:
    params = {
        'context': context(),
        'predict': to_score(),
        'target': 'churn',
        'task': 'classification',
    }
    params.update(kwargs)
    return Request(**params)


def test_a_direct_request_describes_itself():
    request = simple_request()
    assert request.context_rows == 20
    assert request.predict_rows == 2
    assert not request.is_relational
    assert request.task.kind == 'classification'
    assert request.task.target.column_name == 'churn'


def test_the_targets_declared_dtype_comes_from_the_schema():
    """The endpoint compares the two and answers 422 when they differ."""
    request = simple_request()
    column = request.schema[request.entity_table]['churn']
    assert request.task.target.dtype == column.dtype


def test_a_request_whose_predict_half_carries_the_target_is_refused():
    predict = to_score().assign(churn=[0, 1])
    with pytest.raises(StructuredDataError) as error:
        simple_request(predict=predict)
    assert 'carries the target column' in str(error.value)


def test_a_context_without_the_target_is_refused():
    with pytest.raises(StructuredDataError) as error:
        simple_request(target='absent')
    assert 'no target column' in str(error.value)


def test_a_context_without_any_known_targets_is_refused():
    unknown_targets = context().assign(churn=pd.NA)
    with pytest.raises(StructuredDataError) as error:
        simple_request(context=unknown_targets)
    assert 'known target' in str(error.value)


def test_a_predict_column_the_context_never_had_is_refused():
    predict = to_score().assign(surprise=[1.0, 2.0])
    with pytest.raises(StructuredDataError) as error:
        simple_request(predict=predict)
    assert 'has never been given' in str(error.value)


def test_an_unknown_task_kind_lists_the_known_ones():
    with pytest.raises(StructuredDataError) as error:
        simple_request(task='clustering')
    assert 'classification' in str(error.value)


def test_a_dict_instead_of_a_frame_names_the_argument_and_the_fix():
    with pytest.raises(StructuredDataError) as error:
        simple_request(context={'a': [1]})
    message = str(error.value)
    assert 'context must be a pandas DataFrame' in message
    assert 'pd.DataFrame' in message


def a_relational_request() -> Request:
    """A request that is relational the way a caller makes one.

    `related_tables=` is refused where it is offered, because nothing could
    send what it built, so the schema is what says a request crosses tables.
    """
    orders = pd.DataFrame({'order_id': [1, 2], 'row_id': [0, 1]})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': context(), 'orders': orders},
            primary_keys={'instance': 'row_id', 'orders': 'order_id'},
            relationships=[
                Relationship('orders', 'row_id', 'instance', 'row_id')
            ],
        )
    return simple_request(schema=schema)


def test_related_tables_make_a_request_relational():
    assert a_relational_request().is_relational


def test_required_capabilities_are_switched_on_only_by_what_is_needed():
    required = required_capabilities(simple_request())
    assert not required.relationships
    assert not required.multiple_tables
    assert not required.forecasting
    assert not required.link_prediction


def test_tabular_refuses_a_relational_request_before_sending_it():
    request = a_relational_request()
    with pytest.raises(UnsupportedRequestError) as error:
        check_request_against_model(
            request, KumoTabular().spec, ('prediction',)
        )
    message = str(error.value)
    assert 'kumo-tabular' in message
    assert 'relationships' in message or 'multiple_tables' in message


def test_relational_accepts_the_same_relational_request():
    check_request_against_model(
        a_relational_request(), KumoRelational().spec, ('prediction',)
    )


def test_related_frames_are_refused_where_they_are_offered():
    """Nothing could send what they built: the relational adapter wants a
    gathered request and the tabular one cannot take a second table."""
    orders = pd.DataFrame({'order_id': [1, 2], 'row_id': [0, 1]})

    with pytest.raises(UnsupportedRequestError, match='orders'):
        simple_request(related_tables={'orders': orders})


def test_a_task_kind_the_model_does_not_serve_is_refused():
    request = simple_request(task='binary_classification')
    with pytest.raises(UnsupportedRequestError) as error:
        check_request_against_model(
            request, KumoTabular().spec, ('prediction',)
        )
    # Tabular takes only the coarse kind and decides binary itself.
    assert 'binary_classification' in str(error.value)
    check_request_against_model(request, KumoRelational().spec, ('prediction',))


def test_an_output_field_the_model_cannot_produce_is_refused_by_name():
    request = simple_request()
    with pytest.raises(UnsupportedRequestError) as error:
        check_request_against_model(
            request, KumoTabular().spec, ('prediction', 'explanation')
        )
    message = str(error.value)
    assert 'explanation' in message
    assert 'probabilities' in message


def test_the_default_outputs_follow_the_task_kind():
    adapter = TabularAdapter()
    assert adapter.default_output(simple_request()) == (
        'prediction',
        'probabilities',
    )
    regression = simple_request(task='regression')
    assert adapter.default_output(regression) == ('prediction',)


def test_an_unknown_model_id_names_the_contract_change_it_would_need():
    client = SDMClient(url='http://localhost:9')
    try:
        with pytest.raises(UnknownModelError) as error:
            client.capabilities('kumo-timeseries')
        message = str(error.value)
        assert 'closed set' in message
        assert 'kumo-tabular' in message
    finally:
        client.close()


def test_capabilities_are_reported_as_the_clients_own_answer():
    client = SDMClient(url='http://localhost:9')
    try:
        capabilities = client.capabilities(KumoTabular())
        assert capabilities.source == 'sdk'
        assert capabilities.single_table
        assert not capabilities.multiple_tables
        assert not capabilities.relationships
        relational = client.capabilities(KumoRelational())
        assert relational.multiple_tables
        assert relational.relationships
        assert 'explanation' in relational.outputs
    finally:
        client.close()


def test_a_plaintext_url_to_a_remote_host_refuses_an_api_key():
    """A credential must not leave over an unencrypted connection."""
    with pytest.raises(StructuredDataError):
        SDMClient(url='http://example.com', api_key='secret')


def test_a_predict_half_missing_a_feature_the_context_carries_is_refused():
    """The endpoint rejects it; name the column here instead."""
    with pytest.raises(StructuredDataError) as error:
        simple_request(predict=to_score().drop(columns=['age']))
    message = str(error.value)
    assert 'age' in message
    assert 'missing' in message


def test_the_target_is_not_required_in_the_predict_half():
    """It is the one context column that must be absent, not present."""
    request = simple_request()
    assert 'churn' not in request.predict.instance_table.columns
