# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Python host reads every result the core gives through its own typed
structures (T3.16, the structure design's fix D and rule 10): a sample of
each ``OperationResult`` variant (:data:`messages.RESULTS`) decodes once
into frozen dataclasses of ``sdm_client._rust.results``, and a result of a
shape the contract does not hold is refused by name, never read as it is."""

from __future__ import annotations

import copy
import dataclasses

import pytest
from messages import RESULTS, SOURCE_FACTS
from sdm_client._rust import results
from sdm_client._rust.codec import CodecError


def _frozen(value: object) -> None:
    assert dataclasses.is_dataclass(value)
    field = dataclasses.fields(value)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, field, None)


def test_every_result_variant_has_a_sample() -> None:
    assert set(RESULTS) == {
        'validate_query',
        'describe_source',
        'edit_schema',
        'materialize',
        'predict',
    }


def test_a_validation_decodes_typed() -> None:
    validated = results.decode(RESULTS['validate_query'])
    assert isinstance(validated, results.Validated)
    _frozen(validated)
    assert validated.accepted is True
    assert validated.query == results.QueryFacts(
        entity_table='users',
        problem_type='forecast',
        windowed=True,
        num_forecasts=1,
        text='PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
    )
    [diagnostic] = validated.diagnostics
    assert (diagnostic.code, diagnostic.severity) == ('W1', 'warning')


def test_a_description_and_an_edit_decode_typed() -> None:
    described = results.decode(RESULTS['describe_source'])
    assert described == results.Described(description=3)
    edited = results.decode(RESULTS['edit_schema'])
    assert edited == results.Edited(schema=4, declared=('users', 'age'))
    _frozen(edited)


def test_a_materialized_request_decodes_typed() -> None:
    gathered = results.decode(RESULTS['materialize'])
    assert isinstance(gathered, results.Gathered)
    _frozen(gathered)
    state = gathered.state
    assert (state.version, state.payload, state.query) == (2, 5, 6)
    assert state.forecast == (results.Step(step=2, payload=7, query=8),)
    assert state.batches == (results.Batch(payload=9, rows=40),)
    assert state.ranking == results.Ranking(
        top_k=3,
        candidate=0,
        value='"a"',
        parts=(results.Part(candidate=1, value='"b"', payload=10, query=11),),
    )
    assert gathered.query is not None
    assert (gathered.query.windowed, gathered.query.num_forecasts) == (False, 2)


def test_a_prediction_decodes_typed() -> None:
    predicted = results.decode(RESULTS['predict'])
    assert predicted == results.Predicted(
        result=12,
        model='kumo-relational',
        request_id='sdm-1',
        metadata=13,
        outputs=('prediction',),
        unscored_candidates=(2, 4),
    )
    _frozen(predicted)


def test_source_facts_read_and_write_as_the_contract_spells_them() -> None:
    facts = results.SourceFacts.from_contract(SOURCE_FACTS)
    _frozen(facts)
    assert (facts.dtype, facts.nullable, facts.unique) == (
        'string',
        False,
        True,
    )
    assert facts.exact_number == 'declared_key'
    assert facts.to_contract() == SOURCE_FACTS
    sparse = {
        key: SOURCE_FACTS[key]
        for key in ('relation', 'dtype', 'received', 'nullable', 'unique')
    }
    assert results.SourceFacts.from_contract(sparse).to_contract() == sparse


@pytest.mark.parametrize('op', sorted(RESULTS))
def test_a_result_of_another_shape_is_refused_by_name(op: str) -> None:
    extra = {**copy.deepcopy(RESULTS[op]), 'unknown': 1}
    with pytest.raises(CodecError):
        results.decode(extra)
    missing = copy.deepcopy(RESULTS[op])
    del missing[next(key for key in missing if key != 'op')]
    with pytest.raises(CodecError):
        results.decode(missing)


def test_an_operation_the_host_does_not_know_is_refused_by_name() -> None:
    with pytest.raises(CodecError):
        results.decode({'op': 'transmogrify'})
