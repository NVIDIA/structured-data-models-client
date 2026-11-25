# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A session's context is chosen once, so what it must exclude is known once.

`fit()` pins a context and every later call scores against it. The request
that opened the session excludes only the rows *it* names, so every call after
the first scores rows sitting in that context with their answers. Unlike the
one-shot path there is no remedy afterwards: the context cannot be changed
once pinned.

Naming the population up front is what lets it be excluded while there is
still a context to exclude it from.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
)
from sdm_client.wire.keys import _named_by, keys_in

RNG = np.random.default_rng(9)


def tabular_session():
    frame = pd.DataFrame(
        {
            'rid': range(600),
            'a': RNG.normal(0, 1, 600),
            'label': RNG.choice([*'ab'], 600),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})
        request = PredictiveQuery(
            'PREDICT t.label FOR EACH t.rid', schema
        ).materialize(indices=[1, 2], num_context_samples=600)
    return request


def relational_session():
    users = pd.DataFrame({'uid': range(200), 'seg': RNG.choice([*'xy'], 200)})
    events = pd.DataFrame(
        {
            'eid': range(900),
            'uid': RNG.integers(0, 200, 900),
            'amt': RNG.gamma(2.0, 10.0, 900),
            'ts': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(RNG.integers(0, 150, 900), 'D'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'events': events},
            primary_keys={'users': 'uid', 'events': 'eid'},
            time_columns={'events': 'ts'},
            relationships=[Relationship('events', 'uid', 'users', 'uid')],
        )
        request = PredictiveQuery(
            'PREDICT users.seg FOR EACH users.uid', schema
        ).materialize(indices=[1, 2], model=KumoRelational())
    return request


def pin(request, model, holdout=None):
    from sdm_client import SDMClient

    class Transport:
        url = 'http://localhost'

        def create_session(self, payload):
            self.created = payload
            return {'session_id': 'test-session'}

        def delete_session(self, session_id):
            assert session_id == 'test-session'
            self.deleted = True

        def close(self):
            pass

    transport = Transport()
    with SDMClient(url=transport.url) as client:
        client._transport.close()
        client._transport = transport
        with client.fit(request, model=model, holdout=holdout):
            pass
    assert transport.deleted
    return transport.created


@pytest.fixture(params=['kumo-tabular', 'kumo-relational'])
def session_input(request):
    factory = (
        tabular_session
        if request.param == 'kumo-tabular'
        else relational_session
    )
    return factory(), request.param


def test_fit_excludes_holdout_and_does_not_mutate_request(session_input):
    request, model = session_input
    population = list(range(50, 90))
    original = (
        copy.deepcopy(request.payloads) if model == 'kumo-relational' else None
    )
    baseline = pin(request, model)
    first_id = baseline['metadata']['request_id']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        held = pin(request, model, population)
    before = keys_in(baseline, 'context')
    after = keys_in(held, 'context')
    excluded = frozenset(_named_by([key]) for key in population)
    assert before & excluded == excluded
    assert after == before - excluded
    assert (
        len(held['context']['instance_table']['rows'])
        == len(baseline['context']['instance_table']['rows']) - 40
    )
    assert any(
        '40 of the rows named as holdout' in str(w.message) for w in caught
    )
    if model == 'kumo-relational':
        for name, table in baseline['context']['related_tables'].items():
            assert len(held['context']['related_tables'][name]['rows']) < len(
                table['rows']
            )
    retried = pin(request, model)
    assert retried['context'] == baseline['context']
    assert retried['schema'] == baseline['schema']
    assert retried['metadata']['request_id'] != first_id
    assert baseline['metadata']['request_id'] == first_id
    if original is not None:
        assert request.payloads == original


@pytest.mark.parametrize('holdout', [None, []])
def test_empty_holdout_preserves_context(session_input, holdout):
    request, model = session_input
    assert (
        pin(request, model, holdout)['context']
        == pin(request, model)['context']
    )


def test_nonmatching_holdout_reports_no_match(session_input):
    request, model = session_input
    baseline = pin(request, model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        held = pin(request, model, [10_000, 10_001])
    assert held['context'] == baseline['context']
    assert any('matched a row' in str(w.message) for w in caught)
