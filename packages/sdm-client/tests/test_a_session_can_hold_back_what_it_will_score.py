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
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.client import _held_back
from sdm_client.wire.keys import _named_by, keys_in

RNG = np.random.default_rng(9)


def tabular_session() -> dict:
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
    return TabularAdapter().session_plan(request, ('prediction',))


def relational_session() -> dict:
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
    return RelationalAdapter().session_plan(request, ('prediction',))


def test_the_named_rows_are_not_in_the_context_that_gets_pinned():
    payload = tabular_session()
    population = list(range(200, 260))
    before = len(payload['context']['instance_table']['rows'])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        removed = _held_back(payload, population)

    after = keys_in(payload, 'context')
    assert removed == 60
    assert len(payload['context']['instance_table']['rows']) == before - 60
    assert not after & frozenset(_named_by([key]) for key in population)


def test_the_related_rows_go_with_the_instance_they_hang_off():
    """A related row whose instance has gone describes nothing."""
    pytest.importorskip('sdm_relational_engine')

    payload = relational_session()
    before = {
        name: len(table['rows'])
        for name, table in payload['context']['related_tables'].items()
    }

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        removed = _held_back(payload, list(range(50, 90)))

    after = {
        name: len(table['rows'])
        for name, table in payload['context']['related_tables'].items()
    }
    assert removed == 40
    assert all(after[name] < before[name] for name in before)


def test_how_many_were_held_back_is_said():
    """A caller who names a population expects it gone, and silence would
    look the same whether none went or all of them did."""
    payload = tabular_session()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _held_back(payload, list(range(200, 260)))

    said = ' '.join(str(w.message) for w in caught)
    assert '60 of the rows named as holdout' in said


def test_naming_nothing_changes_nothing():
    payload = tabular_session()
    before = len(payload['context']['instance_table']['rows'])

    assert _held_back(payload, None) == 0
    assert _held_back(payload, []) == 0
    assert len(payload['context']['instance_table']['rows']) == before


def test_naming_rows_that_are_not_in_the_context_removes_none():
    payload = tabular_session()
    before = len(payload['context']['instance_table']['rows'])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        removed = _held_back(payload, [10_000, 10_001])

    assert removed == 0
    assert len(payload['context']['instance_table']['rows']) == before
    assert not [w for w in caught if 'holdout' in str(w.message)]


# --- and `fit()` has to be the thing that does it ---------------------------


def test_fit_holds_the_population_back_before_it_pins(monkeypatch):
    """Everything above tests the removal. This tests that opening a session
    performs it, which is the only reason the removal exists."""
    from sdm_client import SDMClient

    sent: dict = {}

    class Transport:
        url = 'http://127.0.0.1:9'

        def create_session(self, payload):
            sent['payload'] = payload
            return {'session_id': 's1'}

        def delete_session(self, session_id):
            pass

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

    client = SDMClient('http://127.0.0.1:9', 'none')
    monkeypatch.setattr(client, '_transport', Transport())
    population = list(range(200, 260))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        client.fit(request, model='kumo-tabular', holdout=population)

    pinned = keys_in(sent['payload'], 'context')
    assert pinned
    assert not pinned & frozenset(_named_by([key]) for key in population)


def test_fit_without_a_holdout_pins_what_it_gathered(monkeypatch):
    from sdm_client import SDMClient

    sent: dict = {}

    class Transport:
        url = 'http://127.0.0.1:9'

        def create_session(self, payload):
            sent['payload'] = payload
            return {'session_id': 's1'}

        def delete_session(self, session_id):
            pass

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

    client = SDMClient('http://127.0.0.1:9', 'none')
    monkeypatch.setattr(client, '_transport', Transport())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        client.fit(request, model='kumo-tabular')

    pinned = keys_in(sent['payload'], 'context')
    assert pinned & frozenset(_named_by([key]) for key in range(200, 260))
