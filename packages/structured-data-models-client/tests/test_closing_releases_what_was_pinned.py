# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closing a client releases the contexts it pinned, and an empty batch is
answered in the shape every other batch has.

`SDMClient.close()` closed the transport while its handles were still open, so
the one call that releases a session, `FittedModel.close()`, was the call
that stopped working, and the endpoint held the context until its own expiry.

Separately, an answer with no rows in it dropped the `id` column every other
answer carries, so a join written against the ordinary case broke on the empty
one, which is the case a filter that matched nothing produces.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from sdm_client import KumoTabular, Request, SDMClient
from sdm_client.result import PredictionResult


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'id': [float(index) for index in range(rows)],
            'x': [float(index) for index in range(rows)],
            'y': [index % 2 for index in range(rows)],
        }
    )


class _Endpoint:
    def __init__(self) -> None:
        self.released: list[str] = []
        self.closed = False

    def create_session(self, payload: dict) -> dict:
        return {'session_id': 's1'}

    def delete_session(self, session_id: str) -> None:
        if self.closed:
            raise AssertionError('the transport was closed before the session')
        self.released.append(session_id)

    def close(self) -> None:
        self.closed = True


def _client_with(endpoint: _Endpoint) -> SDMClient:
    client = SDMClient('http://127.0.0.1:1', 'none', timeout=1.0, max_retries=0)
    client._transport = endpoint
    return client


@pytest.fixture
def request_() -> Request:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Request(
            context=_frame(),
            predict=_frame(3)[['id', 'x']],
            target='y',
            task='classification',
        )


def test_closing_a_client_releases_the_sessions_it_pinned(
    request_: Request,
) -> None:
    endpoint = _Endpoint()
    client = _client_with(endpoint)
    fitted = client.fit(request_, model=KumoTabular())

    client.close()

    assert endpoint.released == ['s1']
    assert endpoint.closed
    assert fitted._closed


def test_a_session_already_released_is_not_released_twice(
    request_: Request,
) -> None:
    endpoint = _Endpoint()
    client = _client_with(endpoint)
    fitted = client.fit(request_, model=KumoTabular())
    fitted.close()

    client.close()

    assert endpoint.released == ['s1']


def test_a_session_that_will_not_release_still_closes_the_transport(
    request_: Request,
) -> None:
    endpoint = _Endpoint()

    def refuse(session_id: str) -> None:
        raise RuntimeError('session maintenance is already pending')

    client = _client_with(endpoint)
    client.fit(request_, model=KumoTabular())
    endpoint.delete_session = refuse  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        client.close()
    assert endpoint.closed


def test_an_empty_answer_carries_the_columns_a_join_is_written_against() -> (
    None
):
    empty = PredictionResult(
        model='kumo-tabular',
        request_id='',
        items=(),
        outputs=('prediction', 'probability'),
    ).to_pandas()

    assert list(empty.columns)[:2] == ['row_index', 'id']
    assert 'prediction' in empty.columns
    assert empty.empty
    assert empty.merge(
        _frame(2), left_on='id', right_on='id', how='inner'
    ).empty


def test_an_empty_answer_has_no_id_where_the_scored_half_carries_none() -> None:
    """Whether answers name their rows needs both a key in the schema and
    every one of its columns in the half being sent. Asking only the schema
    gave an empty answer an `id` where a full one had none."""
    from sdm_client import Schema
    from sdm_client.client import _names_its_rows

    keyed = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': keyed}, primary_keys={'instance': 'id'}
        )

        carrying = Request(
            context=keyed,
            predict=keyed.head(0)[['id', 'x']],
            schema=schema,
            target='y',
            task='classification',
        )
        without = Request(
            context=keyed,
            predict=keyed.head(0)[['x']],
            schema=schema,
            target='y',
            task='classification',
        )

    assert _names_its_rows(carrying)
    assert not _names_its_rows(without)


def test_a_gathered_request_always_names_its_rows() -> None:
    """The gathered envelope spells the entity as a transport reference rather
    than as the schema's key, so asking whether the predict half carries the
    key answered no about requests whose every answer names its entity."""
    pytest.importorskip('sdm_relational_engine')

    import numpy as np

    from sdm_client import (
        KumoRelational,
        PredictiveQuery,
        Relationship,
        Schema,
    )
    from sdm_client.client import _names_its_rows

    rng = np.random.default_rng(0)
    users = pd.DataFrame({'uid': range(60), 'seg': rng.choice(['a', 'b'], 60)})
    orders = pd.DataFrame(
        {
            'oid': range(400),
            'uid': rng.integers(0, 60, 400),
            'p': rng.random(400) * 100,
            'ts': pd.date_range('2024-01-01', periods=400, freq='h'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'orders': orders},
            primary_keys={'users': 'uid', 'orders': 'oid'},
            time_columns={'orders': 'ts'},
            relationships=[Relationship('orders', 'uid', 'users', 'uid')],
        )
        gathered = PredictiveQuery(
            'PREDICT users.seg FOR EACH users.uid', schema
        ).materialize(indices=[1, 2], model=KumoRelational())

    assert _names_its_rows(gathered)

    from sdm_client.client import _empty_answer

    built = _empty_answer(
        'kumo-relational', ('prediction',), gathered
    ).to_pandas()
    assert list(built.columns) == ['row_index', 'id', 'prediction']


def test_an_empty_forecast_still_says_which_step_it_is_about() -> None:
    """`forecast_step` is put on the items as the parts are joined, so with no
    items there was nothing to read it from and an empty forecast lost the
    column that says which step an answer is about."""
    from sdm_client.client import _by_step
    from sdm_client.result import PredictionItem, PredictionResult

    class Rewritten:
        parts = [(None, 1), (None, 2)]

    def parts_with(items: tuple) -> list[PredictionResult]:
        return [
            PredictionResult(
                model='m', request_id='r', items=items, outputs=('prediction',)
            )
            for _ in range(2)
        ]

    full = _by_step(
        parts_with((PredictionItem(row_index=0, id='a', prediction=1.0),)),
        Rewritten(),
    ).to_pandas()
    empty = _by_step(parts_with(()), Rewritten()).to_pandas()

    assert list(full.columns) == list(empty.columns)
    assert 'forecast_step' in empty.columns
    assert empty.empty
