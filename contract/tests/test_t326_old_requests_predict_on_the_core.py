# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A request the core did not materialize predicts on the core as the old
client predicts it (T3.26, owner decision 2026-10-23: "I think we should
support old cases. Re-materializing is a common use case, and if they update
the package we should still be backward compatible").

    python -m pytest contract/tests/test_t326_old_requests_predict_on_the_core.py -q

Such a request holds no state the core wrote: one gathered on the old route
in this process, or one an older client gathered and pickled, the
reference's (on its only route) or Phase 2's (on its core, which kept its
own state beside it). Predict on the core rebuilds what it needs from the
request's own public contents, its payloads, queries and parts, and answers
as the old client answers: the same bodies sent, the same items, outputs,
explanation, unscored candidates and metadata, and the same notices.

The pickles and each client's answers are in
``contract/fixtures/old-request-pickles``, written by its ``write.py``;
each schema in them was pickled without its source, which no client
pickles, and is bound to one again here, so the call routes as a call over
a source does.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
from sdm_client import PredictiveQuery, Schema, _routing

ROOT = Path(__file__).resolve().parents[2]
PICKLES = ROOT / 'contract/fixtures/old-request-pickles'
EXPECTED = json.loads((PICKLES / 'expected.json').read_text(encoding='utf-8'))
CORE = ','.join(sorted(_routing.CORE_ROUTED))
#: Whether the old route, which gathers with the relational engine, can
#: run: CI's ``test`` job installs no engine, its ``engine`` job does.
ENGINE = importlib.util.find_spec('sdm_relational_engine') is not None


def writer() -> ModuleType:
    name = 't326_write_old_request_pickles'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, PICKLES / 'write.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


@pytest.fixture
def on_the_core(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    r"""Every core path forced, and each request the core's predict was
    handed, so a test knows the core answered."""
    from sdm_client._rust import predict

    handed: list[Any] = []
    real = predict.predict

    def spied(client: Any, request: Any, *rest: Any) -> Any:
        handed.append(request)
        return real(client, request, *rest)

    monkeypatch.setattr(predict, 'predict', spied)
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    yield handed


def _bound(request: Any, source: Any) -> Any:
    r"""``request`` with each schema it holds, its own and each part's,
    bound to ``source`` in a schema of its tables."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for held in (request, *(request.parts or ())):
            held.schema = Schema(
                dict(held.schema.tables),
                list(held.schema.relationships),
                source=source,
            )
    return request


def _loaded(client: str, name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loaded = pickle.loads((PICKLES / f'{client}-{name}.pkl').read_bytes())
    return _bound(loaded, writer().source())


def test_every_pickle_is_listed_with_its_client_s_answers() -> None:
    r"""Control: each client's pickles are the writer's, of the commit it
    names, and exercise what they name: a request asked in several calls,
    a forecast's steps, a ranking with a repeated candidate and one left
    unscored, an explanation, and an output the endpoint leaves out."""
    names = set(writer().REQUESTS)
    assert set(EXPECTED) == set(writer().CLIENTS)
    for client in writer().CLIENTS:
        written = {
            path.stem.removeprefix(f'{client}-')
            for path in PICKLES.glob(f'{client}-*.pkl')
        }
        assert written == names == set(EXPECTED[client]['requests'])
        assert EXPECTED[client]['protocol'] == 4
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {EXPECTED["reference"]["commit"]}' in env
    assert EXPECTED['phase-2']['commit'] == writer().PHASE_2
    answered = EXPECTED['reference']['requests']
    assert len(answered['batched'][0]['bodies']) == 3
    assert answered['ranking'][0]['answer']['unscored'] == [9]
    assert answered['regression'][1]['answer']['items'][0]['explanation']
    assert any('quantiles' in n for n in answered['batched'][1]['notices'])
    # Phase 2's requests were gathered on its core, and hold its state.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ranking = pickle.loads((PICKLES / 'phase-2-ranking.pkl').read_bytes())
    assert ranking._core_state['version'] == 2
    assert ranking._core is None


@pytest.mark.parametrize('client', ['reference', 'phase-2'])
@pytest.mark.parametrize('name', sorted(EXPECTED['reference']['requests']))
def test_a_request_an_older_client_pickled_predicts_on_the_core_as_it_answered(
    client: str, name: str, on_the_core: list[Any]
) -> None:
    r"""The request loads, and predicting it on the core sends the bodies
    the client that pickled it sent and gives its answer and notices."""
    loaded = _loaded(client, name)
    assert (
        writer().answers(loaded, name) == (EXPECTED[client]['requests'][name])
    )
    assert on_the_core, 'the core did not predict the request'
    assert all(request is loaded for request in on_the_core)


@pytest.mark.parametrize('name', sorted(EXPECTED['reference']['requests']))
def test_a_request_gathered_on_the_old_route_predicts_on_the_core_as_there(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""A request this client gathered on its old route (where no engine is
    installed, the core's, with its state dropped, as such a request holds
    none) predicts on the core as on the old route."""
    query, arguments = writer().REQUESTS[name]
    if not ENGINE:
        monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(writer().source(), writer().TABLES)
        request = PredictiveQuery(query, schema).materialize(
            anchor_time=pd.Timestamp(writer().ANCHOR), **arguments
        )
    for held in (request, *(request.parts or ())):
        held._core = None
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    old = writer().answers(request, name)
    assert old == EXPECTED['reference']['requests'][name]
    from sdm_client._rust import predict

    handed: list[Any] = []
    real = predict.predict

    def spied(client: Any, asked: Any, *rest: Any) -> Any:
        handed.append(asked)
        return real(client, asked, *rest)

    monkeypatch.setattr(predict, 'predict', spied)
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    assert writer().answers(request, name) == old
    assert handed and all(asked is request for asked in handed)
