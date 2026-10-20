# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Python scripted endpoint against its shared vector file
(``contract/scripted-endpoint.json``), which T1.11d's Rust endpoint answers
too, and through the client's real transport over localhost.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/scripted-endpoint.json'


def _module() -> ModuleType:
    name = 'scripted_endpoint'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'contract/runners/scripted_endpoint.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


endpoint = _module()
document = json.loads(VECTORS.read_text(encoding='utf-8'))
vectors = document['vectors']


def test_the_vector_file() -> None:
    assert document['format'] == 'sdm-scripted-endpoint/1'
    names = [v['name'] for v in vectors]
    assert len(names) == len(set(names))
    # Every refusal reason the rule names has a vector.
    reasons = {v['answer'].get('error') for v in vectors if v['status'] == 400}
    assert reasons == set(endpoint.REASONS)


@pytest.mark.parametrize('vector', vectors, ids=[v['name'] for v in vectors])
def test_the_rule_answers_each_vector(vector: dict[str, Any]) -> None:
    status, answer = endpoint.answer(
        vector['method'], vector['path'], vector['body'].encode()
    )
    assert status == vector['status']
    assert json.loads(json.dumps(answer)) == vector['answer']


@pytest.mark.parametrize('vector', vectors, ids=[v['name'] for v in vectors])
def test_the_server_answers_each_vector(vector: dict[str, Any]) -> None:
    with endpoint.serve() as (url, received):
        body = vector['body'].encode() if vector['method'] != 'GET' else None
        request = urllib.request.Request(
            url + vector['path'], data=body, method=vector['method']
        )
        try:
            with urllib.request.urlopen(request) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
    assert status == vector['status']
    assert json.loads(raw) == vector['answer']
    assert received.requests == [
        (vector['method'], vector['path'], body or b'')
    ]


def test_the_client_s_transport_reaches_it() -> None:
    from sdm_client.core.transport import Transport

    vector = vectors[0]
    with endpoint.serve() as (url, received):
        transport = Transport(url, max_retries=0)
        try:
            answer = transport.predict(json.loads(vector['body']))
        finally:
            transport.close()
    assert answer == vector['answer']
    assert [(m, p) for m, p, _ in received.requests] == [
        ('POST', '/v1/predictions')
    ]


probability_vectors = document['probabilities']


@pytest.mark.parametrize(
    'vector', probability_vectors, ids=[v['name'] for v in probability_vectors]
)
def test_an_endpoint_told_to_answer_probabilities_answers_each_vector(
    vector: dict[str, Any],
) -> None:
    # T2.8: the rule as call number ``call``, and served, after as many
    # calls before it, each answered in turn.
    status, answer = endpoint.answer(
        vector['method'],
        vector['path'],
        vector['body'].encode(),
        call=vector['call'],
        probabilities=True,
    )
    assert status == vector['status']
    assert json.loads(json.dumps(answer)) == vector['answer']
    assert [
        list(i.get('probabilities') or {}) for i in answer['predictions']
    ] == [
        list(i.get('probabilities') or {})
        for i in vector['answer']['predictions']
    ]
    with endpoint.serve(probabilities=True) as (url, _):
        for _ in range(vector['call'] + 1):
            request = urllib.request.Request(
                url + vector['path'],
                data=vector['body'].encode(),
                method=vector['method'],
            )
            with urllib.request.urlopen(request) as response:
                status, raw = response.status, response.read()
    assert status == vector['status']
    assert json.loads(raw) == vector['answer']


def test_the_probabilities_vectors_are_not_vacuous() -> None:
    given = [
        v
        for v in probability_vectors
        if 'probabilities' in v['answer']['predictions'][0]
    ]
    assert any(v['call'] > 0 for v in given)
    assert len(given) < len(probability_vectors)
    # Without being told to, the endpoint answers none.
    for vector in given:
        _, answer = endpoint.answer(
            vector['method'], vector['path'], vector['body'].encode()
        )
        assert all('probabilities' not in i for i in answer['predictions'])


quantile_vectors = document['quantiles']


@pytest.mark.parametrize(
    'vector', quantile_vectors, ids=[v['name'] for v in quantile_vectors]
)
def test_an_endpoint_told_to_answer_quantiles_answers_each_vector(
    vector: dict[str, Any],
) -> None:
    # T2.9: the rule as call number ``call``, and served, after as many
    # calls before it, each answered in turn.
    status, answer = endpoint.answer(
        vector['method'],
        vector['path'],
        vector['body'].encode(),
        call=vector['call'],
        quantiles=True,
    )
    assert status == vector['status']
    assert json.loads(json.dumps(answer)) == vector['answer']
    assert [list(i.get('quantiles') or {}) for i in answer['predictions']] == [
        list(i.get('quantiles') or {}) for i in vector['answer']['predictions']
    ]
    with endpoint.serve(quantiles=True) as (url, _):
        for _ in range(vector['call'] + 1):
            request = urllib.request.Request(
                url + vector['path'],
                data=vector['body'].encode(),
                method=vector['method'],
            )
            with urllib.request.urlopen(request) as response:
                status, raw = response.status, response.read()
    assert status == vector['status']
    assert json.loads(raw) == vector['answer']


def test_the_quantiles_vectors_are_not_vacuous() -> None:
    given = [
        v
        for v in quantile_vectors
        if 'quantiles' in v['answer']['predictions'][0]
    ]
    assert any(v['call'] > 0 for v in given)
    assert len(given) < len(quantile_vectors)
    # Without being told to, or told to answer probabilities only, the
    # endpoint answers none.
    for vector in given:
        for told in ({}, {'probabilities': True}):
            _, answer = endpoint.answer(
                vector['method'],
                vector['path'],
                vector['body'].encode(),
                call=vector['call'],
                **told,
            )
            assert all('quantiles' not in i for i in answer['predictions'])


def _served(vector: dict[str, Any], **told: bool) -> tuple[int, Any, list[Any]]:
    r"""``vector``'s request served as call number ``call``, after as many
    calls before it, each answered in turn: the last status and answer,
    and what the endpoint received."""
    with endpoint.serve(**told) as (url, received):
        for _ in range(vector['call'] + 1):
            request = urllib.request.Request(
                url + vector['path'],
                data=vector['body'].encode(),
                method=vector['method'],
            )
            try:
                with urllib.request.urlopen(request) as response:
                    status, raw = response.status, response.read()
            except urllib.error.HTTPError as error:
                status, raw = error.code, error.read()
    return status, json.loads(raw), received.requests


explanation_vectors = document['explanation']


@pytest.mark.parametrize(
    'vector', explanation_vectors, ids=[v['name'] for v in explanation_vectors]
)
def test_an_endpoint_told_to_answer_explanations_answers_each_vector(
    vector: dict[str, Any],
) -> None:
    # T2.13: the rule as call number ``call``, and served.
    status, answer = endpoint.answer(
        vector['method'],
        vector['path'],
        vector['body'].encode(),
        call=vector['call'],
        explanation=True,
    )
    assert status == vector['status']
    assert json.loads(json.dumps(answer)) == vector['answer']
    status, answer, _ = _served(vector, explanation=True)
    assert status == vector['status']
    assert answer == vector['answer']


def test_the_explanation_vectors_are_not_vacuous() -> None:
    given = [
        v
        for v in explanation_vectors
        if 'explanation' in v['answer']['predictions'][0]
    ]
    assert any(v['call'] > 0 for v in given)
    assert any(len(v['answer']['predictions']) > 1 for v in given)
    assert len(given) < len(explanation_vectors)
    # Each row's explanation is its own, and a classification asking for
    # probabilities beside it is given none untold.
    many = next(v for v in given if len(v['answer']['predictions']) > 1)
    explained = [i['explanation'] for i in many['answer']['predictions']]
    assert len({json.dumps(e, sort_keys=True) for e in explained}) == len(
        explained
    )
    # Without being told to, or told everything else, the endpoint answers
    # none.
    for vector in given:
        for told in (
            {},
            {'probabilities': True, 'quantiles': True, 'item_metadata': True},
        ):
            _, answer = endpoint.answer(
                vector['method'],
                vector['path'],
                vector['body'].encode(),
                call=vector['call'],
                **told,
            )
            assert all('explanation' not in i for i in answer['predictions'])
        _, answer, _ = _served(vector)
        assert all('explanation' not in i for i in answer['predictions'])


item_metadata_vectors = document['item_metadata']


@pytest.mark.parametrize(
    'vector',
    item_metadata_vectors,
    ids=[v['name'] for v in item_metadata_vectors],
)
def test_an_endpoint_told_to_answer_item_metadata_answers_each_vector(
    vector: dict[str, Any],
) -> None:
    # T2.13: the rule as call number ``call``, and served.
    status, answer = endpoint.answer(
        vector['method'],
        vector['path'],
        vector['body'].encode(),
        call=vector['call'],
        item_metadata=True,
    )
    assert status == vector['status']
    assert json.loads(json.dumps(answer)) == vector['answer']
    status, answer, _ = _served(vector, item_metadata=True)
    assert status == vector['status']
    assert answer == vector['answer']


def test_the_item_metadata_vectors_are_not_vacuous() -> None:
    given = [
        v
        for v in item_metadata_vectors
        if v['status'] == 200
        and v['answer']['predictions']
        and 'metadata' in v['answer']['predictions'][0]
    ]
    assert any(v['call'] > 0 for v in given)
    assert any(len(v['answer']['predictions']) > 1 for v in given)
    assert any(v['status'] == 400 for v in item_metadata_vectors)
    # Without being told to, or told everything else, no item carries any.
    for vector in given:
        for told in (
            {},
            {'probabilities': True, 'quantiles': True, 'explanation': True},
        ):
            _, answer = endpoint.answer(
                vector['method'],
                vector['path'],
                vector['body'].encode(),
                call=vector['call'],
                **told,
            )
            assert all('metadata' not in i for i in answer['predictions'])


def test_a_stand_in_rule_of_the_earlier_keywords_still_stands_in() -> None:
    # The endpoint passes its rule the explanation and item metadata
    # keywords only when told to, so a test's stand-in taking the call's
    # number, probabilities and quantiles alone still answers for an
    # endpoint told neither.
    seen: list[dict[str, Any]] = []

    def stand_in(
        method: str,
        path: str,
        body: bytes,
        *,
        call: int = 0,
        probabilities: bool = False,
        quantiles: bool = False,
    ) -> tuple[int, Any]:
        seen.append({'call': call})
        return 200, {'stood_in': True}

    vector = vectors[0]
    original = endpoint.answer
    endpoint.answer = stand_in
    try:
        status, answer, _ = _served({**vector, 'call': 1})
    finally:
        endpoint.answer = original
    assert (status, answer) == (200, {'stood_in': True})
    assert seen == [{'call': 0}, {'call': 1}]


embeddings_vectors = document['embeddings']


@pytest.mark.parametrize(
    'vector', embeddings_vectors, ids=[v['name'] for v in embeddings_vectors]
)
def test_an_endpoint_told_to_answer_embeddings_answers_each_vector(
    vector: dict[str, Any],
) -> None:
    # T2.11: the rule as call number ``call``, and served.
    status, answer = endpoint.answer(
        vector['method'],
        vector['path'],
        vector['body'].encode(),
        call=vector['call'],
        embeddings=True,
    )
    assert status == vector['status']
    assert json.loads(json.dumps(answer)) == vector['answer']
    status, answer, _ = _served(vector, embeddings=True)
    assert status == vector['status']
    assert answer == vector['answer']


def test_the_embeddings_vectors_are_not_vacuous() -> None:
    given = [
        v
        for v in embeddings_vectors
        if 'embeddings' in v['answer']['predictions'][0]
    ]
    assert any(v['call'] > 0 for v in given)
    assert any(len(v['answer']['predictions']) > 1 for v in given)
    assert len(given) < len(embeddings_vectors)
    # Each row's embedding is its own.
    many = next(v for v in given if len(v['answer']['predictions']) > 1)
    embedded = [i['embeddings'] for i in many['answer']['predictions']]
    assert len({json.dumps(e) for e in embedded}) == len(embedded)
    # Without being told to, or told everything else, the endpoint answers
    # none.
    for vector in given:
        for told in (
            {},
            {
                'probabilities': True,
                'quantiles': True,
                'explanation': True,
                'item_metadata': True,
            },
        ):
            _, answer = endpoint.answer(
                vector['method'],
                vector['path'],
                vector['body'].encode(),
                call=vector['call'],
                **told,
            )
            assert all('embeddings' not in i for i in answer['predictions'])
        _, answer, _ = _served(vector)
        assert all('embeddings' not in i for i in answer['predictions'])
