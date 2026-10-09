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
