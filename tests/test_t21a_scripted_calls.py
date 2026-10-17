# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's acceptance tests for the scripted endpoint answering several
calls, numbering each from zero, and failing a chosen call (the independent
test author's; T2.8c's cases need it: ``rows_per_call=3`` over 8 entities,
a failure on call 2).

Each request a served endpoint receives is a call, numbered from zero in
the order received. Every call is answered by the one rule
(``contract/runners/scripted_endpoint.py``'s ``answer``), as one call is
today, but the chosen call, which is answered ``500`` with
``{"error": "scripted_failure", "call": <its number>}``: a status the
client's transport does not retry (``_RETRY_STATUS``), so the failed call is
the one numbered. An endpoint told to fail no call answers as today.

The requests are the shared vectors' (``contract/scripted-endpoint.json``),
so each answer is the one the rule gives that request alone. The Rust
endpoint (``sdm-testkit``'s ``endpoint``) is the implementer's to give the
same numbering and failure, with a test of its own, since the pure-Rust
host calls the rule in process and its API for it does not exist yet.

Gating: :data:`PENDING`, a strict xfail; the implementer deletes its one
definition. The unmarked test is a control that passes today.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
VECTORS = {
    v['name']: v
    for v in json.loads((ROOT / 'contract/scripted-endpoint.json').read_text())[
        'vectors'
    ]
}


# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: ``serve(fail_call: int | None = None)``: as today, and the call numbered
#: ``fail_call`` answered :data:`FAILED`.
#: ``answer(method, path, body, *, call=0, fail_call=None)``: the rule for
#: the call numbered ``call``.
FAILED = 500


def failure(call: int) -> dict[str, Any]:
    return {'error': 'scripted_failure', 'call': call}


# ---------------------------------------------------------------------------------


def _module() -> ModuleType:
    name = 't21a_scripted_endpoint'
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

#: Four different requests, so an answer given to the wrong call shows.
CALLS = [
    'regression',
    'regression, instance_id not the first column',
    'binary classification answers the first class',
    'regression',
]


def _post(url: str, vector: dict[str, Any]) -> tuple[int, Any]:
    request = urllib.request.Request(
        url + vector['path'], data=vector['body'].encode(), method='POST'
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _expected(fail_call: int | None) -> list[tuple[int, Any]]:
    return [
        (FAILED, failure(n))
        if n == fail_call
        else (VECTORS[name]['status'], VECTORS[name]['answer'])
        for n, name in enumerate(CALLS)
    ]


def test_the_calls_are_distinct_vectors() -> None:
    r"""Control: the calls are vectors answered 200, two of them different
    from the first, so a misnumbered answer is seen."""
    assert all(VECTORS[name]['status'] == 200 for name in CALLS)
    assert len({VECTORS[name]['answer'].__repr__() for name in CALLS}) == 3


def test_several_calls_are_each_answered_in_order() -> None:
    r"""Exit: an endpoint failing no call answers every call by the rule,
    in order, each one received."""
    with endpoint.serve(fail_call=None) as (url, received):
        answers = [_post(url, VECTORS[name]) for name in CALLS]
    assert answers == _expected(None)
    assert [body for (_, _, body) in received.requests] == [
        VECTORS[name]['body'].encode() for name in CALLS
    ]


@pytest.mark.parametrize('fail_call', [0, 2, 3])
def test_the_chosen_call_fails_and_only_it(fail_call: int) -> None:
    r"""Exit: calls are numbered from zero; the chosen one is answered 500
    naming its number, the first and the last included, and every other
    call by the rule."""
    with endpoint.serve(fail_call=fail_call) as (url, received):
        answers = [_post(url, VECTORS[name]) for name in CALLS]
    assert answers == _expected(fail_call)
    assert len(received.requests) == len(CALLS)


def test_a_call_past_the_last_fails_nothing() -> None:
    r"""Boundary: a chosen call never made fails nothing."""
    with endpoint.serve(fail_call=len(CALLS)) as (url, _):
        answers = [_post(url, VECTORS[name]) for name in CALLS]
    assert answers == _expected(None)


def test_the_rule_numbers_the_call_it_is_given() -> None:
    r"""Exit: the rule in process, as a host without HTTP calls it."""
    vector = VECTORS['regression']
    asked = (vector['method'], vector['path'], vector['body'].encode())
    assert endpoint.answer(*asked, call=1, fail_call=1) == (FAILED, failure(1))
    status, answer = endpoint.answer(*asked, call=0, fail_call=1)
    assert (status, json.loads(json.dumps(answer))) == (200, vector['answer'])
