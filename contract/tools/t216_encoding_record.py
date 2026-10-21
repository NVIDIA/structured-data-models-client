#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records what the reference gives for a model answer written in each
encoding its reader takes into ``contract/t216-encoding-vectors.json``.

Written by T2.16's test author, from the whole-phase review's round-two
finding (an answer sent as UTF-16, UTF-32 or with UTF-8's byte order mark is
refused by the core, and taken by the reference). Run with the reference
checkout's interpreter (the environment of ``contract/reference-env.txt``,
the ``ref-pandas3`` cell), from this repository's root::

    <reference venv>/bin/python -I contract/tools/t216_encoding_record.py \
        --reference <reference checkout>

The reference's transport reads an answer with ``json.loads`` on its bytes,
which takes the encoding ``json.detect_encoding`` finds: UTF-32 or UTF-16 by
their byte order marks, UTF-8 with its mark, or, with no mark, UTF-16 or
UTF-32 of either byte order by where the first four bytes are zero, and
UTF-8 otherwise, decoding with ``surrogatepass``. Each form of
:data:`FORMS` is one of these; two are bytes no such encoding decodes
(``malformed-*``); and four put a lone surrogate in the answer's id and in
its metadata, as a JSON escape and written as itself in UTF-8, UTF-16 and
UTF-32 (``lone-surrogate-*``), which the reference keeps.

For each query of :data:`QUERY_NAMES` (T2.14b's shop, a regression and a
classification) and each form, the public calls ``Schema.from_source``,
``PredictiveQuery.materialize`` and ``SDMClient.predict`` against the
scripted endpoint, which answers by its rule with the answer's JSON text
written in that form: the result's columns and values, its ``request_id``
and ``metadata``, or the refusal. And each answer of :data:`SERVING`
through a Databricks serving client (``serving-*``), whose served model
writes it in ``response_json``: plain, and with the lone surrogates written
as themselves and as escapes.

:func:`observe` is imported by
``contract/tests/test_t216_response_encoding.py``, so ``python-core`` is
observed by the same code, with the core routes forced.
"""

from __future__ import annotations

import argparse
import codecs
import importlib.util
import json
import os
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t216-encoding-vectors.json'
FORMAT = 'sdm-t216-encoding-vectors/1'


def _marked(mark: bytes, codec: str) -> Callable[[str], bytes]:
    r"""The text in ``codec`` after ``mark``, so the byte order is the one
    named, not this machine's."""
    return lambda text: mark + text.encode(codec)


def _odd(text: str) -> bytes:
    r"""UTF-16 with no mark, one byte short: no UTF-16 decodes it."""
    return text.encode('utf-16-le')[:-1]


def _invalid(text: str) -> bytes:
    r"""UTF-8 with a byte no UTF-8 holds inside a text."""
    return text.encode().replace(b'"r1"', b'"r\xff"', 1)


#: Where a lone-surrogate form puts its surrogates: the answer's id (the
#: result's ``request_id``) holds a high one, and its metadata a low one.
LONE_ID = 'x\ud800y'
LONE_NOTE = 'a\udc00b'


def _lone(text: str) -> str:
    r"""The answer's id and metadata each holding a lone surrogate, written
    as itself, not escaped."""
    return text.replace('"r1"', f'"{LONE_ID}"', 1).replace(
        '"metadata": {}', f'"metadata": {{"note": "{LONE_NOTE}"}}', 1
    )


def _lone_escaped(text: str) -> str:
    r"""The same surrogates written as JSON escapes (``\ud800``)."""
    return text.replace('"r1"', '"x\\ud800y"', 1).replace(
        '"metadata": {}', '"metadata": {"note": "a\\udc00b"}', 1
    )


def _lone_in(mark: bytes, codec: str) -> Callable[[str], bytes]:
    r"""The lone surrogates written as themselves in ``codec`` after
    ``mark``, as Python's reader decodes them (``surrogatepass``)."""
    return lambda text: mark + _lone(text).encode(codec, 'surrogatepass')


#: How each form writes the answer's JSON text, by name.
FORMS: dict[str, Callable[[str], bytes]] = {
    'utf-8': lambda text: text.encode('utf-8'),
    'utf-8-sig': lambda text: text.encode('utf-8-sig'),
    'utf-16-le-bom': _marked(codecs.BOM_UTF16_LE, 'utf-16-le'),
    'utf-16-be-bom': _marked(codecs.BOM_UTF16_BE, 'utf-16-be'),
    'utf-16-le': lambda text: text.encode('utf-16-le'),
    'utf-16-be': lambda text: text.encode('utf-16-be'),
    'utf-32-le-bom': _marked(codecs.BOM_UTF32_LE, 'utf-32-le'),
    'utf-32-be-bom': _marked(codecs.BOM_UTF32_BE, 'utf-32-be'),
    'utf-32-le': lambda text: text.encode('utf-32-le'),
    'utf-32-be': lambda text: text.encode('utf-32-be'),
    'malformed-utf-16-odd': _odd,
    'malformed-utf-8-invalid': _invalid,
    'lone-surrogate-escaped': lambda text: _lone_escaped(text).encode(),
    'lone-surrogate-utf-8': _lone_in(b'', 'utf-8'),
    'lone-surrogate-utf-16': _lone_in(codecs.BOM_UTF16_LE, 'utf-16-le'),
    'lone-surrogate-utf-32': _lone_in(codecs.BOM_UTF32_LE, 'utf-32-le'),
}
#: The Databricks serving answers, by name: the scripted rule's answer as
#: the served model writes it in ``response_json``, and that answer with
#: lone surrogates written as themselves and as escapes.
SERVING = {
    'plain': lambda text: text,
    'lone-surrogate': _lone,
    'lone-surrogate-escaped': _lone_escaped,
}
#: The forms whose lone surrogates the core reads as U+FFFD where the
#: reference keeps them (register P1B-24, the owner's rule of 2026-10-21),
#: by the name the vectors give them.
REPLACED = (
    'lone-surrogate-escaped',
    'lone-surrogate-utf-8',
    'lone-surrogate-utf-16',
    'lone-surrogate-utf-32',
    'serving-lone-surrogate',
    'serving-lone-surrogate-escaped',
)
#: The queries, by name: T2.14b's shop's regression (its predictions
#: numbers) and classification (its predictions classes).
QUERY_NAMES = ('windowed', 'condition')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t216_run_python', ROOT / 'contract/runners/run_python.py')


def host() -> ModuleType:
    r"""T2.14b's host-facts recorder, whose shop, queries and endpoint
    these scenarios reuse."""
    return _module(
        't214b_host_record', ROOT / 'contract/tools/t214b_host_record.py'
    )


def _without(value: Any, url: str) -> Any:
    r"""``value`` with the served URL, whose port changes each run, as
    ``<endpoint>``."""
    if isinstance(value, str):
        return value.replace(url, '<endpoint>')
    if isinstance(value, dict):
        return {k: _without(v, url) for k, v in value.items()}
    if isinstance(value, list):
        return [_without(v, url) for v in value]
    return value


def _predicted(request: Any, form: str) -> dict[str, Any]:
    r"""The request predicted against the endpoint answering in ``form``:
    the result's columns and values, or the refusal."""
    from sdm_client import KumoRelational, SDMClient

    observed: dict[str, Any] = {
        'columns': None,
        'values': None,
        'request_id': None,
        'metadata': None,
    }
    with (
        host().endpoint().serve(encode=FORMS[form]) as (url, received),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter('ignore')
        try:
            with SDMClient(url) as client:
                result = client.predict(request, model=KumoRelational())
            frame = result.to_pandas()
            observed['columns'] = [str(c) for c in frame.columns]
            observed['values'] = host().as_json(frame.to_dict(orient='list'))
            observed['request_id'] = result.request_id
            observed['metadata'] = host().as_json(result.metadata)
            observed['refusal'] = None
        except Exception as error:  # noqa: BLE001 - recorded
            observed['refusal'] = _without(host()._refusal(error), url)
        observed['calls'] = len(received.requests)
    return observed


class _Endpoints:
    r"""The part of the Databricks SDK's ``serving_endpoints`` the client's
    serving transport calls: each query answered by the scripted rule, its
    answer's text written by ``write``."""

    def __init__(self, write: Callable[[str], str]) -> None:
        self._write = write
        self.calls = 0

    def query(self, *, name: str, dataframe_records: Any) -> Any:
        self.calls += 1
        body = dataframe_records[0]['request_json'].encode()
        module = host().endpoint()
        status, reply = module.answer('POST', module.PREDICTIONS, body)
        assert status == 200, reply
        return {
            'predictions': [{'response_json': self._write(json.dumps(reply))}]
        }


class _Workspace:
    def __init__(self, write: Callable[[str], str]) -> None:
        self.serving_endpoints = _Endpoints(write)


def _served(request: Any, form: str) -> dict[str, Any]:
    r"""The request predicted by a Databricks serving client whose served
    model writes its answer in ``form``."""
    from sdm_client import KumoRelational, SDMClient

    workspace = _Workspace(SERVING[form])
    observed: dict[str, Any] = {
        'columns': None,
        'values': None,
        'request_id': None,
        'metadata': None,
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            with SDMClient.for_databricks_serving(
                'scripted-serving', workspace_client=workspace
            ) as client:
                result = client.predict(request, model=KumoRelational())
            frame = result.to_pandas()
            observed['columns'] = [str(c) for c in frame.columns]
            observed['values'] = host().as_json(frame.to_dict(orient='list'))
            observed['request_id'] = result.request_id
            observed['metadata'] = host().as_json(result.metadata)
            observed['refusal'] = None
        except Exception as error:  # noqa: BLE001 - recorded
            observed['refusal'] = host()._refusal(error)
    observed['calls'] = workspace.serving_endpoints.calls
    return observed


def answers() -> dict[str, Any]:
    r"""Each query's answer in each form, by ``<query>.<form>``."""
    connection = host()._connection()
    try:
        schema = host()._schema(connection)
        out = {}
        for query in QUERY_NAMES:
            request = host()._materialize(schema, host().QUERIES[query])
            for form in FORMS:
                out[f'{query}.{form}'] = _predicted(request, form)
            for form in SERVING:
                out[f'{query}.serving-{form}'] = _served(request, form)
        return out
    finally:
        connection.close()


def observe(core: bool) -> dict[str, Any]:
    r"""Every scenario, on the core where ``core`` (the DuckDB from-source,
    materialize and predict paths forced, as ``run_python.py`` forces a
    case's routes), else on the routes the interpreter's client takes."""
    if core:
        from sdm_client import _routing

        os.environ['SDM_CORE_ROUTES'] = ','.join(
            _routing.FROM_SOURCE_DUCKDB
            + _routing.MATERIALIZE_DUCKDB
            + _routing.PREDICT_DUCKDB
        )
    try:
        return {'answers': answers()}
    finally:
        os.environ.pop('SDM_CORE_ROUTES', None)


def load() -> dict[str, Any]:
    r"""The vector file, as committed."""
    return json.loads(OUT.read_text(encoding='utf-8'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--reference', type=Path, required=True)
    arguments = parser.parse_args(argv)
    run = runner()
    guard = run._Guard(arguments.reference)
    guard.check('before the first case')
    run._require_reference_client(arguments.reference)
    commit = run._git(arguments.reference)
    if not commit['clean']:
        print('the reference checkout is not clean', file=sys.stderr)
        return 1
    document = {
        'format': FORMAT,
        'about': (
            "What the reference's public calls give when the model endpoint "
            'writes its answer in each encoding json.detect_encoding finds '
            '(UTF-8 with and without its byte order mark, UTF-16 and UTF-32 '
            'of either byte order with and without theirs), in two byte '
            'forms no such encoding decodes, and with lone surrogates in its '
            'id and metadata, over HTTP and from a Databricks serving '
            'endpoint: the result, its request_id and metadata, or the '
            'refusal. '
            'contract/tests/test_t216_response_encoding.py holds python-core '
            'to them, observed by the same code (observe).'
        ),
        'recorded_by': 'contract/tools/t216_encoding_record.py',
        'reference': commit['sha'],
        **observe(core=False),
    }
    guard.check('after the last case')
    OUT.write_text(
        json.dumps(document, indent=1, ensure_ascii=True) + '\n',
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
