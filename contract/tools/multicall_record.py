#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records what T2.8c's multi-call cases give beside their observation
lines into ``contract/t28c-multicall-vectors.json`` (``main``'s 1fc5a686).

Written by T2.8c's test author, before the implementation. Run with the
reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root, a database at a time or all three::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/multicall_record.py \
        --reference ~/code/sdmc-reference [--database duckdb sqlite postgres]

Postgres needs ``SDM_POSTGRES_TEST_DSN``, as ``run_python.py`` does.

The cases are ``contract/cases/s1/multicall.*.json``; each runs its public
calls on each database as ``run_python.py`` makes them (its ``_open``,
``_source`` and arguments; the scripted endpoint failing the case's
``fail_call``). Their lines (``run_python.py --mode python-ref``, committed
in ``contract/reference/``) hold the outcome, each request the endpoint
received and the result's digest. What a line does not hold, and the
package's exits name, is kept here, per case key:

- ``calls``: how many rows each call the endpoint received scores, in
  order (``predict.instance_table``);
- ``result``: the answer through ``to_pandas()``, its columns and its rows
  (every row up to :data:`ROWS_KEPT`, else the first and last
  :data:`ENDS_KEPT` with the count), so the rows numbered across calls are
  held row by row;
- for a refusal, ``details``: ``StructuredDataError.details`` but
  ``answered``, each value as JSON writes it (the ceilings' counts, the
  failed call's ``calls_answered`` and ``rows_answered``, a forecast's
  ``parts_answered``, the 30 MiB body's ``batch_too_large_to_send``);
  ``answered``: ``details['answered']``, the calls (or a forecast's steps)
  answered before the one that failed, projected as ``result`` is;
  ``opening``: the message's first sentence; and ``remedy``: what the
  client says to change for a call past the 30 MiB body (from "A call
  already scores one row" or "What comes down is one of three" to the
  end), or null.

``raised`` holds, by the same keys, the cases :data:`RAISED` names
observed with the client's transport raising a ``ValueError`` on the
named call, an error that is not the client's own, as a transport may
raise one mid-request (the Databricks SDK on a failed OAuth refresh):
the reference raises it as it was, with the answers already received
kept on it as for its own errors. ``--raised`` records that part alone.

:func:`observe` is imported by ``contract/tests/test_prediction_batching.py``,
so ``python-core`` is observed by the same code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t28c-multicall-vectors.json'
CASES = ROOT / 'contract/cases/s1'
FORMAT = 'sdm-t28c-multicall-vectors/1'
PREFIX = 'multicall.'
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: A result of at most this many rows is kept whole.
ROWS_KEPT = 200
#: Of a longer one, this many rows at each end.
ENDS_KEPT = 3
#: The calls that raise an error that is not the client's own, by case:
#: the call's number from zero. A transport may raise one mid-request (the
#: Databricks SDK's ``ValueError`` on a failed OAuth refresh), and the
#: reference keeps the answers already paid for on it (``client.py``'s
#: ``_carry_the_answers_already_paid_for``, by step for a forecast). The
#: third call of ``rows_per_call=3`` over 8 (6 rows answered before it),
#: and the second step's call of the two-step forecast.
RAISED = {
    'multicall.rows-per-call-3': 2,
    'multicall.forecast.one-call': 1,
}
#: What the raised error says.
RAISED_TEXT = "a failure that is not the client's own (scripted)"
#: Where the client's own remedy for a call past the 30 MiB body begins
#: (``relational.py``'s ``_NO_CALL_IS_SMALLER`` and
#: ``_ASK_IN_SMALLER_CALLS`` at the reference).
REMEDIES = ('A call already scores one row', 'What comes down is one of three')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t28c_run_python', ROOT / 'contract/runners/run_python.py')


def endpoint() -> ModuleType:
    return _module(
        'scripted_endpoint', ROOT / 'tests/support/scripted_endpoint.py'
    )


def case_ids() -> list[str]:
    r"""The multi-call cases' ids, in name order."""
    return sorted(
        path.name[: -len('.json')] for path in CASES.glob(f'{PREFIX}*.json')
    )


def remedy(message: str | None) -> str | None:
    r"""The client's remedy in a 30 MiB body refusal, or None."""
    if not message:
        return None
    found = [message.find(mark) for mark in REMEDIES if mark in message]
    return message[min(found) :] if found else None


def opening(message: str | None) -> str | None:
    r"""A message's first sentence, through its first full stop."""
    if not message:
        return None
    at = message.find('. ')
    return message if at < 0 else message[: at + 1]


def _cell(value: Any) -> Any:
    r"""A result cell as JSON writes it: a NumPy scalar as its Python
    value, a missing value as null."""
    if hasattr(value, 'item'):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def projection(result: Any) -> dict[str, Any] | None:
    r"""A ``PredictionResult`` through ``to_pandas()``: its columns, its
    row count and its rows (all, or both ends of a long one)."""
    if result is None or not hasattr(result, 'to_pandas'):
        return None
    frame = result.to_pandas()
    rows = [
        [_cell(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    kept: dict[str, Any] = {
        'columns': [str(name) for name in frame.columns],
        'count': len(rows),
    }
    if len(rows) <= ROWS_KEPT:
        kept['rows'] = rows
    else:
        kept['first'] = rows[:ENDS_KEPT]
        kept['last'] = rows[-ENDS_KEPT:]
    return kept


def _detail(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def _rows_sent(body: bytes) -> int | None:
    try:
        rows = json.loads(body)['predict']['instance_table']['rows']
    except (ValueError, KeyError, TypeError):
        return None
    return len(rows) if isinstance(rows, list) else None


def _raising(client: Any, call: int) -> None:
    r"""``client``'s transport raising ``ValueError`` on the call numbered
    ``call`` from zero, before it is sent; every other call as it is. Both
    of the transport's ways to send a prediction are counted together: the
    old route's ``predict`` and the core route's ``predict_body``."""
    transport = client._transport
    made = [0]
    for name in ('predict', 'predict_body'):
        original = getattr(transport, name, None)
        if original is None:
            continue

        def sent(
            *arguments: Any, _original: Any = original, **named: Any
        ) -> Any:
            made[0] += 1
            if made[0] == call + 1:
                raise ValueError(RAISED_TEXT)
            return _original(*arguments, **named)

        setattr(transport, name, sent)


def observe(
    case: dict[str, Any],
    database: str,
    raise_on_call: int | None = None,
) -> dict[str, Any]:
    """Exercise current-client public calls against a scripted endpoint or source."""
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

    run = runner()
    given = case['input']
    arguments = run._materialize_arguments(given['materialize'])
    from_source = run._from_source_arguments(given.get('from_source', {}))
    observed: dict[str, Any] = {
        'outcome': None,
        'calls': [],
        'result': None,
        'details': None,
        'answered': None,
        'opening': None,
        'remedy': None,
    }
    connection, cleanup, _ = run._open(case, run.load_bases(), database)
    stage = 'from_source'
    try:
        with (
            endpoint().serve(fail_call=given.get('fail_call')) as (
                url,
                received,
            ),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter('ignore')
            try:
                schema = Schema.from_source(
                    run._source(database, connection), **from_source
                )
                run._edit(schema, given.get('edits', []))
                if case['until'] != 'from_source':
                    stage = 'materialize'
                    request = PredictiveQuery(given['pql'], schema).materialize(
                        **arguments
                    )
                if case['until'] == 'predict':
                    stage = 'predict'
                    with SDMClient(url) as client:
                        if raise_on_call is not None:
                            _raising(client, raise_on_call)
                        result = client.predict(request, model=KumoRelational())
                    observed['result'] = projection(result)
                observed['outcome'] = run._outcome('success', stage)
            except Exception as error:  # noqa: BLE001 - the outcome
                observed['outcome'] = run._outcome('refusal', stage, error)
                details = getattr(error, 'details', None)
                details = details if isinstance(details, dict) else {}
                observed['details'] = {
                    str(name): _detail(value)
                    for name, value in sorted(details.items())
                    if name != 'answered'
                }
                observed['answered'] = projection(details.get('answered'))
                observed['opening'] = opening(str(error))
                observed['remedy'] = remedy(str(error))
            observed['calls'] = [
                _rows_sent(body) for _, _, body in received.requests
            ]
    finally:
        cleanup()
    return observed


def load() -> dict[str, Any]:
    r"""The vector file, as committed."""
    return json.loads(OUT.read_text(encoding='utf-8'))
