# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the reference's predict-side rules and results for T1.28b's tests.

Run with the reference checkout's interpreter (the reference environment of
``contract/reference-env.txt``), from this repository's root::

    ~/code/sdmc-reference/.venv/bin/python \\
        packages/structured-data-models-client/tests/t128b_predict_result_record.py

Each case is a DuckDB source (``contract/bases.yaml``'s ``shop``, changed by
the case's statements, or a seeded one), a query, the arguments of
``PredictiveQuery.materialize`` and of ``SDMClient.predict``, and the script
of a local model endpoint (:func:`answer`): a function of the call it
receives, so a host whose call is the reference's is answered with the
same bytes. The reference runs it, and this script keeps (:func:`observe`)
what T1.28b's exits compare: the result through ``to_pandas()`` (columns,
dtypes, rows), its outputs and metadata, the notices the call gave, the
number of calls the endpoint received; or, when the reference refuses, the
refusal's class, code, HTTP status, and for the endpoint's own refusal its
message and details; and the bytes each call was answered with, which
``sdm-testkit``'s ``tests/t128b_predict_result.rs`` replays on rust-host.

It writes two files. ``t128b-predict-result.json`` beside it holds the
cases; the boundary-vector matrix's answers (deep and wide JSON, finite
overflow and non-finite numbers, the endpoint's against a local error,
missing or retyped evidence) also go, each with the reference's public
outcome, into ``contract/response-vectors.json``'s ``boundaries``, beside
its ``vectors`` (which this script leaves as they are).

The reference's outcome depends, for a few cases, on the interpreter and
on pandas, so it is recorded at each pin the tests run at too. With
``--cell``, run by the reference's code (b413e9ad's packages, installed
from ``git archive``) under another interpreter or pandas, this script
observes every case and boundary vector again (each boundary answer
replayed as recorded) and records where that cell differs: a frame's
dtypes on pandas 2 as ``dtypes_pandas2`` (pandas 3's text column is
``str``, pandas 2's ``object``), and any other difference as the whole
outcome under ``by_python`` keyed by the interpreter's version (an answer
nested 5,000 levels, which Python 3.10's and 3.11's JSON reader cannot
read; a probability in Kawi digits, which Python before 3.12 does not know
as digits). Run it after the full recording, under each pin::

    python packages/structured-data-models-client/tests/\
t128b_predict_result_record.py --cell

Nothing here is written by hand but the inputs, and one case's outcome on
the core, which is the orchestrator's policy and not the reference's
(``core_outcome``: Kawi digits refused on every interpreter); every
expected value is the reference's. ``observe`` and ``answer`` are imported by the test, so both
routes are observed and answered by the same code.
"""

from __future__ import annotations

import io
import json
import math
import random
import struct
import subprocess
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name('t128b-predict-result.json')
VECTORS = ROOT / 'contract/response-vectors.json'

ANCHOR = '2024-03-01'
REGRESSION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
BINARY = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
MULTICLASS = 'PREDICT users.country FOR EACH users.user_id'

# The exits, as the brief words them; each case names the one it covers.
LAYOUT = (
    'L1 layout: distinct per-row answers and reordered responses laid out '
    "as the reference's result"
)
CLASSIFICATION = (
    'classification results: the boolean and text predictions (and their '
    "probabilities) over base.binary and base.multiclass equal the reference's"
)
CONTEXT = (
    "empty and constant context, a request with no rows, and the parts' "
    "result with the absent-output notice, as the reference's"
)
PREPARE = (
    "the adapter's _preflight and SDMClient._prepare's model and output "
    "checks: the reference's refusals and codes"
)
BOUNDARY = (
    'boundary-vector matrix: deep and wide JSON, finite overflow and '
    "non-finite numbers, the endpoint's against a local error's provenance, "
    "missing or retyped evidence, as the reference's public outcome"
)
CARRY = (
    'carry-forward: np.longdouble options and an ndarray target_transforms '
    'accepted as the reference accepts them'
)
SIZE = 'carry-forward: the size refusal explains the limit, as the reference'
INTEGERS = (
    "integer predictions: a regression's integers kept exact (past 2**53) "
    "and mixed with floats, in to_pandas() and in .items, as the reference's"
)
ABSENT_VALUES = (
    'a missing, a null and a NaN prediction: .items (None or NaN) and '
    "to_pandas(), as the reference's"
)
NUMERIC_TEXT = (
    'probabilities written as text (float() grammar: digit separators, '
    'Unicode digits, whitespace): accepted and refused as the reference does'
)

#: The register entries a boundary answer's outcome on the core may differ
#: from the reference's by, once the owner approves them (never before):
#: a non-finite number (proposed), an answer nested past the core's reader
#: (approved), a regression's prediction that is not a number (approved).
NOT_FINITE = 'P1B-4'
TOO_DEEP = 'P1B-22'
NOT_A_NUMBER = 'P1B-23'


def shop() -> list[str]:
    r"""``contract/bases.yaml``'s ``shop`` statements."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def contract_case(name: str) -> dict[str, Any]:
    r"""``contract/cases/s1/<name>.json``'s input: its query and the
    arguments of materialize, which rust-host's case runs as well."""
    case = json.loads(
        (ROOT / f'contract/cases/s1/{name}.json').read_text(encoding='utf-8')
    )
    assert case['input']['base'] == 'shop', name
    return {
        'query': case['input']['pql'],
        'materialize': dict(case['input']['materialize']),
    }


def generated(seed: int, users: int = 0) -> list[str]:
    r"""A seeded source: ``users`` (a key, an age, a VARCHAR segment of 3
    to 6 classes, a BOOLEAN ``active``) and ``orders``."""
    r = random.Random(seed)
    n = users or r.randint(10, 40)
    classes = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'][
        : r.randint(3, 6)
    ]
    rows = [
        f"({i}, {r.randint(18, 80)}, '{r.choice(classes)}', "
        f'{"TRUE" if r.random() < 0.5 else "FALSE"})'
        for i in range(n)
    ]
    orders = []
    for i in range(r.randint(n * 2, n * 8)):
        day = r.randint(0, 420)
        orders.append(
            f'({i}, {r.randrange(n)}, {round(r.uniform(1, 200), 2)}, '
            f"TIMESTAMP '2023-01-01' + to_days({day}))"
        )
    return [
        'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age INTEGER, '
        'segment VARCHAR, active BOOLEAN)',
        f'INSERT INTO users VALUES {", ".join(rows)}',
        'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT '
        'REFERENCES users(user_id), price DOUBLE, ts TIMESTAMP)',
        f'INSERT INTO orders VALUES {", ".join(orders)}',
    ]


# --- the endpoint's scripts ---------------------------------------------------
#
# Each script answers one call: (status, body text). ``answer`` reads the
# call as the endpoint would and dispatches on the script's name; anything
# varying per row is a function of the row's position and instance id, so
# every answer to a row is distinct and the same on every host.


def _rows(call: dict[str, Any]) -> list[tuple[int, Any]]:
    table = call['predict']['instance_table']
    at = table['columns'].index('instance_id')
    return [(i, row[at]) for i, row in enumerate(table['rows'])]


def _value(call: dict[str, Any], index: int, instance: Any) -> Any:
    r"""A distinct answer for one row: a float for a regression, the class
    at the row's position for a classification."""
    task = call.get('task') or {}
    if task.get('kind') == 'regression':
        return 0.5 + 1.25 * index + int(instance) / 1024
    classes = task['target']['classes']
    return classes[(index + 1) % len(classes)]


def _probabilities(call: dict[str, Any], index: int) -> dict[str, float]:
    classes = call['task']['target']['classes']
    weights = [(index + 1) * (k + 2) for k in range(len(classes))]
    total = sum(weights)
    return {c: w / total for c, w in zip(classes, weights, strict=True)}


def _items(
    call: dict[str, Any], *, probabilities: bool = True
) -> list[dict[str, Any]]:
    out = []
    classify = (call.get('task') or {}).get('kind') != 'regression'
    for index, instance in _rows(call):
        item: dict[str, Any] = {
            'id': str(instance),
            'row_index': index,
            'prediction': _value(call, index, instance),
        }
        if classify and probabilities:
            item['probabilities'] = _probabilities(call, index)
        out.append(item)
    return out


def _envelope(items: Any, **extra: Any) -> dict[str, Any]:
    return {
        'object': 'prediction.response',
        'id': 'r-t128b',
        'model': 'kumo-relational',
        'metadata': {'served_by': 'scripted', 'took_ms': 12},
        'predictions': items,
        **extra,
    }


def _ordered(items: list[Any], order: str) -> list[Any]:
    if order == 'reversed':
        return items[::-1]
    if order == 'rotated':
        return items[1:] + items[:1]
    if order == 'interleaved':
        return items[1::2] + items[0::2]
    return items


def _deep(text: str, depth: int) -> str:
    r"""``text`` with its placeholder replaced by ``depth`` nested objects,
    ``{"d": {"d": ... "bottom"}}``, as ``json.dumps`` writes them, built as
    text: Python 3.10's and 3.11's encoder cannot write 5,000 levels."""
    nested = '{"d": ' * depth + '"bottom"' + '}' * depth
    return text.replace('"__t128b_nested__"', nested)


def _text(value: Any) -> str:
    return json.dumps(value)


def _with_token(items: list[dict[str, Any]], at: int, token: str) -> str:
    r"""The answer's text with row ``at``'s prediction written as ``token``,
    as it travels (``NaN``, ``Infinity``, ``1e400``, digits past a double)."""
    marker = '"__t128b_token__"'
    items[at]['prediction'] = '__t128b_token__'
    return _text(_envelope(items)).replace(marker, token)


def answer(script: str, body: bytes) -> tuple[int, str]:
    r"""The status and body text answering ``body`` under ``script``."""
    call = json.loads(body)
    kind, _, arg = script.partition(':')
    items = _items(call)
    if kind == 'order':
        return 200, _text(_envelope(_ordered(items, arg)))
    if kind == 'prediction-only':
        return 200, _text(_envelope(_items(call, probabilities=False)))
    if kind == 'deep-metadata':
        envelope = _envelope(_ordered(items, 'reversed'))
        envelope['metadata'] = '__t128b_nested__'
        return 200, _deep(_text(envelope), int(arg))
    if kind == 'deep-item':
        items[0]['other'] = '__t128b_nested__'
        return 200, _deep(_text(_envelope(items)), int(arg))
    if kind == 'wide-metadata':
        envelope = _envelope(items)
        envelope['metadata'] = {f'k{i:06d}': i for i in range(int(arg))}
        return 200, _text(envelope)
    if kind == 'wide-predictions':
        # Members past what the bindings read, many of them, on one answer.
        items[0].update({f'x{i:06d}': i for i in range(int(arg))})
        return 200, _text(_envelope(items))
    if kind == 'token':
        return 200, _with_token(items, 0, arg)
    if kind == 'metadata-token':
        marker = '"__t128b_token__"'
        envelope = _envelope(items)
        envelope['metadata'] = {'score': '__t128b_token__'}
        return 200, _text(envelope).replace(marker, arg)
    if kind == 'row-index-token':
        marker = '"__t128b_token__"'
        items[0]['row_index'] = '__t128b_token__'
        return 200, _text(_envelope(items)).replace(marker, arg)
    if kind == 'status':
        status, _, shape = arg.partition('/')
        bodies = {
            'detail': {
                'code': 'UNSUPPORTED_FEATURE',
                'detail': 'categorical cardinality 13812 exceeds limit 10000',
                'diagnosis': {'stage': 'encode', 'rows': 2},
            },
            'title': {'title': 'Unexpected server error'},
            'error': {'error': {'message': 'model busy', 'type': 'overload'}},
        }
        if shape == 'text':
            return int(status), 'upstream connect error'
        if shape == 'empty':
            return int(status), ''
        return int(status), _text(bodies[shape])
    if kind == 'raw':
        return 200, {
            'text': 'this is not JSON',
            'empty': '',
            'array': '[1, 2]',
            'truncated': _text(_envelope(items))[:-7],
            'null': 'null',
        }[arg]
    if kind == 'predictions':
        # Each row's prediction written as the token given ('MISSING' leaves
        # the member out), in row order.
        tokens = arg.split(',')
        assert len(tokens) == len(items), (arg, len(items))
        for item, token in zip(items, tokens, strict=True):
            item['prediction'] = (
                '__t128b_missing__'
                if token == 'MISSING'
                else f'__t128b_{token}__'
            )
        text = _text(_envelope(items))
        text = text.replace(', "prediction": "__t128b_missing__"', '')
        for token in set(tokens) - {'MISSING'}:
            text = text.replace(f'"__t128b_{token}__"', token)
        return 200, text
    if kind == 'probabilities':
        # Row 0's probabilities, each class's written as the token given
        # (JSON text: a string, a number, null ...), separated by '|'.
        tokens = arg.split('|')
        classes = call['task']['target']['classes']
        assert len(tokens) == len(classes), (arg, classes)
        items[0]['probabilities'] = {
            c: f'__t128b_p{k}__' for k, c in enumerate(classes)
        }
        text = _text(_envelope(items))
        for k, token in enumerate(tokens):
            text = text.replace(f'"__t128b_p{k}__"', token)
        return 200, text
    if kind == 'mutate':
        envelope = _envelope(items)
        _mutate(envelope, arg)
        return 200, _text(envelope)
    raise ValueError(f'no script {script!r}')


def _mutate(envelope: dict[str, Any], how: str) -> None:
    r"""Evidence missing or retyped, one field at a time."""
    items = envelope['predictions']
    first = items[0]
    changes: dict[str, Callable[[], None]] = {
        'no-predictions': lambda: envelope.pop('predictions'),
        'predictions-text': lambda: envelope.update(predictions='none'),
        'no-response-id': lambda: envelope.pop('id'),
        'response-id-number': lambda: envelope.update(id=7),
        'no-model': lambda: envelope.pop('model'),
        'model-number': lambda: envelope.update(model=1),
        'no-metadata': lambda: envelope.pop('metadata'),
        'metadata-list': lambda: envelope.update(metadata=[1]),
        'no-row-index': lambda: first.pop('row_index'),
        'row-index-text': lambda: first.update(row_index='0'),
        'row-index-null': lambda: first.update(row_index=None),
        'row-index-float': lambda: first.update(row_index=0.0),
        'row-index-out-of-range': lambda: first.update(row_index=len(items)),
        'no-item-id': lambda: first.pop('id'),
        'item-id-null': lambda: first.update(id=None),
        'item-id-number': lambda: first.update(id=int(first['id'])),
        'item-id-other-row': lambda: first.update(id='999999'),
        'no-prediction': lambda: first.pop('prediction'),
        'prediction-null': lambda: first.update(prediction=None),
        'prediction-text': lambda: first.update(prediction='high'),
        'prediction-integer': lambda: first.update(prediction=3),
        'prediction-bool': lambda: first.update(prediction=True),
        'prediction-list': lambda: first.update(prediction=[1.0]),
        'duplicate-row': lambda: items.append(dict(first)),
        'missing-row': lambda: items.pop(),
        'forecast-step': lambda: first.update(forecast_step=1),
        'item-not-object': lambda: items.__setitem__(0, [1]),
    }
    changes[how]()


# --- the cases -------------------------------------------------------------

CASES: list[dict[str, Any]] = []


def case(
    name: str,
    covers: str,
    query: str,
    setup: list[str],
    *,
    script: str = 'order:given',
    predict: dict[str, Any] | None = None,
    rust_case: str | None = None,
    divergence: str | None = None,
    items: bool = False,
    core_outcome: dict[str, Any] | None = None,
    transport: str | None = None,
    **materialize: Any,
) -> None:
    CASES.append(
        {
            'name': name,
            'covers': covers,
            'query': query,
            'setup': setup,
            'materialize': materialize,
            'predict': predict or {},
            'script': script,
            'rust_case': rust_case,
            'divergence': divergence,
            # Whether .items is recorded and compared, as well as the frame.
            **({'items': True} if items else {}),
            # The outcome the core gives where it is a policy, not the
            # reference's (a registered difference).
            **({'core_outcome': core_outcome} if core_outcome else {}),
            # 'databricks': the call made through SDMClient.for_databricks_
            # serving and a fake workspace client, in place of HTTP.
            **({'transport': transport} if transport else {}),
        }
    )


def base_case(
    contract: str,
    name: str,
    covers: str,
    *,
    script: str = 'order:given',
    predict: dict[str, Any] | None = None,
    divergence: str | None = None,
    items: bool = False,
    core_outcome: dict[str, Any] | None = None,
) -> None:
    r"""A case over a contract case's own input, so rust-host can replay it:
    its materialize is the case's, its answer the bytes recorded here."""
    spec = contract_case(contract)
    case(
        name,
        covers,
        spec['query'],
        shop(),
        script=script,
        predict=predict,
        rust_case=contract,
        divergence=divergence,
        items=items,
        core_outcome=core_outcome,
        **spec['materialize'],
    )


# L1 layout: distinct per-row answers, in and out of order.
for order in ('given', 'reversed', 'rotated', 'interleaved'):
    base_case(
        'base.regression',
        f'layout.regression.{order}',
        LAYOUT,
        script=f'order:{order}',
    )
    case(
        f'layout.regression-six.{order}',
        LAYOUT,
        REGRESSION,
        shop(),
        script=f'order:{order}',
        indices=[5, 0, 7, 2, 3, 1],
        anchor_time=ANCHOR,
    )
for seed in (21, 22):
    case(
        f'layout.generated-{seed}.reversed',
        LAYOUT,
        'PREDICT SUM(orders.price, 0, 60, days) FOR EACH users.user_id',
        generated(seed, users=24),
        script='order:reversed',
        indices=list(range(23, 3, -1)),
        anchor_time=ANCHOR,
        random_seed=seed,
    )

# Classification results: boolean and text predictions, with probabilities.
for contract in ('base.binary', 'base.multiclass'):
    for order in ('given', 'reversed'):
        base_case(
            contract,
            f'{contract}.{order}',
            CLASSIFICATION,
            script=f'order:{order}',
        )
    base_case(
        contract,
        f'{contract}.prediction-only',
        CLASSIFICATION,
        predict={'outputs': ['prediction']},
        script='prediction-only',
    )
for seed in (31, 32):
    case(
        f'generated-{seed}.boolean-static',
        CLASSIFICATION,
        'PREDICT users.active FOR EACH users.user_id',
        generated(seed),
        script='order:reversed',
        indices=[1, 0, 2],
        random_seed=seed,
    )
    case(
        f'generated-{seed}.multiclass-static',
        CLASSIFICATION,
        'PREDICT users.segment FOR EACH users.user_id',
        generated(seed),
        script='order:interleaved',
        indices=[0, 2, 4, 6],
        random_seed=seed,
    )
    case(
        f'generated-{seed}.binary-count',
        CLASSIFICATION,
        'PREDICT COUNT(orders.*, 0, 30, days) > 2 FOR EACH users.user_id',
        generated(seed),
        script='order:reversed',
        indices=[0, 1, 2],
        anchor_time=ANCHOR,
        random_seed=seed,
    )

# Empty and constant context, a request with no rows, the absent output.
case(
    'context.empty-static-all-scored',
    CONTEXT,
    MULTICLASS,
    shop(),
    indices=list(range(8)),
)
case(
    'context.constant-regression',
    CONTEXT,
    'PREDICT users.age FOR EACH users.user_id',
    [
        *shop(),
        'CREATE OR REPLACE TABLE users AS SELECT user_id, 30 AS age, '
        'country FROM users',
    ],
    indices=[0, 1],
)
case(
    'context.constant-multiclass',
    CONTEXT,
    MULTICLASS,
    [
        *shop(),
        "CREATE OR REPLACE TABLE users AS SELECT user_id, age, 'US' "
        'AS country FROM users',
    ],
    indices=[0, 1],
)
case(
    'context.constant-binary',
    CONTEXT,
    'PREDICT COUNT(orders.*, 0, 30, days) > 1000 FOR EACH users.user_id',
    shop(),
    indices=[0, 1],
    anchor_time=ANCHOR,
)
case(
    'context.eleven-classes',
    CONTEXT,
    MULTICLASS,
    [
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i)::INTEGER '
        "AS age, 'c' || (i % 11)::VARCHAR AS country FROM range(44) t(i)",
        shop()[1],
    ],
    indices=[0, 1],
)
case(
    'context.ten-classes',
    CONTEXT,
    MULTICLASS,
    [
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i)::INTEGER '
        "AS age, 'c' || (i % 10)::VARCHAR AS country FROM range(40) t(i)",
        shop()[1],
    ],
    indices=[0, 1],
)
case(
    'context.no-such-entity',
    CONTEXT,
    REGRESSION,
    shop(),
    indices=[999],
    anchor_time=ANCHOR,
)
base_case(
    'base.binary',
    'absent.probabilities-named',
    CONTEXT,
    predict={'outputs': ['prediction', 'probabilities']},
    script='prediction-only',
)
base_case(
    'base.binary',
    'absent.probabilities-defaulted',
    CONTEXT,
    script='prediction-only',
)
base_case(
    'base.regression',
    'absent.prediction-named',
    CONTEXT,
    predict={'outputs': ['prediction']},
    script='mutate:no-prediction',
    # A missing prediction: .items says None, which the frame shows as NaN.
    items=True,
)

# _prepare's and _preflight's checks.
for name, arguments in (
    ('model-tabular', {'model': 'kumo-tabular'}),
    ('model-unknown', {'model': 'no-such-model'}),
    ('outputs-empty', {'outputs': []}),
    ('outputs-string', {'outputs': 'prediction'}),
    ('outputs-not-names', {'outputs': [1]}),
    ('outputs-unknown', {'outputs': ['bogus']}),
    ('outputs-probabilities', {'outputs': ['probabilities']}),
    ('outputs-quantiles', {'outputs': ['quantiles']}),
    (
        'outputs-explanation-two-rows',
        {'outputs': ['prediction', 'explanation']},
    ),
    ('outputs-duplicate', {'outputs': ['prediction', 'prediction']}),
):
    base_case(
        'base.regression',
        f'prepare.regression.{name}',
        PREPARE,
        predict=arguments,
    )
for name, arguments in (
    ('outputs-probabilities-only', {'outputs': ['probabilities']}),
    ('outputs-quantiles', {'outputs': ['prediction', 'quantiles']}),
):
    base_case(
        'base.binary', f'prepare.binary.{name}', PREPARE, predict=arguments
    )
case(
    'preflight.explanation-one-row',
    PREPARE,
    REGRESSION,
    shop(),
    predict={'outputs': ['prediction', 'explanation']},
    indices=[0],
    anchor_time=ANCHOR,
)
case(
    'preflight.explanation-larger-context',
    PREPARE,
    REGRESSION,
    shop(),
    predict={'outputs': ['prediction', 'explanation']},
    indices=[0],
    anchor_time=ANCHOR,
    num_context_samples=2000,
)
case(
    'prepare.not-a-request',
    PREPARE,
    REGRESSION,
    shop(),
    predict={'request': 'not-a-request'},
    indices=[0, 1],
    anchor_time=ANCHOR,
)

# The boundary-vector matrix, each answering base.regression's call.
BOUNDARIES = [
    ('deep.metadata-100', 'deep-metadata:100', None),
    ('deep.metadata-200', 'deep-metadata:200', TOO_DEEP),
    ('deep.metadata-5000', 'deep-metadata:5000', TOO_DEEP),
    ('deep.item-5000', 'deep-item:5000', TOO_DEEP),
    ('wide.metadata-20000', 'wide-metadata:20000', None),
    ('wide.item-members-20000', 'wide-predictions:20000', None),
    ('finite.max-double', 'token:1.7976931348623157e308', None),
    ('finite.min-subnormal', 'token:5e-324', None),
    ('finite.negative-zero', 'token:-0.0', None),
    (
        'finite.integer-past-int64',
        'token:123456789012345678901234567890',
        NOT_A_NUMBER,
    ),
    ('finite.integer-past-double', 'token:' + '9' * 400, NOT_A_NUMBER),
    (
        'finite.long-mantissa',
        'token:0.1000000000000000055511151231257827',
        None,
    ),
    ('overflow.prediction-1e400', 'token:1e400', NOT_FINITE),
    ('overflow.prediction-minus-1e400', 'token:-1e400', NOT_FINITE),
    ('nonfinite.prediction-NaN', 'token:NaN', NOT_FINITE),
    ('nonfinite.prediction-Infinity', 'token:Infinity', NOT_FINITE),
    ('nonfinite.prediction-minus-Infinity', 'token:-Infinity', NOT_FINITE),
    ('nonfinite.metadata-NaN', 'metadata-token:NaN', NOT_FINITE),
    ('overflow.metadata-1e400', 'metadata-token:1e400', NOT_FINITE),
    (
        'overflow.row-index-past-uint64',
        'row-index-token:18446744073709551616',
        None,
    ),
    ('overflow.row-index-1e400', 'row-index-token:1e400', None),
    ('provenance.endpoint-422-detail', 'status:422/detail', None),
    ('provenance.endpoint-500-title', 'status:500/title', None),
    ('provenance.endpoint-503-error', 'status:503/error', None),
    ('provenance.endpoint-400-text', 'status:400/text', None),
    ('provenance.endpoint-404-empty', 'status:404/empty', None),
    ('provenance.local-not-json', 'raw:text', None),
    ('provenance.local-empty-body', 'raw:empty', None),
    ('provenance.local-array', 'raw:array', None),
    ('provenance.local-truncated', 'raw:truncated', None),
    ('provenance.local-null', 'raw:null', None),
    ('provenance.local-id-other-row', 'mutate:item-id-other-row', None),
    ('provenance.local-duplicate-row', 'mutate:duplicate-row', None),
    ('provenance.local-missing-row', 'mutate:missing-row', None),
    (
        'provenance.local-row-index-out-of-range',
        'mutate:row-index-out-of-range',
        None,
    ),
    ('provenance.local-forecast-step', 'mutate:forecast-step', None),
]
for how in (
    'no-predictions',
    'predictions-text',
    'no-response-id',
    'response-id-number',
    'no-model',
    'model-number',
    'no-metadata',
    'metadata-list',
    'no-row-index',
    'row-index-text',
    'row-index-null',
    'row-index-float',
    'no-item-id',
    'item-id-null',
    'item-id-number',
    'prediction-null',
    'prediction-text',
    'prediction-integer',
    'prediction-bool',
    'prediction-list',
    'item-not-object',
):
    BOUNDARIES.append(
        (
            f'evidence.{how}',
            f'mutate:{how}',
            NOT_A_NUMBER
            if how in ('prediction-text', 'prediction-bool', 'prediction-list')
            else None,
        )
    )
for name, script, divergence in BOUNDARIES:
    base_case(
        'base.regression',
        f'boundary.{name}',
        BOUNDARY,
        script=script,
        divergence=divergence,
        # A null prediction: .items says None, which the frame shows as NaN.
        items=name == 'evidence.prediction-null',
    )


# Integer predictions, exact past 2**53, and mixed with floats.
for name, tokens in (
    ('all-past-2-53', '9007199254740993,-12'),
    ('all-small', '3,4'),
    ('int64-extremes', '9223372036854775807,-9223372036854775808'),
    ('mixed-int-float', '3,2.5'),
    ('mixed-past-2-53-float', '9007199254740993,0.5'),
    ('whole-float-and-int', '3.0,4'),
):
    base_case(
        'base.regression',
        f'integers.{name}',
        INTEGERS,
        script=f'predictions:{tokens}',
        items=True,
    )

# A missing, a null and a NaN prediction.
for name, tokens in (
    ('missing-one', 'MISSING,1.5'),
    ('null-one', 'null,1.5'),
    ('nan-one', 'NaN,1.5'),
    ('missing-all', 'MISSING,MISSING'),
    ('null-all', 'null,null'),
    ('nan-all', 'NaN,NaN'),
    ('missing-and-null', 'MISSING,null'),
    ('null-and-nan', 'null,NaN'),
    ('missing-and-nan', 'MISSING,NaN'),
):
    base_case(
        'base.regression',
        f'absent-values.{name}',
        ABSENT_VALUES,
        script=f'predictions:{tokens}',
        items=True,
    )

# Probabilities written as text, accepted or refused as float() reads it.
for name, token in (
    ('digit-separator', '"0_0.5"'),
    ('arabic-indic-digits', json.dumps('\u0660.\u0665')),
    ('fullwidth-digits', json.dumps('\uff10.\uff15')),
    ('surrounding-spaces', '" 0.5 "'),
    ('tab-and-newline', json.dumps('\t0.5\n')),
    ('plain', '"0.5"'),
    ('exponent', '"5e-1"'),
    ('overflowing-1e999', '"1e999"'),
    ('nan-text', '"nan"'),
    ('infinity-text', '"-Infinity"'),
    ('letters', '"abc"'),
    ('double-separator', '"0__5"'),
    ('trailing-separator', '"0.5_"'),
    ('empty', '""'),
    ('hexadecimal', '"0x1p-1"'),
    ('vulgar-fraction', json.dumps('\u00bd')),
    ('null', 'null'),
    ('boolean', 'true'),
    ('list', '[0.5]'),
):
    base_case(
        'base.binary',
        f'numeric-text.{name}',
        NUMERIC_TEXT,
        script=f'probabilities:{token}|0.5',
        items=True,
    )

#: The Unicode policy across cells (the orchestrator's): a probability in
#: Kawi digits (Unicode 15) is refused INVALID_RESPONSE on the core on
#: every interpreter, where the reference reads it from Python 3.12
#: (Unicode 15) and refuses it on 3.10 and 3.11 (a registered difference);
#: Arabic-Indic digits (numeric-text.arabic-indic-digits, Unicode 1.1, in
#: Unicode 13 like every digit Python 3.10 knows) are read on both routes
#: everywhere.
UNICODE_POLICY = {'outcome': 'refused', 'code': 'INVALID_RESPONSE'}
base_case(
    'base.binary',
    'numeric-text.kawi-digits',
    NUMERIC_TEXT,
    script='probabilities:' + json.dumps('\U00011f50.\U00011f55') + '|0.5',
    items=True,
    core_outcome=UNICODE_POLICY,
)


# The Databricks serving route (SDMClient.for_databricks_serving with a
# fake workspace client, no endpoint): an answer nested 5,000 levels in its
# response_json is P1B-22's refusal on the core on every interpreter
# (INVALID_RESPONSE, field response, reason too_deep), as over HTTP; the
# reference reads it from Python 3.12 and fails its transport before. The
# controls, a response_json that is not JSON and one that is cut short,
# and an ordinary answer, keep the reference's outcome.
DATABRICKS = (
    "the Databricks serving route: P1B-22's depth refusal as over HTTP, and "
    "a response_json that is not JSON as the reference's"
)
TOO_DEEP_POLICY = {
    'outcome': 'refused',
    'code': 'INVALID_RESPONSE',
    'field': 'response',
    'reason': 'too_deep',
}
spec = contract_case('base.regression')
for name, script, policy in (
    ('deep.metadata-5000', 'deep-metadata:5000', TOO_DEEP_POLICY),
    ('deep.item-5000', 'deep-item:5000', TOO_DEEP_POLICY),
    ('malformed.not-json', 'raw:text', None),
    ('malformed.truncated', 'raw:truncated', None),
    ('malformed.empty', 'raw:empty', None),
    ('layout.reversed', 'order:reversed', None),
):
    case(
        f'databricks.{name}',
        DATABRICKS,
        spec['query'],
        shop(),
        script=script,
        divergence='P1B-22' if policy else None,
        core_outcome=policy,
        transport='databricks',
        **spec['materialize'],
    )
del spec


# The carry-forward: option values the reference accepts.
def _carry_cases() -> None:
    import numpy as np

    del np  # values are built in observe, from their names
    for name, options in (
        ('longdouble-num-context', {'num_context_samples': 'longdouble:50'}),
        (
            'longdouble-num-estimators',
            {'inference_config': {'num_estimators': 'longdouble:4'}},
        ),
        (
            'ndarray-target-transforms',
            {'inference_config': {'target_transforms': 'ndarray:quantile'}},
        ),
    ):
        case(
            f'carry.{name}',
            CARRY,
            REGRESSION,
            shop(),
            indices=[0, 1],
            anchor_time=ANCHOR,
            **options,
        )


_carry_cases()


# --- observing ------------------------------------------------------------


def _value_of(spec: Any) -> Any:
    r"""An option value named in the case, built as the NumPy value."""
    import numpy as np

    if isinstance(spec, dict):
        return {k: _value_of(v) for k, v in spec.items()}
    if isinstance(spec, str) and spec.startswith('longdouble:'):
        return np.longdouble(spec.split(':', 1)[1])
    if isinstance(spec, str) and spec.startswith('ndarray:'):
        return np.array(spec.split(':', 1)[1].split(','))
    return spec


@contextmanager
def endpoint(script: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    r"""A local endpoint answering each call by ``script``, recording each
    call it received and the status and text it answered."""
    calls: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            length = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(length)
            status, text = answer(script, body)
            data = text.encode()
            calls.append({'body': body, 'status': status, 'answer': text})
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextmanager
def databricks_workspace(
    script: str,
) -> Iterator[tuple[Any, list[dict[str, Any]]]]:
    r"""A fake ``WorkspaceClient`` whose serving endpoint answers each
    call by ``script``, in one ``response_json`` row, as Model Serving
    does; it records each call and its answer."""
    calls: list[dict[str, Any]] = []

    def answered(records: Any) -> dict[str, Any]:
        body = records[0]['request_json'].encode()
        status, text = answer(script, body)
        calls.append({'body': body, 'status': status, 'answer': text})
        return {'predictions': [{'response_json': text}]}

    class Endpoints:
        def query(self, *, name: str, **kwargs: Any) -> Any:
            return answered(kwargs['dataframe_records'])

    class ApiClient:
        # The core route's call (T3.13): the envelope the core wrote,
        # posted as it is, and the reply's bytes as the SDK's raw answer.
        def do(self, method: str, path: str, **kwargs: Any) -> Any:
            sent = kwargs['data']
            sent = sent.read() if hasattr(sent, 'read') else sent
            records = json.loads(sent)['dataframe_records']
            reply = json.dumps(answered(records)).encode()
            return {'contents': io.BytesIO(reply)}

    class Workspace:
        serving_endpoints = Endpoints()
        api_client = ApiClient()

    yield Workspace(), calls


def _cell(value: Any) -> Any:
    r"""A result cell as JSON: a float's non-finite values as text, a
    NumPy scalar as its Python value, a missing value as ``None``."""
    import pandas as pd

    if hasattr(value, 'item') and not isinstance(value, (list, dict)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if value is None or (
        not isinstance(value, (list, dict)) and pd.isna(value)
    ):
        if isinstance(value, float) and math.isnan(value):
            return {'f64': 'nan'}
        return None
    if isinstance(value, float):
        # Its IEEE-754 bits, as sdm-testkit's result rows record a float.
        return {'f64_bits': struct.pack('>d', value).hex()}
    if isinstance(value, bool):
        return {'bool': value}
    if isinstance(value, int):
        return {'int': str(value)}
    if isinstance(value, str):
        return {'str': value}
    return {'other': repr(value)}


def _refusal(error: BaseException) -> dict[str, Any]:
    from sdm_client.errors import NimRequestError

    out: dict[str, Any] = {
        'outcome': 'refused',
        'class': type(error).__name__,
        'code': getattr(error, 'code', None),
        # The classes it is an instance of: the core's host may raise a
        # subclass of the reference's class for the same code.
        'mro': [c.__name__ for c in type(error).__mro__],
        'status_code': getattr(error, 'status_code', None),
        'message': str(error),
        # The core's rule, when a refusal names one (none on the reference).
        'rule': (getattr(error, 'details', None) or {}).get('rule')
        if isinstance(getattr(error, 'details', None), dict)
        else None,
    }
    if isinstance(error, NimRequestError):
        # The endpoint's own refusal: shown as the old route shows it (E47).
        out['endpoint'] = {
            'message': error.message,
            'details': json.loads(json.dumps(error.details, default=str)),
        }
    return out


def observe(spec: dict[str, Any]) -> dict[str, Any]:
    r"""The case's outcome on the route the environment selects."""
    import duckdb
    import pandas as pd
    from sdm_connectors import DuckDBSource

    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

    connection = duckdb.connect()
    for statement in spec['setup']:
        connection.execute(statement)
    arguments = {k: _value_of(v) for k, v in spec['materialize'].items()}
    for key in ('anchor_time', 'context_anchor_time'):
        if key in arguments:
            arguments[key] = pd.Timestamp(arguments[key])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery(spec['query'], schema).materialize(
                **arguments
            )
    except Exception as error:  # noqa: BLE001 - the outcome is compared
        return {'stage': 'materialize', **_refusal(error), 'calls': []}

    predict = dict(spec['predict'])
    model = predict.pop('model', None) or KumoRelational()
    if predict.pop('request', None) is not None:
        request = 'not-a-request'
    notices: list[str] = []
    databricks = spec.get('transport') == 'databricks'
    opened = (
        databricks_workspace(spec['script'])
        if databricks
        else endpoint(spec['script'])
    )
    with opened as (where, calls):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with (
                    SDMClient.for_databricks_serving(
                        't128b-serving', workspace_client=where
                    )
                    if databricks
                    else SDMClient(where, max_retries=0)
                ) as client:
                    result = client.predict(request, model=model, **predict)
            notices = sorted(
                str(w.message)
                for w in caught
                if type(w.message).__name__ == 'SDMWarning'
            )
        except RecursionError as error:
            out = {'stage': 'predict', **_refusal(error)}
        except Exception as error:  # noqa: BLE001 - the outcome is compared
            out = {'stage': 'predict', **_refusal(error)}
            if spec.get('core_outcome'):
                # What a policy names of the refusal: its field and reason.
                details = getattr(error, 'details', None)
                details = details if isinstance(details, dict) else {}
                out['fields'] = {k: details.get(k) for k in ('field', 'reason')}
        else:
            out = {
                'stage': 'predict',
                'outcome': 'result',
                'model': result.model,
                'outputs': list(result.outputs),
                'metadata': _finite(
                    json.loads(json.dumps(result.metadata, default=str))
                )
                if _shallow(result.metadata)
                else {'depth_or_width': 'past what is recorded'},
                'notices': notices,
                **_frame(result),
            }
            if spec.get('items'):
                out['items'] = _items_of(result)
    out['calls'] = [
        {
            'status': c['status'],
            'answer': c['answer'],
            'output': json.loads(c['body']).get('output'),
        }
        for c in calls
    ]
    return out


def _items_of(result: Any) -> list[dict[str, Any]]:
    r"""``result.items`` as recorded: each item's row, its id, its
    prediction (``None`` kept apart from NaN) and its probabilities, each
    class as ``str()`` writes it, as the value it holds, and its
    probability."""
    out = []
    for item in result.items:
        probabilities = getattr(item, 'probabilities', None)
        out.append(
            {
                'row_index': _cell(item.row_index),
                'id': _cell(item.id),
                'prediction': _cell(item.prediction),
                'prediction_type': type(item.prediction).__name__,
                'probabilities': None
                if probabilities is None
                else [
                    [str(k), _cell(k), _cell(v), type(v).__name__]
                    for k, v in probabilities.items()
                ],
            }
        )
    return out


def _frame(result: Any) -> dict[str, Any]:
    r"""The result through ``to_pandas()``: its columns, dtypes and rows, or
    the class of the error ``to_pandas()`` raised."""
    try:
        frame = result.to_pandas()
    except Exception as error:  # noqa: BLE001 - the outcome is compared
        return {'frame_error': type(error).__name__}
    return {
        'columns': [str(c) for c in frame.columns],
        'dtypes': [str(t) for t in frame.dtypes],
        'rows': [
            [_cell(v) for v in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def _finite(value: Any) -> Any:
    r"""``value`` with each non-finite float written as its name, so the
    recording is JSON (``{'f64': 'nan'}``, ``'inf'`` or ``'-inf'``)."""
    if isinstance(value, float) and not math.isfinite(value):
        name = 'nan' if value != value else ('inf' if value > 0 else '-inf')
        return {'f64': name}
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _shallow(value: Any, depth: int = 0) -> bool:
    r"""Whether a metadata value is small enough to record whole: up to
    100 levels deep (the vector file is read by readers that stop at 128)
    and 1,000 members wide."""
    if depth > 100:
        return False
    if isinstance(value, dict):
        return len(value) <= 1000 and all(
            _shallow(v, depth + 1) for v in value.values()
        )
    if isinstance(value, list):
        return len(value) <= 1000 and all(_shallow(v, depth + 1) for v in value)
    return True


def _size_case() -> dict[str, Any]:
    r"""The payload one byte-step past the limit, from T1.28a's recorder."""
    sys.path.insert(0, str(Path(__file__).parent))
    import t128a_payload_rules_record as t128a

    longest = t128a.at_the_size_limit()
    return {
        'name': 'size.one-past-the-limit',
        'covers': SIZE,
        'query': t128a.REGRESSION,
        'setup': t128a.padded(longest + 1),
        'materialize': {'indices': [0, 1], 'anchor_time': ANCHOR},
        'predict': {},
        'script': 'order:given',
        'rust_case': None,
        'divergence': None,
    }


def main() -> int:
    import sdm_relational_engine  # noqa: F401 - the reference's engine

    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 1000))
    specs = [*CASES, _size_case()]
    recorded = []
    for spec in specs:
        expected = observe(spec)
        recorded.append({**spec, 'expected': expected})
        print(
            spec['name'],
            expected['stage'],
            expected['outcome'],
            expected.get('class'),
            expected.get('code'),
            file=sys.stderr,
        )
    # The answers are kept with the boundary vectors; here, each case's
    # answers but the boundary matrix's, which the vector file holds.
    cases = []
    boundaries = []
    for entry in recorded:
        if entry['covers'] == BOUNDARY:
            boundaries.append(
                {
                    'name': entry['name'],
                    'covers': BOUNDARY,
                    'answers': entry['rust_case'],
                    'script': entry['script'],
                    'divergence': entry['divergence'],
                    'status': entry['expected']['calls'][0]['status'],
                    'body': entry['expected']['calls'][0]['answer'],
                    'reference': {
                        k: v
                        for k, v in entry['expected'].items()
                        if k != 'calls'
                    },
                }
            )
        else:
            cases.append(entry)
    OUT.write_text(
        json.dumps(
            {
                'recorded_by': Path(__file__).name,
                'reference': commit,
                'cases': cases,
            },
            indent=1,
            allow_nan=False,
        )
        + '\n',
        encoding='utf-8',
    )
    document = json.loads(VECTORS.read_text(encoding='utf-8'))
    document['boundaries_recorded_by'] = (
        f'packages/structured-data-models-client/tests/{Path(__file__).name} '
        f'from the reference at {commit}'
    )
    document['boundaries_description'] = (
        "T1.28b's boundary-vector matrix: each a whole answer (status and "
        "body text) to the call base.regression's request makes on the "
        "reference, and the reference's public outcome of SDMClient.predict "
        'given it (the result, or the refusal: class, code, HTTP status, the '
        "endpoint's own message and details). A vector naming a divergence "
        'may end otherwise on the core only once that register entry is '
        'approved.'
    )
    document['boundaries'] = boundaries
    VECTORS.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
        + '\n',
        encoding='utf-8',
    )
    print(
        f'{len(cases)} cases to {OUT}, {len(boundaries)} boundary vectors to '
        f'{VECTORS}',
        file=sys.stderr,
    )
    return 0


def cell() -> int:
    r"""Record where this interpreter's and pandas' reference outcome
    differs from the full recording (``--cell``; see the module's text)."""
    import pandas as pd
    import sdm_relational_engine  # noqa: F401 - the reference's engine

    python = f'{sys.version_info.major}.{sys.version_info.minor}'
    pandas2 = int(pd.__version__.split('.')[0]) < 3
    recorded = json.loads(OUT.read_text(encoding='utf-8'))
    document = json.loads(VECTORS.read_text(encoding='utf-8'))
    specs = {spec['name']: spec for spec in CASES}
    calls = next(
        c['expected']['calls']
        for c in recorded['cases']
        if c['name'] == 'layout.regression.given'
    )
    global answer
    scripted = answer

    def differs(expected: dict[str, Any], got: dict[str, Any]) -> set[str]:
        # The text of a local error names the endpoint's port; not compared.
        return {
            k
            for k in set(expected) | set(got)
            if k not in ('message', 'dtypes_pandas2', 'by_python')
            and expected.get(k) != got.get(k)
        }

    def keep(expected: dict[str, Any], got: dict[str, Any], name: str) -> None:
        different = differs(expected, got)
        if pandas2 and different == {'dtypes'}:
            expected['dtypes_pandas2'] = got['dtypes']
            print(name, 'dtypes on pandas 2:', got['dtypes'], file=sys.stderr)
        elif different:
            got = {k: v for k, v in got.items() if k != 'message'}
            expected.setdefault('by_python', {})[python] = got
            print(
                name,
                f'differs on Python {python}:',
                sorted(different),
                file=sys.stderr,
            )

    for entry in recorded['cases']:
        if entry['name'] not in specs:
            continue  # the size case: its outcome is held, not compared
        answer = scripted
        keep(entry['expected'], observe(specs[entry['name']]), entry['name'])
    for vector in document['boundaries']:
        answer = lambda script, body, v=vector: (v['status'], v['body'])  # noqa: E731
        got = observe(specs[vector['name']])
        got['calls'] = [
            {'status': c['status'], 'answer': None, 'output': c['output']}
            for c in got['calls']
        ]
        reference = {
            **vector['reference'],
            'calls': [
                {
                    'status': vector['status'],
                    'answer': None,
                    'output': c['output'],
                }
                for c in calls
            ],
        }
        keep(reference, got, vector['name'])
        reference.pop('calls')
        for outcome in reference.get('by_python', {}).values():
            outcome.pop('calls', None)
        vector['reference'] = reference
    answer = scripted
    OUT.write_text(
        json.dumps(recorded, indent=1, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    VECTORS.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
        + '\n',
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(cell() if sys.argv[1:] == ['--cell'] else main())
