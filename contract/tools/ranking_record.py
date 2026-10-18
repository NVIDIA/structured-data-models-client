# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Record the reference's ranking answers, predict side, for T2.8's tests.

Run with the reference checkout's interpreter (the reference environment of
``contract/reference-env.txt``), from this repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/ranking_record.py

The S1 ranking cases (``contract/cases/s1/rank.*.json``) are answered by
the scripted endpoint, which gives no probabilities, so they judge the
decomposition, its refusals and each part's payload, but no score. These
vectors judge the predict side (the design's T2.8b row): each is a call
over the ranking cases' data (``contract/bases.yaml``'s ``shop`` and the
cases' ``setup``, on DuckDB), the arguments of ``materialize`` and of
``predict``, an optional edit of the gathered payload a caller can make
(``Request.payloads``), and a script answering each model call
(:func:`answer`), a function of the call's number and body. The reference
runs each, and :func:`observe` keeps what the exits compare: the result
through ``to_pandas()`` and ``.items`` (the ranked candidates, their scores,
any ``scores`` or ``rankings`` an answer carried), its outputs, the
candidates left unscored, the notices' classes, and every call's asked
output fields with the status and text it was answered with, which
``sdm-testkit``'s ``tests/t28_ranking.rs`` replays on rust-host; or the
refusal's class, code and message.

The exits, each vector naming the one it covers (:data:`EXITS`): scores
and rankings, ``top_k`` and the outputs a ranking gives, ``candidates``
left unscored, PD-1 (a NaN score never reaches the order: the reference
refuses a non-finite probability, so a NaN never ranks), a part's failure
said of its candidate, ``prediction_statistic``, ``response_field`` for
``scores`` and ``rankings``, ``instance_id_type`` and ``prediction_type``.

Nothing here is written by hand but the inputs; every expected value is the
reference's. ``observe`` and ``answer`` are imported by
``contract/tests/test_t28_ranking.py``, so both routes are observed and
answered by the same code.

Beside the vectors it records ``controls``: each S1 case refused before
any call (:data:`CONTROLS`), its own call's refusal on the reference (its
message, which the case's line does not keep), and a control call with
the refused argument changed, which the reference gathers, so each case
is refused for the argument it names and not for another reason.
"""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/ranking-vectors.json'
#: The S1 case whose data every vector reads: the shop base and its setup.
DATA_CASE = 'rank.top-k'

ANCHOR = '2023-06-01'
RANK = (
    'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) RANK TOP 2 '
    'FOR EACH users.user_id'
)
BINARY = 'PREDICT COUNT(buys.*, 0, 30, days) > 0 FOR EACH users.user_id'
REGRESSION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
MULTICLASS = 'PREDICT users.country FOR EACH users.user_id'

EXITS = {
    'scores': (
        "scores and rankings: each entity's candidates ordered by the "
        "probability of the part's positive class, ties kept in the "
        "candidates' order, put back together as one answer"
    ),
    'top_k': (
        "top_k: the query's RANK TOP k, or the option in its place, keeps "
        "the k best of each entity's candidates; the outputs a ranking gives"
    ),
    'candidates': (
        'candidates: one part per distinct candidate, a candidate whose '
        'context holds one class left out unscored, a key the query cannot '
        'say refused of its part'
    ),
    'pd1': (
        'PD-1: a NaN score never reaches the order; the reference refuses a '
        "non-finite probability, said of the part's candidate"
    ),
    'parts': "a part's failure said of its candidate, the calls after it unsent",
    'prediction_statistic': (
        "prediction_statistic: refused for a ranking, a regression's sent as "
        'the caller gave it'
    ),
    'response_field': (
        'response_field for rankings: scores and rankings an answer carries '
        "given back as the reference's, or refused where it refuses"
    ),
    'instance_id_type': (
        'instance_id_type: a payload whose instance id is neither an integer '
        "nor text, answered as the reference's"
    ),
    'prediction_type': (
        "prediction_type: a classification's answer not of its target's "
        "type, given back as the reference's"
    ),
}

#: Each part's probability of its positive class, by call then row: ties
#: within a row (entity 1's 0.9, 0.9; entity 2's 0.5, 0.5), and every
#: order of three candidates.
SCORES = [[0.2, 0.9, 0.5], [0.7, 0.9, 0.1], [0.4, 0.3, 0.5]]


def data_case() -> dict[str, Any]:
    r"""The S1 ranking case whose data the vectors read."""
    return json.loads(
        (ROOT / f'contract/cases/s1/{DATA_CASE}.json').read_text(
            encoding='utf-8'
        )
    )


def statements() -> list[str]:
    r"""``contract/bases.yaml``'s ``shop`` and the data case's setup."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    shop = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        shop.append(line[len('  - ') :])
    return [*shop, *data_case()['input']['setup']]


# --- the endpoint's scripts ---------------------------------------------------


def _rows(call: dict[str, Any]) -> list[tuple[int, Any]]:
    table = call['predict']['instance_table']
    at = table['columns'].index('instance_id')
    return [(i, row[at]) for i, row in enumerate(table['rows'])]


def _id(instance: Any) -> str:
    return instance if isinstance(instance, str) else json.dumps(instance)


def _envelope(number: int, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'object': 'prediction.response',
        'id': f'r-t28-{number}',
        'model': 'kumo-relational',
        'metadata': {'served_by': 'scripted'},
        'predictions': items,
    }


def _token(value: Any) -> str:
    r"""A probability as it travels: a number, or a token given as text
    (``NaN``, ``"nan"``)."""
    return value if isinstance(value, str) else repr(value)


def _scores(number: int, call: dict[str, Any], script: dict[str, Any]) -> str:
    r"""Each row's probability of the positive class from the script's
    table (by call, then row); ``null`` leaves the probabilities out, and
    ``carried`` adds item members as JSON text, as :func:`_rule` does."""
    table = script['table']
    if number >= len(table):
        raise ValueError(f'no scores for call {number}')
    classes = call['task']['target']['classes']
    negative, positive = classes[0], classes[-1]
    items = []
    tokens = []
    for index, instance in _rows(call):
        p = table[number][index]
        item: dict[str, Any] = {
            'id': _id(instance),
            'row_index': index,
            'prediction': positive
            if isinstance(p, float) and p >= 0.5
            else negative,
        }
        if 'carried' in script:
            item['__t28_carried__'] = 0
        if p is not None:
            item['probabilities'] = {
                negative: f'__t28_n{index}__',
                positive: f'__t28_p{index}__',
            }
            tokens.append(
                (
                    index,
                    _token(round(1 - p, 12) if isinstance(p, float) else 0.5),
                    _token(p),
                )
            )
        items.append(item)
    text = json.dumps(_envelope(number, items))
    for index, n, p in tokens:
        text = text.replace(f'"__t28_n{index}__"', n).replace(
            f'"__t28_p{index}__"', p
        )
    if 'carried' in script:
        text = text.replace('"__t28_carried__": 0', script['carried'])
    return text


def _rule(number: int, call: dict[str, Any], script: dict[str, Any]) -> str:
    r"""The scripted endpoint's rule (a regression's 1.0, a
    classification's first class) with what the script adds to every item:
    ``carried``, item members as JSON text (``scores``, ``rankings``), and
    ``prediction``, the prediction as JSON text in place of the rule's."""
    task = call['task']
    regression = task['kind'] == 'regression'
    items = []
    for index, instance in _rows(call):
        item: dict[str, Any] = {
            'id': _id(instance),
            'row_index': index,
            'prediction': '__t28_prediction__'
            if 'prediction' in script
            else (1.0 if regression else task['target']['classes'][0]),
        }
        if 'carried' in script:
            item['__t28_carried__'] = 0
        items.append(item)
    text = json.dumps(_envelope(number, items))
    if 'prediction' in script:
        text = text.replace('"__t28_prediction__"', script['prediction'])
    if 'carried' in script:
        text = text.replace('"__t28_carried__": 0', script['carried'])
    return text


def answer(script: dict[str, Any], number: int, body: bytes) -> tuple[int, str]:
    r"""The status and body text answering call ``number`` (from zero)
    under ``script``."""
    call = json.loads(body)
    if script.get('fail_call') == number:
        return 500, json.dumps({'error': 'scripted_failure', 'call': number})
    if script['kind'] == 'scores':
        return 200, _scores(number, call, script)
    if script['kind'] == 'rule':
        return 200, _rule(number, call, script)
    raise ValueError(f'no script {script["kind"]!r}')


@contextmanager
def endpoint(
    script: dict[str, Any],
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    r"""A local endpoint answering each call by ``script``, recording each
    call it received and the status and text it answered."""
    calls: list[dict[str, Any]] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            length = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(length)
            with lock:
                number = len(calls)
                status, text = answer(script, number, body)
                calls.append({'body': body, 'status': status, 'answer': text})
            data = text.encode()
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


# --- the vectors ------------------------------------------------------------

VECTORS: list[dict[str, Any]] = []


def vector(
    name: str,
    covers: str,
    query: str,
    script: dict[str, Any],
    *,
    outputs: list[str] | None = None,
    edit: dict[str, Any] | None = None,
    **materialize: Any,
) -> None:
    assert covers in EXITS, covers
    VECTORS.append(
        {
            'name': name,
            'covers': covers,
            'query': query,
            'materialize': {'anchor_time': ANCHOR, **materialize},
            'predict': {} if outputs is None else {'outputs': outputs},
            'edit': edit,
            'script': script,
        }
    )


def scores(table: list[list[Any]] = SCORES, **more: Any) -> dict[str, Any]:
    return {'kind': 'scores', 'table': table, **more}


def rule(**more: Any) -> dict[str, Any]:
    return {'kind': 'rule', **more}


THREE = {'indices': [0, 1, 2], 'candidates': [0, 1, 2]}

vector('scores.rank-top-2', 'scores', RANK, scores(), **THREE)
vector('top-k.option-1', 'top_k', RANK, scores(), top_k=1, **THREE)
vector('top-k.option-3', 'top_k', RANK, scores(), top_k=3, **THREE)
vector(
    'top-k.outputs-rankings',
    'top_k',
    RANK,
    scores(),
    outputs=['rankings'],
    **THREE,
)
vector(
    'top-k.outputs-prediction',
    'top_k',
    RANK,
    scores(),
    outputs=['prediction'],
    **THREE,
)
vector(
    'top-k.outputs-probabilities',
    'top_k',
    RANK,
    scores(),
    outputs=['probabilities'],
    **THREE,
)
vector('top-k.outputs-none', 'top_k', RANK, scores(), outputs=[], **THREE)
vector(
    'candidates.repeated',
    'candidates',
    RANK,
    scores(),
    indices=[0, 1, 2],
    candidates=[1, 1, 0],
)
vector(
    'candidates.unscored-one',
    'candidates',
    RANK,
    scores(),
    indices=[0, 1, 2],
    candidates=[0, 9, 1],
)
vector(
    'candidates.all-unscored',
    'candidates',
    RANK,
    scores(),
    indices=[0, 1, 2],
    candidates=[9, 8],
)
vector(
    'candidates.text-key',
    'candidates',
    RANK,
    scores(),
    indices=[0, 1, 2],
    candidates=['1', '0'],
)
vector(
    'pd1.nan-first-call',
    'pd1',
    RANK,
    scores([['NaN', 0.9, 0.5], SCORES[1], SCORES[2]]),
    **THREE,
)
vector(
    'pd1.nan-later-call',
    'pd1',
    RANK,
    scores([SCORES[0], [0.7, 'NaN', 0.1], SCORES[2]]),
    **THREE,
)
vector(
    'pd1.nan-as-text',
    'pd1',
    RANK,
    scores([['"nan"', 0.9, 0.5], SCORES[1], SCORES[2]]),
    **THREE,
)
vector(
    'parts.no-probabilities',
    'parts',
    RANK,
    scores([SCORES[0], [0.7, None, 0.1], SCORES[2]]),
    **THREE,
)
vector('parts.failed-call-1', 'parts', RANK, scores(fail_call=1), **THREE)
vector(
    'prediction-statistic.ranking',
    'prediction_statistic',
    RANK,
    scores(),
    prediction_statistic='mean',
    **THREE,
)
vector(
    'prediction-statistic.regression-mean',
    'prediction_statistic',
    REGRESSION,
    rule(),
    indices=[0, 1],
    prediction_statistic='mean',
)
vector(
    'prediction-statistic.edited-not-text',
    'prediction_statistic',
    REGRESSION,
    rule(),
    indices=[0, 1],
    edit={'op': 'prediction_statistic', 'value': 7},
)
vector(
    'response-field.binary-scores',
    'response_field',
    BINARY,
    rule(carried='"scores": [0.25, 0.75]'),
    indices=[0, 1],
)
vector(
    'response-field.binary-rankings',
    'response_field',
    BINARY,
    rule(
        carried='"rankings": [{"candidate": 3, "score": 0.5}, '
        '{"candidate": 1, "score": 0.25}]'
    ),
    indices=[0, 1],
)
vector(
    'response-field.binary-rankings-nan-score',
    'response_field',
    BINARY,
    rule(
        carried='"rankings": [{"candidate": 3, "score": NaN}, '
        '{"candidate": 1, "score": 0.25}]'
    ),
    indices=[0, 1],
)
vector(
    'response-field.binary-scores-nan',
    'response_field',
    BINARY,
    rule(carried='"scores": [NaN]'),
    indices=[0, 1],
)
vector(
    'response-field.regression-scores',
    'response_field',
    REGRESSION,
    rule(carried='"scores": [0.25, 0.75]'),
    indices=[0, 1],
)
vector(
    'response-field.ranking-part-scores',
    'response_field',
    RANK,
    scores(carried='"scores": [0.25]'),
    **THREE,
)
vector(
    'response-field.ranking-part-rankings',
    'response_field',
    RANK,
    scores(carried='"rankings": [{"candidate": 3, "score": 0.5}]'),
    **THREE,
)
vector(
    'instance-id.float',
    'instance_id_type',
    REGRESSION,
    rule(),
    indices=[0, 1],
    edit={'op': 'instance_id', 'value': 1.5},
)
vector(
    'instance-id.bool',
    'instance_id_type',
    REGRESSION,
    rule(),
    indices=[0, 1],
    edit={'op': 'instance_id', 'value': True},
)
vector(
    'prediction-type.binary-float',
    'prediction_type',
    BINARY,
    rule(prediction='0.7'),
    indices=[0, 1],
)
vector(
    'prediction-type.binary-text',
    'prediction_type',
    BINARY,
    rule(prediction='"maybe"'),
    indices=[0, 1],
)
vector(
    'prediction-type.binary-integer',
    'prediction_type',
    BINARY,
    rule(prediction='1'),
    indices=[0, 1],
)
vector(
    'prediction-type.multiclass-integer',
    'prediction_type',
    MULTICLASS,
    rule(prediction='3'),
    indices=[0, 1],
)
vector(
    'prediction-type.multiclass-bool',
    'prediction_type',
    MULTICLASS,
    rule(prediction='true'),
    indices=[0, 1],
)


# --- observing ------------------------------------------------------------


def _cell(value: Any) -> Any:
    r"""A value as JSON, typed: an integer as its digits, a float as its
    IEEE-754 bits (a NaN as ``{'f64': 'nan'}``), text, a boolean, a list
    and a mapping cell by cell, a NumPy scalar as its Python value, a
    missing value as ``None``."""
    if hasattr(value, 'item') and not isinstance(value, (list, dict, tuple)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if value is None:
        return None
    if isinstance(value, bool):
        return {'bool': value}
    if isinstance(value, int):
        return {'int': str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            return {'f64': 'nan'}
        return {'f64_bits': struct.pack('>d', value).hex()}
    if isinstance(value, str):
        return {'str': value}
    if isinstance(value, (list, tuple)):
        return {'list': [_cell(v) for v in value]}
    if isinstance(value, dict):
        return {'map': [[str(k), _cell(v)] for k, v in value.items()]}
    if hasattr(value, 'tolist'):
        return _cell(value.tolist())
    return {'other': repr(value)}


def _refusal(error: BaseException) -> dict[str, Any]:
    details = getattr(error, 'details', None)
    return {
        'outcome': 'refused',
        'class': type(error).__name__,
        'code': getattr(error, 'code', None),
        # The classes it is an instance of: the core's host may raise a
        # subclass of the reference's class for the same code.
        'mro': [c.__name__ for c in type(error).__mro__],
        'message': str(error),
        # The HTTP status a refusal the endpoint gave carries
        # (NimRequestError's status_code), as the user reads it; none on a
        # local error.
        'status_code': getattr(error, 'status_code', None),
        # The core's rule, where a refusal names one (none on the
        # reference): for the reader, never compared.
        'rule': details.get('rule') if isinstance(details, dict) else None,
    }


def _edit(request: Any, edit: dict[str, Any] | None) -> None:
    r"""A caller's edit of the gathered payload (``Request.payloads``)."""
    if edit is None:
        return
    payload = request.payloads[0]
    if edit['op'] == 'prediction_statistic':
        payload.setdefault('output', {})['prediction_statistic'] = edit['value']
    elif edit['op'] == 'instance_id':
        table = payload['predict']['instance_table']
        at = table['columns'].index('instance_id')
        table['rows'][0][at] = edit['value']
    else:
        raise ValueError(f'no edit {edit["op"]!r}')


def _items(result: Any) -> list[dict[str, Any]]:
    r"""``result.items``: each item's row, id, prediction, probabilities,
    scores and rankings, as cells."""
    return [
        {
            name: _cell(getattr(item, name, None))
            for name in (
                'row_index',
                'id',
                'prediction',
                'probabilities',
                'scores',
                'rankings',
            )
        }
        for item in result.items
    ]


def _frame(result: Any) -> dict[str, Any]:
    r"""The result through ``to_pandas()``: its columns, dtypes and rows."""
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


def observe(spec: dict[str, Any]) -> dict[str, Any]:
    r"""The vector's outcome on the route the environment selects."""
    import duckdb
    import pandas as pd
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    for statement in statements():
        connection.execute(statement)
    arguments = dict(spec['materialize'])
    arguments['anchor_time'] = pd.Timestamp(arguments['anchor_time'])
    notices: list[str] = []
    out: dict[str, Any]
    with endpoint(spec['script']) as (url, calls):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            stage = 'materialize'
            try:
                schema = Schema.from_source(DuckDBSource(connection))
                request = PredictiveQuery(spec['query'], schema).materialize(
                    **arguments
                )
                _edit(request, spec['edit'])
                stage = 'predict'
                with SDMClient(url, max_retries=0) as client:
                    result = client.predict(
                        request, model=KumoRelational(), **spec['predict']
                    )
            except Exception as error:  # noqa: BLE001 - the outcome is compared
                out = {'stage': stage, **_refusal(error)}
            else:
                out = {
                    'stage': 'predict',
                    'outcome': 'result',
                    'outputs': list(result.outputs),
                    'unscored': _cell(
                        list(getattr(result, 'unscored_candidates', ()) or ())
                    ),
                    'items': _items(result),
                    **_frame(result),
                }
        notices = [
            type(w.message).__name__
            for w in caught
            if type(w.message).__name__ == 'SDMWarning'
        ]
    out['notices'] = notices
    out['calls'] = [
        {
            'output': json.loads(c['body']).get('output'),
            'status': c['status'],
            'answer': c['answer'],
        }
        for c in calls
    ]
    return out


#: Each S1 case refused before any call, and the control call beside it:
#: the same call with the refused argument changed (``set``, ``None``
#: dropping it) or the query changed (``query``), which the reference
#: gathers, so the case is refused for that argument and nothing else.
CONTROLS = {
    'rank.top-k-zero': {'set': {'top_k': None}},
    'rank.no-candidates': {'set': {'candidates': [0, 1]}},
    'rank.candidates-empty': {'set': {'candidates': [0, 1]}},
    'rank.candidates-not-a-ranking': {'set': {'candidates': None}},
    'rank.most-parts': {'set': {'candidates': [0, 1]}},
    'rank.batched': {'set': {'indices': [0, 1]}},
    # The entity ceiling over a ranking's parts, past it and at it: the
    # same candidates over the eight users the data holds.
    'rank.entities-by-candidates.over': {'set': {'indices': list(range(8))}},
    'rank.entities-by-candidates.at': {'set': {'indices': list(range(8))}},
    'forecast.top-k': {'set': {'top_k': None}},
    'forecast.most-parts': {
        'query': ['FORECAST 201 TIMEFRAMES', 'FORECAST 2 TIMEFRAMES']
    },
}


def gathered(query: str, materialize: dict[str, Any]) -> dict[str, Any]:
    r"""Whether the reference gathers ``query`` with ``materialize``'s
    arguments (the S1 cases' data): ``materialize``'s outcome, its refusal's
    class, code and message, and how many parts a question asked in parts
    is gathered as."""
    import duckdb
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    for statement in statements():
        connection.execute(statement)
    arguments = dict(materialize)
    if 'anchor_time' in arguments:
        arguments['anchor_time'] = pd.Timestamp(arguments['anchor_time'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery(query, schema).materialize(**arguments)
        except Exception as error:  # noqa: BLE001 - the outcome is compared
            return {'stage': 'materialize', **_refusal(error)}
    parts = request.parts
    return {
        'stage': 'materialize',
        'outcome': 'result',
        'parts': None if parts is None else len(parts),
    }


def controls() -> list[dict[str, Any]]:
    r"""Each case of :data:`CONTROLS`, its own call's outcome and its
    control's."""
    out = []
    for case_id, change in CONTROLS.items():
        case = json.loads(
            (ROOT / f'contract/cases/s1/{case_id}.json').read_text(
                encoding='utf-8'
            )
        )
        query = case['input']['pql']
        materialize = dict(case['input']['materialize'])
        control = dict(materialize)
        for name, value in change.get('set', {}).items():
            if value is None:
                control.pop(name, None)
            else:
                control[name] = value
        control_query = query
        if 'query' in change:
            control_query = query.replace(*change['query'])
        out.append(
            {
                'case': case_id,
                'change': change,
                'refused': gathered(query, materialize),
                'control': gathered(control_query, control),
            }
        )
    return out


def main() -> int:
    import sdm_client

    reference = Path.home() / 'code/sdmc-reference'
    if not Path(sdm_client.__file__).resolve().is_relative_to(reference):
        print("run with the reference checkout's interpreter", file=sys.stderr)
        return 2
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    recorded = []
    for spec in VECTORS:
        expected = observe(spec)
        recorded.append({**spec, 'expected': expected})
        print(
            spec['name'],
            expected['stage'],
            expected['outcome'],
            expected.get('class'),
            expected.get('code'),
            len(expected['calls']),
            file=sys.stderr,
        )
    recorded_controls = controls()
    for each in recorded_controls:
        print(
            each['case'],
            each['refused'].get('code'),
            each['control']['outcome'],
            each['control'].get('code'),
            file=sys.stderr,
        )
    OUT.write_text(
        json.dumps(
            {
                'format': 'sdm-ranking-vectors/1',
                'recorded_by': 'contract/tools/ranking_record.py',
                'reference': commit,
                'database': 'duckdb',
                'data_case': DATA_CASE,
                'exits': EXITS,
                'vectors': recorded,
                'controls': recorded_controls,
            },
            indent=1,
            ensure_ascii=False,
            allow_nan=False,
        )
        + '\n',
        encoding='utf-8',
    )
    print(f'{len(recorded)} vectors to {OUT}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
