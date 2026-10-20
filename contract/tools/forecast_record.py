#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records multi-step forecasts into ``contract/t136-forecast-vectors.json``
(T1.36 fix 21: forecast decomposition moves into the core).

Written by T1.36's independent test author before the fix. Two parts, each
run from this repository's root:

    TZ=UTC ~/code/sdmc-reference/.venv/bin/python -I contract/tools/forecast_record.py reference
    TZ=UTC <candidate venv>/bin/python -I contract/tools/forecast_record.py candidate

``reference`` runs each vector through the reference checkout's public calls
(its interpreter, the ``ref-pandas3`` cell) and keeps the answer: the
outcome, each request the scripted endpoint received (its ``payload``-rule
digest, ``empty_null_record.digest``, and the tables it carries), and the
result through ``to_pandas()`` (columns, dtypes, row digest, and the rows
as the core writes cells: an integer's digits, a float's IEEE-754 bits).

``recandidate <commit> <id> ...`` records again, for the vectors named,
what python-core gives (``python_core``) under a later candidate, whose
commit it keeps beside them (``python_core_candidate``), and each one's
``rust_host`` as :data:`RUST_HOST` says now: a vector a later package
changed on the core by design (T2.8's ``count-filtered-3``, whose
``target_filter`` refusal it lifts, and T2.13b's ``sum-entity-filter-3``,
whose ``entity_filter`` refusal it lifts).

``candidate`` runs each vector through this checkout, forced onto the core
(``python-core``) and on the default routes (``python-old``), and keeps
what each gives today: the same projection, the error's message, and the
SHA-256 of each request's raw bytes with ``metadata.request_id``'s value
masked (a fresh id per call). Recorded at fc2c0e2b, before the fix, so
``python-core``'s forecasts are held byte for byte to what the Python
decomposition sends today.

Each vector is ``contract/bases.yaml``'s ``shop`` asked a forecast for
users 0 and 1 at an anchor time; ``case`` is it as a contract case
(``contract/cases/s1/base.regression.json``'s, its question, anchor and
routes the vector's), which ``sdm-testkit``'s ``tests/t136_forecast.rs``
runs on ``rust-host``.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import platform
import re
import struct
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t136-forecast-vectors.json'
TEMPLATE = ROOT / 'contract/cases/s1/base.regression.json'
FORMAT = 'sdm-forecast-vectors/1'
CASE_PREFIX = 't136fc.'
INDICES = [0, 1]
ANCHOR = '2024-03-01'
#: Where the data starts: one week into ``shop``'s orders.
EARLY = '2023-01-10'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


EN = _module('empty_null_record', ROOT / 'contract/tools/empty_null_record.py')


def _v(
    vid: str, pql: str, anchor: str = ANCHOR, **options: Any
) -> dict[str, Any]:
    return {'id': vid, 'pql': pql, 'anchor_time': anchor, 'options': options}


_EACH = 'FOR EACH users.user_id'

#: The vectors. ``rust_host`` says what ``rust-host`` is held to:
#: ``reference`` (the reference's answer), ``python-core`` (python-core's
#: outcome today: a filter the core does not port, so the fix leaves it
#: refused), or ``none`` (an option a case cannot carry).
VECTORS: list[dict[str, Any]] = [
    # Several steps, each aggregation the forecast path takes, each unit.
    _v(
        'sum-days-3',
        f'PREDICT SUM(orders.price, 0, 7, days) FORECAST 3 TIMEFRAMES {_EACH}',
    ),
    _v(
        'sum-days-1',
        f'PREDICT SUM(orders.price, 0, 30, days) FORECAST 1 TIMEFRAMES {_EACH}',
    ),
    _v(
        'sum-offset-days-2',
        f'PREDICT SUM(orders.price, 7, 14, days) FORECAST 2 TIMEFRAMES {_EACH}',
    ),
    _v(
        'count-days-4',
        f'PREDICT COUNT(orders.*, 0, 14, days) FORECAST 4 TIMEFRAMES {_EACH}',
    ),
    _v(
        'avg-weeks-2',
        f'PREDICT AVG(orders.price, 0, 2, weeks) FORECAST 2 TIMEFRAMES {_EACH}',
    ),
    _v(
        'min-months-2',
        f'PREDICT MIN(orders.price, 0, 1, months) FORECAST 2 TIMEFRAMES {_EACH}',
    ),
    _v(
        'max-hours-2',
        f'PREDICT MAX(orders.price, 0, 48, hours) FORECAST 2 TIMEFRAMES {_EACH}',
    ),
    _v(
        'sum-early-anchor-3',
        f'PREDICT SUM(orders.price, 0, 3, days) FORECAST 3 TIMEFRAMES {_EACH}',
        EARLY,
    ),
    # The reference's parts drop an ASSUMING clause (decompose._per_step
    # rebuilds the target alone), so it answers; held as it answers.
    _v(
        'sum-assuming-3',
        f'PREDICT SUM(orders.price, 0, 7, days) FORECAST 3 TIMEFRAMES {_EACH} ASSUMING COUNT(orders.*, 0, 7, days) > 0',
    ),
    # A filtered aggregation keeps its filter in each part, and an entity
    # filter its clause; the core ports neither filter.
    _v(
        'count-filtered-3',
        f'PREDICT COUNT(orders.* WHERE orders.price > 50, 0, 30, days) FORECAST 3 TIMEFRAMES {_EACH}',
    ),
    _v(
        'sum-entity-filter-3',
        f'PREDICT SUM(orders.price, 0, 7, days) FORECAST 3 TIMEFRAMES {_EACH} WHERE users.age > 21',
    ),
    # Refused.
    _v(
        'refuse.zero-steps',
        f'PREDICT SUM(orders.price, 0, 7, days) FORECAST 0 TIMEFRAMES {_EACH}',
    ),
    _v(
        'refuse.past-most-parts',
        f'PREDICT SUM(orders.price, 0, 1, days) FORECAST 201 TIMEFRAMES {_EACH}',
    ),
    _v(
        'refuse.top-k',
        f'PREDICT SUM(orders.price, 0, 7, days) FORECAST 3 TIMEFRAMES {_EACH}',
        top_k=2,
    ),
    _v(
        'refuse.step-outruns-data',
        f'PREDICT SUM(orders.price, 0, 7, days) FORECAST 3 TIMEFRAMES {_EACH}',
        EARLY,
    ),
    _v(
        'refuse.boolean-target',
        f'PREDICT SUM(orders.price, 0, 30, days) > 10 FORECAST 2 TIMEFRAMES {_EACH}',
    ),
    _v(
        'refuse.static-target',
        f'PREDICT users.age FORECAST 2 TIMEFRAMES {_EACH}',
    ),
]
RUST_HOST = {
    # count-filtered-3 (a target with WHERE) is the reference's on both core
    # hosts from T2.8, which lifts target_filter for a ranking's questions;
    # sum-entity-filter-3 (an entity filter) from T2.13b, which lifts
    # entity_filter.
    'refuse.top-k': 'none',
}


def case_for(vector: dict[str, Any]) -> dict[str, Any]:
    r"""The vector as a contract case: ``base.regression``'s, its question
    and anchor the vector's, on every DuckDB route of the three calls."""
    case = copy.deepcopy(json.loads(TEMPLATE.read_text(encoding='utf-8')))
    case['id'] = CASE_PREFIX + vector['id']
    case['status'] = 'diagnostic'
    case['paths'] = [
        'schema.from-source/all-tables/duckdb',
        'query.materialize/indices-forecast/duckdb',
        'client.predict/forecast/duckdb',
    ]
    case['cells'] = []
    case['input']['pql'] = vector['pql']
    case['input']['materialize'] = {
        'indices': list(INDICES),
        'anchor_time': vector['anchor_time'],
    }
    case['core']['materialize']['indices'] = EN.core_indices(INDICES)
    nanos = _nanos(vector['anchor_time'])
    case['core']['materialize']['anchor_time'] = {
        't': 'ts',
        'v': str(nanos),
        'unit': 'ns',
        'zone': None,
    }
    case['expect'] = {'outcome': 'success', 'stage': 'predict'}
    case['why'] = f'T1.36 fix 21 vector {vector["id"]}'
    case['source'] = 'contract/tools/forecast_record.py'
    case.pop('register', None)
    return case


def _nanos(day: str) -> int:
    from datetime import datetime, timezone

    moment = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return int(moment.timestamp()) * 1_000_000_000


# --- one run --------------------------------------------------------------------

_REQUEST_ID = re.compile(rb'("request_id"\s*:\s*")[^"]*(")')


def masked_bytes(body: bytes) -> bytes:
    r"""``body`` with ``metadata.request_id``'s value replaced, the one part
    of a request drawn fresh per call; every other byte kept."""
    return _REQUEST_ID.sub(rb'\1<masked>\2', body)


def _number(token: str) -> dict[str, str]:
    return {'number': token}


def _cell(value: Any) -> Any:
    import numpy as np

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return {'bool': bool(value)}
    if isinstance(value, (int, np.integer)):
        return {'i64': str(int(value))}
    if isinstance(value, (float, np.floating)):
        return {'f64_bits': struct.pack('>d', float(value)).hex()}
    return {'utf8': str(value)}


def _shop() -> list[str]:
    return EN.shop()


def drive(
    vector: dict[str, Any],
    *,
    core: bool = False,
    routes: list[str] | None = None,
) -> dict[str, Any]:
    r"""``vector`` through the public calls of the ``sdm_client`` this
    interpreter imports, against the scripted endpoint; forced onto the
    core's DuckDB routes with ``core``, or onto ``routes`` when given. Its outcome (with the message),
    each request received and the result."""
    import duckdb
    import pandas as pd
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
    from sdm_connectors import DuckDBSource

    serve = _module(
        'scripted_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
    ).serve
    runner = _module('run_python', ROOT / 'contract/runners/run_python.py')
    previous = os.environ.pop('SDM_CORE_ROUTES', None)
    if routes is not None:
        os.environ['SDM_CORE_ROUTES'] = ','.join(routes)
    elif core:
        os.environ['SDM_CORE_ROUTES'] = ','.join(EN.core_routes({}))
    connection = duckdb.connect()
    stage = 'from_source'
    answer: dict[str, Any] = {'payloads': [], 'result': None}
    try:
        for statement in _shop():
            connection.execute(statement)
        with (
            serve() as (url, received),
            warnings.catch_warnings(record=True) as shown,
        ):
            warnings.simplefilter('always')
            try:
                schema = Schema.from_source(DuckDBSource(connection))
                stage = 'materialize'
                request = PredictiveQuery(vector['pql'], schema).materialize(
                    indices=list(INDICES),
                    anchor_time=pd.Timestamp(vector['anchor_time']),
                    **vector['options'],
                )
                stage = 'predict'
                with SDMClient(url) as client:
                    result = client.predict(request, model=KumoRelational())
                projected = runner._result_projection(result)
                frame = result.to_pandas()
                projected['cells'] = [
                    [_cell(value) for value in row]
                    for row in frame.itertuples(index=False, name=None)
                ]
                answer['result'] = projected
                answer['outcome'] = {
                    'result': 'success',
                    'stage': stage,
                    'error': None,
                    'code': None,
                    'message': None,
                }
            except Exception as error:  # noqa: BLE001 - the call's outcome
                code = getattr(error, 'code', None)
                answer['outcome'] = {
                    'result': 'refusal',
                    'stage': stage,
                    'error': type(error).__name__,
                    'code': code if isinstance(code, str) else None,
                    'message': str(error),
                }
            for method, path, body in received.requests:
                parsed = json.loads(
                    body, parse_int=_number, parse_float=_number
                )
                answer['payloads'].append(
                    {
                        'method': method,
                        'path': path,
                        'digest': EN.digest(parsed),
                        'tables': EN.tables_with_rows(parsed),
                        'masked_sha256': hashlib.sha256(
                            masked_bytes(body)
                        ).hexdigest(),
                    }
                )
            answer['warnings'] = [type(w.message).__name__ for w in shown]
    finally:
        connection.close()
        os.environ.pop('SDM_CORE_ROUTES', None)
        if previous is not None:
            os.environ['SDM_CORE_ROUTES'] = previous
    return answer


def public(answer: dict[str, Any]) -> dict[str, Any]:
    r"""What the reference is compared on: the outcome without its message
    (decision H17), each request's digest and tables, and the result."""
    outcome = {k: v for k, v in answer['outcome'].items() if k != 'message'}
    return {
        'outcome': outcome,
        'payloads': [
            {k: p[k] for k in ('method', 'path', 'digest', 'tables')}
            for p in answer['payloads']
        ],
        'result': answer['result'],
    }


# --- recording ------------------------------------------------------------------


def observed_predicted(observation: dict[str, Any]) -> dict[str, Any]:
    r"""An observation's ``core.predicted`` as the hosts are compared on
    it: the body by its ``payload``-rule digest and tables (the request id
    masked), its JSON type, the model, metadata and result; and the
    outcome's result and code."""
    core = observation.get('core') or {}
    predicted = core.get('predicted') or {}
    body = predicted.get('body')
    return {
        'outcome': {
            'result': observation['outcome']['result'],
            'code': observation['outcome']['code'],
        },
        'body_type': type(body).__name__,
        'body': (
            {'digest': EN.digest(body), 'tables': EN.tables_with_rows(body)}
            if isinstance(body, dict)
            else None
        ),
        'model': predicted.get('model'),
        'metadata': predicted.get('metadata'),
        'result': predicted.get('result'),
    }


def observe(vectors: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    r"""The vectors the reference answers and the core is held to, as
    cases through ``run_python.py --mode mode`` in this interpreter: each
    observation's :func:`observed_predicted`, by vector id."""
    import tempfile

    document = json.loads(OUT.read_text(encoding='utf-8'))
    held = {
        v['id']
        for v in document['vectors']
        if v['rust_host'] == 'reference'
        and v['reference']['outcome']['result'] == 'success'
    }
    with tempfile.TemporaryDirectory(prefix='t136fc-') as tmp:
        cases = Path(tmp) / 'cases'
        cases.mkdir()
        for vector in vectors:
            if vector['id'] in held:
                case = case_for(vector)
                (cases / f'{case["id"]}.json').write_text(
                    json.dumps(case), encoding='utf-8'
                )
        out = Path(tmp) / 'out.jsonl'
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(ROOT / 'contract/runners/run_python.py'),
                '--mode',
                mode,
                '--out',
                str(out),
                str(cases),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        if done.returncode != 0:
            raise RuntimeError(f'{mode}: {done.stdout}\n{done.stderr}')
        return {
            json.loads(line)['case'][len(CASE_PREFIX) :]: observed_predicted(
                json.loads(line)
            )
            for line in out.read_text(encoding='utf-8').splitlines()
        }


def _environment() -> dict[str, str]:
    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    return {
        'python': platform.python_version(),
        'pandas': pd.__version__,
        'pyarrow': pa.__version__,
        'numpy': np.__version__,
        'duckdb': duckdb.__version__,
    }


def _git(directory: Path) -> str:
    return subprocess.run(
        ['git', '-C', str(directory), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _load() -> dict[str, Any]:
    if OUT.exists():
        return json.loads(OUT.read_text(encoding='utf-8'))
    return {}


def main(argv: list[str]) -> int:
    import sdm_client

    part = argv[0] if argv else ''
    document = _load()
    by_id = {v['id']: v for v in document.get('vectors', [])}
    if part == 'reference':
        reference = Path.home() / 'code/sdmc-reference'
        where = Path(sdm_client.__file__).resolve()
        if not where.is_relative_to(reference):
            print("sdm_client is not the reference checkout's", file=sys.stderr)
            return 1
        document['reference'] = _git(reference)
        document['reference_environment'] = _environment()
        for vector in VECTORS:
            entry = by_id.setdefault(vector['id'], {})
            entry.update(copy.deepcopy(vector))
            entry['rust_host'] = RUST_HOST.get(vector['id'], 'reference')
            entry['case'] = case_for(vector)
            entry['reference'] = public(drive(vector))
            print(vector['id'], entry['reference']['outcome'], file=sys.stderr)
    elif part == 'candidate':
        document['candidate'] = _git(ROOT)
        document['candidate_environment'] = _environment()
        for vector in VECTORS:
            entry = by_id[vector['id']]
            entry['python_core'] = drive(vector, core=True)
            entry['python_old'] = drive(vector)
            print(
                vector['id'],
                entry['python_core']['outcome']['result'],
                entry['python_core']['outcome']['code'],
                file=sys.stderr,
            )
    elif part == 'recandidate':
        # A later candidate's python-core for the vectors named, which it
        # changed by design, with the candidate's commit beside each.
        if len(argv) < 3 or any(i not in by_id for i in argv[2:]):
            print('usage: recandidate <commit> <id> ...', file=sys.stderr)
            return 2
        for vector in VECTORS:
            if vector['id'] not in argv[2:]:
                continue
            entry = by_id[vector['id']]
            entry['python_core'] = drive(vector, core=True)
            entry['python_core_candidate'] = argv[1]
            entry['rust_host'] = RUST_HOST.get(vector['id'], 'reference')
            print(
                vector['id'],
                entry['python_core']['outcome']['result'],
                entry['python_core']['outcome']['code'],
                file=sys.stderr,
            )
    elif part == 'observation':
        # python-core's observation (run_python.py --mode python-core) of
        # each forecast the reference answers, recorded after fix 21: the
        # shape rust-host's observation is held to.
        document['observation_candidate'] = _git(ROOT)
        observed = observe(
            [v for v in VECTORS if v['id'] in by_id], 'python-core'
        )
        for vector in VECTORS:
            if vector['id'] in observed:
                by_id[vector['id']]['python_core_observation'] = observed[
                    vector['id']
                ]
    else:
        print(
            'usage: forecast_record.py '
            'reference|candidate|recandidate|observation',
            file=sys.stderr,
        )
        return 2
    document['format'] = FORMAT
    document['recorded_by'] = 'contract/tools/forecast_record.py'
    document['template'] = 'contract/cases/s1/base.regression.json'
    document['indices'] = INDICES
    document['payload_exclusions'] = EN.exclusions()
    document['vectors'] = [by_id[v['id']] for v in VECTORS]
    OUT.write_text(json.dumps(document, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
