#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Register P1B-13's accuracy benchmark (decision E73(a), T1.29).

On demand and local only, like ``scripts/live/rfm_smoke.py``, whose
redaction it uses: it never runs in CI, gates nothing in CI, and no locked
check reads it. P1B-13 (decision E66(a)) has the core read a null where the
sampler's walk tells rows apart as no link, where the reference's sampler
numbers such rows out of step; the owner approved it on condition that model
accuracy holds, benchmarked before Phase 1 acceptance, by this protocol
(fixed 2026-10-12 before any result was seen):

- **Endpoint**: the serving endpoint the owner authorised, called through
  the client's own Databricks serving transport
  (``SDMClient.for_databricks_serving``) with the workspace and credentials
  of a local Databricks CLI profile. Its name comes from ``--endpoint`` or
  ``SDM_BENCHMARK_ENDPOINT``, the profile from ``--profile`` or
  ``SDM_BENCHMARK_PROFILE``, never from this repository; a live run needs
  both, so it never falls back to the SDK's default configuration. Nothing in the
  workspace is created or changed: every call is a prediction.
- **Data** (:data:`DATASETS`): a generated DuckDB dataset holding null keys
  (``orders.order_id``, the key of the table the walk reads by foreign key)
  and null foreign keys (``orders.item_id``, looked up at the walk's second
  hop), which are the two nulls P1B-13 changes; the same dataset without
  nulls as the control (each null key replaced by a fresh key, each null
  foreign key by the item drawn for that row, every other value equal); and
  AdventureWorks (``contract/fixtures/adventureworks``, as
  ``base.adventureworks`` loads and edits it).
- **Comparison**: for each dataset, task and seed, the same request built on
  the old route and on the core route (``SDM_CORE_ROUTES``), each
  ``materialize`` given the same ``random_seed`` (paired seeds,
  :data:`SEEDS` by default), the same entities, anchor time, options and
  inference configuration, sent to the same endpoint with the same model.
- **Metrics**: MAE for a regression, AUROC for a binary task (the positive
  class's probability as the score), macro-F1 for a multiclass task (the
  predicted class), each against labels computed in DuckDB from the data
  past the anchor time, as the core's labels are read (a window's rows after
  its start and not after its end), or the static target's own value.
- **Pass** (:func:`verdict`): for every dataset and task, the core is no more
  than 2% worse than the old route, relative, on the metric averaged over
  the seeds (MAE: ``(core - old) / old``; AUROC and macro-F1:
  ``(old - core) / old``); and on the no-null control each task's paired
  difference ``core - old``, averaged over the seeds, has a 95% bootstrap
  interval over the entities (:data:`BOOTSTRAP` resamples, a fixed
  generator) that holds 0: the controls agree within noise. The same
  interval is reported for every other unit too, for information only, so
  a marginal result on the null dataset can be judged; it gates nothing
  there. The requests the two routes send are also compared, request id
  aside: on the control they are expected to be one request.
- **Scoring** (T1.29 review round 1): an answer no metric can score (a
  regression prediction that is not a finite number, a binary score that
  is not a finite probability, a class that is not text) is refused before
  any metric; a unit whose metric is undefined or not finite fails; and
  the relative worsening from a zero baseline follows the metric's
  direction, so an improvement never fails.

Calls and predictions are counted before any is made (:func:`plan`); the
run refuses a plan of more than ``--max-predictions`` (default 500, and
never above 500). The requests are counted again where they are sent
(:class:`Wire`, wrapping ``requests.Session.send`` in this process): each
prediction may send one request, so an SDK retry on a throttled endpoint
is refused before it is sent, and none past the plan; the summary records
the requests sent. The route sent first alternates from seed to seed, and
a control whose two routes send different requests fails the run, since it
then isolates nothing. By
default it makes 28 calls and 448 predictions: the generated pair three
tasks each, AdventureWorks its notebook's regression, two seeds, two routes,
16 entities a call.

    scripts/live/null_link_benchmark.py --dry-run
    scripts/live/null_link_benchmark.py --endpoint NAME --profile NAME \
        --raw-dir DIR --summary FILE [--seeds 0,1] [--entities 16]
        [--max-predictions 500] [--timeout SECONDS]

``--dry-run`` builds every request on both routes and answers it in process
by the scripted endpoint's rule, with a probability for each class a
classification asks for so the binary scoring path runs too, with no
network access (it opens no socket
and reads no Databricks configuration): it shows the plan, whether each pair
of requests is one request, and the metrics of the scripted answers, which
say nothing of a model. A live run writes, under ``--raw-dir``, each call's
predictions beside their labels (``calls.jsonl``), and to ``--summary`` the
numbers alone (``sdm-p1b13-benchmark/1``): no endpoint, workspace, host,
profile, token or request id is written anywhere, and every line printed
passes through ``rfm_smoke``'s redaction.

Exit status: 0 when the benchmark passes, 1 when it fails or a call fails,
2 for bad usage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT_VARIABLE = 'SDM_BENCHMARK_ENDPOINT'
PROFILE_VARIABLE = 'SDM_BENCHMARK_PROFILE'
FORMAT = 'sdm-p1b13-benchmark/1'
#: The paired seeds each request is materialized with, on both routes.
SEEDS = (0, 1)
#: Entities asked about in each call.
ENTITIES = 16
#: The relative worsening the core may show on a metric (E73(a)).
TOLERANCE = 0.02
#: The bootstrap of the controls' paired differences: resamples and seed.
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261012
#: The generated dataset's generator seed and shape.
GENERATOR_SEED = 2029
USERS, ITEMS = 120, 40
#: The share of orders whose key, and whose item, is null.
NULL_KEYS, NULL_FOREIGN_KEYS = 0.08, 0.12
#: The anchor time of the generated dataset's temporal tasks.
ANCHOR = '2024-03-01'
#: AdventureWorks' anchor time, inside its order dates (2022-06 to 2025-06).
AW_ANCHOR = '2025-03-01'
#: Where AdventureWorks' case, whose input the benchmark reuses, lives.
CASES = ROOT / 'contract/cases/s1'


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


smoke = _module('rfm_smoke', ROOT / 'scripts/live/rfm_smoke.py')
say = smoke.say
redact = smoke.redact


def _runner() -> ModuleType:
    r"""``contract/runners/run_python.py``: the bases and the case's calls."""
    return _module('run_python', ROOT / 'contract/runners/run_python.py')


def core_routes() -> list[str]:
    r"""The paths the core route forces: every DuckDB path of
    ``Schema.from_source`` (every table, or tables named, with the catalog's
    keys), ``materialize`` with indices and ``predict``, as the client's
    own route groups name them (``sdm_client._routing``). Declared here,
    not read from the cases, whose routes follow their admission (T1.29
    review of part 1: AdventureWorks, diagnostic in part 1, forces no
    route)."""
    from sdm_client import _routing

    return sorted(
        {
            *_routing.FROM_SOURCE_DUCKDB,
            *_routing.FROM_SOURCE_DUCKDB_TABLES,
            *_routing.MATERIALIZE_DUCKDB,
            *_routing.PREDICT_DUCKDB,
        }
    )


# --- the data -----------------------------------------------------------------


@dataclass
class Dataset:
    r"""A dataset: how to build its DuckDB connection and schema, and its
    tasks."""

    name: str
    connect: Callable[[], Any]
    from_source: dict[str, Any]
    edits: list[dict[str, Any]]
    tasks: list[Task]
    #: Whether it holds the nulls P1B-13 changes (the controls do not).
    nulls: bool
    #: The dataset a control is the control of.
    control_of: str | None = None


@dataclass
class Task:
    r"""A question, its kind and metric, its anchor time and options, and the
    labels of the entities it asks about."""

    name: str
    kind: str  # regression, binary or multiclass
    pql: str
    anchor: str | None
    labels: Callable[[Any], dict[Any, Any]]
    options: dict[str, Any] = field(default_factory=dict)


def generated_rows(nulls: bool) -> dict[str, list[tuple[Any, ...]]]:
    r"""The generated dataset's rows, with or without its nulls: the same
    draws either way, each null replaced in the control by a fresh key or
    by the item drawn for that row."""
    import numpy as np

    rng = np.random.default_rng(GENERATOR_SEED)
    categories = ['bikes', 'parts', 'apparel', 'gear']
    regions = ['north', 'south', 'east', 'west']
    items = [
        (i, categories[i % 4], float(round(rng.uniform(5, 120), 2)))
        for i in range(ITEMS)
    ]
    start = np.datetime64('2023-07-01T00:00:00')
    end = np.datetime64('2024-05-31T00:00:00')
    span = int((end - start) / np.timedelta64(1, 'm'))
    users, orders = [], []
    order_id = 0
    for user in range(USERS):
        region = int(rng.integers(0, 4))
        rate = float(rng.gamma(2.0, 0.3))  # orders a month
        taste = (region + int(rng.integers(0, 2))) % 4
        count = int(rng.poisson(rate * 11))
        spend = []
        for _ in range(count):
            preferred = rng.random() < 0.7
            pool = [i for i in items if i[1] == categories[taste]]
            item = (
                pool[int(rng.integers(0, len(pool)))]
                if preferred
                else (items[int(rng.integers(0, ITEMS))])
            )
            quantity = int(rng.integers(1, 4 if rate < 1 else 7))
            at = start + np.timedelta64(int(rng.integers(0, span)), 'm')
            null_key = rng.random() < NULL_KEYS
            null_item = rng.random() < NULL_FOREIGN_KEYS
            orders.append(
                (
                    None if (nulls and null_key) else order_id,
                    user,
                    None if (nulls and null_item) else item[0],
                    str(at),
                    quantity,
                )
            )
            order_id += 1
            if str(at) < ANCHOR:
                spend.append(item[2] * quantity)
        average = sum(spend) / len(spend) if spend else 0.0
        tier = 0 if average < 60 else (1 if average < 160 else 2)
        if rng.random() < 0.1:
            tier = int(rng.integers(0, 3))
        users.append(
            (
                user,
                int(rng.integers(18, 80)),
                regions[region],
                ['low', 'mid', 'high'][tier],
            )
        )
    # Fewer orders than the core's preview of a table (sdm-schema's
    # PREVIEW_ROWS, 1,000), as the protocol fixed them on 2026-10-12, when
    # the core refused to choose a time column by the whole table; from
    # T1.32e it reads the whole table as the reference does.
    if len(orders) >= 1_000:
        raise ValueError('the generated orders reach the core preview')
    return {'users': users, 'items': items, 'orders': orders}


def generated_connection(nulls: bool) -> Any:
    import duckdb
    import pandas as pd

    rows = generated_rows(nulls)
    connection = duckdb.connect()
    frames = {
        'users': pd.DataFrame(
            rows['users'], columns=['user_id', 'age', 'region', 'segment']
        ).astype({'user_id': 'int64', 'age': 'int32'}),
        'items': pd.DataFrame(
            rows['items'], columns=['item_id', 'category', 'price']
        ).astype({'item_id': 'int64'}),
        'orders': pd.DataFrame(
            rows['orders'],
            columns=['order_id', 'user_id', 'item_id', 'ts', 'quantity'],
        ).astype(
            {
                'order_id': 'Int64',
                'user_id': 'int64',
                'item_id': 'Int64',
                'quantity': 'int64',
            }
        ),
    }
    frames['orders']['ts'] = pd.to_datetime(frames['orders']['ts'])
    for name, frame in frames.items():
        connection.register('frame', frame)
        connection.execute(f'CREATE TABLE {name} AS SELECT * FROM frame')
        connection.unregister('frame')
    return connection


def _window(table: str, value: str, entity: str, anchor: str) -> str:
    r"""A window's rows after its start and not after its end, as the
    core's labels read them (``sdm_engine::labels``)."""
    return (
        f'SELECT {entity}, {value} FROM {table} WHERE ts > TIMESTAMP '
        f"'{anchor}' AND ts <= TIMESTAMP '{anchor}' + INTERVAL 30 DAY"
    )


def _sum_labels(connection: Any, sql: str, keys_sql: str) -> dict[Any, Any]:
    totals = dict(connection.execute(sql).fetchall())
    return {
        key: float(totals.get(key) or 0)
        for (key,) in connection.execute(keys_sql).fetchall()
    }


def generated_tasks() -> list[Task]:
    users = 'SELECT user_id FROM users'

    def regression(connection: Any) -> dict[Any, Any]:
        return _sum_labels(
            connection,
            f'SELECT user_id, SUM(quantity) FROM ({_window("orders", "quantity", "user_id", ANCHOR)}) GROUP BY user_id',
            users,
        )

    def binary(connection: Any) -> dict[Any, Any]:
        counts = regression(connection)
        active = dict(
            connection.execute(
                f'SELECT user_id, COUNT(*) FROM ({_window("orders", "1", "user_id", ANCHOR)}) GROUP BY user_id'
            ).fetchall()
        )
        return {key: bool(active.get(key, 0) > 0) for key in counts}

    def multiclass(connection: Any) -> dict[Any, Any]:
        return dict(
            connection.execute('SELECT user_id, segment FROM users').fetchall()
        )

    return [
        Task(
            'regression',
            'regression',
            'PREDICT SUM(orders.quantity, 0, 30, days) FOR EACH users.user_id',
            ANCHOR,
            regression,
        ),
        Task(
            'binary',
            'binary',
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
            ANCHOR,
            binary,
        ),
        Task(
            'multiclass',
            'multiclass',
            'PREDICT users.segment FOR EACH users.user_id',
            None,
            multiclass,
        ),
    ]


def adventureworks() -> Dataset:
    run_python = _runner()
    case = json.loads((CASES / 'base.adventureworks.json').read_text())
    given = case['input']
    details = 'adventureworks_sales_order_details'

    def regression(connection: Any) -> dict[Any, Any]:
        window = (
            f'SELECT ProductID, OrderQty FROM {details} WHERE OrderDate > '
            f"TIMESTAMP '{AW_ANCHOR}' AND OrderDate <= TIMESTAMP '{AW_ANCHOR}' "
            '+ INTERVAL 30 DAY'
        )
        return _sum_labels(
            connection,
            f'SELECT ProductID, SUM(OrderQty) FROM ({window}) GROUP BY ProductID',
            'SELECT ProductID FROM adventureworks_products',
        )

    return Dataset(
        'adventureworks',
        lambda: run_python._connection(case, run_python.load_bases()),
        given['from_source'],
        given['edits'],
        [
            Task(
                'regression',
                'regression',
                given['pql'],
                AW_ANCHOR,
                regression,
                {
                    'inference_config': copy.deepcopy(
                        given['materialize']['inference_config']
                    )
                },
            )
        ],
        nulls=False,
    )


#: The generated dataset's keys (its time column, ``orders.ts``, is
#: inferred, as the shop base's is), given so a key holding
#: nulls is still the table's key on both routes, whose key check refuses a
#: key with nulls unless told not to check (``check_keys=False``, as the
#: client's own P1B-13 tests build their sources).
GENERATED_SCHEMA = {
    'primary_keys': {
        'users': 'user_id',
        'items': 'item_id',
        'orders': 'order_id',
    },
    'check_keys': False,
}


def datasets() -> list[Dataset]:
    return [
        Dataset(
            'generated-nulls',
            lambda: generated_connection(True),
            GENERATED_SCHEMA,
            [],
            generated_tasks(),
            nulls=True,
        ),
        Dataset(
            'generated-control',
            lambda: generated_connection(False),
            GENERATED_SCHEMA,
            [],
            generated_tasks(),
            nulls=False,
            control_of='generated-nulls',
        ),
        adventureworks(),
    ]


DATASETS = ('generated-nulls', 'generated-control', 'adventureworks')


def entities(task: Task, labels: dict[Any, Any], count: int) -> list[Any]:
    r"""The entities a task asks about, fixed by the generator seed and the
    labels, never by a route: for a binary task half of each class, for a
    multiclass one each class in turn, for a regression half with a nonzero
    label where there are enough, the rest drawn from all."""
    import numpy as np

    rng = np.random.default_rng(GENERATOR_SEED + len(task.name))
    keys = sorted(labels, key=str)
    if task.kind == 'regression':
        nonzero = [k for k in keys if labels[k]]
        zero = [k for k in keys if not labels[k]]
        take = min(count // 2, len(nonzero))
        chosen = list(rng.choice(nonzero, take, replace=False)) if take else []
        chosen += list(rng.choice(zero, count - take, replace=False))
        return sorted((_plain(k) for k in chosen), key=str)
    groups: dict[Any, list[Any]] = {}
    for key in keys:
        groups.setdefault(labels[key], []).append(key)
    order = sorted(groups, key=str)
    pools = {g: list(rng.permutation(groups[g])) for g in order}
    chosen = []
    while len(chosen) < count and any(pools.values()):
        for group in order:
            if pools[group] and len(chosen) < count:
                chosen.append(_plain(pools[group].pop()))
    return sorted(chosen, key=str)


def plan(
    sets: list[Dataset], seeds: Sequence[int], count: int
) -> tuple[int, int]:
    r"""The calls and predictions the run makes: one call per dataset,
    task, seed and route, each asking about ``count`` entities."""
    calls = sum(len(d.tasks) for d in sets) * len(seeds) * 2
    return calls, calls * count


# --- one call -----------------------------------------------------------------


def masked_digest(body: Any) -> str:
    r"""The SHA-256 of a request with its request id left out (E35)."""
    body = copy.deepcopy(body)
    if isinstance(body, dict) and isinstance(body.get('metadata'), dict):
        body['metadata'].pop('request_id', None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()


def build(
    dataset: Dataset, task: Task, route: str, seed: int, keys: list[Any]
) -> Any:
    r"""The task's request on ``route``, materialized with ``seed``."""
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema, _routing
    from sdm_connectors import DuckDBSource

    run_python = _runner()
    if route == 'core':
        os.environ[_routing.ENVIRONMENT] = ','.join(core_routes())
    else:
        os.environ.pop(_routing.ENVIRONMENT, None)
    connection = dataset.connect()
    schema = Schema.from_source(
        DuckDBSource(connection),
        **run_python._from_source_arguments(dataset.from_source),
    )
    run_python._edit(schema, dataset.edits)
    arguments: dict[str, Any] = {
        'indices': [int(k) for k in keys],
        'random_seed': seed,
        **copy.deepcopy(task.options),
    }
    if task.anchor is not None:
        arguments['anchor_time'] = pd.Timestamp(task.anchor)
    return PredictiveQuery(task.pql, schema).materialize(**arguments)


#: The most predictions a run may send: E73(a)'s "a few hundred"; a larger
#: --max-predictions is refused.
PREDICTION_CEILING = 500


class WireBudget(smoke.SmokeError):  # type: ignore[misc, name-defined]
    r"""A request the run may not send: a second one for one prediction (an
    SDK retry), or one past the budget. Not an ``OSError``, so the SDK's
    retry loop and the client's transport pass it straight out."""


class Wire:
    r"""Counts, and bounds, the prediction requests that reach the network
    (the protocol review's item 4): ``requests.Session.send`` is wrapped,
    for this script's process alone, so each request to a serving endpoint's
    ``invocations`` is counted where it is sent, whatever the SDK retries
    above it. Each prediction may send one; a second is refused before it is
    sent, as is any past ``limit``. Its URL is never kept or shown."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.sent = 0
        self._allowed = 0
        self._original: Any = None

    def install(self) -> None:
        import requests

        original = requests.Session.send
        wire = self

        def send(session: Any, request: Any, **kwargs: Any) -> Any:
            if wire.counts(getattr(request, 'url', '')):
                if wire._allowed < 1:
                    raise WireBudget('a prediction would send a second request')
                if wire.sent >= wire.limit:
                    raise WireBudget('the run would pass its request budget')
                wire._allowed -= 1
                wire.sent += 1
            return original(session, request, **kwargs)

        self._original = original
        requests.Session.send = send  # type: ignore[method-assign, assignment]

    def uninstall(self) -> None:
        if self._original is not None:
            import requests

            requests.Session.send = self._original  # type: ignore[method-assign]
            self._original = None

    @staticmethod
    def counts(url: str) -> bool:
        from urllib.parse import urlparse

        path = urlparse(str(url)).path
        return '/serving-endpoints/' in path and path.endswith('/invocations')

    def allow_one(self) -> int:
        r"""Allows the next prediction one request; the count before it."""
        self._allowed = 1
        return self.sent


class DryTransport:
    r"""The dry run's endpoint: the scripted endpoint's rule, in process,
    with a probability for each class a classification asks for (the rule
    gives none), so the binary task's scoring path is read offline as a
    live answer is (the protocol review's item 9)."""

    serves_sessions = False
    url = 'in-process'

    def __init__(self) -> None:
        self._scripted = smoke._Recorded()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        answer: dict[str, Any] = self._scripted.predict(payload)
        target = (payload.get('task') or {}).get('target') or {}
        classes = target.get('classes')
        asked = (payload.get('output') or {}).get('fields') or []
        if isinstance(classes, list) and classes and 'probabilities' in asked:
            for item in answer.get('predictions', []):
                first = 0.2 + 0.6 * ((int(item['row_index']) * 7) % 5) / 4
                rest = (1 - first) / max(1, len(classes) - 1)
                item['probabilities'] = {
                    str(c): (first if at == len(classes) - 1 else rest)
                    for at, c in enumerate(classes)
                }
        return answer

    def close(self) -> None:
        pass


class Recording:
    r"""Wraps a client's transport so each request's digest is kept (never
    the request): which route's request reached the endpoint."""

    def __init__(self, transport: Any) -> None:
        self._transport = transport
        self.digests: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transport, name)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.digests.append(masked_digest(payload))
        answer: dict[str, Any] = self._transport.predict(payload)
        return answer


def outputs_of(task: Task) -> list[str] | None:
    return (
        None if task.kind == 'regression' else ['prediction', 'probabilities']
    )


def read(
    frame: Any, task: Task, dry: bool = False
) -> dict[Any, dict[str, Any]]:
    r"""Each entity's answer: its prediction, and for a binary task the
    positive class's probability (a column named for the class ``true``,
    whatever its case); a binary answer with none is refused, never scored
    by its hard label, in a dry run too, whose transport answers with
    probabilities (``dry`` is kept for the callers that pass it)."""
    found: dict[Any, dict[str, Any]] = {}
    positive = [
        c for c in frame.columns if str(c).lower() == 'probability_true'
    ]
    for _, row in frame.iterrows():
        answer: dict[str, Any] = {'prediction': _plain(row['prediction'])}
        if task.kind == 'binary':
            if len(positive) == 1:
                answer['score'] = float(row[positive[0]])
            else:
                raise smoke.SmokeError(
                    'a binary answer holds no probability of its positive class'
                )
        _scorable(task, answer)
        found[_plain(row['id'])] = answer
    return found


def _scorable(task: Task, answer: dict[str, Any]) -> None:
    r"""Refuses an answer no metric can score (fix round 1, Codex P1): a
    regression's prediction that is not a finite number (the client passes
    a non-finite one through, as the reference does), a binary score that
    is not a finite probability, a class that is not text, so the run
    fails loudly before any metric is computed."""
    prediction = answer['prediction']
    if task.kind == 'regression' and not _number(prediction):
        raise smoke.SmokeError('a prediction is not a finite number')
    if task.kind == 'binary' and not (
        _number(answer['score']) and 0 <= answer['score'] <= 1
    ):
        raise smoke.SmokeError('a score is not a finite probability')
    if task.kind == 'multiclass' and not isinstance(prediction, str):
        raise smoke.SmokeError('a predicted class is not text')


def _plain(value: Any) -> Any:
    item = getattr(value, 'item', None)
    return item() if callable(item) else value


# --- the metrics --------------------------------------------------------------


def mae(truth: list[float], predicted: list[float]) -> float:
    return sum(abs(t - p) for t, p in zip(truth, predicted, strict=True)) / len(
        truth
    )


def auroc(truth: list[bool], scores: list[float]) -> float | None:
    r"""The area under the ROC curve, by ranks (ties share their mean rank);
    None when the entities hold one class."""
    positives = sum(1 for t in truth if t)
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    at = 0
    while at < len(order):
        end = at
        while (
            end + 1 < len(order) and scores[order[end + 1]] == scores[order[at]]
        ):
            end += 1
        for i in range(at, end + 1):
            ranks[order[i]] = (at + end) / 2 + 1
        at = end + 1
    total = sum(r for r, t in zip(ranks, truth, strict=True) if t)
    return (total - positives * (positives + 1) / 2) / (positives * negatives)


def macro_f1(truth: list[Any], predicted: list[Any]) -> float:
    r"""The mean F1 over the classes the labels or the predictions hold."""
    classes = sorted({*map(str, truth), *map(str, predicted)})
    scores = []
    for c in classes:
        tp = sum(
            1
            for t, p in zip(truth, predicted, strict=True)
            if str(t) == c and str(p) == c
        )
        fp = sum(
            1
            for t, p in zip(truth, predicted, strict=True)
            if str(t) != c and str(p) == c
        )
        fn = sum(
            1
            for t, p in zip(truth, predicted, strict=True)
            if str(t) == c and str(p) != c
        )
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def metric(
    kind: str, truth: list[Any], answers: list[dict[str, Any]]
) -> float | None:
    r"""The task's metric, or None where it is undefined: AUROC over one
    class, or a value that is not a finite number (fix round 1, Codex P1),
    which :func:`verdict` fails rather than compares."""
    if kind == 'regression':
        found: float | None = mae(
            [float(t) for t in truth], [float(a['prediction']) for a in answers]
        )
    elif kind == 'binary':
        found = auroc([bool(t) for t in truth], [a['score'] for a in answers])
    else:
        found = macro_f1(truth, [a['prediction'] for a in answers])
    return found if found is not None and math.isfinite(found) else None


def worse(kind: str, old: float, core: float) -> float:
    r"""How much worse the core is, relative to the old route: positive is
    worse, negative better. MAE is better lower, AUROC and macro-F1 higher.
    From a zero baseline (fix round 1, Codex P2) the direction decides: an
    MAE of 0 the core exceeds is infinitely worse; an AUROC or macro-F1 of
    0 the core exceeds is an improvement, counted as 100% better (-1.0),
    never as worse; equal is 0."""
    if old == 0:
        if core == old:
            return 0.0
        higher = core > old
        if kind == 'regression':
            return math.inf if higher else -1.0
        return -1.0 if higher else math.inf
    return (core - old) / old if kind == 'regression' else (old - core) / old


def bootstrap(
    kind: str,
    truth: list[Any],
    old: list[list[dict[str, Any]]],
    core: list[list[dict[str, Any]]],
) -> tuple[float, float] | None:
    r"""The 95% interval of ``core - old`` on the metric averaged over the
    seeds, resampling the entities with replacement, the same resample for
    every seed and both routes (paired). Resamples on which the metric is
    undefined (one class, for AUROC) are skipped; None when too few are
    left."""
    import numpy as np

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(truth)
    found = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, n, n)
        sample = [truth[i] for i in pick]
        diffs = []
        for o, c in zip(old, core, strict=True):
            a = metric(kind, sample, [o[i] for i in pick])
            b = metric(kind, sample, [c[i] for i in pick])
            if a is None or b is None:
                break
            diffs.append(b - a)
        else:
            found.append(sum(diffs) / len(diffs))
    if len(found) < BOOTSTRAP // 2:
        return None
    found.sort()
    return found[int(0.025 * len(found))], found[int(0.975 * len(found)) - 1]


# --- the run ------------------------------------------------------------------


def verdict(units: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    r"""Whether the benchmark passes, and every reason it does not."""
    reasons = []
    for unit in units:
        where = f'{unit["dataset"]}/{unit["task"]}'
        # An undefined or non-finite metric, or a relative worsening that
        # is no number, fails: NaN compares false with anything (fix round
        # 1, Codex P1).
        if not all(
            _number(unit[key]) for key in ('old', 'core')
        ) or not _number(unit['worse'], infinite=True):
            reasons.append(f'{where}: the metric is undefined')
            continue
        if unit['worse'] > TOLERANCE:
            reasons.append(f'{where}: the core is {unit["worse"]:.2%} worse')
        if unit['control'] and False in (unit.get('one_request') or []):
            # The control isolates P1B-13 only where both routes send one
            # request (the protocol review's item 6).
            reasons.append(
                f"{where}: the control's two routes sent different requests"
            )
        if unit['control']:
            interval = unit['interval']
            if interval is None or not interval[0] <= 0 <= interval[1]:
                reasons.append(
                    f'{where}: the control does not agree within noise'
                )
    return not reasons, reasons


def json_safe(value: Any) -> Any:
    r"""``value`` with each non-finite number written as text (``inf``,
    ``-inf``, ``nan``), so the summary is JSON any reader reads."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _number(value: Any, infinite: bool = False) -> bool:
    r"""Whether ``value`` is a real number: finite, or with ``infinite`` any
    but NaN; never a boolean or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not math.isnan(value) and (infinite or math.isfinite(value))


def run(
    sets: list[Dataset],
    seeds: Sequence[int],
    count: int,
    client: Any,
    recording: Recording,
    raw: Any,
    dry: bool = False,
    wire: Wire | None = None,
) -> list[dict[str, Any]]:
    r"""Every unit's numbers. Each prediction is one request: the
    recording's digests grow by exactly one per prediction and, on a live
    run (``wire``), the requests counted where they are sent by exactly one
    (an empty answer sending none, or an SDK retry sending more, stops the
    run). The route sent first alternates from one seed to the next, so an
    endpoint warming or caching favours neither."""
    from sdm_client import KumoRelational

    units = []
    for dataset in sets:
        for task in dataset.tasks:
            labels = task.labels(dataset.connect())
            keys = entities(task, labels, count)
            truth = [labels[k] for k in keys]
            by_route: dict[str, list[list[dict[str, Any]]]] = {
                'old': [],
                'core': [],
            }
            digests: dict[str, list[str]] = {'old': [], 'core': []}
            for turn, seed in enumerate(seeds):
                for route in (
                    ('old', 'core') if turn % 2 == 0 else ('core', 'old')
                ):
                    request = build(dataset, task, route, seed, keys)
                    recorded = len(recording.digests)
                    before = wire.allow_one() if wire is not None else 0
                    result = client.predict(
                        request,
                        model=KumoRelational(),
                        outputs=outputs_of(task),
                    )
                    if len(recording.digests) != recorded + 1 or (
                        wire is not None and wire.sent != before + 1
                    ):
                        raise smoke.SmokeError(
                            'a prediction did not send exactly one request'
                        )
                    answers = read(result.to_pandas(), task, dry)
                    missing = [k for k in keys if k not in answers]
                    if missing:
                        raise smoke.SmokeError(
                            'an answer left out an entity asked about'
                        )
                    ordered = [answers[k] for k in keys]
                    by_route[route].append(ordered)
                    digests[route].append(recording.digests[-1])
                    for key, label, answer in zip(
                        keys, truth, ordered, strict=True
                    ):
                        raw.write(
                            json.dumps(
                                {
                                    'dataset': dataset.name,
                                    'task': task.name,
                                    'route': route,
                                    'seed': seed,
                                    'entity': key,
                                    'label': label,
                                    **answer,
                                },
                                default=str,
                            )
                            + '\n'
                        )
            per = {
                route: [metric(task.kind, truth, a) for a in by_route[route]]
                for route in by_route
            }
            mean = {
                route: (
                    None
                    if any(v is None for v in values)
                    else sum(values) / len(values)  # type: ignore[arg-type]
                )
                for route, values in per.items()
            }
            control = dataset.control_of is not None
            units.append(
                {
                    'dataset': dataset.name,
                    'task': task.name,
                    'metric': {
                        'regression': 'MAE',
                        'binary': 'AUROC',
                        'multiclass': 'macro-F1',
                    }[task.kind],
                    'entities': len(keys),
                    'seeds': list(seeds),
                    'nulls': dataset.nulls,
                    'control': control,
                    'old_per_seed': per['old'],
                    'core_per_seed': per['core'],
                    'old': mean['old'],
                    'core': mean['core'],
                    'worse': (
                        None
                        if mean['old'] is None or mean['core'] is None
                        else worse(task.kind, mean['old'], mean['core'])
                    ),
                    'one_request': [
                        o == c
                        for o, c in zip(
                            digests['old'], digests['core'], strict=True
                        )
                    ],
                    # The paired difference's interval, for every unit:
                    # it gates the controls (E73(a)); on the others it is
                    # reported only, so a marginal result can be judged.
                    'interval': bootstrap(
                        task.kind, truth, by_route['old'], by_route['core']
                    ),
                    'interval_gates': control,
                }
            )
            u = units[-1]
            say(
                f'{u["dataset"]}/{u["task"]}: {u["metric"]} old {_shown(u["old"])}, '
                f'core {_shown(u["core"])}, worse {_shown(u["worse"])}, '
                f'one request per seed {u["one_request"]}'
                + f', interval {u["interval"]}'
                + ('' if control else ' (reported only)')
            )
    return units


def _shown(value: Any) -> str:
    return 'undefined' if value is None else f'{float(value):.4f}'


def _parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = smoke._Parser(
        description="P1B-13's accuracy benchmark (E73(a))."
    )
    parser.add_argument(
        '--endpoint',
        help=f'the serving endpoint name; default ${ENDPOINT_VARIABLE}',
    )
    parser.add_argument(
        '--profile',
        help=f'the Databricks CLI profile; default ${PROFILE_VARIABLE}',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='build and answer every request offline',
    )
    parser.add_argument(
        '--raw-dir',
        type=Path,
        help="where each call's predictions and labels go",
    )
    parser.add_argument(
        '--summary', type=Path, help='where the numbers-only summary goes'
    )
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--entities', type=int, default=ENTITIES)
    parser.add_argument('--max-predictions', type=int, default=500)
    parser.add_argument('--timeout', type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        seeds = [int(s) for s in arguments.seeds.split(',') if s.strip()]
    except ValueError:
        smoke._usage('--seeds is a list of whole numbers')
    if not seeds or len(set(seeds)) != len(seeds):
        smoke._usage('--seeds names each seed once')
    if arguments.entities < 4 or arguments.timeout <= 0:
        smoke._usage('--entities is 4 or more and --timeout positive')
    if not 0 < arguments.max_predictions <= PREDICTION_CEILING:
        smoke._usage(f'--max-predictions is at most {PREDICTION_CEILING}')
    environ = dict(os.environ)
    endpoint = arguments.endpoint or environ.get(ENDPOINT_VARIABLE) or None
    profile = arguments.profile or environ.get(PROFILE_VARIABLE) or None
    redact.know(endpoint, profile)
    smoke.know_environment(environ)
    for name, value in environ.items():
        if name.startswith('SDM_BENCHMARK_'):
            redact.know_host(value)
    smoke._quiet_the_world()
    if not arguments.dry_run and (
        endpoint is None
        or profile is None
        or arguments.raw_dir is None
        or arguments.summary is None
    ):
        say(
            'a live run needs --endpoint and --profile (or their variables), '
            '--raw-dir and --summary',
            file=sys.stderr,
        )
        return 2
    sets = datasets()
    calls, predictions = plan(sets, seeds, arguments.entities)
    say(f'plan: {calls} calls, {predictions} predictions, seeds {seeds}')
    if predictions > arguments.max_predictions:
        say(
            f'the plan exceeds --max-predictions ({arguments.max_predictions})',
            file=sys.stderr,
        )
        return 2
    wire: Wire | None = None
    try:
        if arguments.dry_run:
            smoke._no_network()
            from sdm_client import SDMClient

            transport = DryTransport()
            client = SDMClient.__new__(SDMClient)
            recording = Recording(transport)
            client._start(recording)
            raw = open(os.devnull, 'w', encoding='utf-8')  # noqa: SIM115
        else:
            smoke.know_profile(environ, profile)
            client = smoke._client(endpoint, profile, arguments.timeout)
            wire = Wire(calls)
            wire.install()
            recording = Recording(client._transport)
            client._transport = recording
            arguments.raw_dir.mkdir(parents=True, exist_ok=True)
            raw = (arguments.raw_dir / 'calls.jsonl').open(
                'w', encoding='utf-8'
            )
        with raw:
            say(
                ('dry run: ' if arguments.dry_run else 'live: ')
                + 'sending each request on both routes'
            )
            units = run(
                sets,
                seeds,
                arguments.entities,
                client,
                recording,
                raw,
                arguments.dry_run,
                wire,
            )
        if not arguments.dry_run:
            client.close()
    except BaseException as failed:  # noqa: BLE001 - reported, redacted
        if isinstance(failed, KeyboardInterrupt):
            say('interrupted', file=sys.stderr)
            return 1
        say(f'benchmark failed: {smoke._diagnosis(failed)}', file=sys.stderr)
        return 1
    finally:
        if wire is not None:
            wire.uninstall()
    # A dry run's answers are the scripted rule's, which say nothing of a
    # model: it gives no verdict.
    passed, reasons = (None, []) if arguments.dry_run else verdict(units)
    summary = {
        'format': FORMAT,
        'protocol': 'decisions E73(a)',
        'dry_run': arguments.dry_run,
        'tolerance': TOLERANCE,
        'bootstrap': {
            'resamples': BOOTSTRAP,
            'seed': BOOTSTRAP_SEED,
            'level': 0.95,
        },
        'generator': {
            'seed': GENERATOR_SEED,
            'users': USERS,
            'items': ITEMS,
            'null_keys': NULL_KEYS,
            'null_foreign_keys': NULL_FOREIGN_KEYS,
            'anchor': ANCHOR,
            'adventureworks_anchor': AW_ANCHOR,
        },
        'calls': calls,
        'predictions': predictions,
        # The prediction requests counted where they were sent (none on a
        # dry run, which sends nothing), and the predictions they carried.
        'requests_sent': None if wire is None else wire.sent,
        'predictions_sent': (
            None if wire is None else wire.sent * arguments.entities
        ),
        'units': units,
        'passed': passed,
        'reasons': reasons,
    }
    text = (
        json.dumps(json_safe(summary), indent=1, default=str, allow_nan=False)
        + '\n'
    )
    if arguments.summary is not None:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary.write_text(redact(text), encoding='utf-8')
    for reason in reasons:
        say(f'benchmark failed: {reason}', file=sys.stderr)
    if passed is None:
        say('dry run done: the scripted answers give no verdict')
        return 0
    say('benchmark passed' if passed else 'benchmark failed')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
