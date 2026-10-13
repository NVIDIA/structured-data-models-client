#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.32e part 1's real-data parity benchmark: a sanity check, not a gate.

On demand and local only, like ``scripts/live/null_link_benchmark.py``,
whose pieces it reuses (the wire budget and retry refusal, the dry run's
in-process endpoint, the metrics, the paired bootstrap) and
``scripts/live/rfm_smoke.py``'s redaction: it never runs in CI, gates
nothing in CI, and no locked check reads it. It asks the same questions of
real data, never generated, on the old route and on the core route
(``SDM_CORE_ROUTES``), with paired seeds and the same endpoint, model,
entities, anchor time and options, and says for each dataset and task
whether the core answers as well as the old route.

- **Data** (:data:`DATASETS`), read from local files under ``--data-dir``
  and loaded into a DuckDB database in a temporary directory; none is
  committed, downloaded or written anywhere else:

  - ``adventureworks-full``: Microsoft's AdventureWorks sample at full
    size, every row of the four files ``contract/fixtures/adventureworks``
    subsets (``DIR/adventureworks/``: ``Customer.csv``, ``Product.csv``,
    ``SalesOrderHeader.csv``, ``SalesOrderDetail.csv``, each checked
    against the SHA-256 ``build.py`` pins), loaded as ``build.py`` and the
    quickstart load them (its ``frames``: nullable dtypes, every ``*Date``
    a timestamp, each order line given its order's ``CustomerID`` and
    ``OrderDate``) with their natural nulls (``SalesPersonID``,
    ``ProductSubcategoryID``, ``Color`` among them), and described as
    ``base.adventureworks`` describes the subset (its tables, keys, time
    columns, relationships and edits). ``build.py`` fetches the files
    (about 23 MB) when run once with ``--source DIR/adventureworks`` and no
    ``--write``; this script never downloads.
  - ``rel-hm``: RelBench's H&M (``DIR/rel-hm/``: ``article.parquet``,
    ``customer.parquet``, ``transactions.parquet``), a local sample or the
    full tables, described by inference alone (keys, time column and links
    by name; ``transactions`` has no key, and is read by foreign key as the
    reference reads a keyless table).

- **Tasks** (:data:`TASKS`), fixed before any data is read, each with its
  labels computed in DuckDB from the rows past the anchor time as the
  core's labels read them (a window's rows after its start and not after
  its end), or a static target's own value: on AdventureWorks the
  quickstart's regression (``SUM(OrderQty)`` over 30 days per product), a
  binary (will a product sell within 90 days) and a multiclass (a
  product's ``Color``); on H&M a regression (``SUM(price)`` over 30 days
  per customer), a binary (will a customer buy within 30 days) and a
  multiclass (an article's ``index_group_name``).
- **Comparison**: for each dataset, task and seed, the request built on the
  old route and on the core route, each ``materialize`` given the same
  ``random_seed`` (:data:`SEEDS` by default) and entities
  (``null_link_benchmark.entities``), sent to the same endpoint; the route
  sent first alternates from seed to seed.
- **Metrics**: MAE for a regression, AUROC for a binary task (the positive
  class's probability as the score), macro-F1 for a multiclass task, each
  route's averaged over the seeds with a 95% bootstrap interval over the
  entities, and the paired difference ``core - old`` with its own.
- **Verdict** (:func:`judge`), per unit: INCONCLUSIVE under the sanity
  floor, when the old route's metric averaged over the seeds is near
  random (AUROC below :data:`AUROC_FLOOR`; macro-F1 at or below chance,
  ``1 / k`` for the ``k`` classes the labels hold; MAE no lower than
  predicting the labels' mean for every entity) or undefined, whatever the
  core answered; otherwise FAIL when the core is more than
  :data:`TOLERANCE` worse, relative, and the paired difference's interval
  lies wholly on the worse side, or its metric is undefined; PASS
  otherwise. The floor reads the old route's evidence first. A route that
  refuses a task is reported with the refusal's class, code and rule as an
  operational error, which fails the run, and its unit is INCONCLUSIVE:
  never an accuracy verdict.
- **Budget and safety**: calls and predictions are counted before any data
  is read (:func:`plan`), and a plan over ``--max-predictions`` (default
  and most :data:`PREDICTION_CEILING`, 500) is refused; the requests are
  counted again where they are sent (``null_link_benchmark.Wire``), so an
  SDK retry is refused before it is sent, as is any request past the plan.
  Nothing written names an endpoint, workspace, host, profile, token or
  request id: every line passes through ``rfm_smoke``'s redaction.

    scripts/live/real_data_benchmark.py --dry-run --data-dir DIR
    scripts/live/real_data_benchmark.py --data-dir DIR --endpoint NAME \
        --profile NAME --raw-dir DIR --summary FILE [--dataset NAME ...]
        [--seeds 0,1] [--entities 16] [--max-predictions 500]
        [--timeout SECONDS]

``--dry-run`` reads the data, describes each dataset on both routes (the
tables' rows, each route's keys and time columns, or its refusal), builds
every request on both routes and answers it in process by the scripted
endpoint's rule, with no network access (it opens no socket and reads no
Databricks configuration): its metrics say nothing of a model. A live run
writes each call's predictions beside their labels under ``--raw-dir``
(``calls.jsonl``), and the summary (``sdm-t132e-real-data-benchmark/1``)
to ``--summary``.

Exit status: 0 when every unit passes or is inconclusive and no route
refused (a dry run when it completes with no refusal), 1 when a unit fails,
a route refuses (an operational error, recorded apart from the verdicts) or
a call fails, 2 for bad usage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FORMAT = 'sdm-t132e-real-data-benchmark/1'
ENDPOINT_VARIABLE = 'SDM_BENCHMARK_ENDPOINT'
PROFILE_VARIABLE = 'SDM_BENCHMARK_PROFILE'


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _module(
    'null_link_benchmark', ROOT / 'scripts/live/null_link_benchmark.py'
)
smoke = base.smoke
say = smoke.say
redact = smoke.redact
Wire = base.Wire
WireBudget = base.WireBudget
#: The most predictions one run may send; a larger --max-predictions is
#: refused.
PREDICTION_CEILING: int = base.PREDICTION_CEILING

#: The datasets a run may name: real data only, never generated.
DATASETS = ('adventureworks-full', 'rel-hm')
#: The paired seeds each request is materialized with, on both routes.
SEEDS = (0, 1)
#: Entities asked about in each call.
ENTITIES = 16
#: The relative worsening over which a significantly worse core fails.
TOLERANCE = 0.05
#: Under this AUROC the old route is near random.
AUROC_FLOOR = 0.6
#: The bootstrap of each unit's intervals: resamples and seed.
BOOTSTRAP = base.BOOTSTRAP
BOOTSTRAP_SEED = base.BOOTSTRAP_SEED
#: AdventureWorks' anchor time, inside its order dates (2022-06 to 2025-06).
AW_ANCHOR = '2025-03-01'
#: H&M's anchor time, inside its transactions (to 2020-09-22).
HM_ANCHOR = '2020-08-15'
#: Where ``build.py`` and the AdventureWorks case live.
BUILD = ROOT / 'contract/fixtures/adventureworks/build.py'
AW_CASE = ROOT / 'contract/cases/s1/base.adventureworks.json'
#: Each dataset's files under ``--data-dir``.
HM_FILES = ('article', 'customer', 'transactions')
AW_FILES = (
    'Customer.csv',
    'Product.csv',
    'SalesOrderHeader.csv',
    'SalesOrderDetail.csv',
)


# --- the tasks ------------------------------------------------------------


def _window(
    table: str, time_column: str, columns: str, anchor: str, days: int
) -> str:
    r"""A window's rows after its start and not after its end, as the core's
    labels read them (``sdm_engine::labels``)."""
    return (
        f'SELECT {columns} FROM {table} WHERE "{time_column}" > TIMESTAMP '
        f"'{anchor}' AND \"{time_column}\" <= TIMESTAMP '{anchor}' + "
        f'INTERVAL {days} DAY'
    )


def _sums(
    connection: Any, window: str, entity: str, value: str, keys: str
) -> dict[Any, Any]:
    totals = dict(
        connection.execute(
            f'SELECT "{entity}", SUM({value}) FROM ({window}) GROUP BY 1'
        ).fetchall()
    )
    return {
        key: float(totals.get(key) or 0)
        for (key,) in connection.execute(keys).fetchall()
    }


def _active(
    connection: Any, window: str, entity: str, keys: str
) -> dict[Any, Any]:
    seen = {
        key
        for (key,) in connection.execute(
            f'SELECT DISTINCT "{entity}" FROM ({window})'
        ).fetchall()
    }
    return {key: key in seen for (key,) in connection.execute(keys).fetchall()}


def _classes(connection: Any, sql: str) -> dict[Any, Any]:
    return {
        key: str(value)
        for key, value in connection.execute(sql).fetchall()
        if value is not None
    }


def aw_tasks() -> list[Any]:
    case = json.loads(AW_CASE.read_text(encoding='utf-8'))
    given = case['input']
    details = 'adventureworks_sales_order_details'
    products = 'SELECT "ProductID" FROM adventureworks_products'
    return [
        base.Task(
            'regression',
            'regression',
            given['pql'],
            AW_ANCHOR,
            lambda c: _sums(
                c,
                _window(details, 'OrderDate', '*', AW_ANCHOR, 30),
                'ProductID',
                '"OrderQty"',
                products,
            ),
            {
                'inference_config': copy.deepcopy(
                    given['materialize']['inference_config']
                )
            },
        ),
        base.Task(
            'binary',
            'binary',
            'PREDICT COUNT(sales_order_details.*, 0, 90, days) > 0 '
            'FOR EACH products.ProductID',
            AW_ANCHOR,
            lambda c: _active(
                c,
                _window(details, 'OrderDate', '*', AW_ANCHOR, 90),
                'ProductID',
                products,
            ),
        ),
        base.Task(
            'multiclass',
            'multiclass',
            'PREDICT products.Color FOR EACH products.ProductID',
            None,
            lambda c: _classes(
                c, 'SELECT "ProductID", "Color" FROM adventureworks_products'
            ),
        ),
    ]


def hm_tasks() -> list[Any]:
    customers = 'SELECT customer_id FROM customer'
    return [
        base.Task(
            'regression',
            'regression',
            'PREDICT SUM(transactions.price, 0, 30, days) '
            'FOR EACH customer.customer_id',
            HM_ANCHOR,
            lambda c: _sums(
                c,
                _window('transactions', 't_dat', '*', HM_ANCHOR, 30),
                'customer_id',
                'price',
                customers,
            ),
        ),
        base.Task(
            'binary',
            'binary',
            'PREDICT COUNT(transactions.*, 0, 30, days) > 0 '
            'FOR EACH customer.customer_id',
            HM_ANCHOR,
            lambda c: _active(
                c,
                _window('transactions', 't_dat', '*', HM_ANCHOR, 30),
                'customer_id',
                customers,
            ),
        ),
        base.Task(
            'multiclass',
            'multiclass',
            'PREDICT article.index_group_name FOR EACH article.article_id',
            None,
            lambda c: _classes(
                c, 'SELECT article_id, index_group_name FROM article'
            ),
        ),
    ]


#: Each dataset's tasks, fixed before any data is read.
TASKS: dict[str, Callable[[], list[Any]]] = {
    'adventureworks-full': aw_tasks,
    'rel-hm': hm_tasks,
}


def plan(
    names: Sequence[str], seeds: Sequence[int], entities: int
) -> tuple[int, int]:
    r"""The calls and predictions a run makes, from the datasets' fixed
    tasks, reading no data: one call per dataset, task, seed and route,
    each asking about ``entities`` entities."""
    calls = sum(len(TASKS[name]()) for name in names) * len(seeds) * 2
    return calls, calls * entities


# --- the data -------------------------------------------------------------


class MissingData(Exception):
    r"""A dataset's file is missing or is not the one pinned."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_adventureworks(directory: Path, database: Any) -> None:
    r"""Every row of the four source files into ``database``, as
    ``build.py`` loads and types them, each table ``adventureworks_<name>``
    as ``base.adventureworks`` names them."""
    build = _module('adventureworks_build', BUILD)
    for name in AW_FILES:
        path = directory / name
        if not path.is_file():
            raise MissingData(
                f'adventureworks: {name} is missing; fetch it with '
                'contract/fixtures/adventureworks/build.py --source '
                f'{directory}'
            )
        expected = build.FILES[name][1]
        if _sha256(path) != expected:
            raise MissingData(f'adventureworks: {name} is not the pinned file')
    for table, frame in build.frames(directory).items():
        name = f'adventureworks_{table}'
        kinds = [build.duckdb_type(frame[c].dtype) for c in frame.columns]
        columns = ', '.join(
            '"{}" {}'.format(column.replace('"', '""'), kind)
            for column, kind in zip(frame.columns, kinds, strict=True)
        )
        database.execute(f'CREATE TABLE {name} ({columns})')
        database.register('frame', frame)
        database.execute(f'INSERT INTO {name} SELECT * FROM frame')
        database.unregister('frame')


def load_hm(directory: Path, database: Any) -> None:
    r"""RelBench H&M's three tables into ``database``, as the files hold
    them."""
    for name in HM_FILES:
        path = directory / f'{name}.parquet'
        if not path.is_file():
            raise MissingData(f'rel-hm: {name}.parquet is missing')
        database.execute(
            f'CREATE TABLE {name} AS SELECT * FROM read_parquet(?)',
            [str(path)],
        )


def dataset(name: str, data_dir: Path, scratch: Path) -> Any:
    r"""Dataset ``name`` as ``null_link_benchmark`` runs one: its rows
    loaded once into a DuckDB file under ``scratch``, each connection a
    fresh one to that file, read only."""
    import duckdb

    path = scratch / f'{name}.duckdb'
    database = duckdb.connect(str(path))
    try:
        if name == 'adventureworks-full':
            load_adventureworks(data_dir / 'adventureworks', database)
            given = json.loads(AW_CASE.read_text(encoding='utf-8'))['input']
            from_source, edits = given['from_source'], given['edits']
        else:
            load_hm(data_dir / 'rel-hm', database)
            from_source, edits = {}, []
    finally:
        database.close()
    return base.Dataset(
        name,
        lambda: duckdb.connect(str(path), read_only=True),
        from_source,
        edits,
        TASKS[name](),
        nulls=False,
    )


def rows_of(connection: Any) -> dict[str, int]:
    r"""Each table's rows, as DuckDB counts them."""
    tables = [
        name
        for (name,) in connection.execute(
            'SELECT table_name FROM duckdb_tables() ORDER BY table_name'
        ).fetchall()
    ]
    return {
        name: int(
            connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        )
        for name in tables
    }


# --- refusals -------------------------------------------------------------


def refusal(failed: BaseException) -> dict[str, Any]:
    r"""What a refusal is, from an allowlist: its class, its code, and the
    named fields that say which rule or request refused (no value, no
    text the client or an endpoint wrote)."""
    found: dict[str, Any] = {'error': type(failed).__name__}
    code = getattr(failed, 'code', None)
    if isinstance(code, str):
        found['code'] = code
    fields = getattr(failed, 'details', None)
    if isinstance(fields, dict):
        for key in ('operation', 'rule', 'request', 'field', 'check', 'reason'):
            value = fields.get(key)
            if isinstance(value, str):
                found[key] = value
    return found


def describe(dataset: Any, route: str) -> dict[str, Any]:
    r"""``dataset`` described on ``route``: each table's key and time
    column, or the refusal."""
    from sdm_client import Schema, _routing
    from sdm_connectors import DuckDBSource

    run_python = base._runner()
    if route == 'core':
        os.environ[_routing.ENVIRONMENT] = ','.join(base.core_routes())
    else:
        os.environ.pop(_routing.ENVIRONMENT, None)
    started = time.monotonic()
    try:
        schema = Schema.from_source(
            DuckDBSource(dataset.connect()),
            **run_python._from_source_arguments(dataset.from_source),
        )
        run_python._edit(schema, dataset.edits)
    except Exception as failed:  # noqa: BLE001 - reported by its class
        return {'refused': refusal(failed)}
    return {
        'seconds': round(time.monotonic() - started, 2),
        'tables': {
            name: {
                'primary_key': table.primary_key,
                'time_column': table.time_column,
            }
            for name, table in sorted(schema.tables.items())
        },
        'relationships': len(schema.relationships),
    }


# --- the verdict ----------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def floor(kind: str, truth: list[Any]) -> float:
    r"""The old route's metric at random: AUROC's :data:`AUROC_FLOOR`,
    macro-F1 at chance (``1 / k`` for the ``k`` classes ``truth`` holds),
    or the MAE of predicting ``truth``'s mean for every entity."""
    if kind == 'binary':
        return AUROC_FLOOR
    if kind == 'multiclass':
        return 1 / len({str(t) for t in truth})
    numbers = [float(t) for t in truth]
    mean = _mean(numbers)
    return _mean([abs(t - mean) for t in numbers])


def under_floor(kind: str, truth: list[Any], old: float) -> bool:
    r"""Whether the old route's metric is near random, so no verdict."""
    at = floor(kind, truth)
    if kind == 'binary':
        return old < at
    if kind == 'multiclass':
        return old <= at + 1e-9
    return old >= at - 1e-9


def _per_seed(
    kind: str, truth: list[Any], answers: list[list[dict[str, Any]]]
) -> float | None:
    found = [base.metric(kind, truth, a) for a in answers]
    if not found or any(v is None for v in found):
        return None
    return _mean(found)  # type: ignore[arg-type]


def judge(
    kind: str,
    truth: list[Any],
    old: list[list[dict[str, Any]]],
    core: list[list[dict[str, Any]]],
) -> str:
    r"""One unit's verdict, ``'PASS'``, ``'FAIL'`` or ``'INCONCLUSIVE'``:
    inconclusive under the sanity floor (:func:`under_floor`, on the old
    route's metric averaged over the seeds) or where the old route's metric
    is undefined or its answers are not one list per seed aligned with
    ``truth``, whatever the core answered; otherwise a failure where the
    core did not answer every seed the old route did about every entity,
    or its metric is undefined, or it is more than :data:`TOLERANCE` worse,
    relative, with the paired difference's 95% bootstrap interval wholly
    on the worse side (or no interval to say otherwise); a pass
    otherwise."""

    def aligned(answers: list[list[dict[str, Any]]]) -> bool:
        return bool(answers) and all(len(a) == len(truth) for a in answers)

    old_mean = _per_seed(kind, truth, old) if aligned(old) else None
    if old_mean is None or under_floor(kind, truth, old_mean):
        return 'INCONCLUSIVE'
    # Paired seeds: the core answered every seed the old route did, each
    # about every entity, or it fails.
    if not aligned(core) or len(core) != len(old):
        return 'FAIL'
    core_mean = _per_seed(kind, truth, core)
    if core_mean is None:
        return 'FAIL'
    if base.worse(kind, old_mean, core_mean) <= TOLERANCE:
        return 'PASS'
    interval = base.bootstrap(kind, truth, old, core)
    if interval is None:
        return 'FAIL'
    low, high = interval
    worse_side = high < 0 if kind != 'regression' else low > 0
    return 'FAIL' if worse_side else 'PASS'


def interval(
    kind: str, truth: list[Any], answers: list[list[dict[str, Any]]]
) -> tuple[float, float] | None:
    r"""The 95% bootstrap interval of a route's metric averaged over the
    seeds, resampling the entities with replacement (the same resamples as
    the paired difference's); None when too few resamples are defined."""
    import numpy as np

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(truth)
    found = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, n, n)
        sample = [truth[i] for i in pick]
        values = [
            base.metric(kind, sample, [a[i] for i in pick]) for a in answers
        ]
        if any(v is None for v in values):
            continue
        found.append(_mean(values))  # type: ignore[arg-type]
    if len(found) < BOOTSTRAP // 2:
        return None
    found.sort()
    return found[int(0.025 * len(found))], found[int(0.975 * len(found)) - 1]


# --- the run --------------------------------------------------------------


def run(
    sets: list[Any],
    seeds: Sequence[int],
    count: int,
    client: Any,
    recording: Any,
    raw: Any,
    wire: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    r"""Each dataset's description on both routes, and every unit's
    numbers and verdict. Each prediction is one request, counted as
    ``null_link_benchmark.run`` counts it; a route that refuses to build a
    request is recorded with its refusal, and its unit is not scored."""
    from sdm_client import KumoRelational

    described = []
    units = []
    for dataset in sets:
        connection = dataset.connect()
        rows = rows_of(connection)
        routes = {route: describe(dataset, route) for route in ('old', 'core')}
        described.append(
            {'dataset': dataset.name, 'rows': rows, 'routes': routes}
        )
        say(f'{dataset.name}: rows {rows}')
        for route, found in routes.items():
            say(f'{dataset.name}: {route} route describes {found}')
        for task in dataset.tasks:
            labels = task.labels(connection)
            keys = base.entities(task, labels, count)
            truth = [labels[k] for k in keys]
            answers: dict[str, list[list[dict[str, Any]]]] = {
                'old': [],
                'core': [],
            }
            refused: dict[str, dict[str, Any]] = {}
            digests: dict[str, list[str]] = {'old': [], 'core': []}
            for turn, seed in enumerate(seeds):
                for route in (
                    ('old', 'core') if turn % 2 == 0 else ('core', 'old')
                ):
                    if route in refused:
                        continue
                    try:
                        request = base.build(dataset, task, route, seed, keys)
                    except Exception as failed:  # noqa: BLE001 - recorded
                        refused[route] = refusal(failed)
                        continue
                    recorded = len(recording.digests)
                    before = wire.allow_one() if wire is not None else 0
                    result = client.predict(
                        request,
                        model=KumoRelational(),
                        outputs=base.outputs_of(task),
                    )
                    if len(recording.digests) != recorded + 1 or (
                        wire is not None and wire.sent != before + 1
                    ):
                        raise smoke.SmokeError(
                            'a prediction did not send exactly one request'
                        )
                    read = base.read(result.to_pandas(), task)
                    if any(k not in read for k in keys):
                        raise smoke.SmokeError(
                            'an answer left out an entity asked about'
                        )
                    ordered = [read[k] for k in keys]
                    answers[route].append(ordered)
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
            units.append(
                unit(dataset.name, task, keys, truth, answers, refused, digests)
            )
            u = units[-1]
            say(
                f'{u["dataset"]}/{u["task"]}: {u["metric"]} old '
                f'{_shown(u["old"])} {u["old_interval"]}, core '
                f'{_shown(u["core"])} {u["core_interval"]}, difference '
                f'{u["difference_interval"]}: {u["verdict"]}'
                + (f' (refused: {u["refused"]})' if u['refused'] else '')
            )
    return described, units


def unit(
    dataset: str,
    task: Any,
    keys: list[Any],
    truth: list[Any],
    answers: dict[str, list[list[dict[str, Any]]]],
    refused: dict[str, dict[str, Any]],
    digests: dict[str, list[str]],
) -> dict[str, Any]:
    r"""One unit's numbers and verdict. The verdict is of quality alone:
    the sanity floor reads the old route's evidence first, so a unit whose
    old route is near random (or refused, or undefined) is INCONCLUSIVE
    whatever the core did; a unit a route refused is never judged on
    accuracy either. A refusal is an operational failure, ``operational``
    ``'FAIL'`` beside the verdict (``'OK'`` with none), its reasons in
    ``operational_errors``, which fails the run as an error (:func:`main`),
    never as an accuracy verdict."""
    kind = task.kind
    old, core = answers['old'], answers['core']
    scored = not refused
    old_mean = (
        _per_seed(kind, truth, old)
        if old and all(len(a) == len(truth) for a in old)
        else None
    )
    near_random = old_mean is None or under_floor(kind, truth, old_mean)
    if near_random or not scored:
        verdict = 'INCONCLUSIVE'
    else:
        verdict = judge(kind, truth, old, core)
    errors = [
        f'the {route} route refused: {why}' for route, why in refused.items()
    ]
    return {
        'dataset': dataset,
        'task': task.name,
        'kind': kind,
        'metric': {
            'regression': 'MAE',
            'binary': 'AUROC',
            'multiclass': 'macro-F1',
        }[kind],
        'entities': len(keys),
        'classes': len({str(t) for t in truth})
        if kind != 'regression'
        else None,
        'floor': floor(kind, truth),
        'old_per_seed': [base.metric(kind, truth, a) for a in old],
        'core_per_seed': [base.metric(kind, truth, a) for a in core],
        'old': _per_seed(kind, truth, old) if old else None,
        'core': _per_seed(kind, truth, core) if core else None,
        'old_interval': interval(kind, truth, old) if old else None,
        'core_interval': interval(kind, truth, core) if core else None,
        'difference_interval': (
            base.bootstrap(kind, truth, old, core) if scored else None
        ),
        'one_request': [
            o == c
            for o, c in zip(digests['old'], digests['core'], strict=False)
        ],
        'refused': refused,
        'near_random': near_random,
        'operational': 'FAIL' if errors else 'OK',
        'operational_errors': errors,
        'verdict': verdict,
    }


def _shown(value: Any) -> str:
    return 'undefined' if value is None else f'{float(value):.4f}'


def _redacted(value: Any) -> Any:
    r"""``value`` with every text in it, keys included, redacted."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {_redacted(k): _redacted(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted(item) for item in value]
    return value


def summary_text(summary: dict[str, Any]) -> str:
    r"""The summary as written to ``--summary``: JSON, each text in it
    redacted before it is written, then every line passed through
    ``redact``, so no host, endpoint or profile is written and the text
    stays JSON."""
    text = json.dumps(
        base.json_safe(_redacted(summary)),
        indent=1,
        default=str,
        allow_nan=False,
    )
    return '\n'.join(redact(line) for line in text.split('\n')) + '\n'


# --- the command line -----------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = smoke._Parser(
        description="T1.32e's real-data parity benchmark."
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
        '--dataset',
        action='append',
        help=f'a dataset to run, repeatable; default every one of {DATASETS}',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='where the local files are: adventureworks/ and rel-hm/',
    )
    parser.add_argument(
        '--raw-dir', type=Path, help="where each call's predictions go"
    )
    parser.add_argument('--summary', type=Path, help='where the summary goes')
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--entities', type=int, default=ENTITIES)
    parser.add_argument(
        '--max-predictions', type=int, default=PREDICTION_CEILING
    )
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
        smoke._usage(
            f'--max-predictions is at least 1 and at most {PREDICTION_CEILING}'
        )
    names = arguments.dataset or list(DATASETS)
    for name in names:
        if name not in DATASETS:
            # A dataset's name is never a secret: named, so the caller sees
            # which was refused (a generated one is never real data).
            smoke._usage(
                f'--dataset {name} is not one of {", ".join(DATASETS)}'
            )
    if len(set(names)) != len(names):
        smoke._usage('--dataset names each dataset once')
    calls, predictions = plan(names, seeds, arguments.entities)
    say(f'plan: {calls} calls, {predictions} predictions, seeds {seeds}')
    if predictions > arguments.max_predictions:
        smoke._usage(
            f'the plan ({predictions} predictions) exceeds --max-predictions '
            f'({arguments.max_predictions})'
        )
    if arguments.data_dir is None:
        smoke._usage('--data-dir names where the local files are')
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
    wire: Any = None
    try:
        with tempfile.TemporaryDirectory(prefix='sdm-t132e-') as scratch:
            if arguments.dry_run:
                smoke._no_network()
            try:
                sets = [
                    dataset(name, arguments.data_dir, Path(scratch))
                    for name in names
                ]
            except MissingData as missing:
                say(f'benchmark failed: {missing}', file=sys.stderr)
                return 1
            if arguments.dry_run:
                from sdm_client import SDMClient

                client = SDMClient.__new__(SDMClient)
                recording = base.Recording(base.DryTransport())
                client._start(recording)
                raw = open(os.devnull, 'w', encoding='utf-8')  # noqa: SIM115
            else:
                smoke.know_profile(environ, profile)
                client = smoke._client(endpoint, profile, arguments.timeout)
                wire = Wire(calls)
                wire.install()
                recording = base.Recording(client._transport)
                client._transport = recording
                arguments.raw_dir.mkdir(parents=True, exist_ok=True)
                raw = (arguments.raw_dir / 'calls.jsonl').open(
                    'w', encoding='utf-8'
                )
            with raw:
                say(
                    ('dry run: ' if arguments.dry_run else 'live: ')
                    + 'each request on both routes'
                )
                described, units = run(
                    sets,
                    seeds,
                    arguments.entities,
                    client,
                    recording,
                    raw,
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
    failed_units = [u for u in units if u['verdict'] == 'FAIL']
    # A route's refusal fails the run as an error, apart from the verdicts.
    errors = [
        f'{u["dataset"]}/{u["task"]}: {why}'
        for u in units
        for why in u['operational_errors']
    ]
    summary = {
        'format': FORMAT,
        'brief': 'T1.32e part 1, scope 2: a sanity check, not a gate',
        'dry_run': arguments.dry_run,
        'tolerance': TOLERANCE,
        'auroc_floor': AUROC_FLOOR,
        'bootstrap': {
            'resamples': BOOTSTRAP,
            'seed': BOOTSTRAP_SEED,
            'level': 0.95,
        },
        'anchors': {'adventureworks-full': AW_ANCHOR, 'rel-hm': HM_ANCHOR},
        'calls': calls,
        'predictions': predictions,
        'requests_sent': None if wire is None else wire.sent,
        'datasets': described,
        'units': units,
        'errors': errors,
        'verdicts': {
            verdict: sum(u['verdict'] == verdict for u in units)
            for verdict in ('PASS', 'FAIL', 'INCONCLUSIVE')
        },
    }
    text = summary_text(summary)
    if arguments.summary is not None:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary.write_text(text, encoding='utf-8')
    for error in errors:
        say(f'benchmark error: {error}', file=sys.stderr)
    if arguments.dry_run:
        say(
            'dry run done: the scripted answers say nothing of a model; '
            f'verdicts {summary["verdicts"]}'
        )
        return 1 if errors else 0
    for u in failed_units:
        say(f'unit failed: {u["dataset"]}/{u["task"]}', file=sys.stderr)
    say(f'benchmark done: verdicts {summary["verdicts"]}')
    return 1 if failed_units or errors else 0


if __name__ == '__main__':
    sys.exit(main())
