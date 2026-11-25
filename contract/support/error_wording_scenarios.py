#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error wording scenarios: independent inputs and current-client test helpers."""

from __future__ import annotations

import importlib.util
import re
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / 'contract/cases/s1'
DATABASES = ('duckdb', 'sqlite', 'postgres')
#: own cases.
OWN = (
    'wording.anchor-after-data',
    'wording.binary-one-class',
    'wording.float32-classification',
    'wording.float32-regression',
    'wording.key-missing-table',
    'wording.key-none',
    'wording.key-number',
    'wording.rank-text-candidate',
    'wording.time-key',
    'wording.time-missing-column',
    'wording.time-number',
)
#: Committed S1 cases the core route refuses as the reference does, in its
#: code's generic sentence where the reference words the refusal, measured
#: on the base (15668f86), or whose warnings it words otherwise (M4).
COMMITTED = (
    'date.day-first-then-packed',
    'decimal.compare-finer-eq',
    'decimal.compare-past-precision',
    'decimal.compare-past-precision-gt',
    'edit.key-missing-column',
    'edit.key-time-column',
    'edit.leave-out-the-key',
    'edit.link-not-the-key',
    'edit.meaning-not-carried',
    'edit.meaning-unknown',
    'edit.time-column-not-a-time',
    'empty.indices',
    'key.uint64-above-int64-max',
    'rank.entities-by-candidates.at',
    'rule.composite-key-given',
    'sample.anchor-entity.context-time',
    'sample.anchor-entity.no-time-column',
    'sample.anchor-entity.temporal',
    'sample.context-anchor-time.entity',
    'sample.context-anchor-time.too-early',
    'sample.exclude-cols.key',
    'sample.exclude-cols.list',
    'sample.exclude-cols.text-columns',
    'sample.exclude-cols.unknown-column',
    'sample.exclude-cols.unknown-table',
    'sample.max-pq-iterations.text',
    'sample.max-pq-iterations.zero',
    'sample.num-hops.bool',
    'sample.num-hops.fraction',
    'sample.num-hops.text',
    'sample.num-hops.zero',
    'sample.num-neighbors.empty',
    'sample.num-neighbors.number',
    'sample.seed.negative',
    'shape.star.customers',
    'target.filter.sum',
)
CASE_IDS = (*OWN, *COMMITTED)
#: FX-4's probes, by name: the statements after the shop base, a table
#: ``events`` whose ``f`` is null in the 1,000 rows the preview reads and
#: holds the infinity in the 1,500 after, read in UTC.
PROBES = {
    f'infinity.{kind.lower()}.{sign}': [
        "SET GLOBAL TimeZone = 'UTC'",
        'CREATE TABLE events AS SELECT i::BIGINT AS event_id, CASE WHEN '
        f"i < 1000 THEN NULL ELSE {kind} '{value}' END AS f "
        'FROM range(2500) t(i)',
    ]
    for kind in ('TIMESTAMPTZ', 'TIMESTAMP')
    for sign, value in (('positive', 'infinity'), ('negative', '-infinity'))
}
#: The routes a from-source case forces on DuckDB, which the probes take
#: on the core.
FROM_SOURCE_ROUTES = (
    'schema.from-source/all-tables/duckdb',
    'schema.from-source/catalog-keys/duckdb',
)
#: The code a core warning's or refusal's text is prefixed with on
#: python-core, left out so a warning is held by its words.
CODE_PREFIX = re.compile(r'^\[[A-Z][A-Z0-9_]*\] ')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def options() -> ModuleType:
    r"""``query_option_scenarios.py``, whose runner, endpoint and JSON helpers this
    recorder shares."""
    return _module(
        'query_option_scenarios',
        ROOT / 'contract/support/query_option_scenarios.py',
    )


def said(shown: list[Any]) -> list[list[str]]:
    r"""Every warning shown, as its class name and text (a core code's
    prefix left out), once each, in the order first given."""
    found: list[list[str]] = []
    for warning in shown:
        entry = [
            warning.category.__name__,
            CODE_PREFIX.sub('', str(warning.message)),
        ]
        if entry not in found:
            found.append(entry)
    return found


def observe(case: dict[str, Any], database: str) -> dict[str, Any]:
    """Exercise current-client public calls against a scripted endpoint or source."""
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

    helpers = options()
    run = helpers.runner()
    given = case['input']
    arguments = run._materialize_arguments(given['materialize'])
    from_source = run._from_source_arguments(given.get('from_source', {}))
    observed: dict[str, Any] = {
        'outcome': None,
        'message': None,
        'details': None,
        'warnings': [],
    }
    connection, cleanup, _ = run._open(case, run.load_bases(), database)
    stage = 'from_source'
    try:
        with (
            helpers.endpoint().serve(
                fail_call=given.get('fail_call'),
                probabilities=given.get('probabilities', False),
                quantiles=given.get('quantiles', False),
                explanation=given.get('explanation', False),
                item_metadata=given.get('item_metadata', False),
            ) as (url, _received),
            warnings.catch_warnings(record=True) as shown,
        ):
            warnings.simplefilter('always')
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
                        client.predict(request, model=KumoRelational())
                observed['outcome'] = run._outcome('success', stage)
            except Exception as error:  # noqa: BLE001 - the outcome
                observed['outcome'] = run._outcome('refusal', stage, error)
                details = getattr(error, 'details', None)
                details = details if isinstance(details, dict) else {}
                observed['details'] = {
                    str(name): helpers.as_json(value)
                    for name, value in sorted(details.items())
                }
                observed['message'] = str(error)
            observed['warnings'] = said(shown)
    finally:
        cleanup()
    return observed


def probe(name: str) -> dict[str, Any]:
    """Exercise current-client public calls against a scripted endpoint or source."""
    from sdm_client import Schema

    run = options().runner()
    case = {'id': name, 'input': {'base': 'shop', 'setup': PROBES[name]}}
    observed: dict[str, Any] = {'outcome': None, 'warnings': []}
    connection, cleanup, _ = run._open(case, run.load_bases(), 'duckdb')
    try:
        with warnings.catch_warnings(record=True) as shown:
            warnings.simplefilter('always')
            try:
                Schema.from_source(run._source('duckdb', connection))
                observed['outcome'] = run._outcome('success', 'from_source')
            except Exception as error:  # noqa: BLE001 - the outcome
                observed['outcome'] = run._outcome(
                    'refusal', 'from_source', error
                )
            observed['warnings'] = said(shown)
    finally:
        cleanup()
    return observed
