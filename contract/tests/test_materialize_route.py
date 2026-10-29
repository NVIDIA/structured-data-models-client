# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""base.regression materialized on the core against the reference (T1.11c).

Run where the contract tests run (CI's ``engine`` job until T1.18):

    python -m pytest contract/tests -q

On ``python-core`` the call's ``materialize`` runs on the core, and from
T1.11d its predict too, which sends the core's call to the scripted
endpoint (``test_predict_route.py`` holds what predict does). Each claim
of materialize and what holds it:

- The payload reaches the endpoint as the reference's: python-core's
  public projection (outcome, warning classes, schema, every request the
  endpoint received, parsed with each number as its text, the result and
  the query-log counts) equals the committed reference observation's, with
  ``capture_check.projection``, whose one exclusion is the value of
  ``metadata.request_id`` (decision E35, approved by the owner on
  2026-10-10, ``contract/compare-policy.yaml``'s one exclusion; the field
  must be present alike with a value of the same type).
- The two core hosts agree: python-core's materialize inputs, step trace
  and request state, payload included, are the ones
  ``contract/pinned/base.regression.core.json`` pins, which
  ``sdm-testkit``'s ``tests/duckdb_provider.rs`` holds rust-host to.
- The core's SQL for materialize returns the reference's results (decision
  H9's count and results, before T1.11e's comparator): the same number of
  statements in the same steps; the index list, the time bounds and the
  reservoir sample text for text, as the reference writes them; and for each
  read against a key set, the key set is the reference's own rows (its
  digest, under the reference's column names, is the one the reference's
  capture recorded), and the result is the reference statement's on those
  rows: equal in order where the statement orders its rows totally, equal as
  a multiset where it does not. The texts of those reads differ, which H9
  leaves diagnostic.

The case is diagnostic until T1.29 (``compare.py`` reports, and gates
none of, its differences); these tests gate the claims above in CI.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pyarrow as pa
import pytest
from _actual_core import checked_operations, checked_view
from test_t44_legacy_pin_inputs import (
    assert_corrected_inputs,
    assert_pinned_fetch_shapes,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
CASE = ROOT / 'contract/cases/s1/base.regression.json'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))
PINNED = ROOT / 'contract/pinned/base.regression.core.json'


def _module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python')
capture_check = _module('capture_check')


def _without_plan_b_reads(observation: dict[str, Any]) -> dict[str, Any]:
    r"""``observation``'s projection (``capture_check.py``), its query log's
    statements counted without those the core marked as plan B's reads of
    a table as it stands (``compare.py``'s ``LOCATE``, the one rule,
    register P3A-1)."""
    path = ROOT / 'scripts/rust_core/compare.py'
    if 'compare' not in sys.modules:
        spec = importlib.util.spec_from_file_location('compare', path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules['compare'] = module
        spec.loader.exec_module(module)
    locate = sys.modules['compare'].LOCATE
    reads = sum(
        1
        for statement in observation['driver']['capture']['statements']
        if statement.get('purpose') == locate
    )
    projection = capture_check.projection(observation)
    logged = projection['query_log']
    return {
        **projection,
        'query_log': {**logged, 'statements': logged['statements'] - reads},
    }


def _reference() -> dict[str, Any]:
    for line in REFERENCE.read_text().splitlines():
        observation = json.loads(line)
        if observation['case'] == 'base.regression':
            return observation
    raise AssertionError('no reference observation of base.regression')


def _core(tmp_path: Path) -> dict[str, Any]:
    out = tmp_path / 'python-core.jsonl'
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNERS / 'run_python.py'),
            '--mode',
            'python-core',
            '--out',
            str(out),
            str(CASE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    [observation] = [json.loads(line) for line in out.read_text().splitlines()]
    return observation


def test_base_regression_reaches_the_reference_s_payload_on_the_core(
    tmp_path: Path,
) -> None:
    reference = _reference()
    core = _core(tmp_path)
    assert checked_operations(core) == [
        'describe_source',
        'materialize',
        'predict',
    ]
    assert core['core']['error'] is None
    logical = checked_view(core)
    materialized = logical['materialized']
    assert materialized['version'] == 2
    assert materialized['query'] == json.loads(CASE.read_text())['input']['pql']
    # Not vacuous: one request, of the payload's 112 examples and 2 to
    # predict, reached the endpoint, and the reference's carried a request
    # id of its own.
    [payload] = core['public']['payloads']
    assert len(payload['parsed']['context']['instance_table']['rows']) == 112
    assert len(payload['parsed']['predict']['instance_table']['rows']) == 2
    assert (
        'request_id' in reference['public']['payloads'][0]['parsed']['metadata']
    )
    assert _without_plan_b_reads(core) == capture_check.projection(reference)
    # The request the endpoint received is the core's payload, with what
    # predict adds: the model, the outputs, a request id, and only the
    # inference options the model reads.
    sent = json.loads(json.dumps(payload['parsed']))
    built = materialized['payload']
    for section in ('task', 'schema', 'context', 'predict'):
        assert sent[section] == built[section], section
    # The two core hosts agree on everything the core did and answered.
    pinned = json.loads(PINNED.read_text())
    assert_corrected_inputs(
        logical['inputs'], pinned['inputs'], materialized=materialized
    )
    before = copy.deepcopy(core)
    assert_pinned_fetch_shapes(
        logical['fetches'], pinned['fetches'], observation=core
    )
    assert core == before
    assert core['core']['described'] == pinned['described']
    assert materialized == pinned['materialized']


def test_historical_plan_comparison_checks_actual_binding_evidence(tmp_path):
    observation = _core(tmp_path)
    logical = checked_view(observation)
    pinned = json.loads(PINNED.read_text())['fetches']
    assert_pinned_fetch_shapes(
        logical['fetches'], pinned, observation=observation
    )
    if not any('bindings' in fetch for fetch in logical['fetches']):
        return  # The production build has no test-only binding observer.
    with pytest.raises(AssertionError, match='complete actual observation'):
        assert_pinned_fetch_shapes(logical['fetches'], pinned)
    position = next(
        i
        for i, fetch in enumerate(observation['core']['fetches'])
        if fetch.get('bindings', {}).get('owners')
    )
    for fault in (
        'run',
        'missing',
        'foreign',
        'schema',
        'digest',
        'unknown',
        'null',
    ):
        changed = copy.deepcopy(observation)
        fetch = changed['core']['fetches'][position]
        facts = fetch['bindings']
        if fault == 'run':
            facts['run'] += len(changed['identity']['operations'])
        elif fault == 'missing':
            facts['owners'] = []
        elif fault == 'foreign':
            facts['owners'][0]['attachment'] += 900_000
        elif fault == 'schema':
            facts['owners'][0]['fields'][0]['nullable'] = not facts['owners'][
                0
            ]['fields'][0]['nullable']
        elif fault == 'digest':
            facts['owners'][0]['values']['ordered'] = 'not a digest'
        elif fault == 'unknown':
            facts['unobserved'] = True
        else:
            fetch['bindings'] = None
        supplied = copy.deepcopy(logical['fetches'])
        if fault == 'null':
            # The older logical reader drops a null supplement. The helper
            # must still reject it in the actual raw observation.
            supplied[position].pop('bindings')
        else:
            supplied[position]['bindings'] = copy.deepcopy(fetch['bindings'])
        before = copy.deepcopy((changed, supplied))
        with pytest.raises(AssertionError):
            assert_pinned_fetch_shapes(supplied, pinned, observation=changed)
        assert (changed, supplied) == before, fault


# --- the core's SQL against the reference's ---------------------------------

#: The reference's names for the seeds' role-named columns.
_REFERENCE_NAMES = {
    'key': '__kumo_id__',
    'batch': '__kumo_batch__',
    'end': '__kumo_end__',
    'start': '__kumo_start__',
    'time': '__kumo_time__',
}


def _shop() -> Any:
    connection = duckdb.connect()
    for statement in run_python.load_bases()['shop']:
        connection.execute(statement)
    return connection


def _materialize_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[dict[str, Any], list[pa.Table], pa.Table]]:
    r"""base.regression's call on the core, in this process: each fetch the
    DuckDB provider answered, with the key sets it was handed and its
    answer."""
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema, _routing
    from sdm_client._rust.providers.duckdb import DuckDBProvider
    from sdm_connectors import DuckDBSource

    case = json.loads(CASE.read_text())
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(case['routes']))
    answer = DuckDBProvider.__call__
    recorded: list[tuple[dict[str, Any], list[pa.Table], pa.Table]] = []

    def recording(self: Any, step: dict[str, Any], keys: Any = None) -> Any:
        tables = {
            attachment: pa.RecordBatchReader._import_from_c_capsule(
                stream.__arrow_c_stream__()
            ).read_all()
            for attachment, stream in (keys or {}).items()
        }
        table = answer(self, step, tables)
        recorded.append((step, list(tables.values()), table))
        return table

    monkeypatch.setattr(DuckDBProvider, '__call__', recording)
    given = case['input']
    schema = Schema.from_source(DuckDBSource(_shop()))
    PredictiveQuery(given['pql'], schema).materialize(
        indices=given['materialize']['indices'],
        anchor_time=pd.Timestamp(given['materialize']['anchor_time']),
    )
    return recorded


def _as_reference(seeds: pa.Table, shape: list[dict[str, Any]]) -> pa.Table:
    r"""Seeds under the reference's column names, in its order, as the
    reference builds its key table (``pa.table`` of arrays)."""
    renamed = {
        _REFERENCE_NAMES[name]: seeds.column(name)
        for name in seeds.column_names
    }
    return pa.table({field['name']: renamed[field['name']] for field in shape})


def test_the_core_s_sql_for_materialize_returns_the_reference_s_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _reference()['driver']['capture']
    recorded = _materialize_recording(monkeypatch)
    # Ten fetches for the schema (test_schema_route.py), then the reads
    # the core marks as plan B's (purpose `locate`, the one rule, register
    # P3A-1), then materialize's six, as the reference's sixteen statements.
    fresh = sum(1 for step, _, _ in recorded if step.get('purpose') == 'locate')
    assert len(recorded) == 16 + fresh
    assert len(reference['statements']) == 16
    ours = [
        entry for entry in recorded[10:] if entry[0].get('purpose') != 'locate'
    ]
    theirs = reference['statements'][10:]
    bindings = iter(reference['bindings'])
    requests = [step['request']['request'] for step, _, _ in ours]
    assert requests == [
        'describe',
        'time_bounds',
        'sample_entities',
        'neighbors',
        'lookup',
        'neighbors',
    ]
    for (step, keys, answer), statement in zip(ours, theirs, strict=True):
        [plan] = step['plan']['statements']
        [result] = statement['results']
        assert answer.num_rows == result['rows'], step['request']
        if not keys:
            # Read without a key set: the reference's own text.
            assert plan['text'] == statement['sql'], step['request']
            continue
        [seeds] = keys
        binding = next(bindings)
        mirrored = _as_reference(seeds, binding['schema'])
        assert run_python._arrow_digest(mirrored) == binding['digest']
        # The reference statement, on its own key table of these rows.
        connection = _shop()
        cursor = connection.cursor()
        cursor.register(binding['name'], mirrored)
        expected = cursor.execute(statement['sql']).to_arrow_table()
        assert expected.schema.names == answer.schema.names
        assert expected.num_rows == answer.num_rows
        if 'ORDER BY tmp.__kumo_batch__' in statement['sql']:
            assert answer.to_pylist() == expected.to_pylist()
            assert run_python._arrow_digest(answer) == result['digest']
        else:
            key = lambda row: json.dumps(row, sort_keys=True, default=str)  # noqa: E731
            assert sorted(map(key, answer.to_pylist())) == sorted(
                map(key, expected.to_pylist())
            )
    assert next(bindings, None) is None
