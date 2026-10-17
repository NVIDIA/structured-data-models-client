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
    assert core['identity']['operations'] == [
        'describe_source',
        'materialize',
        'predict',
    ]
    assert core['core']['error'] is None
    materialized = core['core']['materialized']
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
    assert capture_check.projection(core) == capture_check.projection(reference)
    # The request the endpoint received is the core's payload, with what
    # predict adds: the model, the outputs, a request id, and only the
    # inference options the model reads.
    sent = json.loads(json.dumps(payload['parsed']))
    built = materialized['payload']
    for section in ('task', 'schema', 'context', 'predict'):
        assert sent[section] == built[section], section
    # The two core hosts agree on everything the core did and answered.
    pinned = json.loads(PINNED.read_text())
    for field in ('inputs', 'fetches', 'described', 'materialized'):
        assert core['core'][field] == pinned[field], field


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
    # Ten fetches for the schema (test_schema_route.py), then
    # materialize's six, as the reference's sixteen statements.
    assert len(recorded) == 16
    assert len(reference['statements']) == 16
    ours = recorded[10:]
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
