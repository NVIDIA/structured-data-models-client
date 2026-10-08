# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P75.capture: the boundary recordings exist for every corpus scenario, the
live ones and the synthetic ones included, and hold what today's Python
does.

- ``capture_sampling.py --out DIR`` writes exactly the committed
  ``tests/parity/sampling/``.
- Replaying every transcript through today's Python here, with the checks'
  own interceptor, gives each recorded call: the same method, ``describe``'s
  inputs and tables, the same statements with the same results, and the same
  outputs in canonical form. Every scenario over a graph makes both calls.
- An in-memory call is given the graph's tables themselves.
- The synthetic scenarios of task 1 are in the corpus, in memory and on
  DuckDB (``CONTRACT.md``, "Synthetic scenarios").
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from _driver import (
    CAPTURE,
    RECORDINGS,
    REPO,
    TRANSCRIPTS,
    Interceptor,
    canon_frame,
    capture_module,
    difference,
    python_outputs,
    read_arrow,
    read_recording,
    recorded_tables,
    replaying,
    scenario_dir,
    strict_json,
    transcripts,
)


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob('*'))
        if path.is_file() and '__pycache__' not in path.parts
    }


def run_capture(out: Path, hashseed: str | None = None) -> None:
    env = dict(os.environ)
    if hashseed is not None:
        env['PYTHONHASHSEED'] = hashseed
    proc = subprocess.run(
        [sys.executable, str(CAPTURE), '--out', str(out)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0, (
        f'capture_sampling.py --out failed ({proc.returncode}):\n'
        f'{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}'
    )


def compare_trees(want: dict[str, bytes], got: dict[str, bytes], what: str):
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    differ = sorted(k for k in set(want) & set(got) if want[k] != got[k])
    assert not (missing or extra or differ), (
        f'{what}: {len(missing)} files missing, such as {missing[:5]}; '
        f'{len(extra)} extra, such as {extra[:5]}; {len(differ)} differ, '
        f'such as {differ[:5]}'
    )


def test_capture_writes_the_committed_recordings() -> None:
    assert CAPTURE.is_file(), f'{CAPTURE} does not exist'
    assert RECORDINGS.is_dir(), f'{RECORDINGS} does not exist'
    with tempfile.TemporaryDirectory(prefix='sdm-p75-capture-') as tmp:
        run_capture(Path(tmp))
        compare_trees(
            tree(RECORDINGS),
            tree(Path(tmp)),
            'capture_sampling.py --out against tests/parity/sampling',
        )


# --- Today's Python, replayed here ------------------------------------------------


@cache
def replayed() -> dict[str, dict[str, Any]]:
    r"""Every transcript replayed under the checks' interceptor, with the
    implementer's ``describe``: by scenario, its backend and calls."""
    describe = capture_module().describe
    out: dict[str, dict[str, Any]] = {}
    with replaying(Interceptor(describe)) as replay:
        for path in transcripts():
            transcript = json.loads(path.read_text('utf-8'))
            reason, calls = replay(path)
            assert reason is None, (
                f'{transcript["name"]} does not replay with the interceptor '
                f'on: {reason}'
            )
            out[transcript['name']] = {
                'backend': transcript['backend'],
                'data': transcript['data'],
                'calls': calls,
            }
    return out


def test_every_scenario_has_its_recording() -> None:
    names = {json.loads(p.read_text('utf-8'))['name'] for p in transcripts()}
    assert len(names) >= 71, f'only {len(names)} transcripts'
    recorded = {}
    for path in sorted(RECORDINGS.rglob('boundary.json')):
        recorded[read_recording(path.parent)['scenario']] = path.parent
    missing = sorted(names - set(recorded))
    extra = sorted(set(recorded) - names)
    assert not missing and not extra, (
        f'recordings missing for {missing[:10]}; recordings without a '
        f'transcript: {extra[:10]}'
    )
    misplaced = [
        name
        for name, directory in recorded.items()
        if directory != scenario_dir(RECORDINGS, name)
    ]
    assert not misplaced, f'recordings in the wrong directory: {misplaced}'


def test_every_graph_scenario_makes_both_calls() -> None:
    found = replayed()
    lacking = [
        name
        for name, scenario in found.items()
        if scenario['backend'] != 'tabular'
        and {c['method'] for c in scenario['calls']}
        != {'sample_target', 'sample_subgraph'}
    ]
    assert not lacking, f'scenarios without both calls: {lacking}'
    live = [
        name
        for name, scenario in found.items()
        if scenario['backend'] in ('databricks', 'lakebase', 'snowflake')
        and scenario['calls']
    ]
    assert len(live) >= 24, f'only {len(live)} live scenarios make calls'


def _table_problem(want: Any, got: Any) -> str | None:
    if not want.schema.equals(got.schema, check_metadata=True):
        return f'schema {got.schema} instead of {want.schema}'
    return difference(canon_frame(want), canon_frame(got))


def _call_problem(
    directory: Path, recorded: dict[str, Any], call: dict[str, Any]
) -> str | None:
    if recorded['method'] != call['method']:
        return f'method {recorded["method"]} instead of {call["method"]}'
    inputs = strict_json(call['inputs_json'], 'inputs_json')
    if json.dumps(recorded['inputs']) != json.dumps(inputs):
        return 'inputs differ from what describe() gives'
    tables = recorded_tables(directory, recorded)
    if list(tables) != list(call['tables']):
        return f'tables {list(tables)} instead of {list(call["tables"])}'
    for name, table in call['tables'].items():
        found = _table_problem(table, tables[name])
        if found:
            return f'table {name}: {found}'
    if len(recorded['steps']) != len(call['steps']):
        return (
            f'{len(recorded["steps"])} statements recorded, '
            f'{len(call["steps"])} run'
        )
    for j, (r, s) in enumerate(
        zip(recorded['steps'], call['steps'], strict=True)
    ):
        if r['sql'] != s['sql']:
            return f'statement {j}: SQL differs'
        if json.dumps(r['params']) != json.dumps(s['params']):
            return f'statement {j}: parameters differ'
        if json.dumps(r['error']) != json.dumps(s['error']):
            return f'statement {j}: error {r["error"]} instead of {s["error"]}'
        if (r['result'] is None) != (s['result'] is None):
            return f'statement {j}: result present in only one'
        if s['result'] is not None:
            found = _table_problem(
                s['result'], read_arrow(directory / r['result'])
            )
            if found:
                return f'statement {j}: result {found}'
    return difference(call['outputs'], recorded['outputs'], '$.outputs')


def test_recordings_hold_what_todays_python_does() -> None:
    found = replayed()
    problems = []
    compared = 0
    for name, scenario in found.items():
        directory = scenario_dir(RECORDINGS, name)
        if not (directory / 'boundary.json').is_file():
            problems.append(f'{name}: no recording')
            continue
        recorded = read_recording(directory)
        if recorded['scenario'] != name:
            problems.append(
                f'{name}: the recording names {recorded["scenario"]}'
            )
            continue
        calls = scenario['calls']
        if len(recorded['calls']) != len(calls):
            problems.append(
                f'{name}: {len(recorded["calls"])} calls recorded, '
                f'{len(calls)} made'
            )
            continue
        for i, (r, c) in enumerate(zip(recorded['calls'], calls, strict=True)):
            found_problem = _call_problem(directory, r, c)
            if found_problem:
                problems.append(
                    f'{name} call {i} ({c["method"]}): {found_problem}'
                )
            else:
                compared += 1
    assert not problems, (
        f"{len(problems)} recorded calls differ from today's Python:\n"
        + '\n'.join(problems[:25])
    )
    assert compared >= 230, f'only {compared} calls compared'


def test_in_memory_calls_take_the_graph_tables() -> None:
    import pyarrow as pa

    problems = []
    seen = 0
    for name, scenario in replayed().items():
        for i, call in enumerate(scenario['calls']):
            if call['backend'] != 'local':
                continue
            seen += 1
            graph = call['graph']
            assert graph is not None, f'{name}: no graph seen for the sampler'
            for table_name, table in graph.tables.items():
                want = pa.Table.from_pandas(table._data, preserve_index=False)
                got = call['tables'].get(table_name)
                if got is None:
                    problems.append(f'{name} call {i}: no table {table_name}')
                    continue
                found = difference(canon_frame(want), canon_frame(got))
                if found:
                    problems.append(f'{name} call {i} {table_name}: {found}')
    assert seen >= 20, f'only {seen} in-memory calls'
    assert not problems, '\n'.join(problems[:20])


# --- The synthetic scenarios ---------------------------------------------------------


def _frames(scenario: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    r"""The scenario's data by table name: the graph's own frames in memory,
    the generator's frames on a database."""
    if call['backend'] == 'local':
        return {n: t._data for n, t in call['graph'].tables.items()}
    from _parity import data as datagen

    spec = scenario['data']
    return dict(datagen.generate(spec['generator'], spec['seed']))


def _fans_out(scenario: dict[str, Any], call: dict[str, Any]) -> bool:
    if call['method'] != 'sample_subgraph':
        return False
    sampler = call['sampler']
    hops = [n for n in call['arguments']['num_neighbors'] if n > 0]
    if not hops:
        return False
    frames = _frames(scenario, call)
    sampled = {t['name']: t for t in call['outputs'].get('tables', [])}
    for src, fkey, dst in [tuple(e) for e in call['graph'].edges]:
        if src in sampler.time_column_dict or dst not in sampled:
            continue
        if src not in frames or fkey not in frames[src]:
            continue
        key = sampler.primary_key_dict.get(dst)
        names = sampled[dst]['df']['names']
        if key not in names:
            continue
        keys = set(sampled[dst]['df']['columns'][names.index(key)]['values'])
        counts = frames[src][fkey].value_counts(dropna=True)
        fan = max((int(counts.get(k, 0)) for k in keys), default=0)
        if fan > min(hops):
            return True
    return False


def _draws(call: dict[str, Any]) -> bool:
    r"""Whether today's in-memory sampler gives other neighbors for the call
    under the next seed: ``relationallib`` drew random numbers."""
    arguments = dict(call['arguments'])
    seed = arguments.get('random_seed')
    arguments['random_seed'] = (0 if seed is None else seed) + 1
    other = call['sampler'].sample_subgraph(**arguments)
    again = python_outputs('sample_subgraph', other, None, [])
    again['warnings'] = call['outputs']['warnings']
    return difference(call['outputs'], again) is not None


def _float32s(part: dict[str, Any]) -> list[float]:
    target = part['target']
    if target['type'] != 'float':
        return []
    return [
        struct.unpack('>f', bytes.fromhex(v))[0]
        for v in target['values']
        if v != 'nan'
    ]


def _fractional(call: dict[str, Any]) -> bool:
    if call['method'] != 'sample_target' or 'error' in call['outputs']:
        return False
    values = _float32s(call['outputs']['train']) + _float32s(
        call['outputs']['test']
    )
    return any(abs(v) != float('inf') and v != int(v) for v in values)


def _temporal_tables(call: dict[str, Any]) -> bool:
    return len(call['sampler'].time_column_dict) >= 2


def _entity_ties(call: dict[str, Any]) -> bool:
    if call['method'] != 'sample_target' or 'error' in call['outputs']:
        return False
    arguments = call['arguments']
    if 'entity' not in (
        arguments.get('train_anchor_time'),
        arguments.get('test_anchor_time'),
    ):
        return False
    times = [
        t
        for part in ('train', 'test')
        for t in call['outputs'][part]['anchor_time']
        if t is not None
    ]
    return len(times) != len(set(times))


PROPERTIES = {
    'static fan-out above the hop count': None,
    'fractional amounts': _fractional,
    'two or more temporal tables': _temporal_tables,
    "anchor_time='entity' with ties": _entity_ties,
}


@pytest.mark.parametrize('backend', ['memory', 'duckdb'])
@pytest.mark.parametrize('what', list(PROPERTIES))
def test_synthetic_scenarios(backend: str, what: str) -> None:
    found = replayed()
    holding = []
    for name, scenario in found.items():
        if scenario['backend'] != backend:
            continue
        for call in scenario['calls']:
            if what == 'static fan-out above the hop count':
                ok = _fans_out(scenario, call) and (
                    backend != 'memory' or _draws(call)
                )
            else:
                ok = PROPERTIES[what](call)
            if ok:
                holding.append(name)
                break
    assert holding, (
        f'no {backend} scenario of the corpus has {what} '
        f'(tests/rust_core/sampling/driver/CONTRACT.md, "Synthetic scenarios")'
    )
    recorded = [
        name
        for name in holding
        if (scenario_dir(RECORDINGS, name) / 'boundary.json').is_file()
        and (scenario_dir(TRANSCRIPTS, name) / 'transcript.json').is_file()
    ]
    assert recorded, f'{holding} have no transcript or recording'
