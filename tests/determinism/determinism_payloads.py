# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the payloads the determinism checks compare, in a process of its own.

The checks run this file as a script, so that every run starts from a fresh
interpreter: string hashing, import order and any module-level state are new
each time, as they are for two users on two machines. Whether a module named
``torch`` can be imported is decided here, before anything that might sort is
imported.

    python determinism_payloads.py snapshot --torch fake --repeat 10 --out a.json
    python determinism_payloads.py pushdown --backend sqlite --workdir d --out b.json

``snapshot`` is every scenario of ``scripts/snapshot_wire_payloads.py`` plus
in-memory relational scenarios whose data has timestamp ties among a parent's
neighbors. ``pushdown`` is relational scenarios over tables left in SQLite or
DuckDB, sampled there with SQL. ``--repeat`` builds everything that many times
in this one process and exits with status 3 if any build differs from the
first.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SNAPSHOT_SCRIPT = REPO / 'scripts' / 'snapshot_wire_payloads.py'
FAKE_TORCH = HERE / 'fake_torch'

TORCH_MODES = ('ambient', 'fake', 'absent')
DIFFERENT_IN_ONE_PROCESS = 3


# --- torch -------------------------------------------------------------------


def set_torch(mode: str) -> str:
    r"""Make ``torch`` importable as the fake, not importable, or leave it.

    Returns where ``torch`` resolves now, so a check can prove the mode it
    asked for is the mode that ran.
    """
    if mode not in TORCH_MODES:
        raise ValueError(f'unknown torch mode {mode!r}')
    if 'torch' in sys.modules and sys.modules['torch'] is not None:
        raise RuntimeError('torch was imported before its mode was set')
    if mode == 'fake':
        sys.path.insert(0, str(FAKE_TORCH))
        torch = importlib.import_module('torch')
        if not getattr(torch, 'IS_DETERMINISM_FAKE', False):
            raise RuntimeError(f'expected the fake torch, got {torch!r}')
    elif mode == 'absent':
        # A `None` entry makes `import torch` raise ImportError and
        # `importlib.util.find_spec('torch')` return None, which is what an
        # environment without torch looks like to both ways of asking.
        sys.modules['torch'] = None  # type: ignore[assignment]
    return torch_origin()


def torch_origin() -> str:
    try:
        spec = importlib.util.find_spec('torch')
    except (ImportError, ValueError):
        return 'absent'
    if spec is None:
        return 'absent'
    return str(spec.origin)


# --- shared pieces -----------------------------------------------------------


def snapshot_module() -> ModuleType:
    r"""``scripts/snapshot_wire_payloads.py``, loaded from its path."""
    spec = importlib.util.spec_from_file_location(
        '_snapshot_wire_payloads', SNAPSHOT_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relational_available() -> bool:
    try:
        importlib.import_module('sdm_relational_engine')
    except ImportError:
        return False
    return True


def tie_frames(num_users: int = 8) -> tuple[Any, Any]:
    r"""Users whose orders arrive several at the same instant.

    Each user has five orders at each of twelve weekly timestamps, every order
    with its own price. A sampler that keeps fewer than five neighbors has to
    choose among orders that share a parent and a timestamp, and the price
    of the order it chose is in the payload.
    """
    import pandas as pd

    users = pd.DataFrame(
        {
            'user_id': range(num_users),
            'age': [30.0 + i for i in range(num_users)],
        }
    )
    rows = []
    for user in range(num_users):
        for week in range(12):
            for _ in range(5):
                rows.append(
                    {
                        'order_id': len(rows),
                        'user_id': user,
                        'price': 10.0 + len(rows),
                        'ordered_at': pd.Timestamp('2026-01-01')
                        + pd.Timedelta(weeks=week),
                    }
                )
    orders = pd.DataFrame(rows)
    per_instant = orders.groupby(['user_id', 'ordered_at']).size()
    assert per_instant.min() > 1, 'the fixture must hold timestamp ties'
    return users, orders


# (query, materialize options) per tie scenario. Fewer neighbors per hop than
# orders per instant, so ties decide what is sampled.
TIE_SCENARIOS: dict[str, tuple[str, dict[str, Any]]] = {
    'ties/relational-count-two-hops': (
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
        {'indices': [0, 1, 2], 'num_neighbors': [3, 3], 'random_seed': 11},
    ),
    'ties/relational-sum-one-hop': (
        'PREDICT SUM(orders.price, 0, 14, days) FOR EACH users.user_id',
        {'indices': [3, 4], 'num_neighbors': [4], 'random_seed': 5},
    ),
}


def _relational_payloads(
    snapshot: ModuleType, request: Any
) -> dict[str, list[dict[str, Any]]]:
    from sdm_client.adapters.relational import RelationalAdapter

    adapter = RelationalAdapter()
    return {
        label: [
            snapshot._normalize(payload)
            for payload in adapter.prediction_plan(request, outputs)
        ]
        for label, outputs in snapshot._output_sets(adapter, request)
    }


def tie_cases(snapshot: ModuleType) -> dict[str, list[dict[str, Any]]]:
    from sdm_client import PredictiveQuery, Schema

    users, orders = tie_frames()
    recorded: dict[str, list[dict[str, Any]]] = {}
    for name, (query, options) in TIE_SCENARIOS.items():
        schema = Schema.from_data({'users': users, 'orders': orders})
        request = PredictiveQuery(query, schema).materialize(**options)
        for label, payloads in _relational_payloads(snapshot, request).items():
            recorded[f'{name}{label}'] = payloads
    return recorded


def snapshot_payloads() -> dict[str, Any]:
    snapshot = snapshot_module()
    recorded: dict[str, Any] = dict(snapshot.collect())
    if relational_available():
        recorded.update(tie_cases(snapshot))
    return recorded


# --- push-down ---------------------------------------------------------------

PUSHDOWN_TABLES = {'users': 'app_users', 'orders': 'app_orders'}

PUSHDOWN_SCENARIOS: dict[str, tuple[str, dict[str, Any]]] = {
    'pushdown/sum-30-days': (
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        {'indices': [1, 2, 3], 'random_seed': 7},
    ),
    'pushdown/count-14-days-few-neighbors': (
        'PREDICT COUNT(orders.*, 0, 14, days) FOR EACH users.user_id',
        {'indices': [4, 5], 'num_neighbors': [3, 3], 'random_seed': 23},
    ),
}


def pushdown_frames() -> tuple[Any, Any]:
    r"""Forty users, so the context is drawn from more rows than it keeps."""
    users, orders = tie_frames(num_users=40)
    users['user_id'] = users['user_id'] + 1
    orders['user_id'] = orders['user_id'] + 1
    return users, orders


def _sqlite_source(workdir: Path) -> Any:
    from adbc_driver_sqlite import dbapi
    from sdm_connectors import SQLiteSource

    users, orders = pushdown_frames()
    connection = dbapi.connect(str(workdir / 'pushdown.sqlite'))
    with connection.cursor() as cursor:
        cursor.adbc_ingest('app_users', users, mode='create')
        cursor.adbc_ingest('app_orders', orders, mode='create')
    connection.commit()
    return SQLiteSource(connection)


def _duckdb_source(workdir: Path) -> Any:
    import duckdb
    from sdm_connectors import DuckDBSource

    users, orders = pushdown_frames()
    connection = duckdb.connect(str(workdir / 'pushdown.duckdb'))
    connection.register('users_frame', users)
    connection.register('orders_frame', orders)
    connection.execute('CREATE TABLE app_users AS SELECT * FROM users_frame')
    connection.execute('CREATE TABLE app_orders AS SELECT * FROM orders_frame')
    connection.unregister('users_frame')
    connection.unregister('orders_frame')
    return DuckDBSource(connection)


def pushdown_payloads(backend: str, workdir: Path) -> dict[str, Any]:
    from sdm_client import PredictiveQuery, Schema

    snapshot = snapshot_module()
    workdir.mkdir(parents=True, exist_ok=True)
    if backend == 'sqlite':
        source = _sqlite_source(workdir)
    elif backend == 'duckdb':
        source = _duckdb_source(workdir)
    else:
        raise ValueError(f'unknown push-down backend {backend!r}')
    if source.backend != backend:
        raise RuntimeError(f'built a {source.backend} source for {backend}')
    schema = Schema.from_source(source, PUSHDOWN_TABLES)
    recorded: dict[str, Any] = {}
    for name, (query, options) in PUSHDOWN_SCENARIOS.items():
        request = PredictiveQuery(query, schema).materialize(**options)
        for label, payloads in _relational_payloads(snapshot, request).items():
            recorded[f'{name}{label}'] = payloads
    return recorded


# --- comparing -----------------------------------------------------------------


def first_difference(left: Any, right: Any, path: str = '') -> str | None:
    r"""Where two parsed payloads first differ, or None if they are equal."""
    if type(left) is not type(right):
        return f'{path or "<root>"}: {type(left).__name__} != {type(right).__name__}'
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                return f'{path}/{key}: present on one side only'
            found = first_difference(left[key], right[key], f'{path}/{key}')
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f'{path}: length {len(left)} != {len(right)}'
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            found = first_difference(a, b, f'{path}[{index}]')
            if found:
                return found
        return None
    if left != right:
        return f'{path}: {left!r} != {right!r}'
    return None


def describe_differences(
    left: dict[str, Any], right: dict[str, Any], limit: int = 10
) -> list[str]:
    lines = []
    for name in sorted(set(left) | set(right)):
        if name not in left or name not in right:
            lines.append(f'{name}: produced on one side only')
            continue
        found = first_difference(left[name], right[name])
        if found:
            lines.append(f'{name}: {found}')
    return lines[:limit]


def run_payloads(
    tmp_path: Path, name: str, *arguments: str, hash_seed: str = '0'
) -> dict[str, Any]:
    r"""Run this file in a fresh interpreter and return what it built."""
    out = tmp_path / f'{name}.json'
    environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *arguments]
        + ['--out', str(out)],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, (
        f'building payloads failed (exit {result.returncode}):\n'
        f'{result.stdout}\n{result.stderr}'
    )
    return json.loads(out.read_text())


# --- entry point ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('snapshot', 'pushdown'))
    parser.add_argument('--torch', choices=TORCH_MODES, default='ambient')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--backend', choices=('sqlite', 'duckdb'))
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)

    origin = set_torch(args.torch)
    warnings.simplefilter('ignore')

    runs = []
    for attempt in range(max(1, args.repeat)):
        if args.kind == 'snapshot':
            payloads = snapshot_payloads()
        else:
            if args.backend is None or args.workdir is None:
                parser.error('pushdown needs --backend and --workdir')
            payloads = pushdown_payloads(
                args.backend, args.workdir / f'run-{attempt}'
            )
        # Parsed-equal is the criterion, so compare what JSON gives back.
        runs.append(json.loads(json.dumps(payloads, default=str)))

    first = runs[0]
    for attempt, run in enumerate(runs[1:], start=2):
        differences = describe_differences(first, run)
        if differences:
            print(
                f'run {attempt} of {len(runs)} in one process differs from '
                f'run 1:\n  ' + '\n  '.join(differences),
                file=sys.stderr,
            )
            return DIFFERENT_IN_ONE_PROCESS

    args.out.write_text(
        json.dumps(
            {
                'torch': origin,
                'relational': relational_available(),
                'payloads': first,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
