#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.32e part 1's old-against-core probe of the key rules (decision E49),
a local pre-push probe, not a CI gate: ``Schema.from_source``,
``materialize`` and ``predict`` over generated DuckDB sources whose keys
hold nulls or repeat, on the old route and on the core route of one
environment, compared by outcome, schema, warnings and the request sent.

    python -I contract/tools/key_rules_probe.py [--seed 33] [--n 50]

Run with the candidate's environment. Each source is ``accounts`` (40
rows) and ``payments`` (120 to 3,000 rows) linked by ``account_id``; one of
the two keys is an integer, text or a decimal, holds 0 to 12 nulls or 0 to
40 repeats (the payments' after the first 1,000 rows of a large table, or
anywhere), and a fifth name the payments' key in ``primary_keys``. The
question is ``base.adventureworks``'s shape: a 30-day sum per account. What
is compared, per source: the outcome (success, or the exception's class
and code, at the call that refused), both tables' keys and duplicate-key
counts, the links, the warnings' classes and bodies (the core's ``[CODE]``
prefix set aside), and the SHA-256 of the request sent to the in-process
scripted endpoint, its request id left out. It prints every difference
and exits 1 on any, or on no evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PQL = 'PREDICT SUM(payments.amount, 0, 30, days) FOR EACH accounts.account_id'
ANCHOR = '2024-01-01'


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def key(kind: str, value: str) -> str:
    return {
        'int': f'({value})::BIGINT',
        'text': f"'k' || ({value})::VARCHAR",
        'decimal': f'({value})::DECIMAL(18, 0)',
    }[kind]


def value(n_rows: int, nulls: int, repeats: int, start: int) -> str:
    r"""Row ``i``'s key: null for the last ``nulls`` rows, a repeat of the
    row before for the ``repeats`` before them, past ``start``."""
    nul = f'i >= {n_rows - nulls}' if nulls else 'FALSE'
    rep = (
        f'i >= {n_rows - nulls - repeats} AND i >= {start}'
        if repeats
        else 'FALSE'
    )
    return f'CASE WHEN {nul} THEN NULL WHEN {rep} THEN i - 1 ELSE i END'


def generated(seed: int, n: int) -> dict[str, dict[str, Any]]:
    rnd = random.Random(seed)
    out = {}
    for i in range(n):
        rows = rnd.choice([120, 200, 1000, 3000])
        which = rnd.choice(['payments', 'payments', 'accounts'])
        kind = rnd.choice(['int', 'int', 'text', 'decimal'])
        nulls = rnd.choice([0, 0, 1, 2, 12])
        repeats = rnd.choice([0, 0, 3, 40])
        late = rows > 1000 and rnd.random() < 0.5
        start = 1000 if late else 0

        account = key(
            kind if which == 'accounts' else 'int',
            value(40, nulls, repeats, start) if which == 'accounts' else 'i',
        )
        payment = key(
            kind if which == 'payments' else 'int',
            value(rows, nulls, repeats, start) if which == 'payments' else 'i',
        )
        account_ref = key(kind if which == 'accounts' else 'int', 'i % 40')
        named = which == 'payments' and rnd.random() < 0.2
        out[f'k{i:02d}'] = {
            'statements': [
                f'CREATE TABLE accounts AS SELECT {account} AS account_id, '
                "['north', 'south'][1 + i % 2] AS region FROM range(40) t(i)",
                f'CREATE TABLE payments AS SELECT {payment} AS payment_id, '
                f'{account_ref} AS account_id, TIMESTAMP '
                f"'2023-06-01' + to_minutes((i * {432000 // rows})::INTEGER) AS paid_at, "
                f'(i % 13)::DOUBLE AS amount FROM range({rows}) t(i)',
            ],
            'primary_keys': {'payments': 'payment_id'} if named else None,
            'indices': [3, 5]
            if kind == 'int' or which == 'payments'
            else ['k3', 'k5'],
        }
    return out


def probe(source: dict[str, Any], route: str) -> dict[str, Any]:
    import duckdb
    import pandas as pd
    from sdm_client import (
        KumoRelational,
        PredictiveQuery,
        Schema,
        SDMClient,
        _routing,
    )
    from sdm_connectors import DuckDBSource

    base = _module(
        'null_link_benchmark', ROOT / 'scripts/live/null_link_benchmark.py'
    )
    if route == 'core':
        os.environ[_routing.ENVIRONMENT] = ','.join(base.core_routes())
    else:
        os.environ.pop(_routing.ENVIRONMENT, None)
    connection = duckdb.connect()
    for statement in source['statements']:
        connection.execute(statement)
    got: dict[str, Any] = {'stage': 'from_source'}
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        try:
            arguments = (
                {'primary_keys': source['primary_keys']}
                if source['primary_keys']
                else {}
            )
            schema = Schema.from_source(DuckDBSource(connection), **arguments)
            got['tables'] = {
                name: [t.primary_key, t.duplicate_keys, t.time_column]
                for name, t in sorted(schema.tables.items())
            }
            got['links'] = sorted(
                [
                    r.source_table,
                    r.source_column,
                    r.target_table,
                    r.target_column,
                ]
                for r in schema.relationships
            )
            got['stage'] = 'materialize'
            request = PredictiveQuery(PQL, schema).materialize(
                indices=source['indices'],
                anchor_time=pd.Timestamp(ANCHOR),
                random_seed=0,
            )
            got['stage'] = 'predict'
            client = SDMClient.__new__(SDMClient)
            recording = base.Recording(base.DryTransport())
            client._start(recording)
            client.predict(request, model=KumoRelational())
            got['request'] = recording.digests[-1]
            got['outcome'] = 'success'
        except Exception as error:  # noqa: BLE001 - any is an answer
            details = getattr(error, 'details', None)
            got['outcome'] = [
                type(error).__name__,
                getattr(error, 'code', None),
                details.get('rule') if isinstance(details, dict) else None,
            ]
    # The sampler's warnings are the core's own words (E41(i)): compared
    # by class alone; every other by its body.
    got['warnings'] = [
        [w.category.__name__]
        if w.category.__name__ == 'UserWarning'
        else [w.category.__name__, re.sub(r'^\[[A-Z_]+\] ', '', str(w.message))]
        for w in seen
    ]
    return got


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--seed', type=int, default=33)
    parser.add_argument('--n', type=int, default=50)
    args = parser.parse_args()
    sources = generated(args.seed, args.n)
    counts: dict[str, int] = {}
    differ = 0
    for name, source in sources.items():
        old, core = probe(source, 'old'), probe(source, 'core')
        kind = 'agree' if old == core else 'DIFFER'
        if old != core and core['outcome'][2:] == ['wire_dtype']:
            # A key declined leaves a decimal column a feature, which the
            # core does not send yet (wire_dtype, the decimals family's,
            # T1.32b; register P1B-3).
            kind = 'known: wire_dtype'
        elif old != core:
            differ += 1
            print(f'{name}: {json.dumps(source)}\n  old {old}\n  core {core}')
        outcome = (
            'success'
            if old['outcome'] == 'success'
            else f'{old["stage"]} {old["outcome"][1]}'
        )
        label = f'{kind} ({outcome})'
        counts[label] = counts.get(label, 0) + 1
    print(json.dumps(counts, indent=1))
    return 1 if differ or not sources else 0


if __name__ == '__main__':
    sys.exit(main())
