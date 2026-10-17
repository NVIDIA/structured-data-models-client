# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1d: a Postgres row read's types in the Postgres provider's vocabulary.

The reference reads Postgres through psycopg, whose description names each
column by its Postgres type (``int8``, ``numeric(38,10)``, ``int4[]``).
``compare.py`` holds an Arrow read's schema to such a row read through the
Arrow types the Python host's Postgres provider builds each type as
(:func:`compare.postgres_arrow_name`): equal types pass, a retyped column
and a name outside the vocabulary fail.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')


def _preview() -> dict[str, Any]:
    r"""python-ref's preview of ``orders`` on Postgres, a row read."""
    for line in REFERENCE.read_text().splitlines():
        observation = json.loads(line)
        if observation['case'] == 'base.regression@postgres':
            for statement in observation['driver']['capture']['statements']:
                if statement['sql'].endswith(
                    'FROM "public"."orders" LIMIT 1000'
                ):
                    return statement
    raise AssertionError('no reference line holds the statement')


def _failures(ours: dict[str, Any], theirs: dict[str, Any]) -> list[str]:
    report = compare.Report({'sql.result': 'gate'})
    compare._results(
        report,
        'at',
        ours,
        theirs,
        False,
        ('rust-host', 'python-ref'),
        database='postgres',
    )
    return report.texts


def test_a_postgres_row_read_s_types_are_held_to_the_schema() -> None:
    statement = _preview()
    read = statement['results'][0]
    assert read['values']['types'] == ['int8', 'int8', 'float8', 'timestamp']
    arrow = dict(
        statement,
        results=[
            {
                'method': 'query_arrow',
                'rows': read['rows'],
                'schema': [
                    {'name': name, 'type': kind}
                    for name, kind in zip(
                        read['values']['columns'],
                        ['int64', 'int64', 'double', 'timestamp[us]'],
                        strict=True,
                    )
                ],
                'values': dict(read['values'], types=None),
            }
        ],
    )
    assert _failures(arrow, statement) == []
    retyped = copy.deepcopy(arrow)
    retyped['results'][0]['schema'][0]['type'] = 'int32'
    assert any(
        'differs from the types' in t for t in _failures(retyped, statement)
    )
    foreign = copy.deepcopy(statement)
    foreign['results'][0]['values']['types'][0] = 'BIGINT'
    assert any('no Arrow name' in t for t in _failures(arrow, foreign))


def test_the_vocabulary_reads_precision_arrays_lengths_and_composites() -> None:
    name = compare.postgres_arrow_name
    assert name('numeric(38,10)') == 'decimal128(38, 10)'
    assert name('numeric(40,2)') == 'decimal256(40, 2)'
    assert name('int4[]') == 'list<item: int32>'
    assert name('varchar(3)') == 'string'
    assert name('sdm_b133392301bc7b7d9ca02548cea099c5') == 'string'
    # A timestamptz is built in UTC; an interval kept as months, days and
    # nanoseconds; a name of no type is none.
    assert name('timestamptz') == 'timestamp[us, tz=UTC]'
    assert name('interval') == 'month_day_nano_interval'
    assert name('sdm_x') is None
