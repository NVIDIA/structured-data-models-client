# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only P3F-2 parity, with unchanged comparator and reached-key checks."""

from __future__ import annotations

import copy
import functools
import importlib.util
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / 'docs/design/rust-core/register.yaml'


@functools.cache
def comparison():
    name = '_p3f2_existing_compare'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'scripts/rust_core/compare.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@functools.cache
def authority():
    """Intersect only P3F-2 fields with the existing owner's approved reader."""
    document = yaml.safe_load(REGISTER.read_text())
    [entry] = [e for e in document['entries'] if e['id'] == 'P3F-2']
    approved = comparison().more_exact(REGISTER)
    owner_approved = (
        entry['status'] == 'approved'
        and entry['approved_by'] == 'owner'
        and entry['decision_ref']['row'] in comparison()._owner_rows()
    )
    found = {}
    for item in entry['more_exact']:
        fields = set(item['fields'])
        for runner in ('python-core', 'rust-host'):
            if runner not in item['runners']:
                continue
            admitted = (
                tuple(
                    f
                    for f in approved.get((item['case'], runner), ())
                    if f in fields
                    or f.removeprefix(comparison()._NULL_SIDE) in fields
                )
                if owner_approved
                else ()
            )
            if not admitted:
                continue
            found[item['case'], runner] = admitted
    return found


def payloads(line):
    result = []
    for payload in line['public']['payloads']:
        parsed = copy.deepcopy(payload['parsed'])
        if isinstance(parsed, dict) and isinstance(
            parsed.get('metadata'), dict
        ):
            parsed['metadata'].pop('request_id', None)
        result.append([payload['method'], payload['path'], parsed])
    return result


def number(value):
    if isinstance(value, dict) and set(value) == {'number'}:
        return Decimal(value['number'])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Decimal(str(value))
    return value


def row_columns(body, fields, root):
    """Resolve only admitted scalar row paths through this actual table layout."""
    prefix = (
        ['public.payloads', '*', '2']
        if root == 'public.payloads'
        else ['core.predicted.body', '*']
    )
    columns = set()
    for field in fields:
        parts = field.removeprefix(comparison()._NULL_SIDE).split('/')
        if parts[: len(prefix)] != prefix:
            continue
        tail = parts[len(prefix) :]
        if not (
            len(tail) == 6
            and tail[0] in ('context', 'predict')
            and tail[1] == 'related_tables'
            and tail[3:5] == ['rows', '*']
            and tail[5].isdigit()
        ):
            continue
        table = body[tail[0]]['related_tables'].get(tail[2])
        if table is None:
            continue
        position = int(tail[5])
        if position >= len(table['columns']):
            return (
                'approved FK position is outside the actual row columns',
                set(),
            )
        columns.add((tail[2], table['columns'][position]))
    return None, columns


def _links(old, new, fields, root):
    problem, columns = row_columns(new, fields, root)
    if problem is not None:
        return problem, 0
    filled = 0
    for table, column in columns:
        relations = [
            r
            for r in new['schema']['relationships']
            if r.get('source_table') == table
            and r['source_columns'] == [column]
        ]
        if len(relations) != 1 or len(relations[0]['target_columns']) != 1:
            return 'approved FK has no unique single-column relationship', 0
        relation = relations[0]
        target, key = relation['target_table'], relation['target_columns'][0]
        values = []
        for half in ('context', 'predict'):
            sources = new[half]['related_tables'].get(table)
            if sources is None:
                continue
            targets = new[half]['related_tables'].get(
                target, {'columns': [], 'rows': []}
            )
            old_sources = old[half]['related_tables'][table]
            source_columns = sources['columns']
            position = source_columns.index(column)
            instance = source_columns.index('instance_id')
            target_columns = targets['columns']
            keys = Counter(
                (
                    number(row[target_columns.index('instance_id')]),
                    number(row[target_columns.index(key)]),
                )
                for row in targets['rows']
            )
            for before, row in zip(
                old_sources['rows'], sources['rows'], strict=True
            ):
                previous, value = before[position], row[position]
                values.append(value)
                if previous is not None and number(previous) != number(value):
                    return 'a previously valid FK changed or disappeared', 0
                if isinstance(value, bool):
                    return (
                        'FK key is Boolean rather than its original numeric type',
                        0,
                    )
                if value is not None:
                    if keys[number(row[instance]), number(value)] != 1:
                        return (
                            'FK does not name one reached target in the same instance',
                            0,
                        )
                    filled += previous is None
        old_dtype = old['schema']['related_tables'][table]['columns'][column][
            'dtype'
        ]
        new_dtype = new['schema']['related_tables'][table]['columns'][column][
            'dtype'
        ]
        if new_dtype != old_dtype and not (
            old_dtype == 'float64'
            and new_dtype == 'int64'
            and values
            and all(value is not None for value in values)
        ):
            return 'FK dtype changed beyond null-free integer restoration', 0
    return None, filled


def payload_problem(want: dict[str, Any], got: dict[str, Any], case_key: str):
    """Compare every request field; only actual approved FK improvements differ."""
    fields = authority().get((case_key, 'python-core'), ())
    old, new = payloads(want), payloads(got)
    difference = comparison()._path_of(
        comparison()._lossless(new),
        comparison()._lossless(old),
        'public.payloads',
        fields,
    )
    if difference is not None or not fields:
        return difference
    filled = 0
    for before, after in zip(old, new, strict=True):
        problem, count = _links(before[2], after[2], fields, 'public.payloads')
        if problem is not None:
            return problem
        filled += count
    if not filled:
        return 'approved affected case did not exercise any valid FK fill'
    return None


def rust_problems(document):
    """Only approved Rust cases delegate; unlisted ones retain their Rust check."""
    result = {}
    for case, line in document['observed'].items():
        admitted = authority().get((case, 'rust-host'))
        if admitted is None or case not in document['reference']:
            continue
        fields = admitted
        old = [p[2] for p in payloads(document['reference'][case])]
        body = line['core']['predicted'].get('body')
        new = [] if body is None else [copy.deepcopy(body)]
        for value in new:
            if isinstance(value.get('metadata'), dict):
                value['metadata'].pop('request_id', None)
        problem = comparison()._path_of(
            comparison()._lossless(new),
            comparison()._lossless(old),
            'core.predicted.body',
            fields,
        )
        filled = 0
        if problem is None:
            for before, after in zip(old, new, strict=True):
                problem, count = _links(
                    before, after, fields, 'core.predicted.body'
                )
                if problem is not None:
                    break
                filled += count
            if problem is None and not filled:
                problem = (
                    'approved affected case did not exercise any valid FK fill'
                )
        result[case] = problem
    return result


if __name__ == '__main__':
    import json

    if sys.argv[1:] != ['--rust-pairs']:
        raise SystemExit(
            'use --rust-pairs with observed/reference JSON on stdin'
        )
    json.dump(rust_problems(json.load(sys.stdin)), sys.stdout)
