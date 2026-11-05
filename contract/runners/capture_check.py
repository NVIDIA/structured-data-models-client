#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that driver capture is complete and does not change public results.

Compare current-client observations with capture enabled and disabled:
    python contract/runners/capture_check.py --on FILE --off FILE

Only the random request ID value is masked; presence and JSON type still match.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# Capture conservation masks only the random request ID value.
# Its presence and JSON type remain part of the comparison.
Paths = tuple[tuple[str, ...], ...]
RANDOM_PATHS: Paths = (('metadata', 'request_id'),)


def _json_type(value: Any) -> str:
    r"""A parsed value's JSON type; a number is ``{"number": text}``."""
    if isinstance(value, dict) and set(value) == {'number'}:
        return 'number'
    if isinstance(value, dict):
        return 'object'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, bool):
        return 'boolean'
    if value is None:
        return 'null'
    return 'string'


def masked(parsed: Any, paths: Paths | None = None) -> Any:
    r"""A parsed prediction payload with the value at each of ``paths``
    (by default:data:`RANDOM_PATHS`) replaced by its JSON type, as
    conservation compares it. The same rule holds the core's call (the payload it writes for the
    endpoint, ``core.predicted.body``) across the core runners and to its
    pin: one value, ``metadata.request_id``, under E35, approved by
    the owner on 2026-10-10."""
    parsed = copy.deepcopy(parsed)
    for path in RANDOM_PATHS if paths is None else paths:
        node = parsed
        for part in path[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        # Only the value: an absent field stays absent, and a present
        # one keeps its type, so a deleted or retyped field differs.
        if isinstance(node, dict) and path[-1] in node:
            node[path[-1]] = {
                'excluded value of type': _json_type(node[path[-1]])
            }
    return parsed


def _payloads(public: dict[str, Any]) -> list[Any]:
    return [
        (payload['method'], payload['path'], masked(payload['parsed']))
        for payload in public['payloads']
    ]


def projection(observation: dict[str, Any]) -> dict[str, Any]:
    r"""What the capture must not change of one observation."""
    public = observation['public']
    return {
        'outcome': observation['outcome'],
        'warnings': observation['host']['warnings'],
        'schema': public['schema'],
        'payloads': _payloads(public),
        'result': public['result'],
        'query_log': None
        if observation['driver']['query_log'] is None
        else {
            key: observation['driver']['query_log'][key]
            for key in ('statements', 'bindings')
        },
    }


def complete(observation: dict[str, Any]) -> list[str]:
    r"""Why a captured observation's capture is not complete, if it is
    not."""
    driver = observation['driver']
    capture, logged = driver['capture'], driver['query_log']
    if capture is None:
        return ['the capture was off']
    problems = [f'could not record: {what}' for what in capture['incomplete']]
    if logged is None:
        # Every line has its log: DuckDB's own, or off DuckDB the driver's
        # statement log; a capture held to none is incomplete.
        problems.append('an observation without its query log')
        return problems
    if len(capture['statements']) != logged['statements']:
        problems.append(
            f'{len(capture["statements"])} statements captured, '
            f'{logged["statements"]} in the query log'
        )
    if len(capture['bindings']) != logged['bindings']:
        problems.append(
            f'{len(capture["bindings"])} bindings captured, '
            f'{logged["bindings"]} in the query log'
        )
    if logged['differing']:
        problems.append(
            f'statements {logged["differing"]} ran as text other than the '
            'capture recorded'
        )
    return problems


def _first_difference(on: Any, off: Any, path: str) -> str | None:
    if isinstance(on, dict) and isinstance(off, dict):
        for key in sorted(set(on) | set(off)):
            if key not in on or key not in off:
                return f'{path}/{key}: present in one run only'
            found = _first_difference(on[key], off[key], f'{path}/{key}')
            if found:
                return found
        return None
    if isinstance(on, (list, tuple)) and isinstance(off, (list, tuple)):
        if len(on) != len(off):
            return f'{path}: {len(on)} items against {len(off)}'
        for index, (a, b) in enumerate(zip(on, off, strict=True)):
            found = _first_difference(a, b, f'{path}/{index}')
            if found:
                return found
        return None
    return None if on == off else f'{path}: differs'


def _cases(ids: list[str], name: str) -> list[str]:
    problems = []
    if not ids:
        problems.append(f'the {name} run holds no case')
    repeated = sorted({case for case in ids if ids.count(case) > 1})
    if repeated:
        problems.append(f'the {name} run holds {repeated} more than once')
    return problems


#: One captured observation and its capture-off twin (``None`` when the
#: capture-off run lacks the case).
Pair = tuple[dict[str, Any], dict[str, Any] | None]


def _check(
    on_ids: list[str],
    off_ids: list[str],
    required: tuple[str, ...] | list[str],
    pairs: Iterable[Pair],
) -> list[str]:
    r""":func:`check` over the two runs' case ids and their observations
    in pairs, taken one pair at a time, so a caller reading files holds
    one pair in memory, never a whole run."""
    problems = _cases(on_ids, 'captured') + _cases(off_ids, 'uncaptured')
    if on_ids != off_ids:
        problems.append('the two runs hold different cases')
    on_set, off_set = set(on_ids), set(off_ids)
    for case in required:
        if case not in off_set or case not in on_set:
            problems.append(f'{case}: required, and not in both runs')
    for item, other in pairs:
        case = item['case']
        problems.extend(f'{case}: {p}' for p in complete(item))
        if other is None:
            continue
        if other['driver']['capture'] is not None:
            problems.append(f'{case}: the run without capture captured')
        found = _first_difference(
            projection(item), projection(other), 'projection'
        )
        if found:
            problems.append(f'{case}: capture on and off differ at {found}')
    return problems


def check(
    on: list[dict[str, Any]],
    off: list[dict[str, Any]],
    required: tuple[str, ...] | list[str] = (),
) -> list[str]:
    r"""Every problem with the capture over two runs of the same cases,
    each case in ``required`` among them."""
    by_case = {item['case']: item for item in off}
    return _check(
        [item['case'] for item in on],
        [item['case'] for item in off],
        required,
        ((item, by_case.get(item['case'])) for item in on),
    )


def _lines(path: Path) -> Iterator[tuple[int, bytes]]:
    r"""Each non-empty line of ``path`` with the offset it starts at."""
    with path.open('rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return
            if line.strip(b'\r\n'):
                yield offset, line


def _index(path: Path) -> list[tuple[str, int]]:
    r"""Each observation's case id and the offset of its line, in file
    order; one line is parsed at a time."""
    return [(json.loads(line)['case'], offset) for offset, line in _lines(path)]


def _at(path: Path, offset: int) -> dict[str, Any]:
    with path.open('rb') as handle:
        handle.seek(offset)
        loaded: dict[str, Any] = json.loads(handle.readline())
        return loaded


def _summary(item: dict[str, Any]) -> str | None:
    capture = item['driver']['capture']
    logged = item['driver']['query_log']
    if capture is None:
        return None
    if logged is None:
        return (
            f'{item["case"]}: {len(capture["statements"])} statements and '
            f'{len(capture["bindings"])} bindings captured on '
            f'{capture["cursors"]} derived cursors; no query log'
        )
    return (
        f'{item["case"]}: {len(capture["statements"])} statements and '
        f'{len(capture["bindings"])} bindings captured on '
        f'{capture["cursors"]} derived cursors; query log '
        f'{logged["statements"]} and {logged["bindings"]}; '
        f'{logged["compared"]} statement texts compared with it'
    )


def check_files(
    on: Path, off: Path, required: tuple[str, ...] | list[str] = ()
) -> tuple[list[str], list[str]]:
    r""":func:`check` over two observation files, and each captured
    case's summary line, reading one pair of observations at a time: a
    family's runs reach hundreds of megabytes, which parsed whole took
    gigabytes (time and feature family: 6.6 GB on Linux)."""
    on_index, off_index = _index(on), _index(off)
    off_at = dict(off_index)
    summaries: list[str] = []

    def pairs() -> Iterator[Pair]:
        for case, offset in on_index:
            item = _at(on, offset)
            summary = _summary(item)
            if summary is not None:
                summaries.append(summary)
            where = off_at.get(case)
            yield item, None if where is None else _at(off, where)

    problems = _check(
        [case for case, _ in on_index],
        [case for case, _ in off_index],
        required,
        pairs(),
    )
    return problems, summaries


# --- the warehouse captures ------------------------------


#: Where the captures are committed: a protected path (contract/protected-
#: paths.txt), changed only in a landing whose trailer names it.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--require', action='append', default=[])
    arguments = parser.parse_args(argv)
    problems, summaries = check_files(
        arguments.on, arguments.off, arguments.require
    )
    for problem in problems:
        print(problem, file=sys.stderr)
    for summary in summaries:
        print(summary)
    print('capture: ' + ('FAILED' if problems else 'complete and conserving'))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
