#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Checks the driver capture on a runner's observations (design, "Driver
capture"; decision Q2):

    python contract/runners/capture_check.py --on FILE --off FILE

``--on`` holds a run with the capture, ``--off`` the same cases run with
``--no-capture``. It holds, case by case:

- completeness: the capture recorded every statement it saw and could
  record every result read, and its statement and binding counts equal
  DuckDB's own query log;
- conservation: capture on and off give the same public projection (the
  outcome, warnings, public ``Schema``, each request the scripted endpoint
  received, and the public result) and the same query-log counts, so the
  capture changed no value and no execution count.

A payload is compared parsed, leaving out only ``metadata.request_id``,
which the client draws at random for every request (:data:`RANDOM_PATHS`);
its raw SHA-256 differs with it, so it is not compared.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

#: The payload paths the client fills at random, left out of conservation.
RANDOM_PATHS = (('metadata', 'request_id'),)


def _payloads(public: dict[str, Any]) -> list[Any]:
    found = []
    for payload in public['payloads']:
        parsed = copy.deepcopy(payload['parsed'])
        for path in RANDOM_PATHS:
            node = parsed
            for part in path[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
            if isinstance(node, dict):
                node.pop(path[-1], None)
        found.append((payload['method'], payload['path'], parsed))
    return found


def projection(observation: dict[str, Any]) -> dict[str, Any]:
    r"""What the capture must not change of one observation."""
    public = observation['public']
    return {
        'outcome': observation['outcome'],
        'warnings': observation['host']['warnings'],
        'schema': public['schema'],
        'payloads': _payloads(public),
        'result': public['result'],
        'query_log': observation['driver']['query_log'],
    }


def complete(observation: dict[str, Any]) -> list[str]:
    r"""Why a captured observation's capture is not complete, if it is
    not."""
    driver = observation['driver']
    capture, logged = driver['capture'], driver['query_log']
    if capture is None:
        return ['the capture was off']
    problems = [f'could not record: {what}' for what in capture['incomplete']]
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


def check(on: list[dict[str, Any]], off: list[dict[str, Any]]) -> list[str]:
    r"""Every problem with the capture over two runs of the same cases."""
    problems = []
    by_case = {item['case']: item for item in off}
    if [item['case'] for item in on] != [item['case'] for item in off]:
        problems.append('the two runs hold different cases')
    for item in on:
        case = item['case']
        problems.extend(f'{case}: {p}' for p in complete(item))
        other = by_case.get(case)
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


def _read(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--off', type=Path, required=True)
    arguments = parser.parse_args(argv)
    on, off = _read(arguments.on), _read(arguments.off)
    problems = check(on, off)
    for problem in problems:
        print(problem, file=sys.stderr)
    for item in on:
        capture = item['driver']['capture']
        logged = item['driver']['query_log']
        print(
            f'{item["case"]}: {len(capture["statements"])} statements and '
            f'{len(capture["bindings"])} bindings captured on '
            f'{capture["cursors"]} derived cursors; query log '
            f'{logged["statements"]} and {logged["bindings"]}'
        )
    print('capture: ' + ('FAILED' if problems else 'complete and conserving'))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
