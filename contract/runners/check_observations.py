#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate Python native-client observations against explicit current outcomes.

This consumes the existing runner format, including SQL fetch traces.
Expected outcomes are supplied separately and never inferred from observations.
Historical approvals, frozen clients and diagnostic waivers are not consulted.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

RUNNER = Path(__file__).with_name('run_python.py')
NATIVE_RUNNERS = {'python-core'}


def runner_module():
    spec = importlib.util.spec_from_file_location('observation_runner', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError('nonfinite JSON number')


def read_json(text):
    return json.loads(
        text, object_pairs_hook=strict_object, parse_constant=reject_constant
    )


def read_observations(path, runner):
    """Reject malformed, empty, duplicate and mixed-runner evidence."""
    validate = runner.validator('observation')
    records = {}
    who = set()
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            where = f'{path.name}:{number}: runner failure'
            try:
                item = read_json(line)
                problem = runner._invalid(validate, item)
            except (ValueError, RecursionError):
                raise ValueError(f'{where}: invalid JSON observation') from None
            if problem:
                raise ValueError(f'{where}: invalid observation: {problem}')
            key = item['case']
            if key in records:
                raise ValueError(f'{where}: duplicate case {key}')
            records[key] = item
            who.add(item['identity']['runner'])
    if not records or len(who) != 1 or not who <= NATIVE_RUNNERS:
        raise ValueError(f'{path.name}: expected one nonempty native runner')
    identities = [item['identity'] for item in records.values()]
    for field in (
        'commit',
        'contract',
        'native',
        'versions',
        'platform',
        'cell',
    ):
        if any(
            identity.get(field) != identities[0].get(field)
            for identity in identities[1:]
        ):
            raise ValueError(
                f'{path.name}: mixed build/runtime identity: {field}'
            )
    return who.pop(), records


def check(
    path,
    cases,
    expected,
    database,
    runner,
    cell=None,
    commit=None,
    native_sha256=None,
):
    """Validate inventory, identity, expected outcomes and native execution."""
    who, records = read_observations(path, runner)
    selected = {
        runner.case_key(case['id'], database): (case, digest)
        for case, digest in cases
        if database in runner.case_legs(case)
    }
    if not selected or set(records) != set(selected):
        raise ValueError(
            f'{path.name}: observations do not match selected cases'
        )
    wanted = expected.get(who)
    if not isinstance(wanted, dict) or set(wanted) != set(selected):
        raise ValueError(
            f'{path.name}: explicit outcomes must cover every case'
        )
    bases = runner.digest(runner.BASES.read_bytes())
    commit = commit or runner._git(runner.REPO)['sha']
    problems = []
    for key, item in records.items():
        case, digest = selected[key]
        identity, core, outcome = (
            item['identity'],
            item['core'],
            item['outcome'],
        )
        failures = []
        if identity['commit']['sha'] != commit:
            failures.append('candidate commit differs')
        if (
            who == 'python-core'
            and native_sha256 is not None
            and (identity.get('native') or {}).get('sha256') != native_sha256
        ):
            failures.append('native artifact hash differs')
        if identity['inputs'] != {'case': digest, 'bases': bases}:
            failures.append('input hashes differ')
        if runner.stale_client(identity['contract']):
            failures.append('native contract version differs')
        if identity.get('reference') is not None:
            failures.append('unexpected frozen-client identity')
        if cell is not None and identity.get('cell') != cell:
            failures.append('runtime cell differs')
        if who == 'python-core' and (
            identity['native_loaded'] is not True or not identity.get('native')
        ):
            failures.append('native module identity missing')
        if wanted[key] != outcome:
            failures.append('outcome differs from explicit expectation')
        if outcome['result'] == 'success' and (
            outcome['error'] is not None or outcome['code'] is not None
        ):
            failures.append('success has an error')
        if who == 'python-core' and (
            outcome['stage'] is None
            or (outcome['result'] == 'refusal' and outcome['error'] is None)
        ):
            failures.append('public outcome incomplete')
        if not isinstance(core, dict):
            failures.append('native trace missing')
        else:
            operations = identity['operations']
            # Successful calls must complete every declared operation.
            # Refusals can stop early, but must retain the operation that ran.
            if outcome['result'] == 'success' and not set(case['core']) <= set(
                operations
            ):
                failures.append('declared native operation missing')
            if not operations and case['core']:
                failures.append('no native operation recorded')
            if any(
                core.get(field) is not None
                for field in ('described', 'materialized')
            ) and not core.get('fetches'):
                failures.append('native fetch trace missing')
            if outcome['result'] == 'success':
                for operation, field in (
                    ('describe_source', 'described'),
                    ('materialize', 'materialized'),
                    ('predict', 'predicted'),
                ):
                    if operation in case['core'] and core.get(field) is None:
                        failures.append(f'{operation} result missing')
                if core.get('error') is not None:
                    failures.append('native error contradicts success')
        driver = item.get('driver')
        if driver is not None:
            capture, logged = driver.get('capture'), driver.get('query_log')
            if capture is None or logged is None:
                failures.append('SQL capture or query log missing')
            else:
                if capture['incomplete'] or logged['differing']:
                    failures.append(
                        'SQL capture incomplete or differs from execution'
                    )
                for field in ('statements', 'bindings'):
                    if len(capture[field]) != logged[field]:
                        failures.append(
                            f'SQL {field} count differs from execution'
                        )
        elif isinstance(core, dict) and core.get('fetches'):
            failures.append('SQL execution evidence missing')
        if (
            who == 'python-core'
            and isinstance(core, dict)
            and core.get('predicted')
        ):
            predicted = core['predicted']
            calls = predicted.get('calls', [predicted['body']])
            received = [
                entry['parsed']
                for entry in (item.get('public') or {}).get('payloads', [])
            ]
            paths = (('metadata', 'request_id'),)
            if [_payload(value, paths) for value in calls] != [
                _payload(value, paths) for value in received
            ]:
                failures.append(
                    'endpoint payload differs from native prediction'
                )
        problems.extend(f'{key}: {problem}' for problem in failures)
    return problems


_INTEGER = re.compile(r'-?(?:0|[1-9][0-9]*)')


def _payload(value, paths):
    """Ignore only request-ID values, preserving their presence and type."""
    value = copy.deepcopy(value)
    for path in paths:
        node = value
        for part in path[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and path[-1] in node:
            original = node[path[-1]]
            kind = type(original).__name__
            if isinstance(original, dict) and set(original) == {'number'}:
                kind = 'number'
            node[path[-1]] = {'excluded request id type': kind}
    return _lossless(value)


def _lossless(value: Any) -> Any:
    r"""A parsed payload with each number (``{"number": text}``) as its kind
    (an integer token or a float token) and its exact decimal value, so
    ``1e-05`` and ``1e-5`` agree, and ``1`` and ``1.0`` do not, nor ``-0.0``
    and ``0.0`` (its sign kept); nothing passes through a double."""
    if isinstance(value, dict):
        if set(value) == {'number'} and isinstance(value['number'], str):
            text = value['number']
            kind = 'integer' if _INTEGER.fullmatch(text) else 'float'
            value = Decimal(text)
            # The sign too, which Decimal equality ignores for zero, so
            # -0.0 and 0.0 differ while 1e-05 and 1e-5 agree.
            return ['number', kind, value, value.is_signed()]
        return {key: _lossless(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_lossless(item) for item in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cases', type=Path, nargs='+', required=True)
    parser.add_argument('--expect', type=Path, required=True)
    parser.add_argument('--database', default='duckdb')
    parser.add_argument('--cell')
    parser.add_argument(
        '--commit', help='candidate commit; defaults to the checkout HEAD'
    )
    parser.add_argument(
        '--native-sha256', help='installed Python native artifact hash'
    )
    parser.add_argument('observations', type=Path)
    args = parser.parse_args(argv)
    runner = runner_module()
    try:
        cases = runner.load_cases(args.cases)
        expected = read_json(args.expect.read_text(encoding='utf-8'))
        if not isinstance(expected, dict):
            raise ValueError(
                'expected outcomes must be an object keyed by runner'
            )
        problems = check(
            args.observations,
            cases,
            expected,
            args.database,
            runner,
            args.cell,
            args.commit,
            args.native_sha256,
        )
    except (OSError, ValueError, RecursionError, runner.RunnerFailure) as error:
        print(f'FAIL: {error}')
        return 1
    for problem in problems:
        print(f'FAIL: {problem}')
    return int(bool(problems))


if __name__ == '__main__':
    raise SystemExit(main())
