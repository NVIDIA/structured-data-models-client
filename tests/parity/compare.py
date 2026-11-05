# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate installed-client evidence and compare language behavior, fail closed."""

import argparse
import json
import math
import operator
import re
from pathlib import Path

from identity import (
    IdentityError,
    canonical,
    canonical_scoring_payload,
)
from shop_fixtures import load_fixture

from contract import (
    cells,
    deferred,
    effective_case,
    evidence_problem,
    expectation,
    manifest,
    records,
)


def difference(left, right, tolerance, path='$'):
    if isinstance(left, bool) or isinstance(right, bool):
        return None if type(left) is type(right) and left == right else path
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if left == right:
            return None
        if isinstance(left, int) and isinstance(right, int):
            return path
        if tolerance['absolute'] == 0 and tolerance['relative'] == 0:
            return path
        return (
            None
            if math.isclose(
                left,
                right,
                abs_tol=tolerance['absolute'],
                rel_tol=tolerance['relative'],
            )
            else path
        )
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + '.keys'
        for name in left:
            found = difference(
                left[name], right[name], tolerance, path + '.' + name
            )
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path + '.length'
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            found = difference(a, b, tolerance, f'{path}[{index}]')
            if found:
                return found
        return None
    return None if left == right else path


def equivalent_snowflake_reader_integers(old, new, connector):
    """Prove lossless signed integer encodings without changing recorded evidence."""
    if connector not in ('snowflake', 'snowflake-semantic'):
        return False
    if old['data']['columns'] != new['data']['columns']:
        return False
    if len(old['schema']) != len(new['schema']):
        return False
    if len(old['data']['rows']) != len(new['data']['rows']):
        return False
    if difference(old['data'], new['data'], {'absolute': 0, 'relative': 0}):
        return False

    def bounds(dtype):
        integer = re.fullmatch(r'int(8|16|32|64)', dtype)
        if integer:
            limit = 2 ** (int(integer[1]) - 1)
            return -limit, limit - 1
        decimal = re.fullmatch(
            r'decimal128\(([1-9]|[12][0-9]|3[0-8]), 0\)', dtype
        )
        if decimal:
            limit = 10 ** int(decimal[1]) - 1
            return -limit, limit
        return None

    normalized = False
    for before, after in zip(old['schema'], new['schema'], strict=True):
        if before[0] != after[0] or before[1] == after[1]:
            continue
        if not (before[1].startswith('int') or after[1].startswith('int')):
            continue
        left, right = bounds(before[1]), bounds(after[1])
        if left is None or right is None:
            continue
        at = old['data']['columns'].index(before[0])
        for a, b in zip(old['data']['rows'], new['data']['rows'], strict=True):
            x, y = a[at], b[at]
            if x is None and y is None:
                continue
            if (
                type(x) is not int
                or type(y) is not int
                or x != y
                or not left[0] <= x <= left[1]
                or not right[0] <= y <= right[1]
            ):
                break
        else:
            after[1] = before[1]
            normalized = True
    return normalized


def equivalent_reader_time_units(old, new):
    """Normalize only exact timestamp values with compatible declared resolutions."""
    exact = {'absolute': 0, 'relative': 0}
    if difference(old['data'], new['data'], exact) is not None:
        return
    if len(old['schema']) != len(new['schema']):
        return
    for before, after in zip(old['schema'], new['schema'], strict=True):
        if before[0] != after[0] or before[1] == after[1]:
            continue
        a = re.fullmatch(r'timestamp\[(s|ms|us|ns)(, tz=[^]]+)?\]', before[1])
        b = re.fullmatch(r'timestamp\[(s|ms|us|ns)(, tz=[^]]+)?\]', after[1])
        if not a or not b or a[2] != b[2]:
            continue
        digits = min(
            {'s': 0, 'ms': 3, 'us': 6, 'ns': 9}[a[1]],
            {'s': 0, 'ms': 3, 'us': 6, 'ns': 9}[b[1]],
        )
        at = old['data']['columns'].index(before[0])
        for row in old['data']['rows']:
            value = row[at]
            if value is None:
                continue
            match = (
                re.fullmatch(
                    r'[+-]?\d{4,}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})?',
                    value,
                )
                if isinstance(value, str)
                else None
            )
            if not match or any(c != '0' for c in (match[1] or '')[digits:]):
                break
        else:
            # Caller compares all other schema fields and every original value.
            after[1] = before[1]


def equivalent_raw_observations(old, new, fixture):
    """Prove only reader representations against named fixture slots, on copies."""
    import datetime
    from decimal import Decimal, InvalidOperation, localcontext
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    notes = set()
    exact = {'absolute': 0, 'relative': 0}

    def instant(value, dtype):
        match = re.fullmatch(r'timestamp\[(s|ms|us|ns), tz=([^]]+)\]', dtype)
        if not match:
            return None
        factor = {'s': 10**9, 'ms': 10**6, 'us': 1000, 'ns': 1}[match[1]]
        if type(value) is int:
            return value * factor if -(2**63) <= value < 2**63 else None
        if not isinstance(value, str):
            return None
        text = re.fullmatch(
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})',
            value,
        )
        if not text:
            return None
        try:
            date = datetime.datetime.fromisoformat(
                text[1] + text[3].replace('Z', '+00:00')
            )
            delta = date - datetime.datetime(
                1970, 1, 1, tzinfo=datetime.timezone.utc
            )
            nanos = (delta.days * 86400 + delta.seconds) * 10**9 + int(
                (text[2] or '').ljust(9, '0')
            )
            return (
                nanos
                if nanos % factor == 0 and -(2**63) <= nanos // factor < 2**63
                else None
            )
        except ValueError:
            return None

    def adjust(frame, source, key, schema):
        if key not in frame['columns']:
            return
        source_key = source['columns'].index(key)
        known = {}
        for row in source['rows']:
            identity = row[source_key]
            if type(identity) is not int or identity in known:
                return
            known[identity] = row
        key_at = frame['columns'].index(key)
        for row in frame['rows']:
            identity = row[key_at]
            if type(identity) is not int or identity not in known:
                continue
            declared = known[identity]
            for at, column in enumerate(frame['columns']):
                if column not in source['columns']:
                    continue
                wanted = declared[source['columns'].index(column)]
                dtype = schema.get(column, '')
                if (
                    wanted is None
                    and dtype in ('float', 'double', 'float32', 'float64')
                    and row[at] == {'float': 'nan'}
                ):
                    row[at] = None
                    notes.add('nullable numeric missingness')
                spec = source.get('types', {}).get(column, '')
                if (
                    spec in {'list<float32>', 'list<float64>'}
                    and dtype in {'list<item: float>', 'list<item: double>'}
                    and isinstance(wanted, list)
                    and isinstance(row[at], list)
                    and len(wanted) == len(row[at])
                ):
                    normalized = [
                        None
                        if declared is None and value == {'float': 'nan'}
                        else value
                        for value, declared in zip(row[at], wanted, strict=True)
                    ]
                    if normalized != row[at]:
                        notes.add('nullable numeric list elements')
                        row[at] = normalized
                if (
                    spec == 'date32'
                    and dtype == 'date32[day]'
                    and isinstance(wanted, str)
                ):
                    try:
                        datetime.date.fromisoformat(wanted)
                        if row[at] in (wanted + 'T00:00', wanted + 'T00:00:00'):
                            row[at] = wanted
                            notes.add(
                                'date32 inspection as exact midnight datetime'
                            )
                    except ValueError:
                        pass
                if spec.startswith('timestamp[') and wanted is not None:
                    desired = instant(wanted, dtype)
                    if (
                        desired is not None
                        and instant(row[at], dtype) == desired
                    ):
                        if row[at] != wanted:
                            notes.add('timestamp value encoding')
                        row[at] = wanted

    for name, source in fixture['tables'].items():
        if name not in old.get('reader', {}) or name not in new.get(
            'reader', {}
        ):
            continue
        a, b = old['reader'][name], new['reader'][name]
        key = fixture['primary_keys'][name]
        for record, reading in ((old, a), (new, b)):
            schema = dict(reading['schema'])
            adjust(reading['data'], source, key, schema)
            request = (record.get('semantic') or {}).get('request')
            # Only public pre-wire single-table inspection, never wire payloads.
            if (
                name == 'users'
                and isinstance(request, dict)
                and set(request) == {'context', 'predict', 'target', 'task'}
            ):
                for half in ('context', 'predict'):
                    adjust(request[half], source, key, schema)
        if difference(a['data'], b['data'], exact) is not None:
            continue
        if len(a['schema']) != len(b['schema']):
            continue
        for left, right in zip(a['schema'], b['schema'], strict=True):
            column = left[0]
            if (
                column != right[0]
                or left[1] == right[1]
                or column not in source['columns']
            ):
                continue
            at = source['columns'].index(column)
            observed_at = a['data']['columns'].index(column)
            zones = [
                re.fullmatch(r'timestamp\[(s|ms|us|ns), tz=([^]]+)\]', t)
                for t in (left[1], right[1])
            ]
            if (
                all(zones)
                and zones[0][1] == zones[1][1]
                and source.get('types', {})
                .get(column, '')
                .startswith('timestamp[')
            ):
                try:
                    for zone in zones:
                        ZoneInfo(zone[2])
                    expected_rows = {
                        r[source['columns'].index(key)]: r[at]
                        for r in source['rows']
                    }
                    verified = True
                    observed = False
                    for reading in (a, b):
                        schema = dict(reading['schema'])
                        columns = reading['data']['columns']
                        for row in reading['data']['rows']:
                            identity = row[columns.index(key)]
                            value = row[columns.index(column)]
                            if (
                                type(identity) is not int
                                or identity not in expected_rows
                            ):
                                verified = False
                                break
                            wanted = expected_rows[identity]
                            if value is None and wanted is None:
                                continue
                            desired = instant(wanted, schema[column])
                            if (
                                desired is None
                                or instant(value, schema[column]) != desired
                            ):
                                verified = False
                                break
                            observed = True
                    if verified and observed:
                        notes.add(
                            'timezone display label differs; exact instants preserved'
                        )
                        right[1] = left[1]
                        continue
                except (ValueError, ZoneInfoNotFoundError):
                    pass
            if {left[1], right[1]} == {'null', 'string'}:
                if (
                    source['rows']
                    and all(row[at] is None for row in source['rows'])
                    and all(
                        row[observed_at] is None for row in a['data']['rows']
                    )
                ):
                    right[1] = left[1]
                    notes.add('all-null field storage')
                continue
            types = [
                re.fullmatch(
                    r'decimal128\(([1-9]|[12][0-9]|3[0-8]), ([0-9]+)\)', t
                )
                for t in (left[1], right[1])
            ]
            if not all(types) or types[0][2] != types[1][2]:
                continue
            precision = min(int(t[1]) for t in types)
            scale = int(types[0][2])
            if scale > precision:
                continue
            try:
                with localcontext() as context:
                    context.prec = 80
                    for row in a['data']['rows']:
                        value = row[observed_at]
                        if value is None:
                            continue
                        if type(value) is int:
                            decimal = Decimal(value)
                        elif type(value) is float and math.isfinite(value):
                            decimal = Decimal.from_float(value)
                        elif isinstance(value, dict) and set(value) == {
                            'decimal'
                        }:
                            decimal = Decimal(value['decimal'])
                        else:
                            raise ValueError('decimal observation is not exact')
                        unscaled = decimal.scaleb(scale)
                        if (
                            not decimal.is_finite()
                            or unscaled != unscaled.to_integral_value()
                            or abs(unscaled) >= 10**precision
                        ):
                            raise ValueError(
                                'decimal does not fit both schemas'
                            )
                right[1] = left[1]
                notes.add(
                    'decimal precision metadata; exact values fit both types'
                )
            except (InvalidOperation, ValueError, TypeError):
                continue
    return sorted(notes)


def assess(
    data,
    candidate,
    *,
    live=True,
    selected=None,
    candidate_commit,
):
    scenarios = {case['id']: case for case in data['scenarios']}
    required = set(cells(data))
    unknown = set(candidate) - required
    if unknown:
        raise ValueError(f'unknown candidate cells: {sorted(unknown)}')
    result = []
    artifacts = {}
    for cell, record in candidate.items():
        identity = record.get('artifact', {})
        if not identity.get('package_hash'):
            continue
        pinned = (
            identity.get('commit'),
            identity.get('package_hash'),
            identity.get('native_hash'),
            identity.get('contract'),
        )
        previous = artifacts.setdefault(cell[3], pinned)
        if previous != pinned:
            raise ValueError('candidate records mix artifacts for ' + cell[3])
        if (
            candidate_commit is not None
            and identity.get('commit') != candidate_commit
        ):
            raise ValueError(
                'candidate observation is not the requested commit'
            )
    for cell in sorted(required):
        row = {
            'cell': list(cell),
            'status': 'incomplete',
            'reason': 'missing observation',
        }
        result.append(row)
        if deferred(data, cell):
            row.update(
                status='deferred',
                reason='manifest-approved Java Snowflake deferral',
            )
            continue
        if selected is not None and cell not in selected:
            row['reason'] = (
                'outside requested scope; full goal remains incomplete'
            )
            continue
        case = effective_case(scenarios[cell[0]], cell[1])
        old = candidate.get((*cell[:3], 'python'))
        new = candidate.get(cell)
        if cell[3] == 'python':
            if new is None:
                continue
            problem = evidence_problem(new, case, live)
            expected = expectation(case, cell[2], 'python', live=live)
            wanted = expected.get(
                'outcome',
                'refused' if expected['status'] == 'refused' else 'passed',
            )
            if problem:
                row['reason'] = problem
            elif new.get('status') != wanted or (
                expected['status'] == 'refused'
                and any(
                    new.get('error', {}).get(name) != expected[name]
                    for name in ('stage', 'code', 'rule')
                )
            ):
                row.update(
                    status='failed',
                    reason='unexpected current-client outcome or refusal',
                )
            else:
                row.update(
                    status='passed',
                    verdict='Current contract verified',
                    reason='artifact, lifecycle and manifest checks passed',
                )
            continue
        if old is None or new is None:
            continue
        problem = evidence_problem(old, case, live) or evidence_problem(
            new, case, live
        )
        if problem:
            row['reason'] = problem
            continue
        if old['source_snapshot'] != new['source_snapshot']:
            row['reason'] = 'different source snapshots'
            continue
        if live and old.get('endpoint') != new.get('endpoint'):
            row['reason'] = 'different model versions or transports'
            continue
        expected = expectation(case, cell[2], cell[3], live=live)
        python_expected = expectation(case, cell[2], 'python', live=live)
        wanted = expected.get(
            'outcome',
            'refused' if expected['status'] == 'refused' else 'passed',
        )
        old_wanted = python_expected.get(
            'outcome',
            'refused' if python_expected['status'] == 'refused' else 'passed',
        )
        if new.get('status') != wanted or old.get('status') != old_wanted:
            row.update(
                status='failed',
                reason='unexpected outcome or missing Python behavior',
            )
            continue
        if (
            expected['status'] == 'refused'
            or python_expected['status'] == 'refused'
        ):
            if any(
                record.get('error', {}).get(name) != spec[name]
                for record, spec in ((old, python_expected), (new, expected))
                if spec['status'] == 'refused'
                for name in ('stage', 'code', 'rule')
            ):
                row.update(
                    status='failed',
                    reason='unexpected language refusal stage/code/rule',
                )
                continue

        if expected['status'] == python_expected['status'] == 'refused' and any(
            expected[name] != python_expected[name]
            for name in ('stage', 'code', 'rule')
        ):
            row.update(
                status='passed',
                verdict='Declared language difference',
                reason='each language matches its explicit manifest expectation; no agreement claimed',
            )
            continue

        # Differences must have executable case-specific policies before they
        # can waive anything. A prose difference ID alone never grants a pass.
        import copy

        if case.get('input_mode') == 'raw-reader':
            from contract import HERE

            old, new = copy.deepcopy(old), copy.deepcopy(new)
            row['reader_representation'] = equivalent_raw_observations(
                old, new, load_fixture(HERE / case['fixture'])
            )
        old_semantic, new_semantic = (
            copy.deepcopy(old.get('semantic')),
            copy.deepcopy(new.get('semantic')),
        )
        if live and case.get('supplement') and wanted == 'passed':
            from supplements import normalize_response_ids

            for semantic in (old_semantic, new_semantic):
                normalize_response_ids(semantic, case)
        if (
            cell[2] == 'relational'
            and case.get('comparison') == 'fixture-source-identity'
            and isinstance(old_semantic, dict)
            and isinstance(new_semantic, dict)
            and old_semantic.get('request') is not None
            and new_semantic.get('request') is not None
            and difference(
                old_semantic['request'],
                new_semantic['request'],
                {'absolute': 0, 'relative': 0},
            )
            is not None
        ):
            from contract import HERE

            fixture = load_fixture(HERE / case['fixture'])
            if case.get('input_mode') == 'raw-reader':
                # The untouched reader rows, not prepared fixture types, are
                # the source of sampled identities in these journeys.
                old_rows = {
                    name: value['data'] for name, value in old['reader'].items()
                }
                new_rows = {
                    name: value['data'] for name, value in new['reader'].items()
                }
                if (
                    difference(
                        old_rows, new_rows, {'absolute': 0, 'relative': 0}
                    )
                    is not None
                ):
                    row.update(
                        status='failed',
                        reason='raw reader rows differ before identity proof',
                    )
                    continue
                fixture['tables'] = old_rows
            try:
                for semantic in (old_semantic, new_semantic):
                    semantic['request'] = canonical(
                        semantic['request'], fixture
                    )
            except (IdentityError, KeyError, TypeError) as error:
                row.update(
                    status='failed',
                    reason='identity proof failed: ' + str(error),
                )
                continue
        if (
            live
            and case.get('supplement')
            and cell[2] == 'relational'
            and wanted == 'passed'
        ):
            from contract import HERE

            fixture = load_fixture(HERE / case['fixture'])
            try:
                for before, after in zip(
                    old_semantic['transport'],
                    new_semantic['transport'],
                    strict=True,
                ):
                    if (
                        before['operation'] in {'fit', 'predict'}
                        and difference(
                            before['payload'],
                            after['payload'],
                            {'absolute': 0, 'relative': 0},
                        )
                        is not None
                    ):
                        before['payload'] = canonical(
                            [before['payload']], fixture
                        )[0]
                        after['payload'] = canonical(
                            [after['payload']], fixture
                        )[0]
                    elif (
                        before['operation'] == 'score'
                        and difference(
                            before['payload'],
                            after['payload'],
                            {'absolute': 0, 'relative': 0},
                        )
                        is not None
                    ):
                        before['payload'] = canonical_scoring_payload(
                            before['payload'],
                            old['semantic']['transport'][0]['payload'][
                                'schema'
                            ],
                            fixture,
                        )
                        after['payload'] = canonical_scoring_payload(
                            after['payload'],
                            new['semantic']['transport'][0]['payload'][
                                'schema'
                            ],
                            fixture,
                        )
            except (IdentityError, KeyError, TypeError) as error:
                row.update(
                    status='failed',
                    reason='transport identity proof failed: ' + str(error),
                )
                continue
        if live and case.get('input_mode') == 'raw-reader':
            from supplements import wire_inputs

            from contract import HERE

            old_calls, new_calls = (
                wire_inputs(old, case),
                wire_inputs(new, case),
            )
            old_semantic = old_semantic or {}
            new_semantic = new_semantic or {}
            old_semantic.pop('wire_calls', None)
            new_semantic.pop('wire_calls', None)
            fixture = load_fixture(HERE / case['fixture'])
            if cell[2] == 'relational' and old_calls:
                # Reader representations were proven equivalent above; identity
                # conversion still checks all wire features and foreign-key edges.
                fixture['tables'] = {
                    name: {**fixture['tables'][name], **value['data']}
                    for name, value in old['reader'].items()
                }
                pinned = [None, None]
                try:
                    for before, after in zip(old_calls, new_calls, strict=True):
                        op = before['operation']
                        if op == 'fit':
                            pinned = [
                                copy.deepcopy(c['payload']['schema'])
                                for c in (before, after)
                            ]
                        if op in {'predict', 'fit'}:
                            if (
                                difference(
                                    before['payload'],
                                    after['payload'],
                                    {'absolute': 0, 'relative': 0},
                                )
                                is not None
                            ):
                                for call in (before, after):
                                    call['payload'] = canonical(
                                        [call['payload']], fixture
                                    )[0]
                        elif op == 'score':
                            for call, schema in zip(
                                (before, after), pinned, strict=True
                            ):
                                call['payload'] = canonical_scoring_payload(
                                    call['payload'], schema, fixture
                                )
                except (
                    IdentityError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as error:
                    row.update(
                        status='failed',
                        reason='actual model input identity proof failed: '
                        + str(error),
                    )
                    continue
            old_semantic['model_inputs'] = old_calls
            new_semantic['model_inputs'] = new_calls

        exact = {'absolute': 0, 'relative': 0}
        old_results = (
            old_semantic.pop('results', None)
            if isinstance(old_semantic, dict)
            else None
        )
        new_results = (
            new_semantic.pop('results', None)
            if isinstance(new_semantic, dict)
            else None
        )
        changed = difference(old_semantic, new_semantic, exact)
        changed = changed or difference(
            old_results, new_results, case['tolerance'], '$.results'
        )
        from contract import HERE

        fixture = load_fixture(HERE / case['fixture'])

        def readers(record, fixture=fixture):
            values = {}
            for name, reading in record.get('reader', {}).items():
                data = copy.deepcopy(reading['data'])
                at = data['columns'].index(fixture['primary_keys'][name])
                data['rows'].sort(key=operator.itemgetter(at))
                values[name] = {
                    'schema': copy.deepcopy(reading['schema']),
                    'data': data,
                }
            return values

        old_readers, new_readers = readers(old), readers(new)
        integer_encoding = False
        for name in old_readers.keys() & new_readers.keys():
            # Prove original row order, before the existing fixture-key ordering.
            raw_old = copy.deepcopy(old['reader'][name])
            raw_new = copy.deepcopy(new['reader'][name])
            if equivalent_snowflake_reader_integers(raw_old, raw_new, cell[1]):
                new_readers[name]['schema'] = raw_new['schema']
                integer_encoding = True
            equivalent_reader_time_units(old_readers[name], new_readers[name])
        changed = changed or difference(
            old_readers, new_readers, exact, '$.reader'
        )

        row.update(
            status='failed' if changed else 'passed',
            verdict='Cross-language agreement'
            if not changed
            else 'Different language results',
            reason=f'difference at {changed}'
            if changed
            else 'Lossless Snowflake signed integer / scale-zero Decimal128 reader encoding'
            if integer_encoding
            else 'matching evidence',
        )
    scoped = [
        row
        for row in result
        if selected is None or tuple(row['cell']) in selected
    ]

    def complete(rows):
        return bool(rows) and all(
            row['status'] in ('passed', 'deferred') for row in rows
        )

    return {
        'version': 1,
        'scope_complete': complete(scoped),
        'goal_complete': live and complete(result),
        'coverage_mode': 'live' if live else 'local',
        'comparison': 'current-clients',
        'candidate_commit': candidate_commit,
        'cells': result,
    }


def label(row, mode):
    if row.get('status') == 'passed':
        return (
            row.get('verdict', 'Current contract verified')
            if mode == 'live'
            else 'Local check only'
        )
    return row.get('status', 'not applicable')


def matrix(report):
    indexed = {tuple(row['cell']): row for row in report['cells']}
    lines = [
        '# Prediction compatibility evidence',
        '',
        '<!-- Generated by tests/parity/compare.py; do not edit. -->',
        '',
        '| Case / connector | Python Relational | Python Tabular | Java Relational | Java Tabular |',
        '|---|---|---|---|---|',
    ]
    for case, connector in sorted({key[:2] for key in indexed}):
        values = [
            label(
                indexed.get((case, connector, model, language), {}),
                report['coverage_mode'],
            )
            + indexed.get((case, connector, model, language), {}).get(
                'build_label', ''
            )
            for language in ('python', 'java')
            for model in ('relational', 'tabular')
        ]
        lines.append(
            '| ' + ' | '.join([case + ' / ' + connector, *values]) + ' |'
        )
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path(__file__).with_name('manifest.json'),
    )
    parser.add_argument(
        '--comparison',
        choices=('current-clients',),
        default='current-clients',
    )
    parser.add_argument('--candidate-commit', required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--local', action='store_true')
    for axis in ('case', 'connector', 'model', 'language'):
        parser.add_argument('--' + axis, action='append')
    args = parser.parse_args()
    data = manifest(args.manifest)
    selected = {
        cell
        for cell in cells(data)
        if all(
            not getattr(args, axis) or cell[i] in getattr(args, axis)
            for i, axis in enumerate(('case', 'connector', 'model', 'language'))
        )
    }
    report = assess(
        data,
        records(args.candidate),
        live=not args.local,
        selected=selected,
        candidate_commit=args.candidate_commit,
    )
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    if args.matrix:
        args.matrix.write_text(matrix(report))
    return 0 if report['scope_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
