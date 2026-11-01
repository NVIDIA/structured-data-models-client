# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small shared manifest and observation contract for both language runners."""

import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(',', ':'), allow_nan=False
        ).encode()
    ).hexdigest()


def manifest(path=HERE / 'manifest.json'):
    data = json.loads(Path(path).read_text())
    assert data['version'] == 1
    ids = [case['id'] for case in data['scenarios']]
    assert len(ids) == len(set(ids)), 'duplicate case IDs'
    for case in data['scenarios']:
        for field in (
            'task',
            'entry',
            'journey',
            'fixture',
            'options',
            'expected',
            'baseline',
            'differences',
            'tolerance',
        ):
            assert field in case, (case['id'], field)
        assert set(case['apis']) == set(data['languages'])
        assert set(case['models']) <= set(data['models'])
        assert set(case['connectors']) <= set(data['connectors'])
        assert case['expected']['status'] in ('supported', 'refused')
        if case['expected']['status'] == 'refused':
            assert set(('stage', 'code', 'rule')) <= set(case['expected'])
        assert (
            case['tolerance']['absolute'] == 0
            and case['tolerance']['relative'] == 0
        ), 'nonzero tolerance requires explicit prediction-column policy'
    return data


def cells(data):
    for case in data['scenarios']:
        for connector, model, language in itertools.product(
            case['connectors'], case['models'], data['languages']
        ):
            yield (case['id'], connector, model, language)


def key(record):
    return tuple(
        record[name] for name in ('case', 'connector', 'model', 'language')
    )


def deferred(data, cell):
    return any(
        cell[3] == item['language'] and cell[1] in item['connectors']
        for item in data['deferrals']
    )


def effective_case(case, connector):
    active = {key: value for key, value in case.items() if key != 'variants'}
    active.update(case.get('variants', {}).get(connector, {}))
    return active


def fixture_hash(case, root=HERE):
    return digest(
        {
            'fixture': json.loads((root / case['fixture']).read_text()),
            'case': {
                key: value for key, value in case.items() if key != 'variants'
            },
        }
    )


def records(path):
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(
        json.loads((HERE / 'observation.schema.json').read_text())
    )
    out = {}
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        validator.validate(record)
        identity = key(record)
        if identity in out:
            raise ValueError(f'duplicate observation: {identity}')
        out[identity] = record
    return out


def evidence_problem(record, case, phase, live):
    if record.get('version') != 1 or record.get('phase') != phase:
        return 'invalid observation version/phase'
    if record.get('fixture_hash') != fixture_hash(case):
        return 'fixture/options identity differs'
    artifact = record.get('artifact', {})
    if not artifact.get('commit') or not artifact.get('package_hash'):
        return 'missing installed artifact identity'
    if phase == 'baseline':
        if artifact['commit'] != BASELINE:
            return 'baseline is not the frozen reference'
    else:
        if not artifact.get('native_hash') or not isinstance(
            artifact.get('contract'), int
        ):
            return 'missing candidate native identity'
        route = record.get('route', {})
        if (
            not route.get('installed')
            or route.get('scope') != 'whole-journey'
            or not route.get('workers')
        ):
            return 'missing whole-journey route proof'
        if route.get('unobserved_subprocesses') != []:
            return 'unobserved child process prevents route proof'
        if route.get('legacy_hits') != []:
            return 'legacy computation executed'
    if not record.get('source_snapshot'):
        return 'missing source snapshot'
    if record.get('status') == 'passed':
        semantic = record.get('semantic') or {}
        if not semantic.get('schema') or semantic.get('request') is None:
            return 'missing schema or request evidence'
        fixture = json.loads((HERE / case['fixture']).read_text())
        reader = record.get('reader', {})
        if set(reader) != set(fixture['tables']):
            return 'missing raw reader evidence'
        for name, reading in reader.items():
            if {field[0] for field in reading.get('schema', [])} != set(
                fixture['tables'][name]['columns']
            ):
                return 'missing reader schema fields'
        expected_lifecycle = ['materialize']
        if case['journey'] == 'restore-predict':
            expected_lifecycle.append('restore')
        if live:
            expected_lifecycle.extend(
                ['fit', 'predict', 'predict', 'close']
                if case['journey'] == 'fit-predict-close'
                else ['predict']
            )
        if semantic.get('lifecycle') != expected_lifecycle:
            return 'journey lifecycle is incomplete'
        expected_results = expected_lifecycle.count('predict')
        if len(semantic.get('results', [])) != expected_results:
            return 'prediction evidence is incomplete'
    if live and case['expected']['status'] == 'supported':
        endpoint = record.get('endpoint', {})
        if record.get('execution') != 'live' or endpoint.get('kind') != 'live':
            return 'local/stub evidence cannot certify live parity'
        if (
            not endpoint.get('model')
            or not endpoint.get('version')
            or not endpoint.get('transport')
        ):
            return 'missing model endpoint identity'
    return None
