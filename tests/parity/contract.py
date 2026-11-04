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
        assert case.get('input_mode', 'prepared') in {
            'prepared',
            'raw-reader',
        }, 'unknown input preparation mode'
        assert case.get('schema_input', 'frame') in {'frame', 'source'}
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
        assert set(case.get('expected_by_model', {})) <= set(case['models'])
        for expected in [
            case['expected'],
            *case.get('expected_by_model', {}).values(),
        ]:
            assert expected['status'] in ('supported', 'refused')
            if expected['status'] == 'refused':
                assert set(('stage', 'code', 'rule')) <= set(expected)
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


def expectation(case, model, language=None):
    """Model capabilities are explicit; candidate language exceptions stay separate."""
    expected = case.get('expected_by_model', {}).get(model, case['expected'])
    if language is not None:
        expected = case.get('expected_by_language', {}).get(language, expected)
    return expected


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


def evidence_problem(record, case, phase, live, *, baseline_commit=BASELINE):
    if record.get('version') != 1 or record.get('phase') != phase:
        return 'invalid observation version/phase'
    if record.get('fixture_hash') != fixture_hash(case):
        return 'fixture/options identity differs'
    artifact = record.get('artifact', {})
    if not artifact.get('commit') or not artifact.get('package_hash'):
        return 'missing installed artifact identity'
    if phase == 'baseline':
        if artifact['commit'] != baseline_commit:
            return 'baseline is not the requested reference commit'
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
        if live and case.get('supplement'):
            details = semantic.get('result_details', [])
            if len(details) != expected_results:
                return 'missing public result item evidence'
            fields = {
                'row_index',
                'id',
                'forecast_step',
                'prediction',
                'probabilities',
                'scores',
                'rankings',
                'embeddings',
                'quantiles',
                'explanation',
                'metadata',
            }
            for detail, result in zip(
                details, semantic['results'], strict=True
            ):
                if (
                    len(detail.get('items', [])) != len(result.get('rows', []))
                    or detail.get('outputs') != case['outputs']
                    or not detail.get('request_id')
                    or not isinstance(detail.get('metadata'), dict)
                    or not detail.get('model')
                    or not all(
                        set(item) == fields for item in detail.get('items', [])
                    )
                ):
                    return 'incomplete public result item fields'
                for output in case['outputs']:
                    if not any(
                        item.get(output) not in (None, {}, [], ())
                        for item in detail['items']
                    ):
                        return 'requested result output is absent'
            calls = semantic.get('transport', [])
            expected_calls = (
                ['fit', 'score', 'score', 'close']
                if case['supplement']['kind'] == 'holdout'
                else ['predict'] * case['supplement']['prediction_calls']
            )
            if [call.get('operation') for call in calls] != expected_calls:
                return 'real transport call evidence is incomplete'
            if any(
                call.get('payload', {}).get('output', {}).get('fields')
                != case['outputs']
                for call in calls
                if call['operation'] in {'predict', 'score'}
            ):
                return 'requested outputs were not sent to the real endpoint'
            from supplements import request_ids, response_groups, session_proof

            try:
                response_groups(semantic, case)
                request_ids(semantic, case)
                session_proof(semantic, case)
            except (ValueError, KeyError, TypeError):
                return 'actual response/result correlation is not proven'
            if case['supplement']['kind'] == 'holdout':
                from supplements import assert_holdout_payloads

                asked = semantic['request']
                payloads = asked if isinstance(asked, list) else None
                count = (
                    sum(
                        len(p['context']['instance_table']['rows'])
                        for p in payloads
                    )
                    if payloads
                    else len(asked['context']['rows'])
                )
                try:
                    assert_holdout_payloads(
                        payloads, count, semantic.get('holdout_keys', []), calls
                    )
                except (AssertionError, KeyError, TypeError, ValueError):
                    return 'uploaded holdout exclusion is not proven'

    if live and case.get('input_mode') == 'raw-reader':
        from supplements import wire_inputs

        try:
            wire_inputs(record, case)
        except (ValueError, KeyError, TypeError):
            return 'actual model input capture is missing or invalid'

    expected = expectation(
        case,
        record['model'],
        record['language'] if phase == 'candidate' else None,
    )
    if live and (
        expected['status'] == 'supported'
        or expected.get('stage') in {'fit', 'predict', 'close'}
    ):
        endpoint = record.get('endpoint') or {}
        if record.get('execution') != 'live' or endpoint.get('kind') != 'live':
            return 'local/stub evidence cannot certify live parity'
        if (
            not endpoint.get('model')
            or not endpoint.get('version')
            or not endpoint.get('transport')
        ):
            return 'missing model endpoint identity'
    return None
