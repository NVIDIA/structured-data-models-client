# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private original bytes authenticate a distinct, decoded publication.

These are independently authored actual-format DTO fixtures, not recordings
of SDK execution. No projected or supplied view can authenticate their bytes.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError
from test_t44_materialization_raw_journal import _bytes
from test_t44_prediction_observation import actual_observation, reader
from test_t44_prediction_serving_capture import served_observation
from test_t45_materialization_observations import (
    actual_materialization,
    prediction_materialization_failure,
)
from test_t136_forecast_capture import RUNNER, _module

ROOT = Path(__file__).parents[2]
DATABASE = 'databricks-sql'
NAMES = {
    'catalog': 'private_catalog_t45',
    'schema': 'private_schema_t45',
    'shop_metric_view': 'private_events_t45',
}
COMPARE = _module('t45_private_compare', ROOT / 'scripts/rust_core/compare.py')
T32 = _module('t45_private_pair', ROOT / 'tests/test_t32_live_only.py')
REFUSALS = (ValueError, ValidationError, RUNNER.RunnerFailure)


def publication():
    return RUNNER._sibling('observation_publication')


def encoded(attachment, value):
    # Deliberate spacing distinguishes original bytes from a canonical dump.
    return _bytes(attachment, json.dumps(value, indent=1).encode())


def observation(core, operations):
    # Reuse a complete historical envelope; replace only its actual core and
    # endpoint receipts. The seed claims no new mathematical case execution.
    raw = copy.deepcopy(T32._reference_line())
    raw['case'] = 't45.private.observation@databricks-sql'
    raw['identity'].update(
        runner='python-core',
        reference=None,
        native={'sha256': '0' * 64},
        native_loaded=True,
        operations=operations,
        contract=35,
    )
    raw['core'] = core
    raw['public']['payloads'] = []
    return raw


def materialization():
    """Existing three-Run Source/Population/Request chain, real text slots.

    Task refs and enums stay unchanged. Source relations carry private text
    in a taken value and a rebound input; Task/query and final output reads
    cover the separate byte roles without adding an untyped Task field.
    """
    core = actual_materialization()
    task = core['materialization_preparations'][0]
    request = copy.deepcopy(task)
    relation = {'source': list(NAMES.values())}
    source = {
        'source': 'source',
        'identity': 'source identity',
        'owner': 4,
        'target': {'source': relation, 'column': 'value'},
        'tables': [
            {'source': relation, 'read': relation, 'columns': ['value']}
        ],
    }
    population_data = {
        'source': 'source',
        'identity': 'population identity',
        'owner': 8,
        'rows': 0,
        'batch_rows': 10,
        'target': source['target'],
    }
    task['input_reads'][1] = encoded(
        2,
        'PREDICT private_events_t45.value FOR EACH private_events_t45.id',
    )
    task['value_reads'] = [encoded(10, {'value': 'source', 'data': source})]
    task['output_reads'] = []
    task['owner_takes'] = [{'attachment': 4, 'origin': 'result'}]
    population = copy.deepcopy(task)
    population.update(
        run=1,
        input=encoded(
            9, {'stage': 'population', 'source': {**source, 'owner': 2}}
        ),
        input_reads=[],
        owner_binds=[
            {
                'attachment': 2,
                'producer': {'run': 0, 'attachment': 4, 'origin': 'result'},
            }
        ],
        owner_takes=[{'attachment': 8, 'origin': 'result'}],
        value_reads=[
            encoded(10, {'value': 'population', 'data': population_data})
        ],
    )
    request.update(
        run=2,
        input=encoded(
            9,
            {
                'stage': 'request',
                'population': {**population_data, 'owner': 3},
                'batches': [],
                'values': None,
            },
        ),
        input_reads=[],
        owner_binds=[
            {
                'attachment': 3,
                'producer': {'run': 1, 'attachment': 8, 'origin': 'result'},
            }
        ],
        output_reads=[
            encoded(
                11,
                {
                    'context': {
                        NAMES['shop_metric_view']: {
                            'format': 'arrays',
                            'columns': ['value'],
                            'rows': [[1], [2]],
                        }
                    },
                    'predict': {},
                },
            ),
            encoded(12, 'resolved private_events_t45 query'),
        ],
    )
    core['materialization_preparations'] = [task, population, request]
    core['materialization_entries'][0].update(
        runs=[0, 1, 2], terminal={'run': 2}
    )
    return observation(core, ['prepare_materialize'] * 3)


def prediction(*, serving=False, retries=1, names=NAMES):
    core, operations = served_observation() if serving else actual_observation()
    call = core['predictions'][0]['calls'][0]
    body = json.loads(base64.b64decode(call['input_body_base64']))
    body.setdefault('metadata', {})['source_catalog'] = names['catalog']
    original = json.dumps(body, indent=1).encode()
    call['input_body_base64'] = base64.b64encode(original).decode('ascii')
    call['input_body_sha256'] = hashlib.sha256(original).hexdigest()
    wire = (
        json.dumps(
            {'dataframe_records': [{'request_json': original.decode()}]}
        ).encode()
        if serving
        else original
    )
    received = RUNNER._payload(wire)
    call['body'] = copy.deepcopy(received['parsed'])
    call['body_sha256'] = received['sha256']
    core['predicted']['body'] = copy.deepcopy(call['body'])
    raw = observation(core, operations)
    raw['public']['payloads'] = [
        {
            'method': 'POST',
            'path': '/serving-endpoints/fixture/invocations'
            if serving
            else '/v1/predictions',
            **copy.deepcopy(received),
        }
        for _ in range(retries)
    ]
    RUNNER.validator('observation').validate(raw)
    return raw


def prepare(raw, names=NAMES):
    return publication().prepare_comparison(raw, DATABASE, names)


def test_raw_owned_source_and_projected_views_are_separate():
    raw = materialization()
    before = copy.deepcopy(raw)
    names = dict(NAMES)
    prepared = prepare(raw, names)
    expected = reader().view(raw['core'], raw['identity']['operations'])
    held = RUNNER.Placeheld(NAMES, NAMES.values())
    assert isinstance(prepared, Mapping)
    assert prepared.raw == before and prepared.raw is not raw
    assert prepared.raw_view == expected
    assert prepared.view == held.value(expected)
    assert prepared['case'] == raw['case']
    assert prepared['public'] == held.value(raw['public'])
    assert prepared.view['operations'] == ['materialize']
    assert '{shop_metric_view}' in prepared.view['materialized']['query']
    assert prepared.raw['core']['materialized'] is None
    raw['core']['materialization_preparations'][0]['input']['sha256'] = 'f' * 64
    names['catalog'] = 'changed-after-prepare'
    assert prepared.raw == before
    assert prepared.view == held.value(expected)


@pytest.mark.parametrize(
    'changed',
    [
        'base64',
        'input-hash',
        'parsed-only',
        'value-hash',
        'late-read-hash',
        'output-hash',
        'missing-read',
        'foreign-owner',
        'wrong-terminal',
        'before-done',
    ],
)
def test_raw_materialization_must_validate_before_any_name_projection(changed):
    raw = materialization()
    records = raw['core']['materialization_preparations']
    if changed == 'base64':
        records[1]['input']['base64'] = '!not base64!'
    elif changed == 'input-hash':
        records[1]['input']['sha256'] = '0' * 64
    elif changed == 'parsed-only':
        records[1]['input']['parsed']['source']['tables'][0]['columns'] = ['id']
    elif changed == 'value-hash':
        records[0]['value_reads'][0]['sha256'] = '0' * 64
    elif changed == 'late-read-hash':
        records[0]['input_reads'][1]['sha256'] = '0' * 64
    elif changed == 'output-hash':
        records[2]['output_reads'][0]['sha256'] = '0' * 64
    elif changed == 'missing-read':
        records[0]['input_reads'].pop(1)
    elif changed == 'foreign-owner':
        records[2]['owner_binds'][0]['producer']['run'] = 0
    elif changed == 'wrong-terminal':
        raw['core']['materialization_entries'][0]['terminal'] = {'run': 1}
    else:
        records[1]['events'].reverse()
    before = copy.deepcopy(raw)
    with pytest.raises(REFUSALS):
        prepare(raw)
    assert raw == before


@pytest.mark.parametrize('stored', ['view', 'raw_view', 'validated'])
def test_caller_supplied_authority_cannot_rescue_corrupt_original(stored):
    raw = materialization()
    forged = copy.deepcopy(prepare(raw).view)
    raw['core']['materialization_preparations'][2]['output_reads'].clear()
    raw[stored] = True if stored == 'validated' else forged
    with pytest.raises(REFUSALS):
        prepare(raw)


@pytest.mark.parametrize(
    ('serving', 'retries'), [(False, 1), (True, 1), (True, 2)]
)
def test_original_model_bytes_and_received_retries_are_conserved_first(
    serving, retries
):
    raw = prediction(serving=serving, retries=retries)
    before = copy.deepcopy(raw)
    prepared = prepare(raw)
    assert prepared.raw == before
    original = reader().view(raw['core'], raw['identity']['operations'])
    reader().check_receipts(original['attempts'], raw['public']['payloads'])
    assert prepared.raw_view == original
    assert prepared.view == RUNNER.Placeheld(NAMES, NAMES.values()).value(
        original
    )
    assert len(prepared.raw_view['attempts']) == 1
    assert len(prepared.raw['public']['payloads']) == retries
    assert (
        prepared.raw_view['events'][0]['result']
        == original['events'][0]['result']
    )
    call = raw['core']['predictions'][0]['calls'][0]
    assert NAMES['catalog'].encode() in base64.b64decode(
        call['input_body_base64']
    )
    if serving:
        assert call['input_body_sha256'] != call['body_sha256']
    assert raw == before


@pytest.mark.parametrize(
    'changed',
    [
        'input-bytes',
        'parsed',
        'wire-hash',
        'receipt-hash',
        'receipt-body',
        'missing',
        'call-owner',
        'missing-call',
        'scope',
    ],
)
def test_mapping_never_authenticates_corrupt_actual_model_or_receipt_evidence(
    changed,
):
    raw = prediction(serving=True, retries=2)
    call = raw['core']['predictions'][0]['calls'][0]
    if changed == 'input-bytes':
        call['input_body_base64'] = base64.b64encode(b'{}').decode()
    elif changed == 'parsed':
        call['body']['dataframe_records'][0]['request_json'] += ' '
    elif changed == 'wire-hash':
        call['body_sha256'] = '0' * 64
    elif changed == 'receipt-hash':
        raw['public']['payloads'][1]['sha256'] = '0' * 64
    elif changed == 'receipt-body':
        raw['public']['payloads'][1]['parsed'] = {}
    elif changed == 'missing':
        raw['public']['payloads'].clear()
    elif changed == 'call-owner':
        call['run'] = 2
    elif changed == 'missing-call':
        raw['core']['predictions'][0]['calls'].clear()
    else:
        call['scope']['call_request_id'] = 'another-owner'
    with pytest.raises(REFUSALS):
        prepare(raw)


def test_namespace_mapping_preserves_complete_result_and_non_name_differences():
    raw = prediction()
    other_names = {key: 'other_' + value for key, value in NAMES.items()}
    other = prediction(names=other_names)
    left, right = prepare(raw), prepare(other, other_names)
    report = COMPARE.Report()
    COMPARE._core_agree(report, {'id': raw['case']}, left, right)
    assert report.failures == [], report.texts
    changed = copy.deepcopy(other)
    result = changed['core']['predictions'][0]['results'][0]['result']
    result['rows'].reverse()
    changed['core']['predicted']['result'] = copy.deepcopy(result)
    report = COMPARE.Report()
    COMPARE._core_agree(
        report, {'id': raw['case']}, left, prepare(changed, other_names)
    )
    assert report.failures
    changed = materialization()
    payload = copy.deepcopy(
        changed['core']['materialization_preparations'][2]['output_reads'][0][
            'parsed'
        ]
    )
    payload['context'][NAMES['shop_metric_view']]['rows'][1][0] = 3
    changed['core']['materialization_preparations'][2]['output_reads'][0] = (
        encoded(11, payload)
    )
    report = COMPARE.Report()
    COMPARE._core_agree(
        report,
        {'id': raw['case']},
        prepare(materialization()),
        prepare(changed),
    )
    assert report.failures


@pytest.mark.parametrize('factory', [materialization, prediction])
def test_public_derivative_is_distinct_safe_provenance_and_never_raw_evidence(
    factory,
):
    raw = factory()
    prepared = prepare(raw)
    derivative = publication().public_derivative(
        prepared, source_sha256='a' * 64, line=2
    )
    assert set(derivative) == {
        'format',
        'source',
        'observation',
        'core_summary',
    }
    assert derivative['format'] == 'sdm-published-observation/1'
    assert derivative['source'] == {
        'sha256': 'a' * 64,
        'line': 2,
        'case': raw['case'],
    }
    assert 'core' not in derivative['observation']
    assert derivative['observation'] == {
        key: value for key, value in prepared.items() if key != 'core'
    }
    assert derivative['core_summary'] == {
        'operations': prepared.view['operations'],
        'materialized': prepared.view['materialized'],
        'error': None,
    }
    text = json.dumps(derivative)
    assert all(name not in text for name in NAMES.values())
    assert (
        'input_body_base64' not in text
        and 'materialization_preparations' not in text
    )
    assert 'owner_binds' not in text and 'value_reads' not in text
    assert all(
        snapshot['base64'] not in text
        for record in raw['core'].get('materialization_preparations', [])
        for snapshot in [
            record['input'],
            *record['value_reads'],
            *record['output_reads'],
        ]
    )
    with pytest.raises(REFUSALS):
        prepare(derivative)
    assert (
        RUNNER._invalid(RUNNER.validator('observation'), derivative) is not None
    )


def test_summary_keeps_only_actual_selected_error_kind_and_code():
    raw = observation(
        prediction_materialization_failure(),
        ['prepare_request', 'prepare_materialize'],
    )
    raw['outcome'] = {
        'result': 'refusal',
        'stage': 'materialize',
        'error': 'StructuredDataError',
        'code': 'PROVIDER_ERROR',
    }
    derivative = publication().public_derivative(
        prepare(raw), source_sha256='b' * 64, line=1
    )
    assert derivative['core_summary']['error'] == {
        'kind': 'data_source',
        'code': 'PROVIDER_ERROR',
    }
    assert 'predictions' not in derivative['core_summary']
    assert 'materialization_preparations' not in derivative['core_summary']


def environment(tmp_path):
    config = tmp_path / 'live.json'
    config.write_text(
        json.dumps({'format': 'sdm-live/1', 'legs': {DATABASE: {'driver': {}}}})
    )
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if key not in RUNNER.PRIVATE_ENVIRONMENT
            and not key.startswith('SDM_CAPTURE_')
        },
        'USER': 'fixture_publication_operator_9ac1',
        'LOGNAME': 'fixture_publication_operator_9ac1',
        'SDM_LIVE_CONFIG': str(config),
        **{'SDM_CAPTURE_' + key.upper(): value for key, value in NAMES.items()},
    }


def test_publication_api_requires_and_binds_real_private_file_bytes(
    tmp_path, monkeypatch
):
    raw = [materialization(), prediction(serving=True, retries=2)]
    raw[1]['case'] = 't45.private.second@databricks-sql'
    source = tmp_path / 'observations.private.jsonl'
    # Whitespace and trailing newline belong to the file digest, not a DTO dump.
    original = (
        '\n'.join(
            json.dumps(value, indent=None, separators=(', ', ': '))
            for value in raw
        )
        + '\n'
    ).encode()
    source.write_bytes(original)
    replay = tmp_path / 'fetches.private.jsonl'
    replay.write_bytes(b'{ "source_namespace" : "private_catalog_t45" }\n')
    replay_bytes = replay.read_bytes()
    monkeypatch.setattr(
        RUNNER,
        'live_configuration',
        lambda database: ({}, NAMES, list(NAMES.values())),
    )
    with pytest.raises(REFUSALS):
        RUNNER.publish_observations(raw, DATABASE)
    published = RUNNER.publish_observations(
        raw, DATABASE, private_source=source
    )
    digest = hashlib.sha256(original).hexdigest()
    assert [value['source'] for value in published] == [
        {'sha256': digest, 'line': at, 'case': value['case']}
        for at, value in enumerate(raw, 1)
    ]
    assert (
        source.read_bytes() == original and replay.read_bytes() == replay_bytes
    )
    mismatch = copy.deepcopy(raw)
    mismatch[1]['public']['payloads'].pop()
    # Even a semantically legal retry subset is a different source DTO.
    with pytest.raises(REFUSALS):
        RUNNER.publish_observations(mismatch, DATABASE, private_source=source)
    with pytest.raises(REFUSALS):
        RUNNER.publish_observations(raw[::-1], DATABASE, private_source=source)


def test_publication_cli_keeps_original_and_replay_bytes_and_refuses_overwrite(
    tmp_path,
):
    raw = prediction(serving=True, retries=2)
    source, target = (
        tmp_path / 'raw.private.jsonl',
        tmp_path / 'published.jsonl',
    )
    original = (json.dumps(raw, separators=(', ', ': ')) + '\n').encode()
    source.write_bytes(original)
    replay = tmp_path / 'fetches.private.jsonl'
    replay.write_bytes(b'{"untouched":"private_schema_t45"}\n')
    replay_bytes = replay.read_bytes()
    arguments = [
        sys.executable,
        '-I',
        str(ROOT / 'contract/runners/run_python.py'),
        'publish',
        '--database',
        DATABASE,
        '--on',
        str(source),
        '--out',
        str(target),
    ]
    done = subprocess.run(
        arguments,
        env=environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    derivative = json.loads(target.read_text())
    assert derivative['format'] == 'sdm-published-observation/1'
    assert derivative['source'] == {
        'sha256': hashlib.sha256(original).hexdigest(),
        'line': 1,
        'case': raw['case'],
    }
    assert all(name not in target.read_text() for name in NAMES.values())
    assert (
        source.read_bytes() == original and replay.read_bytes() == replay_bytes
    )
    written = target.read_bytes()
    refused = subprocess.run(
        arguments,
        env=environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert refused.returncode == 1
    assert target.read_bytes() == written and source.read_bytes() == original


def test_publication_cli_checks_decoded_nested_json_for_unmapped_private_text(
    tmp_path,
):
    private = 'fixture_publication_operator_9ac1'
    raw = prediction(serving=True, names={**NAMES, 'catalog': private})
    call = raw['core']['predictions'][0]['calls'][0]
    inner = call['body']['dataframe_records'][0]['request_json']
    escaped = ''.join(f'\\u{ord(character):04x}' for character in private)
    posted = json.dumps(
        {
            'dataframe_records': [
                {'request_json': inner.replace(private, escaped)}
            ]
        }
    ).encode()
    receipt = RUNNER._payload(posted)
    call.update(body=receipt['parsed'], body_sha256=receipt['sha256'])
    raw['core']['predicted']['body'] = copy.deepcopy(call['body'])
    raw['public']['payloads'] = [
        {
            'method': 'POST',
            'path': '/serving-endpoints/fixture/invocations',
            **receipt,
        }
    ]
    # The unescaped private text exists only inside encoded actual evidence;
    # the physical receipt holds escaped nested JSON, which is also private.
    assert private not in json.dumps(raw)
    reader().check_receipts(
        reader().view(raw['core'], raw['identity']['operations'])['attempts'],
        raw['public']['payloads'],
    )
    source, target = (
        tmp_path / 'encoded.private.jsonl',
        tmp_path / 'unsafe.jsonl',
    )
    source.write_text(json.dumps(raw) + '\n')
    original = source.read_bytes()
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            'publish',
            '--database',
            DATABASE,
            '--on',
            str(source),
            '--out',
            str(target),
        ],
        env=environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 1
    assert private not in done.stdout + done.stderr
    assert source.read_bytes() == original


def name_map(tmp_path, source, **changes):
    path = tmp_path / 'names.private.json'
    path.write_text(
        json.dumps(
            {
                'format': 'sdm-private-name-map/1',
                'database': DATABASE,
                'names': NAMES,
                'source_sha256': hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                **changes,
            }
        )
    )
    return path


def test_compare_api_and_cli_accept_only_the_exact_hash_bound_private_source(
    tmp_path,
):
    pair = T32.Pair(tmp_path, DATABASE)
    source = pair.write(pair.answered(None))
    source_bytes, reference_bytes = (
        source.read_bytes(),
        pair.reference.read_bytes(),
    )
    mapping = name_map(tmp_path, source)
    report = COMPARE.compare(
        [pair.cases],
        pair.reference,
        [source],
        database=DATABASE,
        private_name_map=mapping,
    )
    assert report.failures == [], report.texts
    command = [
        sys.executable,
        '-I',
        str(ROOT / 'scripts/rust_core/compare.py'),
        '--cases',
        str(pair.cases),
        '--reference',
        str(pair.reference),
        '--database',
        DATABASE,
        '--private-name-map',
        str(mapping),
        str(source),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    # Same decoded observation, different actual file bytes: stale binding.
    source.write_bytes(b' ' + source_bytes)
    report = COMPARE.compare(
        [pair.cases],
        pair.reference,
        [source],
        database=DATABASE,
        private_name_map=mapping,
    )
    assert report.failures
    refused = subprocess.run(
        command, capture_output=True, text=True, timeout=60
    )
    assert refused.returncode == 1
    assert pair.reference.read_bytes() == reference_bytes
    assert source.read_bytes() == b' ' + source_bytes


def test_private_map_never_remaps_the_reference_file(tmp_path):
    pair = T32.Pair(tmp_path, DATABASE)
    raw = pair.answered(None)
    # Table names are an existing decoded public field, not wire operands.
    raw['public']['schema']['tables'][0]['name'] = NAMES['catalog']
    reference = json.loads(pair.reference.read_text())
    reference['public']['schema']['tables'][0]['name'] = NAMES['catalog']
    pair.reference.write_text(json.dumps(reference) + '\n')
    source = pair.write(raw)
    mapping = name_map(tmp_path, source)
    before = pair.reference.read_bytes()
    plain = COMPARE.compare(
        [pair.cases], pair.reference, [source], database=DATABASE
    )
    assert plain.failures == [], plain.texts
    mapped = COMPARE.compare(
        [pair.cases],
        pair.reference,
        [source],
        database=DATABASE,
        private_name_map=mapping,
    )
    assert 'schema' in [rule for rule, _ in mapped.failures]
    assert pair.reference.read_bytes() == before
    # A separately authored placeholder expectation now agrees; no frozen
    # recording or observed body/hash was rewritten to construct it.
    reference['public']['schema']['tables'][0]['name'] = '{catalog}'
    pair.reference.write_text(json.dumps(reference) + '\n')
    mapped = COMPARE.compare(
        [pair.cases],
        pair.reference,
        [source],
        database=DATABASE,
        private_name_map=mapping,
    )
    assert mapped.failures == [], mapped.texts


def test_byte_free_historical_publication_keeps_existing_ordinary_contract(
    tmp_path, monkeypatch
):
    pair = T32.Pair(tmp_path, DATABASE)
    raw = pair.answered(None)
    original = copy.deepcopy(raw)
    monkeypatch.setattr(
        RUNNER,
        'live_configuration',
        lambda database: ({}, NAMES, list(NAMES.values())),
    )
    assert RUNNER.publish_observations([raw], DATABASE) == [
        RUNNER.Placeheld(NAMES, NAMES.values()).value(raw)
    ]
    assert raw == original


@pytest.mark.parametrize(
    'changed', ['format', 'database', 'hash', 'extra', 'collision']
)
def test_private_map_is_closed_and_leg_and_source_specific(tmp_path, changed):
    pair = T32.Pair(tmp_path, DATABASE)
    source = pair.write(pair.answered(None))
    changes = {
        'format': {'format': 'anything'},
        'database': {'database': 'snowflake'},
        'hash': {'source_sha256': '0' * 64},
        'extra': {'validated': True},
        'collision': {
            'names': {
                'catalog': 'same_private_name',
                'schema': 'same_private_name',
            }
        },
    }[changed]
    report = COMPARE.compare(
        [pair.cases],
        pair.reference,
        [source],
        database=DATABASE,
        private_name_map=name_map(tmp_path, source, **changes),
    )
    assert report.failures


def test_serving_unicode_escapes_are_projected_only_in_decoded_publication():
    raw = prediction(serving=True)
    call = raw['core']['predictions'][0]['calls'][0]
    logical = call['body']['dataframe_records'][0]['request_json']
    private = NAMES['catalog']
    escaped = ''.join(f'\\u{ord(character):04x}' for character in private)
    posted = json.dumps(
        {
            'dataframe_records': [
                {'request_json': logical.replace(private, escaped)}
            ]
        }
    ).encode()
    receipt = RUNNER._payload(posted)
    call.update(body=receipt['parsed'], body_sha256=receipt['sha256'])
    raw['core']['predicted']['body'] = copy.deepcopy(call['body'])
    raw['public']['payloads'] = [
        {
            'method': 'POST',
            'path': '/serving-endpoints/fixture/invocations',
            **receipt,
        }
    ]
    original = copy.deepcopy(raw)
    prepared = prepare(raw)
    derivative = publication().public_derivative(
        prepared,
        source_sha256='c' * 64,
        line=1,
    )
    body = derivative['observation']['public']['payloads'][0]['parsed']
    decoded = json.loads(body['dataframe_records'][0]['request_json'])
    assert private not in json.dumps(decoded)
    assert '{catalog}' in json.dumps(decoded)
    assert prepared.raw == original and raw == original
    reader().check_receipts(
        prepared.raw_view['attempts'],
        raw['public']['payloads'],
    )


def precise_serving(token='1.0000000000000001', *, old=False):
    raw = prediction(serving=True)
    call = raw['core']['predictions'][0]['calls'][0]
    private = NAMES['catalog']
    logical = (
        '{"model":"kumo-relational","metadata":{"request_id":'
        + json.dumps(call['request_id'])
        + ',"source_catalog":'
        + json.dumps(private)
        + '},"precise":'
        + token
        + ',"integer":1,"float":1.0,"zero":-0.0,'
        '"duplicate":1,"duplicate":2,"object":{"number":"1"},'
        '"text":"1.0000000000000001","values":[true,false,null,1,1.0,-0.0,"😺"]}'
    )
    escaped = ''.join(f'\\u{ord(character):04x}' for character in private)
    posted = json.dumps(
        {
            'dataframe_records': [
                {'request_json': logical.replace(private, escaped)}
            ]
        }
    ).encode()
    receipt = RUNNER._payload(posted)
    encoded = logical.encode()
    call.update(
        input_body_base64=base64.b64encode(encoded).decode('ascii'),
        input_body_sha256=hashlib.sha256(encoded).hexdigest(),
        body=copy.deepcopy(receipt['parsed']),
        body_sha256=receipt['sha256'],
    )
    raw['core']['predicted']['body'] = copy.deepcopy(call['body'])
    raw['public']['payloads'] = [
        {
            'method': 'POST',
            'path': '/serving-endpoints/fixture/invocations',
            **receipt,
        }
    ]
    if old:
        raw['core'] = None
        raw['identity'].update(
            runner='python-old', native=None, native_loaded=False, operations=[]
        )
    return raw, logical.replace(private, '{catalog}')


def ordered_numbers(text):
    return json.loads(
        text,
        parse_int=lambda token: ('integer', token),
        parse_float=lambda token: ('float', token),
        object_pairs_hook=lambda pairs: ('members', pairs),
    )


def test_serving_member_collision_is_detected_before_literal_name_replacement():
    held = RUNNER.Placeheld(NAMES, ())
    private = NAMES['catalog']
    text = '{' + json.dumps(private) + ':1,"{catalog}":2}'
    original = {'dataframe_records': [{'request_json': text}]}
    before = copy.deepcopy(original)
    with pytest.raises(publication().PublicationError, match='collide'):
        publication()._project(held, original)
    assert original == before


@pytest.mark.parametrize(
    'old', [False, True], ids=['gathered', 'old-without-core']
)
def test_escaped_serving_name_projection_preserves_all_number_tokens_and_members(
    old,
):
    raw, expected = precise_serving(old=old)
    original = copy.deepcopy(raw)
    prepared = prepare(raw)
    shown = prepared['public']['payloads'][0]['parsed']['dataframe_records'][0][
        'request_json'
    ]
    assert ordered_numbers(shown) == ordered_numbers(expected)
    derivative = publication().public_derivative(
        prepared, source_sha256='d' * 64, line=1
    )
    published = derivative['observation']['public']['payloads'][0]['parsed'][
        'dataframe_records'
    ][0]['request_json']
    assert ordered_numbers(published) == ordered_numbers(expected)
    assert prepared.raw == raw == original
    if not old:
        reader().check_receipts(
            prepared.raw_view['attempts'], raw['public']['payloads']
        )


@pytest.mark.parametrize(
    'old', [False, True], ids=['gathered', 'old-without-core']
)
@pytest.mark.parametrize(
    'token', ['1.0000000000000002', '1', '1.0', '-0.0', '0.0']
)
def test_names_only_projection_cannot_hide_a_non_name_serving_number_difference(
    old, token
):
    first, _ = precise_serving(old=old)
    other, _ = precise_serving(token, old=old)
    before = (copy.deepcopy(first), copy.deepcopy(other))
    left, right = prepare(first), prepare(other)
    report = COMPARE.Report()
    COMPARE._host_fields(
        report,
        {'id': first['case']},
        'python-core',
        left,
        right,
        against='python-old',
    )
    assert [rule for rule, _ in report.failures] == ['payload']
    assert (first, other) == before


def test_each_private_input_file_uses_its_own_hash_bound_map(tmp_path):
    pair = T32.Pair(tmp_path, DATABASE)
    core = pair.answered(None)
    source = pair.write(core)
    old = json.loads(pair.reference.read_text())
    old['identity'].update(
        runner='python-old',
        reference=None,
        native=None,
        native_loaded=False,
        operations=[],
        contract=None,
    )
    old['core'] = None
    second = tmp_path / 'python-old.private.jsonl'
    second.write_text(json.dumps(old) + '\n')
    first_map = name_map(tmp_path, source)
    other_map = tmp_path / 'old-names.private.json'
    other_map.write_text(
        json.dumps(
            {
                'format': 'sdm-private-name-map/1',
                'database': DATABASE,
                'names': NAMES,
                'source_sha256': hashlib.sha256(
                    second.read_bytes()
                ).hexdigest(),
            }
        )
    )
    originals = {
        path: path.read_bytes() for path in (source, second, pair.reference)
    }
    report = COMPARE.compare(
        [pair.cases],
        pair.reference,
        [source, second],
        database=DATABASE,
        private_name_map=[first_map, other_map],
    )
    assert report.failures == [], report.texts
    command = [
        sys.executable,
        '-I',
        str(ROOT / 'scripts/rust_core/compare.py'),
        '--cases',
        str(pair.cases),
        '--reference',
        str(pair.reference),
        '--database',
        DATABASE,
        '--private-name-map',
        str(first_map),
        '--private-name-map',
        str(other_map),
        str(source),
        str(second),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    for maps in (
        [first_map],
        [first_map, first_map],
        [first_map, other_map, other_map],
    ):
        refused = COMPARE.compare(
            [pair.cases],
            pair.reference,
            [source, second],
            database=DATABASE,
            private_name_map=maps,
        )
        assert refused.failures
    assert all(path.read_bytes() == data for path, data in originals.items())
