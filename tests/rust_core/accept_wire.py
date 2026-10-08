# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P4.wire: the Rust wire types are the pinned contract's, and every golden
and corpus payload goes through them unchanged.

- The generator's check mode passes on the committed types and the pinned
  contract, fails on a hand-edited file and on a changed contract, and the
  generator writes back exactly what is committed.
- Its record names the same contract revision and bytes as sdm-api's.
- Every contract schema is a Rust type.
- The wire crate's own tests fail when a golden or corpus payload does not
  round-trip.
- Through the binding, every golden and corpus payload, and every request
  example in the contract, comes back parsed-equal, by value and JSON type;
  extra keys survive where the contract allows them; and what is not a
  request is refused.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from _rc import (
    GOLDEN_PAYLOADS,
    REPO,
    cargo_metadata,
    contract_clone,
    copy_repo,
    corpus_payloads,
    find_crate,
    golden_payloads,
    install_local_wheel,
    json_diff,
    load_spec,
    run,
    sdm_api_provenance,
    sha256,
    transcript_files,
)

# What the corpus held when these checks were written; it may only grow.
MIN_GOLDEN = 9
MIN_CORPUS = 57


@pytest.fixture(scope='module')
def spec_path(tmp_path_factory) -> Path:
    return contract_clone(tmp_path_factory.mktemp('contract'))


@pytest.fixture(scope='module')
def spec(spec_path) -> dict[str, Any]:
    return load_spec(spec_path)


def rust_records(root: Path = REPO) -> list[Path]:
    rust = root / 'rust'
    return sorted(
        p
        for p in rust.glob('**/PROVENANCE.json')
        if 'target' not in p.relative_to(rust).parts
    )


@pytest.fixture(scope='module')
def record_path() -> Path:
    found = rust_records()
    assert len(found) == 1, f'expected one PROVENANCE.json in rust/: {found}'
    return found[0]


@pytest.fixture(scope='module')
def record(record_path) -> dict[str, Any]:
    return json.loads(record_path.read_text())


@pytest.fixture(scope='module')
def generated(record_path, record) -> Path:
    return record_path.parent / record['generated_file']


def generate(root: Path, generator: str, spec: Path, *extra: str):
    return run(
        [sys.executable, generator, '--spec', spec, *extra],
        cwd=root,
        check=False,
    )


# --- Provenance and check mode -------------------------------------------------


def test_the_record_matches_sdm_api_and_the_bytes(record, generated):
    pinned = sdm_api_provenance()
    for key in (
        'source_path',
        'source_repository',
        'source_revision',
        'source_sha256',
    ):
        assert record.get(key) == pinned[key], (
            f'{key} is {record.get(key)!r}, sdm-api has {pinned[key]!r}'
        )
    assert generated.is_file(), f'{generated} is missing'
    assert record['generated_sha256'] == sha256(generated)
    generator = REPO / record['generator']
    assert generator.is_file(), f'{generator} is missing'
    assert record['generator_sha256'] == sha256(generator)


def test_the_generated_file_is_in_the_wire_crate(generated):
    wire = find_crate('wire', cargo_metadata())
    root = Path(wire['manifest_path']).parent.resolve()
    assert generated.resolve().is_relative_to(root), (
        f'{generated} is outside the wire crate at {root}'
    )


def test_check_mode_passes_on_the_committed_types(record, spec_path):
    proc = generate(REPO, record['generator'], spec_path, '--check')
    assert proc.returncode == 0, proc.stderr[-4000:]


def test_every_contract_schema_is_a_rust_type(spec, generated):
    text = generated.read_text()
    missing = [
        name
        for name in spec['components']['schemas']
        if not re.search(
            rf'\bpub\s+(struct|enum|type)\s+{re.escape(name)}\b', text
        )
    ]
    assert not missing, f'no Rust type for {missing}'


def test_the_generator_writes_back_what_is_committed(
    record, record_path, generated, spec_path, tmp_path
):
    root = copy_repo(tmp_path / 'repo')
    proc = generate(root, record['generator'], spec_path)
    assert proc.returncode == 0, proc.stderr[-4000:]
    for path in (generated, record_path):
        rel = path.relative_to(REPO)
        assert (root / rel).read_bytes() == path.read_bytes(), (
            f'regenerating changed {rel}'
        )


def test_check_mode_fails_on_a_hand_edited_file(
    record, generated, spec_path, tmp_path
):
    root = copy_repo(tmp_path / 'repo')
    target = root / generated.relative_to(REPO)
    target.write_text(target.read_text() + '\n// edited by hand\n')
    proc = generate(root, record['generator'], spec_path, '--check')
    assert proc.returncode != 0, 'check mode passed a hand-edited file'


def test_check_mode_fails_on_a_changed_contract_and_follows_it(
    record, generated, tmp_path
):
    r"""A contract with one more schema: check fails, and writing emits it."""
    spec_path = contract_clone(tmp_path / 'contract')
    changed = load_spec(spec_path)
    changed['components']['schemas']['AcceptanceProbe'] = {
        'type': 'object',
        'required': ['probe_field'],
        'properties': {'probe_field': {'type': 'string'}},
        'additionalProperties': False,
    }
    request = changed['components']['schemas']['PredictionRequest']
    request['properties']['acceptance_probe'] = {
        '$ref': '#/components/schemas/AcceptanceProbe'
    }
    # The clone is the pinned revision; the change is a working-tree edit, so
    # the generator still reads the revision from it.
    spec_path.write_text(yaml.safe_dump(changed, sort_keys=False))
    root = copy_repo(tmp_path / 'repo')
    proc = generate(root, record['generator'], spec_path, '--check')
    assert proc.returncode != 0, 'check mode passed a changed contract'
    proc = generate(root, record['generator'], spec_path)
    assert proc.returncode == 0, proc.stderr[-4000:]
    text = (root / generated.relative_to(REPO)).read_text()
    assert re.search(r'\bpub\s+struct\s+AcceptanceProbe\b', text)
    assert 'probe_field' in text and 'acceptance_probe' in text
    proc = generate(root, record['generator'], spec_path, '--check')
    assert proc.returncode == 0, 'check mode failed right after writing'


# --- The wire crate's own round-trip tests --------------------------------------


@pytest.fixture(scope='module')
def wire_copy(tmp_path_factory) -> Path:
    r"""One copy of the repository for the wire crate's tests to run in.

    It builds into its own ``target``, so a test binary can only ever read
    this copy's files: a target shared with another copy could keep a binary
    built there, and with it that copy's paths.
    """
    return copy_repo(tmp_path_factory.mktemp('wire-copy') / 'repo')


def _wire_tests(root: Path) -> Any:
    wire = find_crate('wire', cargo_metadata(root / 'rust'))
    env = {k: v for k, v in os.environ.items() if k != 'CARGO_TARGET_DIR'}
    return run(
        ['cargo', 'test', '-p', wire['name']],
        cwd=root / 'rust',
        env=env,
        check=False,
    )


def _with_unexpected_key(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, 'acceptance_unexpected_key': 1}


@contextlib.contextmanager
def _changed(path: Path, change) -> Iterator[None]:
    original = path.read_bytes()
    data = json.loads(original)
    change(data)
    path.write_text(json.dumps(data, indent=2) + '\n')
    try:
        yield
    finally:
        path.write_bytes(original)


def test_the_wire_tests_pass_on_a_copy(wire_copy):
    proc = _wire_tests(wire_copy)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    ran = sum(int(n) for n in re.findall(r'running (\d+) tests?', proc.stdout))
    assert ran > 0, 'the wire crate has no tests'


def test_the_wire_tests_catch_a_bad_golden_payload(wire_copy):
    def change(data: dict[str, Any]) -> None:
        last = list(data)[-1]
        data[last][-1] = _with_unexpected_key(data[last][-1])

    path = wire_copy / GOLDEN_PAYLOADS.relative_to(REPO)
    with _changed(path, change):
        proc = _wire_tests(wire_copy)
    assert proc.returncode != 0, (
        'the wire tests passed with a golden payload that cannot round-trip'
    )


def test_the_wire_tests_catch_a_bad_corpus_payload(wire_copy):
    def change(transcript: dict[str, Any]) -> None:
        assert transcript['payloads'], 'the last transcript has no payloads'
        transcript['payloads'][-1] = _with_unexpected_key(
            transcript['payloads'][-1]
        )

    path = wire_copy / transcript_files()[-1].relative_to(REPO)
    with _changed(path, change):
        proc = _wire_tests(wire_copy)
    assert proc.returncode != 0, (
        'the wire tests passed with a corpus payload that cannot round-trip'
    )


# --- Through the binding --------------------------------------------------------


@pytest.fixture(scope='module')
def native():
    install_local_wheel()
    sys.modules.pop('sdm_core', None)
    sys.modules.pop('sdm_core._native', None)
    from sdm_core import _native

    return _native


def roundtrip(native, payload: Any) -> Any:
    out = native.roundtrip_payload(json.dumps(payload))
    assert isinstance(out, str), f'roundtrip_payload gave {type(out)}'
    return json.loads(out)


def test_the_corpus_is_all_there():
    assert len(golden_payloads()) >= MIN_GOLDEN
    assert len(corpus_payloads()) >= MIN_CORPUS


@pytest.mark.parametrize(
    ('name', 'payload'),
    [
        pytest.param(name, payload, id=name)
        for name, payload in golden_payloads() + corpus_payloads()
    ],
)
def test_every_golden_and_corpus_payload_round_trips(native, name, payload):
    diff = json_diff(payload, roundtrip(native, payload))
    assert diff is None, f'{name}: {diff}'


def contract_examples(spec: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    schema = spec['components']['schemas']['PredictionRequest']
    if 'example' in schema:
        yield 'components.PredictionRequest.example', schema['example']
    body = spec['paths']['/v1/predictions']['post']['requestBody']
    content = body['content']['application/json']
    for name, example in content.get('examples', {}).items():
        yield f'/v1/predictions:{name}', example['value']


def test_every_contract_request_example_round_trips(native, spec):
    examples = list(contract_examples(spec))
    assert len(examples) >= 5
    for name, payload in examples:
        diff = json_diff(payload, roundtrip(native, payload))
        assert diff is None, f'{name}: {diff}'


def _base() -> dict[str, Any]:
    return copy.deepcopy(golden_payloads()[0][1])


OPEN_OBJECTS = {
    'task': lambda p: p['task'],
    'task.target': lambda p: p['task']['target'],
    'column spec': lambda p: next(
        iter(p['schema']['instance_table']['columns'].values())
    ),
    'output': lambda p: p['output'],
    'inference': lambda p: p.setdefault('inference', {}),
    'metadata': lambda p: p.setdefault('metadata', {}),
    'split metadata': lambda p: p['context'].setdefault('metadata', {}),
}


@pytest.mark.parametrize('where', sorted(OPEN_OBJECTS))
def test_open_objects_keep_their_extra_keys(native, where):
    payload = _base()
    OPEN_OBJECTS[where](payload).update(
        {
            'acceptance_extra': {'n': 1, 'x': 1.5, 'ok': True, 'none': None},
            'acceptance_list': [1, 'a', [2.0]],
        }
    )
    diff = json_diff(payload, roundtrip(native, payload))
    assert diff is None, f'{where}: {diff}'


def _drop(key: str):
    def change(p: dict[str, Any]) -> None:
        del p[key]

    return change


def _set(path: tuple[str, ...], value: Any):
    def change(p: dict[str, Any]) -> None:
        node = p
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return change


NOT_REQUESTS = {
    **{
        f'missing {key}': _drop(key)
        for key in ('model', 'task', 'schema', 'context', 'predict', 'output')
    },
    'unknown top-level key': _set(('acceptance_unexpected',), 1),
    'model outside the enum': _set(('model',), 'kumo-other'),
    'task kind outside the enum': _set(('task', 'kind'), 'clustering'),
    'unknown table format': _set(
        ('context', 'instance_table', 'format'), 'parquet'
    ),
    'unknown key in arrays table data': _set(
        ('predict', 'instance_table', 'acceptance_unexpected'), 1
    ),
    'unknown output field': _set(('output', 'fields'), ['bogus']),
    'primary key of the wrong type': _set(
        ('schema', 'instance_table', 'primary_key'), 5
    ),
    'rows of the wrong type': _set(
        ('context', 'instance_table', 'rows'), 'rows'
    ),
    'unknown key in a relationship': _set(
        ('schema', 'relationships'),
        [
            {
                'source_columns': ['a'],
                'target_table': 't',
                'target_columns': ['b'],
                'acceptance_unexpected': 1,
            }
        ],
    ),
}


@pytest.mark.parametrize('case', sorted(NOT_REQUESTS))
def test_what_is_not_a_request_is_refused(native, case):
    payload = _base()
    NOT_REQUESTS[case](payload)
    with pytest.raises(ValueError):
        native.roundtrip_payload(json.dumps(payload))


@pytest.mark.parametrize('text', ['', '{', '[]', '"request"', 'null'])
def test_text_that_is_not_a_json_object_is_refused(native, text):
    with pytest.raises(ValueError):
        native.roundtrip_payload(text)
