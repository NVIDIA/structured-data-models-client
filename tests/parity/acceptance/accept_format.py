# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P2.format: FORMAT.md exists, and every transcript validates against the
JSON Schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _support import (
    PARITY,
    SCHEMA,
    STEP_OPS,
    TRANSCRIPTS,
    arrow_refs,
    deep,
    execute_steps,
    load,
    read_arrow,
    transcript_files,
)


def _validator() -> Any:
    import jsonschema

    schema = load(SCHEMA)
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def test_format_document_names_the_version_and_every_step_kind() -> None:
    text = (PARITY / 'FORMAT.md').read_text()
    assert 'format_version' in text
    for op in STEP_OPS:
        assert f'`{op}`' in text, f'FORMAT.md does not document {op}'


def test_schema_declares_a_draft_and_is_itself_valid() -> None:
    assert '$schema' in load(SCHEMA)
    _validator()


def test_there_are_transcripts() -> None:
    assert transcript_files(), f'no transcripts under {TRANSCRIPTS}'


@pytest.mark.parametrize(
    'path',
    transcript_files() if TRANSCRIPTS.is_dir() else [],
    ids=lambda p: str(p.relative_to(TRANSCRIPTS)),
)
def test_every_transcript_validates(path: Path) -> None:
    transcript = load(path)
    errors = sorted(_validator().iter_errors(transcript), key=str)
    assert not errors, '\n'.join(e.message for e in errors[:5])
    assert transcript['format_version'] == 1
    for op in (step['op'] for step in transcript['steps']):
        assert op in STEP_OPS
    for index in execute_steps(transcript):
        assert isinstance(transcript['steps'][index]['sql'], str)
    assert isinstance(transcript['payloads'], list)
    assert isinstance(transcript['calls'], list)


@pytest.mark.parametrize(
    'path',
    transcript_files() if TRANSCRIPTS.is_dir() else [],
    ids=lambda p: str(p.relative_to(TRANSCRIPTS)),
)
def test_every_referenced_arrow_file_is_readable(path: Path) -> None:
    transcript = load(path)
    for ref in arrow_refs(transcript):
        target = (path.parent / ref).resolve()
        assert target.is_file(), f'{path.name} references missing {ref}'
        assert target.suffix == '.arrow', ref
        read_arrow(target)
    if transcript['payloads']:
        if transcript.get('result') is None and 'error' in transcript:
            # A run that ends in a refusal stores what it raised instead of a
            # result (FORMAT.md, `error`).
            assert isinstance(transcript['error'], dict), transcript['error']
            assert isinstance(transcript['error'].get('message'), str)
            return
        assert isinstance(transcript.get('result'), str), (
            'a recorded scenario stores its decoded result, or the error it '
            'ended in'
        )
        table, _ = read_arrow(path.parent / transcript['result'])
        assert table.num_rows > 0, 'the decoded result is empty'


def test_the_transcript_directory_holds_only_json_and_arrow() -> None:
    other = [
        p
        for p in TRANSCRIPTS.rglob('*')
        if p.is_file() and p.suffix not in ('.json', '.arrow')
    ]
    assert not other, other


def test_nothing_under_parity_is_python_specific_binary() -> None:
    bad = [
        p
        for p in PARITY.rglob('*')
        if p.suffix in ('.pkl', '.pickle', '.npy', '.npz', '.joblib')
    ]
    assert not bad, bad


def _a_transcript_with_an_execute_step() -> dict[str, Any]:
    for path in transcript_files():
        transcript = load(path)
        if execute_steps(transcript):
            return transcript
    raise AssertionError('no transcript has an execute step')


def _broken(kind: str) -> dict[str, Any]:
    transcript = deep(_a_transcript_with_an_execute_step())
    first = execute_steps(transcript)[0]
    if kind == 'version':
        transcript['format_version'] = 2
    elif kind == 'no-steps':
        del transcript['steps']
    elif kind == 'steps-not-array':
        transcript['steps'] = {'0': transcript['steps'][0]}
    elif kind == 'unknown-op':
        transcript['steps'][first]['op'] = 'vacuum'
    elif kind == 'sql-not-string':
        transcript['steps'][first]['sql'] = 123
    elif kind == 'no-payloads':
        del transcript['payloads']
    elif kind == 'no-name':
        del transcript['name']
    return transcript


@pytest.mark.parametrize(
    'kind',
    [
        'version',
        'no-steps',
        'steps-not-array',
        'unknown-op',
        'sql-not-string',
        'no-payloads',
        'no-name',
    ],
)
def test_the_schema_rejects_a_broken_transcript(kind: str) -> None:
    assert not _validator().is_valid(_broken(kind)), (
        f'the schema accepts a transcript with {kind}'
    )
