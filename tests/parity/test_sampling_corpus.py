# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the ordinary corpus hook against real sampling recordings."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'scripts/parity'
sys.path.insert(0, str(SCRIPTS))

from _parity import sampling  # noqa: E402


def command(*arguments):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / 'run_corpus.py'), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def recording(tmp_path):
    source = sampling.stored(sampling.RECORDINGS)['duckdb/forecast']
    target = tmp_path / 'corpus' / 'selected'
    shutil.copytree(source, target)
    return target


def test_sampling_listing_classifies_all_boundary_recordings():
    process = command('list', '--kind', 'sampling', '--json')
    assert process.returncode == 0, process.stderr
    entries = json.loads(process.stdout)
    assert len(entries) == len(sampling.driver().recording_dirs())
    assert entries
    assert sum(entry['calls'] for entry in entries) > 0
    assert all(entry['kind'] == 'sampling' for entry in entries)
    assert len({entry['name'] for entry in entries}) == len(entries)


def test_sampling_cli_replays_actual_native_calls(recording):
    process = command(
        'replay',
        '--kind',
        'sampling',
        '--corpus',
        str(recording.parent),
        '--only',
        'duckdb/forecast',
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert process.stdout.strip() == 'PASS duckdb/forecast'
    assert '1 passed, 0 failed' in process.stderr


@pytest.mark.parametrize('mutation', ['output', 'sql', 'missing_arrow'])
def test_sampling_cli_rejects_real_recording_mutations(recording, mutation):
    path = recording / 'boundary.json'
    document = json.loads(path.read_bytes())
    call = document['calls'][0]
    if mutation == 'output':
        call['outputs'] = {
            'error': {'code': 'MUTATED', 'message': 'not actual'}
        }
    elif mutation == 'sql':
        call['steps'][0]['sql'] = 'SELECT 918273645'
    else:
        (recording / call['steps'][0]['result']).unlink()
    path.write_text(json.dumps(document))
    process = command(
        'replay', '--kind', 'sampling', '--corpus', str(recording.parent)
    )
    assert process.returncode != 0
    assert 'PASS duckdb/forecast' not in process.stdout


def test_sampling_selection_rejects_empty_missing_and_duplicate(
    recording, tmp_path
):
    empty = tmp_path / 'empty'
    empty.mkdir()
    process = command('replay', '--kind', 'sampling', '--corpus', str(empty))
    assert process.returncode != 0
    assert 'no sampling boundary recordings' in process.stderr
    for selection in ('missing', 'duckdb/forecast,duckdb/forecast'):
        process = command(
            'replay',
            '--kind',
            'sampling',
            '--corpus',
            str(recording.parent),
            '--only',
            selection,
        )
        assert process.returncode != 0


def test_sampling_classification_rejects_duplicate_recording_names(recording):
    shutil.copytree(recording, recording.parent / 'duplicate')
    with pytest.raises(ValueError, match='unique scenario names'):
        sampling.stored(recording.parent)


def test_coverage_producer_includes_sampling_runner():
    spec = importlib.util.spec_from_file_location(
        'parity_coverage', SCRIPTS / 'coverage.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = dict(module._runs())
    assert runs['sampling boundary replay'] == [
        str(SCRIPTS / 'run_corpus.py'),
        'replay',
        '--kind',
        'sampling',
    ]
