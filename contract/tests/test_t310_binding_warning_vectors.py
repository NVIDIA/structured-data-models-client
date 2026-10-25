# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The canonical Java binding recorder preserves warnings before validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _recorder():
    spec = importlib.util.spec_from_file_location(
        't310_binding_warning_recorder',
        ROOT / 'contract/tools/t311_java_record.py',
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_canonical_binding_writer_records_actual_escape_warning_controls(
    tmp_path, monkeypatch
):
    from sdm_client import _native

    recorder = _recorder()
    original = json.loads(recorder.OUT.read_text())
    output = tmp_path / 'vectors.json'
    output.write_text(json.dumps(original))
    monkeypatch.setattr(recorder, 'OUT', output)
    assert recorder.main(['binding']) == 0
    recorded = json.loads(output.read_text())
    assert recorded['reference'] == original['reference']
    vector = recorded['binding']['validate']['s0.boundary.escapes']
    operation = json.loads(vector['operation'])
    import base64

    attachments = {
        int(key): base64.b64decode(value)
        for key, value in vector['attachments'].items()
    }
    run = _native.Run.start(recorder._start(_native, operation), attachments)
    warnings = []
    while True:
        control, carried = run.next()
        message = json.loads(control)
        if message['message'] == 'attached':
            message = json.loads(carried[message['attachment']])
        if message['message'] != 'warnings':
            break
        warnings.append(
            {'digest': hashlib.sha256(control).hexdigest(), 'control': message}
        )
    assert vector['warnings'] == warnings
    assert (
        warnings
        and warnings[0]['control']['warnings'][0]['code']
        == 'INVALID_STRING_ESCAPE'
    )
    assert message['message'] == 'step' and message['step'] == 'done'
    assert vector['digest'] == hashlib.sha256(control).hexdigest()
    assert vector['accepted'] is True
    assert (
        recorded['binding']['validate'][recorder.SCHEMA_CASE]['warnings'] == []
    )


def test_attached_and_multiple_warning_controls_keep_their_order(monkeypatch):
    recorder = _recorder()
    monkeypatch.setattr(recorder, '_texts', lambda: {'one': 'unused'})
    first = {
        'message': 'warnings',
        'warnings': [{'code': 'FIRST', 'fields': {'n': 1}}],
    }
    second = {
        'message': 'warnings',
        'warnings': [{'code': 'SECOND', 'fields': {}}],
    }
    done = {
        'message': 'step',
        'step': 'done',
        'result': {'op': 'validate_query', 'accepted': True, 'diagnostics': []},
    }
    attached = json.dumps({'message': 'attached', 'attachment': 7}).encode()
    second_bytes = json.dumps(second).encode()
    done_bytes = json.dumps(done).encode()
    stream = iter(
        [
            (attached, {7: json.dumps(first).encode()}),
            (second_bytes, {}),
            (done_bytes, {}),
        ]
    )
    native = SimpleNamespace(
        CONTRACT_VERSION=29,
        CoreError=RuntimeError,
        Run=SimpleNamespace(
            start=lambda *args: SimpleNamespace(next=lambda: next(stream))
        ),
    )
    vector = recorder._validate(native)['one']
    assert vector['warnings'] == [
        {'digest': hashlib.sha256(attached).hexdigest(), 'control': first},
        {'digest': hashlib.sha256(second_bytes).hexdigest(), 'control': second},
    ]
    assert vector['digest'] == hashlib.sha256(done_bytes).hexdigest()


def test_an_unexpected_nonterminal_control_is_not_a_warning(monkeypatch):
    recorder = _recorder()
    monkeypatch.setattr(recorder, '_texts', lambda: {'one': 'unused'})
    native = SimpleNamespace(
        CONTRACT_VERSION=29,
        CoreError=RuntimeError,
        Run=SimpleNamespace(
            start=lambda *args: SimpleNamespace(
                next=lambda: (
                    b'{"message":"step","step":"fetch","result":{}}',
                    {},
                )
            )
        ),
    )
    with pytest.raises(ValueError, match='validate_query.*done'):
        recorder._validate(native)
