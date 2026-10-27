# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The canonical Java binding recorder preserves warnings before validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
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
    tmp_path,
):
    from sdm_client import _native

    recorder = _recorder()
    original = json.loads(recorder.OUT.read_text())
    output = tmp_path / 'vectors.json'
    output.write_text(json.dumps(original))
    # CI's parent has the production binding; only the existing kit
    # interpreter has the fail operation the canonical recorder also needs.
    python = os.environ.get('SDM_KIT_PYTHON') or sys.executable
    recorded_run = subprocess.run(
        [
            python,
            '-I',
            '-c',
            'import importlib.util, sys; from pathlib import Path; '
            "spec = importlib.util.spec_from_file_location('recorder', sys.argv[1]); "
            'recorder = importlib.util.module_from_spec(spec); '
            'spec.loader.exec_module(recorder); recorder.OUT = Path(sys.argv[2]); '
            "raise SystemExit(recorder.main(['binding']))",
            str(ROOT / 'contract/tools/t311_java_record.py'),
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert recorded_run.returncode == 0, (
        recorded_run.stdout + recorded_run.stderr
    )
    recorded = json.loads(output.read_text())
    assert recorded['reference'] == original['reference']
    assert recorded['binding']['contract'] == _native.CONTRACT_VERSION
    vector = recorded['binding']['validate']['text.escape-warning']
    operation = json.loads(vector['operation'])
    import base64

    attachments = {
        int(key): base64.b64decode(value)
        for key, value in vector['attachments'].items()
    }
    run = _native.Run.start(recorder._start(_native, operation), attachments)
    control, carried = run.next()
    message = json.loads(control)
    if message['message'] == 'attached':
        message = json.loads(carried[message['attachment']])
    done_control = control
    taken = run.take_warnings()
    assert taken is not None
    warning_control, carried = taken
    warning_message = json.loads(warning_control)
    if warning_message['message'] == 'attached':
        warning_message = json.loads(carried[warning_message['attachment']])
    warnings = [
        {
            'digest': hashlib.sha256(warning_control).hexdigest(),
            'control': warning_message,
        }
    ]
    assert run.take_warnings() is None
    assert vector['warnings'] == warnings
    assert (
        warnings
        and warnings[0]['control']['warnings'][0]['code']
        == 'INVALID_STRING_ESCAPE'
    )
    assert message['message'] == 'step' and message['step'] == 'done'
    assert vector['digest'] == hashlib.sha256(done_control).hexdigest()
    assert vector['accepted'] is True
    assert (
        recorded['binding']['validate'][recorder.SCHEMA_CASE]['warnings'] == []
    )


def test_attached_warning_controls_keep_warning_order_and_done_digest(
    monkeypatch,
):
    recorder = _recorder()
    monkeypatch.setattr(recorder, '_texts', lambda: {'one': 'unused'})
    warned = {
        'message': 'warnings',
        'warnings': [
            {'code': 'FIRST', 'fields': {'n': 1}},
            {'code': 'SECOND', 'fields': {}},
        ],
    }
    done = {
        'message': 'step',
        'step': 'done',
        'result': {'op': 'validate_query', 'accepted': True, 'diagnostics': []},
    }
    attached = json.dumps({'message': 'attached', 'attachment': 7}).encode()
    done_bytes = json.dumps(done).encode()
    native = SimpleNamespace(
        CONTRACT_VERSION=29,
        CoreError=RuntimeError,
        Run=SimpleNamespace(
            start=lambda *args: SimpleNamespace(
                next=lambda: (done_bytes, {}),
                take_warnings=lambda: (
                    attached,
                    {7: json.dumps(warned).encode()},
                ),
            )
        ),
    )
    vector = recorder._validate(native)['one']
    assert vector['warnings'] == [
        {'digest': hashlib.sha256(attached).hexdigest(), 'control': warned}
    ]
    assert vector['digest'] == hashlib.sha256(done_bytes).hexdigest()


def test_warning_controls_are_kept_when_validation_raises(monkeypatch):
    recorder = _recorder()
    monkeypatch.setattr(recorder, '_texts', lambda: {'one': 'unused'})

    class CoreError(Exception):
        code = 'EXPECTED_ERROR'
        control = b'{"message":"error"}'

    def fail():
        raise CoreError()

    warned = b'{"message":"warnings","warnings":[{"code":"FIRST"}]}'
    native = SimpleNamespace(
        CONTRACT_VERSION=29,
        CoreError=CoreError,
        Run=SimpleNamespace(
            start=lambda *args: SimpleNamespace(
                next=fail, take_warnings=lambda: (warned, {})
            )
        ),
    )
    vector = recorder._validate(native)['one']
    assert vector['error'] == 'EXPECTED_ERROR'
    assert vector['warnings'] == [
        {
            'digest': hashlib.sha256(warned).hexdigest(),
            'control': json.loads(warned),
        }
    ]


def test_an_unexpected_nonterminal_control_is_not_a_warning(monkeypatch):
    recorder = _recorder()
    monkeypatch.setattr(recorder, '_texts', lambda: {'one': 'unused'})
    native = SimpleNamespace(
        CONTRACT_VERSION=29,
        CoreError=RuntimeError,
        Run=SimpleNamespace(
            start=lambda *args: SimpleNamespace(
                take_warnings=lambda: None,
                next=lambda: (
                    b'{"message":"step","step":"fetch","result":{}}',
                    {},
                ),
            )
        ),
    )
    with pytest.raises(ValueError, match='validate_query.*done'):
        recorder._validate(native)
