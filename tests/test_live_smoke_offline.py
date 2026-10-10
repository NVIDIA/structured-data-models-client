# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live-model smoke script's offline parts (``scripts/live/rfm_smoke.py``,
decision E27, T1.11d): its argument handling, its redaction, its refusal to
run with no endpoint, and its dry run, which builds the request on the core
and shows what would be sent with every socket connection refused. Nothing
here reaches a model endpoint or reads Databricks configuration; the live
smoke itself is run by hand and gates nothing.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/live/rfm_smoke.py'


def _smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location('rfm_smoke', SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clean_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(('SDM_SMOKE_', 'DATABRICKS_', 'SDM_CORE_'))
    }


def test_an_argument_wins_over_its_variable_and_a_variable_is_read() -> None:
    smoke = _smoke()
    environ = {'SDM_SMOKE_ENDPOINT': 'from-env', 'SDM_SMOKE_PROFILE': 'p-env'}
    given = smoke.parse(['--endpoint', 'from-arg'])
    assert smoke.resolve(given, environ) == ('from-arg', 'p-env')
    assert smoke.resolve(smoke.parse([]), environ) == ('from-env', 'p-env')
    assert smoke.resolve(smoke.parse([]), {}) == (None, None)


def test_bad_usage_is_refused() -> None:
    smoke = _smoke()
    for argv in (['--show-body'], ['--timeout', '0'], ['--route', 'x']):
        with pytest.raises(SystemExit) as refused:
            smoke.parse(argv)
        assert refused.value.code == 2


def test_redaction_removes_secrets_and_anything_shaped_like_one() -> None:
    smoke = _smoke()
    redact = smoke.Redactor()
    redact.know('my-serving-endpoint', 'my-profile', None, '', 'ab')
    said = redact(
        'calling my-serving-endpoint (profile my-profile) at '
        'https://adb-1234567890123456.7.azuredatabricks.net/serving with '
        'Bearer abc.def and dapi0123456789abcdef0123456789abcdef as '
        'someone@example.com on dbc-a1b2c3d4-e5f6.cloud.databricks.com'
    )
    for secret in (
        'my-serving-endpoint',
        'my-profile',
        'adb-1234567890123456',
        'azuredatabricks',
        'abc.def',
        'dapi0123',
        'someone@',
        'example.com',
        'dbc-a1b2c3d4',
    ):
        assert secret not in said, secret
    assert said.count(smoke.REDACTED) >= 6
    # Text of no secret's shape passes; a value too short to be a secret is
    # not one.
    assert redact('rows to predict: 2 ab') == 'rows to predict: 2 ab'


def test_a_failure_is_reported_redacted_with_no_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    smoke = _smoke()
    monkeypatch.setattr(smoke, '_quiet_the_world', lambda: None)

    def failing(route: str) -> Any:
        raise RuntimeError(
            'could not reach my-serving-endpoint at '
            'https://dbc-a1b2c3d4-e5f6.cloud.databricks.com with token '
            'dapi0123456789abcdef0123456789abcdef'
        )

    monkeypatch.setattr(smoke, 'build_request', failing)
    status = smoke.main(['--endpoint', 'my-serving-endpoint'])
    captured = capsys.readouterr()
    assert status == 1
    assert 'smoke failed: RuntimeError' in captured.err
    for secret in ('my-serving-endpoint', 'dbc-a1b2c3d4', 'dapi0123'):
        assert secret not in captured.err + captured.out
    assert 'Traceback' not in captured.err


def test_no_endpoint_is_bad_usage_and_nothing_is_sent() -> None:
    run = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env=_clean_env(),
        check=False,
    )
    assert run.returncode == 2, run.stderr
    assert 'SDM_SMOKE_ENDPOINT' in run.stderr


def test_the_result_check_names_what_is_wrong() -> None:
    import pandas as pd

    smoke = _smoke()
    good = pd.DataFrame(
        {'row_index': [0, 1], 'id': [0, 1], 'prediction': [1.0, 2.5]}
    )
    assert smoke.check(good, [0, 1]) == []
    assert smoke.check(good, [0, 1, 2])
    assert smoke.check(good.assign(prediction=[1.0, float('nan')]), [0, 1])
    assert smoke.check(good[['row_index', 'prediction']], [0, 1])


def test_the_dry_run_builds_the_core_s_request_with_no_connection() -> None:
    run = subprocess.run(
        [sys.executable, str(SCRIPT), '--dry-run'],
        capture_output=True,
        text=True,
        env=_clean_env(),
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'dry run, core route' in run.stdout
    assert 'context rows: 112, rows to predict: 2' in run.stdout
    assert 'smoke passed' in run.stdout
