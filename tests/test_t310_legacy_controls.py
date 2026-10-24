# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic controls select the implementation their output names."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from sdm_client import Schema, _routing

ROOT = Path(__file__).resolve().parents[1]


def _module(relative: str) -> ModuleType:
    path = ROOT / relative
    name = path.stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.mark.parametrize('route', ['old', 'core'])
def test_smoke_dry_run_reaches_the_requested_implementation(
    route: str, tmp_path: Path
) -> None:
    # The command deliberately blocks sockets and changes logging for its
    # entire process. Keep those protections out of the pytest process.
    script = f"""
    import importlib.util
    import json
    import sys
    from sdm_client import _routing

    original = _routing.route
    seen = []

    def audited(paths):
        selected = original(paths)
        operation = paths[0].split('/')[0]
        if operation in (
            'schema.from-source', 'query.materialize', 'client.predict'
        ):
            seen.append((operation, selected))
        return selected

    _routing.route = audited
    spec = importlib.util.spec_from_file_location(
        'rfm_smoke', {str(ROOT / 'scripts/live/rfm_smoke.py')!r}
    )
    smoke = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = smoke
    spec.loader.exec_module(smoke)
    status = smoke.main(['--dry-run', '--route', {route!r}])
    print('route-audit: ' + json.dumps(seen))
    sys.exit(status)
    """
    done = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        env={
            key: value
            for key, value in os.environ.items()
            if key != _routing.ENVIRONMENT
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    seen = json.loads(done.stdout.split('route-audit: ', 1)[1])
    assert {operation for operation, _ in seen} == {
        'schema.from-source',
        'query.materialize',
        'client.predict',
    }
    assert {selected for _, selected in seen} == {route}


@pytest.mark.parametrize('route', ['old', 'core'])
@pytest.mark.parametrize('tool', ['null-link', 'real-data', 'key-rules'])
def test_comparison_builders_select_the_requested_source_route(
    tool: str, route: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    # Start on the opposite route, so inheriting a caller's selection is
    # also wrong. Stop at the real public source boundary, before sampling.
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(_routing.FROM_SOURCE_DUCKDB) if route == 'old' else '',
    )
    seen = []

    class StopAtSource(BaseException):
        pass

    def source_boundary(
        cls: Any, source: Any, *args: Any, **kwargs: Any
    ) -> Any:
        seen.append(_routing.route(_routing.FROM_SOURCE_DUCKDB))
        source.connection.close()
        raise StopAtSource

    monkeypatch.setattr(Schema, 'from_source', classmethod(source_boundary))
    dataset = SimpleNamespace(connect=duckdb.connect, from_source={})
    with pytest.raises(StopAtSource):
        if tool == 'null-link':
            _module('scripts/live/null_link_benchmark.py').build(
                dataset, None, route, 0, [1]
            )
        elif tool == 'real-data':
            _module('scripts/live/real_data_benchmark.py').describe(
                dataset, route
            )
        else:
            _module('contract/tools/key_rules_probe.py').probe(
                {'statements': [], 'primary_keys': {}}, route
            )
    assert seen == [route]


@pytest.mark.parametrize('mode', ['python-ref', 'python-old', 'python-core'])
@pytest.mark.parametrize('previous', [None, 'query.validate/accept/any'])
@pytest.mark.parametrize('fail', [False, True])
def test_sampled_null_vectors_select_each_route_and_restore_the_caller(
    mode: str, previous: str | None, fail: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _module('contract/tools/sampled_null_key_record.py')
    monkeypatch.setattr(recorder, 'VECTORS', [{'id': 'one'}, {'id': 'two'}])
    boundary = SimpleNamespace(
        reset=lambda: SimpleNamespace(operations=[]),
    )
    monkeypatch.setattr(
        recorder, '_runner', lambda: SimpleNamespace(_Boundary=lambda: boundary)
    )
    if previous is None:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(_routing.ENVIRONMENT, previous)
    seen = []

    def observe(vector: dict[str, Any]) -> dict[str, Any]:
        seen.append(_routing.route(_routing.FROM_SOURCE_DUCKDB))
        if fail and vector['id'] == 'two':
            raise RuntimeError('observation failed')
        return {}

    monkeypatch.setattr(recorder, 'observe', observe)
    if fail:
        with pytest.raises(RuntimeError, match='observation failed'):
            recorder.observe_all(mode)
    else:
        assert set(recorder.observe_all(mode)) == {'one', 'two'}
    expected = 'core' if mode == 'python-core' else 'old'
    assert seen == [expected, expected]
    assert os.environ.get(_routing.ENVIRONMENT) == previous
