# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inert acceptance coupling primitives; full P97 product routes are absent."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAGING = Path(__file__).resolve().parents[1] / 'p97-acceptance-staging'
sys.path.insert(0, str(STAGING))
from accept_contract import Rejected, digest, require  # noqa: E402

MISSING_INTERFACES = (
    'complete_public_capture_v1',
    'governed_fixture_decode_v1',
    'complete_binding_replay_v1',
    'complete_rust_replay_v1',
)


def require_phase_interfaces() -> None:
    raise Rejected('coupling.unimplemented:' + ','.join(MISSING_INTERFACES))


def load_switch(root: Path) -> Any:
    path = root / 'tests/rust_core/switch/_switch.py'
    spec = importlib.util.spec_from_file_location('_p97_existing_switch', path)
    require(spec is not None and spec.loader is not None, 'coupling.switch')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Execution:
    pid: int
    returncode: int
    result: Any


def execute(
    command: list[str], request: dict, environment: dict[str, str], cwd: Path
) -> Execution:
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ) as child:
        try:
            output, _ = child.communicate(json.dumps(request), timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
            raise Rejected('coupling.timeout') from None
        require(child.returncode == 0, 'coupling.child_failed')
        try:
            result = json.loads(output)
        except (TypeError, ValueError):
            raise Rejected('coupling.child_output') from None
        return Execution(child.pid, child.returncode, result)


def capture_pair(
    switch: Any,
    command: list[str],
    recipe: dict,
    cwd: Path,
) -> dict[str, Execution]:
    executions = {
        selector: execute(
            command,
            {'recipe': recipe},
            switch.env_for(selector, PYTHONDONTWRITEBYTECODE='1'),
            cwd,
        )
        for selector in (switch.OLD, switch.CORE)
    }
    require(
        executions[switch.OLD].pid != executions[switch.CORE].pid,
        'coupling.same_process',
    )
    return executions


def observe_public_call(
    operation: Callable[[], Any],
    public_code: Any,
    legacy_symbol: Callable[[Any], Any],
    native_functions: tuple[Any, ...],
) -> tuple[Any, dict[str, int]]:
    counts = {'public_calls': 0, 'legacy_calls': 0, 'native_calls': 0}
    public_depth = 0

    def profile(frame: Any, event: str, argument: Any) -> None:
        nonlocal public_depth
        if event == 'call' and frame.f_code is public_code:
            public_depth += 1
            counts['public_calls'] += 1
        if public_depth:
            if event == 'call' and legacy_symbol(frame.f_code):
                counts['legacy_calls'] += 1
            if event == 'c_call' and any(
                argument is function for function in native_functions
            ):
                counts['native_calls'] += 1
        if event == 'return' and frame.f_code is public_code:
            public_depth -= 1

    require(sys.getprofile() is None, 'coupling.profile_already_active')
    sys.setprofile(profile)
    try:
        value = operation()
    finally:
        sys.setprofile(None)
    return value, counts


def require_path(counts: dict[str, int], selector: str) -> None:
    require(counts['public_calls'] > 0, 'coupling.no_public_call')
    require(selector in {'python', 'core'}, 'coupling.selector')
    wanted, forbidden = (
        ('legacy_calls', 'native_calls')
        if selector == 'python'
        else ('native_calls', 'legacy_calls')
    )
    require(counts[wanted] > 0, 'coupling.missing_implementation')
    require(counts[forbidden] == 0, 'coupling.mixed_implementation')


def fixture_bytes(root: Path, manifest: dict[str, str]) -> dict[str, bytes]:
    require(bool(manifest), 'coupling.empty_fixtures')
    contents = {}
    for relative, expected_hash in manifest.items():
        path = root / relative
        require(
            not Path(relative).is_absolute()
            and '..' not in Path(relative).parts
            and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()),
            'coupling.fixture_path',
        )
        try:
            content = path.read_bytes()
        except OSError:
            raise Rejected('coupling.fixture_missing') from None
        require(
            hashlib.sha256(content).hexdigest() == expected_hash,
            'coupling.fixture_changed',
        )
        contents[relative] = content
    return contents


def replay_pair(
    switch: Any,
    commands: dict[str, list[str]],
    root: Path,
    manifest: dict[str, str],
    decode_expected: Callable[[dict[str, bytes]], dict],
    environment: dict[str, str],
) -> dict[str, Execution]:
    require(set(commands) == {'rust', 'binding'}, 'coupling.replay_routes')
    contents = fixture_bytes(root, manifest)
    expected = decode_expected(contents)
    require(bool(expected), 'coupling.empty_expected')
    request = {'files': {name: value.hex() for name, value in contents.items()}}
    results = {}
    for route in ('rust', 'binding'):
        execution = execute(commands[route], request, environment, root)
        require(
            switch.first_difference(expected, execution.result) is None,
            'coupling.replay_disagrees',
        )
        require(
            fixture_bytes(root, manifest) == contents,
            'coupling.fixture_changed',
        )
        results[route] = execution
    require(
        results['rust'].pid != results['binding'].pid, 'coupling.same_process'
    )
    return results


def join_observation(stored: dict, executed: dict) -> None:
    require(digest(stored) == digest(executed), 'coupling.bundle_disagrees')


if __name__ == '__main__':
    try:
        require_phase_interfaces()
    except Rejected as error:
        print(f'REJECTED: {error}')
        raise SystemExit(2) from None
