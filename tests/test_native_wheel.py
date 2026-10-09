# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`scripts/rust_core/native_wheel.py` reads a build's target flags and
refuses `target-cpu` wherever it is set (decision F7), on sample logs, and
leaves each build's wheel and identity in its destination together, on a
stand-in maturin."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent

LOG = """\
   Compiling sdm-core v1.0.0 (/io/rust/crates/sdm-core)
     Running `/tmp/home/.rustup/toolchains/1.97.0/bin/rustc --crate-name sdm_core --edition=2024 crates/sdm-core/src/lib.rs --crate-type lib -C opt-level=3 -C lto=thin -C codegen-units=1 -C metadata=0123abcd{extra}`
     Running `rustc --crate-name _native --crate-type cdylib -C opt-level=3 -C strip=debuginfo -C metadata=4567ef01`
"""


def _script() -> ModuleType:
    path = ROOT / 'scripts/rust_core/native_wheel.py'
    spec = importlib.util.spec_from_file_location('native_wheel', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_flags_of_every_rustc_invocation_are_read() -> None:
    script = _script()
    flags = script._rustc_flags(LOG.format(extra=''))
    assert flags == [
        '-C opt-level=3',
        '-C lto=thin',
        '-C codegen-units=1',
        '-C strip=debuginfo',
    ]
    assert script.target_cpu_refusal({}, {}, flags) is None


def test_target_cpu_is_refused_wherever_it_is_set() -> None:
    script = _script()
    for extra in (' -C target-cpu=native', ' -Ctarget-cpu=native'):
        flags = script._rustc_flags(LOG.format(extra=extra))
        assert script.target_cpu_refusal({}, {}, flags), extra
    clean = script._rustc_flags(LOG.format(extra=''))
    assert script.target_cpu_refusal(
        {'RUSTFLAGS': '-C target-cpu=native'}, {}, clean
    )
    assert script.target_cpu_refusal(
        {}, {'/io/.cargo/config.toml': 'rustflags = ["-Ctarget-cpu=x"]'}, clean
    )


def test_a_log_with_no_rustc_invocation_is_refused() -> None:
    script = _script()
    assert script.target_cpu_refusal({}, {}, script._rustc_flags('Fresh x'))


#: Stands in for maturin: its version, and a build that writes one wheel
#: holding an extension module, under the same name each time, and logs
#: one rustc invocation.
_MATURIN = """\
import pathlib, sys, zipfile
if sys.argv[1:] == ['--version']:
    print('maturin 1.15.0')
    sys.exit(0)
out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])
wheel = out / 'structured_data_models_client-1.0.0-cp310-abi3-linux_x86_64.whl'
with zipfile.ZipFile(wheel, 'w') as archive:
    archive.writestr('sdm_client/_native.abi3.so', b'native')
print('     Running `rustc --crate-name _native -C opt-level=3`')
"""


def test_two_builds_into_one_directory_each_leave_a_wheel_and_its_identity(
    tmp_path: Path,
) -> None:
    tools = tmp_path / 'bin'
    tools.mkdir()
    for name, body in (
        ('maturin', _MATURIN),
        ('rustc', 'print("rustc 1.97.0 (stand-in)")\n'),
    ):
        tool = tools / name
        tool.write_text(f'#!{sys.executable}\n{body}')
        tool.chmod(0o755)
    out = tmp_path / 'out'
    environment = {
        **os.environ,
        'PATH': f'{tools}{os.pathsep}{os.environ["PATH"]}',
    }
    for build in range(2):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / 'scripts/rust_core/native_wheel.py'),
                '--out',
                str(out),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (build, result.stderr)
        wheels = sorted(p.name for p in out.iterdir())
        assert wheels == [
            'structured_data_models_client-1.0.0-cp310-abi3-linux_x86_64.whl',
            'structured_data_models_client-1.0.0-cp310-abi3-linux_x86_64.whl.json',
        ], (build, wheels)
        identity = json.loads((out / wheels[1]).read_text())
        assert identity['wheel'] == wheels[0]
        assert (
            identity['native_sha256'] == hashlib.sha256(b'native').hexdigest()
        )
