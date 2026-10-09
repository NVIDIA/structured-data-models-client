# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`scripts/rust_core/native_wheel.py` reads a build's target flags and
refuses `target-cpu` wherever it is set (decision F7), on sample logs."""

from __future__ import annotations

import importlib.util
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
