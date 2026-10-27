# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selftest reuses lexing while each real guard still reads its own fixtures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def command(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/rust_core/selftest.py'
    spec = importlib.util.spec_from_file_location(
        'selftest_lexical_cache', path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def guarded_tree(command, tmp_path):
    rust = tmp_path / 'rust'
    rust.mkdir()
    (rust / 'clippy.toml').write_bytes(
        (command.REPO_ROOT / 'rust/clippy.toml').read_bytes()
    )
    crate = rust / 'crates/sdm-host'
    (crate / 'src').mkdir(parents=True)
    (crate / 'tests').mkdir()
    (crate / 'Cargo.toml').write_text('[package]\nname = "sdm-host"\n')
    text = (
        'fn lookup(root: &Root) {\n'
        '    #[allow(clippy::disallowed_methods, reason = "zone roots")]\n'
        '    let db = TimeZoneDatabase::from_dir(&root.path);\n'
        '}\n'
    )
    return tmp_path, crate, text


def counted_reader(command, monkeypatch):
    original = command.l0_scan.read_rust
    calls = []

    def read(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(command.l0_scan, 'read_rust', read)
    return calls, read


def test_identical_text_keeps_path_exemption_and_shipped_admission(
    command, guarded_tree, monkeypatch
):
    root, crate, text = guarded_tree
    for relative in ['src/zones.rs', 'src/other.rs', 'tests/guard.rs']:
        (crate / relative).write_text(text)
    calls, reader = counted_reader(command, monkeypatch)
    with command.lexical_cache():
        problems = command.l0_scan.CHECKS['bans'](root)
        assert len(problems) == 1
        assert 'sdm-host/src/other.rs:' in problems[0]
        assert 'shipped code allows clippy::disallowed_methods' in problems[0]
        assert calls == [text]
        assert command.l0_scan.read_rust.cache_info().hits == 2
    assert command.l0_scan.read_rust is reader


def test_same_path_changed_source_reaches_the_real_seeded_guard(
    command, guarded_tree, monkeypatch
):
    root, crate, text = guarded_tree
    path = crate / 'src/zones.rs'
    path.write_text(text)
    changed = text.replace('&root.path', '&other.path')
    calls, _ = counted_reader(command, monkeypatch)
    with command.lexical_cache():
        assert command.l0_scan.CHECKS['bans'](root) == []
        path.write_text(changed)
        problems = command.l0_scan.CHECKS['bans'](root)
        assert len(problems) == 1
        assert 'sdm-host/src/zones.rs:' in problems[0]
        assert 'shipped code allows clippy::disallowed_methods' in problems[0]
        assert calls == [text, changed]


def test_guard_rediscovers_added_and_removed_files_with_cached_text(
    command, guarded_tree, monkeypatch
):
    root, crate, text = guarded_tree
    (crate / 'src/zones.rs').write_text(text)
    calls, _ = counted_reader(command, monkeypatch)
    with command.lexical_cache():
        assert command.l0_scan.CHECKS['bans'](root) == []
        extra = crate / 'src/other.rs'
        extra.write_text(text)
        problems = command.l0_scan.CHECKS['bans'](root)
        assert len(problems) == 1 and 'src/other.rs:' in problems[0]
        extra.unlink()
        assert command.l0_scan.CHECKS['bans'](root) == []
        assert calls == [text]


@pytest.mark.parametrize('fail', [False, True])
def test_main_restores_reader_and_clears_cache_even_on_guard_failure(
    command, guarded_tree, monkeypatch, fail
):
    root, crate, text = guarded_tree
    (crate / 'src/zones.rs').write_text(text)
    calls, reader = counted_reader(command, monkeypatch)
    cached_readers = []
    failure = RuntimeError('guard fixture failure')
    monkeypatch.setattr(command.l0_deps, 'cargo_metadata', lambda _: {})
    monkeypatch.setattr(command, 'deps_cases', lambda _, __: None)
    monkeypatch.setattr(command, 'freeze_cases', lambda _: None)
    monkeypatch.setattr(command, 'parser_name_cases', lambda _, __: None)
    monkeypatch.setattr(command, 'clippy_case', lambda _: None)

    def scan(_):
        cached_readers.append(command.l0_scan.read_rust)
        assert command.l0_scan.CHECKS['bans'](root) == []
        assert command.l0_scan.CHECKS['bans'](root) == []
        if fail:
            raise failure

    monkeypatch.setattr(command, 'scan_cases', scan)
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError) as error:
                command.main([])
            assert error.value is failure
        else:
            assert command.main([]) == 0
        assert command.l0_scan.read_rust is reader
        assert cached_readers[-1].cache_info().currsize == 0
    assert calls == [text, text]
    assert cached_readers[0] is not cached_readers[1]


def test_cache_is_bounded_and_evicted_text_is_lexed_again(command, monkeypatch):
    calls, reader = counted_reader(command, monkeypatch)
    with command.lexical_cache():
        cached = command.l0_scan.read_rust
        for number in range(513):
            cached(f'fn item_{number}() {{}}\n')
        assert cached.cache_info().maxsize == 512
        assert cached.cache_info().currsize == 512
        cached('fn item_0() {}\n')
        assert len(calls) == 514
    assert command.l0_scan.read_rust is reader
    assert cached.cache_info().currsize == 0
