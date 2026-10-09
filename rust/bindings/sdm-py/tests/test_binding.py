# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`sdm_client._native` as any build of it ships: the module its stub
describes, a run of a real operation, the one `CoreError`, and an import
that changes nothing in the process (decision F5)."""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
from typing import Any

import messages
import pytest
from sdm_client import _native

REPO = pathlib.Path(__file__).resolve().parents[4]
STUB = (
    REPO / 'packages/structured-data-models-client/src/sdm_client/_native.pyi'
)
CANARY = 'CANARY-5e0d'


def _public(names: object) -> set[str]:
    return {name for name in names if not name.startswith('_')}  # type: ignore[attr-defined]


def _stub() -> tuple[set[str], dict[str, set[str]]]:
    """The names the stub gives the module, and each class's members."""
    tree = ast.parse(STUB.read_text())
    names: set[str] = set()
    members: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
            members[node.name] = {
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            } | {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
            }
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            names.add(node.target.id)
    return _public(names), members


def test_the_build_is_the_one_named(native_build: str) -> None:
    assert _native.TEST_OPS == (native_build == 'test-ops')


def test_the_stub_describes_the_module() -> None:
    names, members = _stub()
    assert _public(dir(_native)) == names
    for cls in ('Run', 'ArrowStream'):
        declared = getattr(_native, cls).__dict__
        assert {m for m in declared if not m.startswith('__')} | {
            m for m in declared if m.startswith('__arrow')
        } == members[cls], cls


def test_the_contract_version_is_the_contract_crate_s() -> None:
    source = (REPO / 'rust/crates/sdm-contract/src/lib.rs').read_text()
    version = re.search(r'pub const CONTRACT_VERSION: u32 = (\d+);', source)
    assert version
    assert _native.CONTRACT_VERSION == int(version.group(1))
    assert messages.CONTRACT == _native.CONTRACT_VERSION


def test_a_query_is_validated_through_a_run() -> None:
    for query, accepted in [
        ('PREDICT users.age FOR EACH users.id', True),
        ('PREDICT FOR EACH users.id', False),
    ]:
        run = _native.Run.start(*messages.validate(query))
        control, attachments = run.next()
        step = messages.read(control)
        assert (step['message'], step['step']) == ('step', 'done')
        assert step['result']['op'] == 'validate_query'
        assert step['result']['accepted'] is accepted, query
        assert attachments == {}


def test_an_error_the_run_returns_is_raised_as_core_error() -> None:
    run = _native.Run.start(
        *messages.validate('PREDICT users.age FOR EACH users.id')
    )
    run.next()
    with pytest.raises(_native.CoreError) as raised:
        run.next()
    error = raised.value
    assert (error.kind, error.code) == ('protocol', 'RUN_FINISHED')
    assert str(error) == 'protocol: RUN_FINISHED'
    message = messages.read(error.control)
    assert message['message'] == 'error'
    assert (message['error']['kind'], message['error']['code']) == (
        'protocol',
        'RUN_FINISHED',
    )
    assert error.attachments == {}
    assert issubclass(_native.CoreError, Exception)
    assert _native.CoreError.__module__ == 'sdm_client._native'


def test_a_refused_start_quotes_nothing_it_was_given() -> None:
    malformed = messages.start({'op': CANARY})
    other_contract, attachments = messages.validate(CANARY)
    for message, attachments_, code in [
        (b'{"message": "start", "' + CANARY.encode(), {}, 'INVALID_MESSAGE'),
        (malformed, {}, 'INVALID_MESSAGE'),
        (
            other_contract.replace(b'"contract": 3', b'"contract": 99'),
            attachments,
            None,
        ),
    ]:
        with pytest.raises(_native.CoreError) as raised:
            _native.Run.start(message, attachments_)
        error = raised.value
        if code:
            assert error.code == code
        written = f'{error} {error!r} {error.control!r} {error.attachments!r}'
        assert CANARY not in written


def test_an_attachment_is_bytes_or_arrow() -> None:
    start, _ = messages.validate('PREDICT users.age FOR EACH users.id')
    for wrong in ['text', 12, bytearray(b'x'), [b'x']]:
        with pytest.raises(TypeError, match='attachment 1 is a'):
            _native.Run.start(start, {1: wrong, 2: b'{}'})


def test_only_the_test_ops_build_takes_a_test_operation(
    native_build: str,
) -> None:
    start = messages.operation_for_tests('warn', warnings=1)
    attachments = {1: messages.encoded([])}
    if native_build == 'test-ops':
        run = _native.Run.start(start, attachments)
        assert messages.read(run.next()[0])['step'] == 'done'
    else:
        with pytest.raises(_native.CoreError) as raised:
            _native.Run.start(start, attachments)
        assert raised.value.code == 'INVALID_MESSAGE'


#: What the clean-import test runs in a fresh interpreter: everything the
#: process holds that an import could change, read before and after
#: importing `_native` alone, `sdm_client` itself having been imported
#: first. Environment reads are counted twice: through `os.environ` and
#: `os.getenv` in Python, and through the C library's `getenv` by the shim
#: the parent preloads.
_CLEAN_IMPORT = textwrap.dedent(
    """
    import ctypes, json, logging, os, signal, subprocess, sys, threading

    shim = ctypes.CDLL(os.environ['SDM_GETENV_SHIM'])
    shim.shim_stop.restype = ctypes.c_char_p
    libc = ctypes.CDLL(None)

    def native_threads():
        tasks = '/proc/self/task'
        if os.path.isdir(tasks):
            return len(os.listdir(tasks))
        listing = subprocess.run(
            ['ps', '-M', '-p', str(os.getpid())],
            capture_output=True, text=True, check=True,
        )
        return len(listing.stdout.splitlines()) - 1

    def handlers():
        found = {'': [id(h) for h in logging.getLogger().handlers]}
        for name, logger in logging.Logger.manager.loggerDict.items():
            found[name] = [id(h) for h in getattr(logger, 'handlers', [])]
        return found

    def signals():
        # Each signal's handler and flags as the C library holds them, read
        # where struct sigaction keeps them: the handler first, the flags
        # after the mask, which is 4 bytes on macOS and 128 in glibc. The
        # mask itself is left out: glibc copies all 128 bytes of it from
        # the kernel's 8, so the rest is whatever was on its stack.
        width = ctypes.sizeof(ctypes.c_void_p)
        flags = width + (4 if sys.platform == 'darwin' else 128)
        found = {}
        for number in range(1, signal.NSIG):
            buffer = ctypes.create_string_buffer(512)
            if libc.sigaction(number, None, buffer) == 0:
                raw = buffer.raw
                found[number] = (raw[:width] + raw[flags:flags + 4]).hex()
        return found, {
            s.name: repr(signal.getsignal(s)) for s in signal.valid_signals()
            if isinstance(s, signal.Signals)
        }

    class Reads(dict):
        def __init__(self, base):
            super().__init__(base)
            self.read = []
        def __getitem__(self, key):
            self.read.append(key)
            return super().__getitem__(key)
        def get(self, key, default=None):
            self.read.append(key)
            return super().get(key, default)
        def __contains__(self, key):
            self.read.append(key)
            return super().__contains__(key)

    import sdm_client

    # The shim must see a read from another image, or its silence below
    # proves nothing.
    shim.shim_start()
    libc.getenv(b'SDM_SHIM_PROBE')
    probe = shim.shim_stop().decode()

    before = (
        sorted(t.name for t in threading.enumerate()), native_threads(),
        handlers(), signals(),
    )
    environ, os.environ = os.environ, Reads(os.environ)
    shim.shim_start()
    import sdm_client._native
    read_natively = shim.shim_stop().decode().split()
    read_in_python, os.environ = os.environ.read, environ
    after = (
        sorted(t.name for t in threading.enumerate()), native_threads(),
        handlers(), signals(),
    )
    print(json.dumps({
        'probe': probe,
        'threads': [before[0], after[0]],
        'native_threads': [before[1], after[1]],
        'handlers_equal': before[2] == after[2],
        'signals_changed': sorted(
            {
                str(n) for n in before[3][0]
                if before[3][0][n] != after[3][0].get(n)
            }
            | {
                name for name in before[3][1]
                if before[3][1][name] != after[3][1].get(name)
            }
        ),
        'signals_before': {
            str(n): before[3][0][n] for n in before[3][0]
            if before[3][0][n] != after[3][0].get(n)
        },
        'signals_after': {
            str(n): after[3][0].get(n) for n in before[3][0]
            if before[3][0][n] != after[3][0].get(n)
        },
        'read_in_python': read_in_python,
        'read_natively': read_natively,
    }))
    """
)


def _getenv_shim(directory: pathlib.Path) -> pathlib.Path:
    """The shim of `getenv_shim.c`, built for this machine."""
    library = directory / 'getenv_shim.so'
    subprocess.run(
        [
            os.environ.get('CC', 'cc'),
            '-shared',
            '-fPIC',
            '-o',
            str(library),
            str(pathlib.Path(__file__).with_name('getenv_shim.c')),
            *([] if sys.platform == 'darwin' else ['-ldl']),
        ],
        check=True,
    )
    return library


def test_importing_the_module_changes_nothing_in_the_process(
    tmp_path: pathlib.Path,
) -> None:
    shim = _getenv_shim(tmp_path)
    preload = (
        'DYLD_INSERT_LIBRARIES' if sys.platform == 'darwin' else 'LD_PRELOAD'
    )
    result = subprocess.run(
        [sys.executable, '-I', '-c', _CLEAN_IMPORT],
        env={**os.environ, preload: str(shim), 'SDM_GETENV_SHIM': str(shim)},
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    found: dict[str, Any] = json.loads(result.stdout)
    assert found['probe'] == 'SDM_SHIM_PROBE\n', (
        'the getenv shim is not in place'
    )
    threads, native = found['threads'], found['native_threads']
    assert threads[0] == threads[1], 'a Python thread started'
    assert native[0] == native[1], 'a native thread started'
    assert found['handlers_equal'], 'a logging handler was added'
    assert found['signals_changed'] == [], (
        found['signals_before'],
        found['signals_after'],
    )
    assert found['read_in_python'] == []
    # The dynamic loader reads its own variables as it loads any library;
    # the module itself reads none.
    loader = ('DYLD_', 'LD_')
    assert [
        name for name in found['read_natively'] if not name.startswith(loader)
    ] == []
