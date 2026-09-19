# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import sys

import pytest
import sdm_relational_engine.utils.progress_logger as progress_logger
from sdm_relational_engine.utils.progress_logger import PlainProgressLogger

# The ConEmu / Windows Terminal taskbar-progress OSC sequences.
_TASKBAR_ESCAPES = ('\x1b]9;4;3\x07', '\x1b]9;4;0\x07')


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.mark.parametrize(
    'stdout_factory, verbose, expected',
    [
        (io.StringIO, True, False),
        (_FakeTTY, True, True),
        # `verbose=False` is the only control a caller has over this output, so it
        # has to reach the taskbar sequences as well -- they are written outside
        # `on_enter`/`on_exit` and were previously unconditional on a TTY.
        (_FakeTTY, False, False),
    ],
)
def test_taskbar_escapes_only_reach_a_verbose_tty(
    monkeypatch,
    stdout_factory,
    verbose,
    expected,
) -> None:
    # Pin the other half of the gate: `_in_terminal()` is
    # `not in_notebook() and stdout.isatty()`, and a notebook-flavoured test
    # runner would otherwise decide the outcome instead of the fixture.
    monkeypatch.setattr(progress_logger, 'in_notebook', lambda: False)
    stdout = stdout_factory()
    monkeypatch.setattr(sys, 'stdout', stdout)

    with PlainProgressLogger('msg', verbose=verbose):
        pass

    written = stdout.getvalue()
    for escape in _TASKBAR_ESCAPES:
        assert (escape in written) is expected


def test_in_terminal_survives_a_stdout_without_isatty(monkeypatch) -> None:
    from sdm_relational_engine.utils.progress_logger import _in_terminal

    class _NoIsatty:
        def write(self, text: str) -> int:
            return len(text)

        def flush(self) -> None:
            pass

    monkeypatch.setattr(sys, 'stdout', _NoIsatty())
    assert _in_terminal() is False


def test_in_terminal_is_false_in_a_notebook(monkeypatch) -> None:
    monkeypatch.setattr(progress_logger, 'in_notebook', lambda: True)
    monkeypatch.setattr(sys, 'stdout', _FakeTTY())
    assert progress_logger._in_terminal() is False
