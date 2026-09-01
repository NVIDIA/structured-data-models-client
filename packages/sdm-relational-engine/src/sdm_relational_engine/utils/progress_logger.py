# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import sys
import time
from abc import ABC, abstractmethod
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.padding import Padding
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    Task,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from typing_extensions import Self


def _in_terminal() -> bool:
    r"""Whether stdout is an interactive terminal outside a notebook.

    Guards the taskbar-progress OSC sequences: ``in_notebook()`` alone does not
    cover a plain script whose stdout is a pipe or a file, where the escape
    codes end up in the captured output as literal bytes.
    """
    from sdm_relational_engine import in_notebook

    if in_notebook():
        return False
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        # A replaced or already-closed stdout.
        return False


class ProgressLogger(ABC):
    r"""An abstract base class for logging progress updates."""

    def __init__(self, msg: str, verbose: bool = True) -> None:
        self.msg: str = msg
        self.verbose: bool = verbose

        self.logs: list[str] = []

        self.start_time: float | None = None
        self.end_time: float | None = None

        # Handle nested loggers gracefully:
        self._depth: int = 0

        # Internal progress bar cache:
        self._progress_bar_msg: str | None = None
        self._total: int = 0
        self._current: int = 0

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @classmethod
    def default(cls, msg: str, verbose: bool = True) -> 'ProgressLogger':
        r"""The default progress logger for the current environment."""
        from sdm_relational_engine import (
            in_streamlit_notebook,
            in_vnext_notebook,
        )

        if in_streamlit_notebook():
            return StreamlitProgressLogger(msg, verbose)
        if in_vnext_notebook():
            return PlainProgressLogger(msg, verbose)
        return RichProgressLogger(msg, verbose)

    @property
    def duration(self) -> float:
        r"""The current/final duration."""
        assert self.start_time is not None
        if self.end_time is not None:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time

    def __enter__(self) -> Self:
        self._depth += 1
        if self._depth == 1:
            self.start_time = time.perf_counter()
        # TUI only: piped into a file, a CI log or a subprocess these escape
        # codes are literal garbage bytes in the captured output. Silenced by
        # `verbose=False` too, which otherwise leaves the taskbar sequences as
        # the one thing a caller cannot turn off.
        if self._depth == 1 and self.verbose and _in_terminal():
            sys.stdout.write('\x1b]9;4;3\x07')
            sys.stdout.flush()
        if self._depth == 1 and self.verbose:
            self.on_enter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._depth -= 1
        if self._depth == 0:
            self.end_time = time.perf_counter()
        if self._depth == 0 and self.verbose:
            self.on_exit(error=exc_val is not None)
        if self._depth == 0 and self.verbose and _in_terminal():
            sys.stdout.write('\x1b]9;4;0\x07')
            sys.stdout.flush()

    def log(self, msg: str) -> None:
        r"""Logs a new message."""
        self.logs.append(msg)
        if self.verbose:
            self.on_log(msg)

    def init_progress(self, msg: str, total: int) -> None:
        r"""Initializes a progress bar."""
        if self._progress_bar_msg is not None:
            raise RuntimeError('Current progress not yet finished')
        self._progress_bar_msg = msg
        self._current = 0
        self._total = total
        if self.verbose:
            self.on_init_progress(msg, total)

    def step(self) -> None:
        r"""Increments an active progress bar."""
        assert self._progress_bar_msg is not None
        self._current += 1
        if self.verbose:
            self.on_step(self._progress_bar_msg, self._current, self._total)
        if self._current >= self._total:
            self._progress_bar_msg = None
            self._current = self._total = 0

    @abstractmethod
    def on_enter(self) -> None:
        pass

    @abstractmethod
    def on_exit(self, error: bool) -> None:
        pass

    @abstractmethod
    def on_log(self, msg: str) -> None:
        pass

    @abstractmethod
    def on_init_progress(self, msg: str, total: int) -> None:
        pass

    @abstractmethod
    def on_step(self, msg: str, current: int, total: int) -> None:
        pass


class PlainProgressLogger(ProgressLogger):
    RESET: str = '\x1b[0m'
    BOLD: str = '\x1b[1m'
    DIM: str = '\x1b[2m'
    RED: str = '\x1b[31m'
    GREEN: str = '\x1b[32m'
    CYAN: str = '\x1b[36m'

    def on_enter(self) -> None:
        from sdm_relational_engine import in_vnext_notebook

        msg = self.msg.replace('[bold]', self.BOLD)
        msg = msg.replace('[/bold]', self.RESET + self.CYAN)
        msg = self.CYAN + msg + self.RESET
        print(msg, end='\n' if in_vnext_notebook() else '', flush=True)

    def on_exit(self, error: bool) -> None:
        from sdm_relational_engine import in_vnext_notebook

        if error:
            msg = f'❌ {self.RED}({self.duration:.2f}s){self.RESET}'
        else:
            msg = f'✅ {self.GREEN}({self.duration:.2f}s){self.RESET}'

        if in_vnext_notebook():
            print(f'{self.DIM}↳{self.RESET} {msg}', flush=True)
        else:
            print(f' {msg}', flush=True)

    def on_log(self, msg: str) -> None:
        from sdm_relational_engine import in_vnext_notebook

        msg = f'{self.DIM}↳ {msg}{self.RESET}'

        if in_vnext_notebook():
            print(msg, flush=True)
        else:
            print(f'\n{msg}', end='', flush=True)

    def on_init_progress(self, msg: str, total: int) -> None:
        from sdm_relational_engine import in_vnext_notebook

        msg = f'{self.DIM}↳ {msg}{self.RESET}'

        if in_vnext_notebook():
            print(msg, flush=True)
        else:
            print(f'\n{msg} {self.DIM}[{self.RESET}', end='', flush=True)

    def on_step(self, msg: str, current: int, total: int) -> None:
        from sdm_relational_engine import in_vnext_notebook

        if in_vnext_notebook():
            return

        msg = f'{self.DIM}#{self.RESET}'
        if current == total:
            msg += f'{self.DIM}]{self.RESET}'

        print(msg, end='', flush=True)


class ColoredMofNCompleteColumn(MofNCompleteColumn):
    def __init__(self, style: str = 'green') -> None:
        super().__init__()
        self.style = style

    def render(self, task: Task) -> Text:
        return Text(str(super().render(task)), style=self.style)


class ColoredTimeRemainingColumn(TimeRemainingColumn):
    def __init__(self, style: str = 'cyan') -> None:
        super().__init__()
        self.style = style

    def render(self, task: Task) -> Text:
        return Text(str(super().render(task)), style=self.style)


class RichProgressLogger(ProgressLogger):
    def __init__(
        self,
        msg: str,
        verbose: bool = True,
        refresh_per_second: int = 10,
    ) -> None:
        super().__init__(msg=msg, verbose=verbose)

        self.refresh_per_second = refresh_per_second

        self._progress: Progress | None = None
        self._task: TaskID | None = None

        self._live: Live | None = None
        self._exception: bool = False

    def on_enter(self) -> None:
        self._live = Live(
            self,
            refresh_per_second=self.refresh_per_second,
            vertical_overflow='visible',
        )
        self._live.start()

    def on_exit(self, error: bool) -> None:
        self._exception = error

        if self._progress is not None:
            self._progress.stop()

        if self._live is not None:
            self._live.update(self, refresh=True)
            self._live.stop()

        self._progress = None
        self._task = None
        self._live = None

    def on_log(self, msg: str) -> None:
        pass

    def on_init_progress(self, msg: str, total: int) -> None:
        self._progress = Progress(
            TextColumn(f'   ↳ {msg}', style='dim'),
            BarColumn(bar_width=None),
            ColoredMofNCompleteColumn(style='dim'),
            TextColumn('•', style='dim'),
            ColoredTimeRemainingColumn(style='dim'),
        )
        self._task = self._progress.add_task('Progress', total=total)

    def on_step(self, msg: str, current: int, total: int) -> None:
        assert self._progress is not None
        assert self._task is not None
        self._progress.update(self._task, advance=1)

        if current == total:
            self._progress.stop()
            self._progress = None
            self._task = None

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:

        table = Table.grid(padding=(0, 1))

        icon: Text | Padding
        if self._exception:
            style = 'red'
            icon = Text('❌', style=style)
        elif self.end_time is not None:
            style = 'green'
            icon = Text('✅', style=style)
        else:
            style = 'cyan'
            icon = Padding(Spinner('dots', style=style), (0, 1, 0, 0))

        title = Text.from_markup(
            f'{self.msg} ({self.duration:.2f}s)',
            style=style,
        )
        table.add_row(icon, title)

        for log in self.logs:
            table.add_row('', Text(f'↳ {log}', style='dim'))

        yield table

        if self._progress is not None:
            yield self._progress.get_renderable()


class StreamlitProgressLogger(ProgressLogger):
    def __init__(
        self,
        msg: str,
        verbose: bool = True,
    ) -> None:
        super().__init__(msg=msg, verbose=verbose)

        self._status: Any = None
        self._progress: Any = None

    @staticmethod
    def _sanitize_text(msg: str) -> str:
        return re.sub(r'\[/?bold\]', '**', msg)

    def on_enter(self) -> None:
        import streamlit as st

        # Adjust layout for prettier output:
        st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)

        self._status = st.status(
            f':blue[{self._sanitize_text(self.msg)}]',
            expanded=True,
        )

    def on_exit(self, error: bool) -> None:
        if self._status is not None:
            label = f'{self._sanitize_text(self.msg)} ({self.duration:.2f}s)'
            self._status.update(
                label=f':red[{label}]' if error else f':green[{label}]',
                state='error' if error else 'complete',
                expanded=True,
            )

    def on_log(self, msg: str) -> None:
        if self._status is not None:
            self._status.write(msg)

    def on_init_progress(self, msg: str, total: int) -> None:
        if self._status is not None:
            self._progress = self._status.progress(
                value=0.0,
                text=f'{msg} [{0}/{total}]',
            )

    def on_step(self, msg: str, current: int, total: int) -> None:
        if self._progress is not None:
            self._progress.progress(
                value=min(current / total, 1.0),
                text=f'{msg} [{current}/{total}]',
            )


STREAMLIT_CSS = """
<style>
/* Fix horizontal scrollbar */
.stExpander summary {
    width: auto;
}

/* Fix paddings/margins */
.stExpander summary {
    padding: 0.75rem 1rem 0.5rem;
}
.stExpander p {
    margin: 0px 0px 0.2rem;
}
.stExpander [data-testid="stExpanderDetails"] {
    padding-bottom: 1.45rem;
}
.stExpander .stProgress div:first-child {
    padding-bottom: 4px;
}

/* Fix expand icon position */
.stExpander summary svg {
    height: 1.5rem;
}

/* Fix summary icons */
.stExpander summary [data-testid="stExpanderIconCheck"] {
    font-size: 1.8rem;
    margin-top: -3px;
    color: rgb(21, 130, 55);
}
.stExpander summary [data-testid="stExpanderIconError"] {
    font-size: 1.8rem;
    margin-top: -3px;
    color: rgb(255, 43, 43);
}
.stExpander summary span:first-child span:first-child {
    width: 1.6rem;
}

/* Add border between title and content */
.stExpander [data-testid="stExpanderDetails"] {
    border-top: 1px solid rgba(30, 37, 47, 0.2);
    padding-top: 0.5rem;
}

/* Fix title font size */
.stExpander summary p {
    font-size: 1rem;
}

/* Gray out content */
.stExpander [data-testid="stExpanderDetails"] {
    color: rgba(30, 37, 47, 0.5);
}

/* Fix progress bar font size */
.stExpander .stProgress p {
    line-height: 1.6;
    font-size: 1rem;
    color: rgba(30, 37, 47, 0.5);
}
</style>
"""
