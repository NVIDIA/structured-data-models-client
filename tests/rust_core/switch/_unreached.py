# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.unreached's guard: notes ported code that the packages themselves run.

Loaded into pytest with ``-p _unreached`` (this directory on
``PYTHONPATH``), it watches the whole session; run as

    python tests/rust_core/switch/_unreached.py replay --out FILE

it watches ``run_corpus.py replay``. Either way it writes a report to
``$SDM_P9_UNREACHED_OUT`` (or ``--out``).

``sys.monitoring`` reports each function, class body, lambda, comprehension
and module body as it starts or resumes. For ported code (``ported.tsv``,
``CONTRACT.md``, "Ported code") the guard walks up the stack to the first
frame that is ported code, test or harness code, or code of a package's
``src`` tree (``sdm_core``'s Python included):

- ported code: the ported code was called from ported code, which is judged
  where it started;
- test or harness code (``tests/``, ``packages/*/tests/``, ``scripts/``,
  any ``conftest.py``): a test calling the old implementation directly;
- package code: the packages reached ported code at runtime, which is a
  breach.

Tests under ``tests/rust_core/`` (the shadow harness and the earlier phases'
comparisons with today's Python) are not judged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _switch import PortedCode, add_paths  # noqa: E402

EXCLUDED = ('tests/rust_core/',)


class Guard:
    def __init__(self) -> None:
        self.code = PortedCode()
        self.context = 'startup'
        self.breaches: dict[str, dict[str, Any]] = {}
        self.counts = {'nested': 0, 'direct': 0, 'orphan': 0, 'excluded': 0}
        self.contexts: set[str] = set()
        self._tool: int | None = None

    def _event(self, code: Any, offset: int) -> Any:
        kind, symbol = self.code.kind(code)
        if kind != PortedCode.PORTED:
            return sys.monitoring.DISABLE
        if self.context.startswith(EXCLUDED):
            self.counts['excluded'] += 1
            return None
        frame = sys._getframe(1).f_back
        while frame is not None:
            above, where = self.code.kind(frame.f_code)
            if above == PortedCode.PORTED:
                self.counts['nested'] += 1
                return None
            if above == PortedCode.TEST:
                self.counts['direct'] += 1
                return None
            if above == PortedCode.LIBRARY:
                found = self.breaches.setdefault(
                    symbol, {'count': 0, 'called_from': where, 'contexts': []}
                )
                found['count'] += 1
                if (
                    len(found['contexts']) < 5
                    and self.context not in found['contexts']
                ):
                    found['contexts'].append(self.context)
                return None
            frame = frame.f_back
        self.counts['orphan'] += 1
        return None

    def install(self) -> None:
        monitoring = sys.monitoring
        for tool in (5, 2, 1):
            if monitoring.get_tool(tool) is None:
                monitoring.use_tool_id(tool, 'sdm-p9-unreached')
                self._tool = tool
                break
        assert self._tool is not None, 'no sys.monitoring tool id is free'
        events = monitoring.events
        for event in (events.PY_START, events.PY_RESUME):
            monitoring.register_callback(self._tool, event, self._event)
        monitoring.set_events(self._tool, events.PY_START | events.PY_RESUME)

    def uninstall(self) -> None:
        if self._tool is None:
            return
        monitoring = sys.monitoring
        monitoring.set_events(self._tool, 0)
        for event in (monitoring.events.PY_START, monitoring.events.PY_RESUME):
            monitoring.register_callback(self._tool, event, None)
        monitoring.free_tool_id(self._tool)
        self._tool = None

    def report(self) -> dict[str, Any]:
        return {
            'breaches': self.breaches,
            'counts': self.counts,
            'contexts': len(self.contexts),
        }

    def enter(self, context: str) -> None:
        self.context = context
        self.contexts.add(context)


# The guard watches from the moment pytest loads this plugin, before any
# conftest or test module imports the packages.
GUARD: Guard | None = None
if os.environ.get('SDM_P9_UNREACHED_OUT') and 'pytest' in sys.modules:
    GUARD = Guard()
    GUARD.install()
    GUARD.enter('collection')


def _out() -> Path:
    return Path(os.environ['SDM_P9_UNREACHED_OUT'])


try:
    import pytest

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(item: Any, nextitem: Any) -> Any:
        if GUARD is not None:
            GUARD.enter(item.nodeid)
        yield
        if GUARD is not None:
            GUARD.enter('between tests')

    OUTCOMES: dict[str, int] = {
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'error': 0,
    }

    def pytest_runtest_logreport(report: Any) -> None:
        # The implementer's shadow tests count for no outcome (P9.tests).
        if report.nodeid.startswith('tests/rust_core/shadow/'):
            return
        if report.when == 'call':
            OUTCOMES[report.outcome] = OUTCOMES.get(report.outcome, 0) + 1
        elif report.skipped:
            OUTCOMES['skipped'] += 1
        elif report.failed:
            OUTCOMES['error'] += 1

    def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
        if GUARD is None:
            return
        GUARD.uninstall()
        report = GUARD.report()
        report['exitstatus'] = int(exitstatus)
        report['tests'] = int(session.testscollected)
        report['outcomes'] = OUTCOMES
        _out().write_text(json.dumps(report, indent=1))

except ImportError:  # pragma: no cover - the replay runs without pytest
    pass


def replay(out: Path) -> int:
    r"""``run_corpus.py replay`` under the guard, each transcript its own
    context."""
    add_paths()
    guard = Guard()
    guard.install()
    guard.enter('replay startup')
    import _parity.corpus as corpus
    import run_corpus

    original = corpus.check

    def check(path: Path) -> Any:
        guard.enter(f'replay {path.parent.relative_to(corpus.TRANSCRIPTS)}')
        try:
            return original(path)
        finally:
            guard.enter('replay between transcripts')

    corpus.check = check
    run_corpus.check = check
    status = run_corpus.main(['replay'])
    guard.uninstall()
    report = guard.report()
    report['exitstatus'] = status
    out.write_text(json.dumps(report, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    rep = commands.add_parser('replay')
    rep.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    return replay(Path(args.out))


if __name__ == '__main__':
    raise SystemExit(main())
