# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 7.8 checks share (``CONTRACT.md`` beside this file).

The sampling gap list and its entries, the branches the Phase 7.7 verifier
found reachable or wrongly explained, citations of tests in a reason, and
measuring which tests take which branch: the host suites, run under
``coverage run`` with a context per test.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

# The real coverage.py, imported before anything puts scripts/parity, which
# holds a coverage.py of its own, on the path.
import coverage

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GAPS = REPO / 'scripts' / 'parity' / 'sampling_coverage_gaps.json'
COVERAGE_JSON = REPO / 'tests' / 'parity' / 'coverage.json'
ENGINE = 'packages/sdm-relational-engine/src/sdm_relational_engine/rfm'
HARNESS = Path(
    os.environ.get(
        'SDM_RC_HARNESS', str(Path.home() / '.local/share/sdm-dbx/rust-core')
    )
)
VERIFIER = HARNESS / 'investigations' / 'P77-coverage-gaps-verifier.md'

#: The host suites, as the gate's P78.suite-rest runs them (CI's test job).
HOST_SUITES = (
    'packages/structured-data-models-api/tests',
    'packages/structured-data-models-pql/tests',
    'packages/structured-data-models-client/tests',
    'packages/structured-data-models-connectors/tests',
    'tests',
)
HOST_MARKERS = 'not live_nim and not live_warehouse'


# --- Branches ---------------------------------------------------------------------


@dataclass(frozen=True)
class Branch:
    r"""A branch as ``scripts/parity/coverage.py`` numbers it, with the text
    its two lines start with (checked, so a moved line fails loudly rather
    than silently checking another branch)."""

    module: str
    start: int
    end: int
    first: str
    second: str

    @property
    def pair(self) -> tuple[int, int]:
        return (self.start, self.end)

    def __str__(self) -> str:
        return (
            f'{self.module.removeprefix(ENGINE + "/")} {self.start}->{self.end}'
        )


#: The six branches the verifier found an offline public call reaches, by
#: name (``CONTRACT.md``, "Names").
REACHABLE: dict[str, Branch] = {
    'text-of-lists': Branch(
        f'{ENGINE}/base/sampler.py',
        1165,
        1166,
        'if len(ser) == 0 or pd.api.types.is_list_like(ser.iloc[0]):',
        'return ser',
    ),
    'list-column-on-sql': Branch(
        f'{ENGINE}/base/sql_sampler.py',
        390,
        391,
        'if dtype is not None and dtype.is_list():',
        'continue',
    ),
    'no-valid-key': Branch(
        f'{ENGINE}/backend/local/graph_store.py',
        234,
        241,
        'if table.has_time_column():',
        'error_msg += (',
    ),
    'entity-times-at-max': Branch(
        f'{ENGINE}/base/sampler.py',
        479,
        480,
        'if earliest is None or earliest == pd.Timestamp.max:',
        'return None',
    ),
    'sqlite-unseeded': Branch(
        f'{ENGINE}/backend/sqlite/sampler.py',
        283,
        284,
        'if entity_ids is None and random_seed is None:',
        'order_by = None',
    ),
    'postgres-unseeded': Branch(
        f'{ENGINE}/backend/postgres/sampler.py',
        175,
        177,
        'if random_seed is not None:',
        'frame = _frame(',
    ),
}

#: The one entry the verifier found unreachable with a false clause.
NAIVE_UTC = Branch(
    f'{ENGINE}/base/utils.py',
    65,
    66,
    'if isinstance(value.dtype, pd.DatetimeTZDtype):',
    "return value.dt.tz_convert('UTC').dt.tz_localize(None)",
)

#: What each of the seven entries' reasons said at the base that the
#: verifier showed false (``investigations/P77-coverage-gaps-verifier.md``),
#: besides "Unreachable".
FALSE_CLAUSES: dict[str, tuple[str, ...]] = {
    'text-of-lists': (
        'a list column never reaches the sampler from sdm_client',
    ),
    'list-column-on-sql': ('so no table the SQL sampler reads has one',),
    'no-valid-key': ('sdm_client refuses both before it builds the graph',),
    # Its sentence on a time column holding no time is true; what was false
    # is "Unreachable", which every still-listed entry of the six drops.
    'entity-times-at-max': (),
    'sqlite-unseeded': ('there is no output to compare',),
    'postgres-unseeded': ('there is no output to compare',),
    'naive-utc': (
        "the only Series reaching to_naive_utc is the subgraph's anchor times",
    ),
}


def check_lines(branches: dict[str, Branch]) -> None:
    r"""The branches are where this file says they are."""
    for name, branch in branches.items():
        lines = (
            (REPO / _current_path(branch.module))
            .read_text('utf-8')
            .splitlines()
        )
        got = (
            lines[branch.start - 1].strip(),
            lines[branch.end - 1].strip(),
        )
        assert got[0].startswith(branch.first) and got[1].startswith(
            branch.second
        ), (
            f'{branch} ({name}) is now {got}, not '
            f'{(branch.first, branch.second)}: the engine changed under the '
            'checks'
        )


# --- The list -----------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    module: str
    pair: tuple[int, int]
    reason: str

    def __str__(self) -> str:
        return (
            f'{self.module.removeprefix(ENGINE + "/")} '
            f'{self.pair[0]}->{self.pair[1]}'
        )


def entries() -> list[Entry]:
    gaps = json.loads(GAPS.read_text('utf-8'))
    assert isinstance(gaps, dict), f'{GAPS} is not an object'
    out = []
    seen = set()
    for module, listed in gaps.items():
        module = _historical_path(module)
        assert module not in seen, 'duplicate projected gap module'
        seen.add(module)
        for entry in listed:
            branch = entry['branch']
            out.append(
                Entry(
                    module,
                    (int(branch[0]), int(branch[1])),
                    str(entry['reason']),
                )
            )
    return out


def entry_of(branch: Branch) -> Entry | None:
    found = [
        e
        for e in entries()
        if e.module == branch.module and e.pair == branch.pair
    ]
    assert len(found) <= 1, f'{branch} is listed {len(found)} times'
    return found[0] if found else None


_REACHABLE_ONLY = re.compile(r'\breachable\s+only\b', re.I)
#: A test cited by its pytest node id, in backticks.
_CITATION = re.compile(r'`((?:tests|packages)/[^`\s]+?\.py::[^`\s]+)`')


def says_reachable_only(reason: str) -> bool:
    return _REACHABLE_ONLY.search(reason) is not None


def citations(reason: str) -> list[str]:
    return list(dict.fromkeys(_CITATION.findall(reason)))


# --- Coverage ---------------------------------------------------------------------


def _relative(path: str) -> str:
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = REPO / absolute
    return _historical_path(str(absolute.resolve().relative_to(REPO.resolve())))


def executed(report: dict[str, Any]) -> dict[str, set[tuple[int, int]]]:
    r"""``module -> branches taken`` from coverage's JSON report."""
    out = {}
    for path, body in report['files'].items():
        key = _relative(path)
        assert key not in out, 'duplicate projected coverage module'
        out[key] = {
            tuple(branch) for branch in body.get('executed_branches', [])
        }
    return out


def corpus_missing() -> dict[str, set[tuple[int, int]]]:
    r"""``module -> branches untaken`` by the corpus, the fixture cases and
    the PQL corpus, as the committed ``tests/parity/coverage.json`` says."""
    report = json.loads(COVERAGE_JSON.read_text('utf-8'))
    out = {}
    for crate in report['crates'].values():
        for module, body in crate['modules'].items():
            key = _historical_path(module)
            assert key not in out, 'duplicate projected coverage module'
            out[key] = {tuple(b) for b in body.get('missing_branches', [])}
    return out


_RCFILE = """\
[run]
branch = True
source = {source}
data_file = {data}
"""


@dataclass(eq=False)
class Measured:
    r"""One measured pytest run."""

    returncode: int
    outcomes: dict[str, str]
    core_calls: set[str]
    rcfile: Path
    data: Path
    tail: str
    _reports: dict[str, dict[str, set[tuple[int, int]]]] = field(
        default_factory=dict
    )

    def _report(self, pattern: str | None) -> dict[str, set[tuple[int, int]]]:
        key = '' if pattern is None else pattern
        if key not in self._reports:
            cov = coverage.Coverage(
                data_file=str(self.data), config_file=str(self.rcfile)
            )
            cov.load()
            out = (
                Path(tempfile.mkdtemp(prefix='sdm-p78-report-')) / 'report.json'
            )
            with contextlib.redirect_stdout(io.StringIO()):
                cov.json_report(
                    outfile=str(out),
                    contexts=None if pattern is None else [pattern],
                )
            self._reports[key] = executed(json.loads(out.read_text('utf-8')))
        return self._reports[key]

    def taken(self) -> dict[str, set[tuple[int, int]]]:
        r"""Every branch any part of the run took, in any context, the
        collection of tests included."""
        return self._report(None)

    def taken_by(self, nodeid: str) -> dict[str, set[tuple[int, int]]]:
        r"""The branches the test took in its setup, call or teardown."""
        return self._report(f'^{re.escape(nodeid)}\\|')


def measure_tests(args: list[str], watch_core: list[str] = ()) -> Measured:
    r"""Run ``pytest args`` from the repository root under ``coverage run``
    with branch coverage of the engine's ``rfm`` package, a context per
    test, and the plugin beside this file."""
    scratch = Path(tempfile.mkdtemp(prefix='sdm-p78-measure-'))
    rcfile = scratch / 'coveragerc'
    data = scratch / 'data'
    rcfile.write_text(
        _RCFILE.format(
            source=(REPO / _current_path(ENGINE)).resolve(), data=data
        ),
        'utf-8',
    )
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(HERE), *([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])]
    )
    env['SDM_P78_MEASURE_OUT'] = str(scratch)
    env['SDM_P78_WATCH_CORE'] = json.dumps(list(watch_core))
    env.pop('COVERAGE_PROCESS_START', None)
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'coverage',
            'run',
            f'--rcfile={rcfile}',
            '-m',
            'pytest',
            *args,
            '-q',
            '-p',
            'no:cacheprovider',
            '-p',
            '_gap_list_plugin',
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    outcomes_file = scratch / 'outcomes.json'
    assert outcomes_file.is_file(), (
        f'the measured run wrote no outcomes (exit {proc.returncode}):\n'
        f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
    )
    return Measured(
        proc.returncode,
        json.loads(outcomes_file.read_text('utf-8')),
        set(json.loads((scratch / 'core_calls.json').read_text('utf-8'))),
        rcfile,
        data,
        proc.stdout[-3000:] + proc.stderr[-3000:],
    )


@cache
def host_suites() -> Measured:
    r"""The host suites, measured once per process."""
    return measure_tests([*HOST_SUITES, '-m', HOST_MARKERS])


def _current_path(path: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'switch'))
    from _switch import current_package_path

    return current_package_path(path)


def _historical_path(path: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'switch'))
    from _switch import PACKAGE_RENAMES

    for old, current in PACKAGE_RENAMES.items():
        prefix = f'packages/{current}/'
        if path.startswith(prefix):
            return f'packages/{old}/' + path[len(prefix) :]
    return path
