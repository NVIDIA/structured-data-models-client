# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 7.7 checks share (``CONTRACT.md`` beside this file).

The three branches the sampling gap list claimed unreachable, where they
are, a watch that notes which of their lines run while a transcript
replays, and finding the transcripts that take each one.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUST_CORE = REPO / 'tests' / 'rust_core'
SAMPLING = RUST_CORE / 'sampling'
DRIVER = SAMPLING / 'driver'
LINK = RUST_CORE / 'link'
REQUEST = RUST_CORE / 'request'
PARITY_SCRIPTS = REPO / 'scripts' / 'parity'
for _path in (PARITY_SCRIPTS, RUST_CORE, SAMPLING, DRIVER, LINK, REQUEST):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

TRANSCRIPTS = REPO / 'tests' / 'parity' / 'transcripts'
SAMPLING_RECORDINGS = REPO / 'tests' / 'parity' / 'sampling'
REQUEST_RECORDINGS = REPO / 'tests' / 'parity' / 'requests'
GAPS = PARITY_SCRIPTS / 'sampling_coverage_gaps.json'
COVERAGE_JSON = REPO / 'tests' / 'parity' / 'coverage.json'

_ENGINE = 'packages/sdm-relational-engine/src/sdm_relational_engine/rfm'


@dataclass(frozen=True)
class Branch:
    r"""A branch as ``coverage.py`` numbers it, with the text its two lines
    start with (checked, so a moved line fails loudly rather than silently
    checking another branch), and the function it is in."""

    module: str
    start: int
    end: int
    first: str
    second: str
    function: str

    @property
    def pair(self) -> tuple[int, int]:
        return (self.start, self.end)

    def __str__(self) -> str:
        return f'{self.module} {self.start}->{self.end}'


#: The three branches, by the name of the scenarios that take them.
BRANCHES: dict[str, Branch] = {
    # `_seeded_order_by` of the SQLite sampler: the entity key's source
    # column has no dtype (`SQLiteTable._to_dtype` gives none for a declared
    # type with NUMERIC affinity), so the key's dtype is looked up instead.
    'sqlite-declared-key': Branch(
        f'{_ENGINE}/backend/sqlite/sampler.py',
        244,
        245,
        'if dtype is None:',
        'dtype = self.table_dtype_dict[table_name].get(key)',
        '_seeded_order_by',
    ),
    # `_first_uncastable` runs out of values without finding one that will
    # not cast alone, so the refusal names no index.
    'decimal-mixed-indices': Branch(
        f'{_ENGINE}/rfm.py',
        262,
        267,
        'for position, value in enumerate(values):',
        'return None',
        '_first_uncastable',
    ),
    # `_get_query_time_offset_dict` finds two edge types from the aggregated
    # table to the entity table: a table linked to itself.
    'self-link': Branch(
        f'{_ENGINE}/base/sampler.py',
        428,
        429,
        'if len(edge_types) != 1:',
        'raise ValueError(',
        '_add_time_offset',
    ),
}


def check_lines() -> None:
    r"""The branches named above are where this file says they are."""
    for name, branch in BRANCHES.items():
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


# --- Which lines run ----------------------------------------------------------


class LineWatch:
    r"""Notes, with ``sys.monitoring``, which of the branches' second lines
    run, in any thread, while it is installed.

    ``hits`` holds the name of each branch whose second line ran since the
    last ``clear()``. Every other line is disabled the first time it is
    reported, so the watch costs little.
    """

    def __init__(self) -> None:
        #: By line: the module's path below ``sdm_relational_engine`` and
        #: the branch's name.
        self.targets: dict[int, list[tuple[str, str]]] = {}
        for name, b in BRANCHES.items():
            below = b.module.split('/src/', 1)[1]
            self.targets.setdefault(b.end, []).append((below, name))
        self.hits: set[str] = set()
        self._tool: int | None = None

    def clear(self) -> None:
        self.hits = set()

    def _line(self, code: Any, line: int) -> Any:
        filename = code.co_filename.replace('\\', '/')
        for below, name in self.targets.get(line, ()):
            if filename.endswith('/' + below):
                if code.co_name == BRANCHES[name].function:
                    self.hits.add(name)
                return None
        return sys.monitoring.DISABLE

    @contextlib.contextmanager
    def installed(self) -> Iterator[LineWatch]:
        monitoring = sys.monitoring
        for tool in (5, 2):
            if monitoring.get_tool(tool) is None:
                monitoring.use_tool_id(tool, 'sdm-p77-lines')
                self._tool = tool
                break
        assert self._tool is not None, 'no sys.monitoring tool id is free'
        line_event = monitoring.events.LINE
        monitoring.register_callback(self._tool, line_event, self._line)
        monitoring.set_events(self._tool, line_event)
        try:
            yield self
        finally:
            monitoring.set_events(self._tool, 0)
            monitoring.register_callback(self._tool, line_event, None)
            monitoring.free_tool_id(self._tool)
            self._tool = None


# --- Transcripts ----------------------------------------------------------------


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text('utf-8'))


def all_transcripts() -> list[Path]:
    return sorted(TRANSCRIPTS.rglob('transcript.json'))


def recording_dir(root: Path, transcript_path: Path) -> Path:
    return root / transcript_path.parent.relative_to(TRANSCRIPTS)


def queries_of(transcript: dict[str, Any]) -> list[str]:
    return [
        call['args']['query']
        for call in transcript.get('calls', [])
        if call.get('call') == 'PredictiveQuery'
        and isinstance(call.get('args', {}).get('query'), str)
    ]


def materialize_calls(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        call.get('args', {})
        for call in transcript.get('calls', [])
        if call.get('call') == 'PredictiveQuery.materialize'
    ]


_ENTITY = re.compile(r'\bFOR\s+EACH\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)', re.I)


def entity_of(query: str) -> tuple[str, str] | None:
    r"""The ``(table, key)`` a query predicts for."""
    found = _ENTITY.search(query)
    return None if found is None else (found[1], found[2])


def error_of(transcript: dict[str, Any]) -> dict[str, Any] | None:
    error = transcript.get('error')
    return error if isinstance(error, dict) else None


def normalize_type(declared: Any) -> str:
    r"""A declared type, upper case, without white space."""
    return re.sub(r'\s+', '', str(declared or '')).upper()


_PRAGMA = re.compile(r'^\s*PRAGMA\s+table_info\s*\((.*)\)\s*;?\s*$', re.I)


def declared_types(path: Path) -> dict[tuple[str, str], str]:
    r"""``(table, column) -> declared type`` from the transcript's recorded
    ``PRAGMA table_info`` results, the table named as the statement names
    it, unquoted and without its schema."""
    from _parity.transcript import read_arrow

    transcript = read(path)
    out: dict[tuple[str, str], str] = {}
    for step in transcript.get('steps', []):
        if step.get('op') != 'execute' or not isinstance(step.get('sql'), str):
            continue
        found = _PRAGMA.match(step['sql'])
        if found is None or not step.get('result'):
            continue
        table = found[1].strip().split('.')[-1].strip().strip('"`[]')
        for row in read_arrow(path.parent / step['result']).to_pylist():
            out[(table, str(row['name']))] = str(row.get('type') or '')
    return out


# --- Replays with the watch on ---------------------------------------------------


@dataclass
class Replay:
    r"""One replay of a transcript under the watch."""

    path: Path
    reason: str | None
    hits: set[str]
    #: For the SQLite sampler's seeded draws: ``(table, key, source name,
    #: whether the key is a column of the source, the source column's
    #: dtype)``, one per call of ``_seeded_order_by``.
    seeded: list[tuple[str, str, str, bool, Any]] = field(default_factory=list)


def _sqlite_sampler_class() -> type:
    from sdm_relational_engine.rfm.backend.sqlite import sampler

    found = [
        value
        for value in vars(sampler).values()
        if isinstance(value, type) and '_seeded_order_by' in vars(value)
    ]
    assert len(found) == 1, f'no single SQLite sampler class: {found}'
    return found[0]


def replay_all(paths: list[Path]) -> dict[Path, Replay]:
    r"""Replay each of ``paths`` (``_parity.corpus.check``) with the line
    watch installed and the SQLite sampler's ``_seeded_order_by`` noted."""
    from _parity.corpus import check

    cls = _sqlite_sampler_class()
    original = vars(cls)['_seeded_order_by']
    seeded: list[tuple[str, str, str, bool, Any]] = []

    def noted(self: Any, table_name: str, *args: Any, **kwargs: Any) -> Any:
        key = self.primary_key_dict[table_name]
        source = self.source_table_dict[table_name].get(key)
        name = str(self.source_name_dict[table_name])
        seeded.append(
            (
                table_name,
                key,
                name.split('.')[-1].strip().strip('"`[]'),
                source is not None,
                None if source is None else source.dtype,
            )
        )
        return original(self, table_name, *args, **kwargs)

    out = {}
    watch = LineWatch()
    cls._seeded_order_by = noted
    try:
        with watch.installed():
            for path in paths:
                watch.clear()
                seeded.clear()
                reason = check(path)
                out[path] = Replay(path, reason, set(watch.hits), list(seeded))
    finally:
        cls._seeded_order_by = original
    return out


# --- The three kinds of scenario ------------------------------------------------

#: The declared types the SQLite scenarios give an entity key, normalized:
#: ``DECIMAL(10,0)``, ``NUMERIC`` and none.
SQLITE_DECLARED = ('DECIMAL(10,0)', 'NUMERIC', '')

INDICES_REFUSAL = re.compile(
    r"'indices' must hold values matching the primary key of "
    r"'(?P<table>[A-Za-z_]\w*)', which is (?P<dtype>decimal128\([^)]*\)\S*)"
)

SELF_LINK_REFUSAL = re.compile(
    r'Could not find a unique foreign key from table '
    r"'(?P<table>[A-Za-z_]\w*)' to '(?P=table)'"
)


@dataclass
class Found:
    r"""The transcripts of each kind, and the replays that showed it."""

    sqlite: dict[str, Path]
    #: Which of ``SQLITE_DECLARED`` each SQLite transcript's seeded draws
    #: took the branch on.
    sqlite_types: dict[str, set[str]]
    indices: dict[str, Path]
    self_link: dict[str, Path]
    replays: dict[Path, Replay]

    def every(self) -> dict[str, Path]:
        return {**self.sqlite, **self.indices, **self.self_link}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _indices_candidate(transcript: dict[str, Any]) -> bool:
    error = error_of(transcript)
    return (
        transcript.get('backend') == 'duckdb'
        and error is not None
        and INDICES_REFUSAL.search(str(error.get('message', ''))) is not None
    )


def _self_link_candidate(transcript: dict[str, Any]) -> bool:
    error = error_of(transcript)
    return (
        transcript.get('backend') == 'memory'
        and error is not None
        and SELF_LINK_REFUSAL.search(str(error.get('message', ''))) is not None
    )


@cache
def found() -> Found:
    r"""Every transcript of the three kinds (``CONTRACT.md``, "Names").

    Every SQLite transcript replays under the watch, and so does every
    candidate of the other two kinds, so the branch decides, not the name.
    """
    paths = all_transcripts()
    by_path = {path: read(path) for path in paths}
    wanted = [
        path
        for path, t in by_path.items()
        if t.get('backend') == 'sqlite'
        or _indices_candidate(t)
        or _self_link_candidate(t)
    ]
    replays = replay_all(wanted)
    sqlite: dict[str, Path] = {}
    sqlite_types: dict[str, set[str]] = {}
    indices: dict[str, Path] = {}
    self_link: dict[str, Path] = {}
    for path in wanted:
        transcript = by_path[path]
        name = transcript['name']
        replay = replays[path]
        if transcript.get('backend') == 'sqlite':
            if 'sqlite-declared-key' not in replay.hits:
                continue
            declared = declared_types(path)
            types = set()
            for table, key, source, physical, dtype in replay.seeded:
                if not physical or dtype is not None:
                    continue
                spelled = declared.get((source, key))
                if spelled is None:
                    spelled = declared.get((table, key))
                if spelled is None:
                    continue
                spelled = normalize_type(spelled)
                if spelled in SQLITE_DECLARED:
                    types.add(spelled)
            if types:
                sqlite[name] = path
                sqlite_types[name] = types
        elif _indices_candidate(transcript):
            if 'decimal-mixed-indices' in replay.hits:
                indices[name] = path
        elif _self_link_candidate(transcript):
            if 'self-link' in replay.hits:
                self_link[name] = path
    return Found(sqlite, sqlite_types, indices, self_link, replays)


def describe_found() -> str:
    f = found()
    return (
        f'SQLite declared keys {sorted(f.sqlite_types.items())}; '
        f'DuckDB decimal mixed indices {sorted(f.indices)}; '
        f'in-memory self links {sorted(f.self_link)}'
    )


def mixed_indices(transcript: dict[str, Any]) -> list[list[Any]]:
    r"""Each materialize call's ``indices`` holding both an ``int`` and a
    ``str``."""
    out = []
    for args in materialize_calls(transcript):
        indices = args.get('indices')
        if isinstance(indices, list) and (
            any(_is_int(v) for v in indices)
            and any(isinstance(v, str) for v in indices)
        ):
            out.append(indices)
    return out


def gap_entries() -> dict[str, set[tuple[int, int]]]:
    gaps = json.loads(GAPS.read_text('utf-8'))
    assert isinstance(gaps, dict), f'{GAPS} is not an object'
    out = {}
    for module, entries in gaps.items():
        key = _historical_path(module)
        assert key not in out, 'duplicate projected gap module'
        out[key] = {tuple(entry['branch']) for entry in entries}
    return out


def coverage_modules(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for entry in report['crates'].values():
        for module, body in entry['modules'].items():
            key = _historical_path(module)
            assert key not in out, 'duplicate projected coverage module'
            out[key] = body
    return out


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
