# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 8.5 acceptance checks share.

- The guard: no ported Python runs while the host runs.
- The recorder: it sits on ``sdm_core._native``, on ``Source._run`` and on
  the model's stand-in, holds the host to ``CONTRACT.md``, and keeps every
  binding call.
- Replaying a transcript through the host, and through today's Python.
- The recordings: writing them, and replaying them through a binding.

The interface these rely on is ``CONTRACT.md``, beside this file. Nothing
here imports pandas, pyarrow or the packages at import time, so
``_redrive.py`` can use it in a process where they cannot be imported.

    python tests/rust_core/request/_request.py capture --out DIR

writes the recordings of every transcript under ``DIR``.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import fnmatch
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pkgutil
import re
import shutil
import sys
import threading
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUST_CORE = HERE.parent
REPO = HERE.parents[2]
RUST = REPO / 'rust'
PARITY = REPO / 'tests' / 'parity'
TRANSCRIPTS = PARITY / 'transcripts'
RECORDINGS = PARITY / 'requests'
FIXTURES = PARITY / 'fixtures'
GOLDEN = REPO / 'tests' / 'golden' / 'wire_payloads.json'
SCRIPTS = REPO / 'scripts' / 'parity'
HOST = SCRIPTS / 'request_host.py'
EXCLUSIONS = SCRIPTS / 'request_fixture_exclusions.json'
GAPS = SCRIPTS / 'request_coverage_gaps.json'
PORTED = HERE / 'ported.tsv'
FIXTURE_TEST = RUST / 'crates' / 'core' / 'tests' / 'accept_request_fixtures.rs'
REPLAY_TEST = RUST / 'crates' / 'core' / 'tests' / 'request_replay.rs'

for _path in (
    HERE,
    RUST_CORE,
    RUST_CORE / 'sampling',
    RUST_CORE / 'sampling' / 'driver',
    PARITY,
    SCRIPTS,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

#: The porting ledger's source roots, and the files it leaves out.
SOURCES = (
    'packages/structured-data-models-pql/src',
    'packages/structured-data-models-relational-engine/src',
    'packages/structured-data-models-client/src',
    'packages/structured-data-models-connectors/src',
    'packages/structured-data-models-api/src',
)
PACKAGE_RENAMES = {
    'sdm-api': 'structured-data-models-api',
    'sdm-client': 'structured-data-models-client',
    'sdm-connectors': 'structured-data-models-connectors',
    'sdm-pql': 'structured-data-models-pql',
    'sdm-relational-engine': 'structured-data-models-relational-engine',
}
SKIP = ('*/generated/*', '*/grammar/PQLGrammar*.py', '*/__pycache__/*')
#: The packages imported before the guard watches, so no import runs under it.
PACKAGES = (
    'sdm_api',
    'sdm_connectors',
    'predictive_query',
    'sdm_relational_engine',
    'sdm_client',
)

#: The crates whose ledger rows are ported code.
CORE_CRATES = frozenset(
    {'pql', 'schema', 'dialects', 'sampling', 'labels', 'request', 'core'}
)
#: The operations of the request fixture cases.
REQUEST_PREFIXES = ('client.', 'decompose.', 'payload.', 'tabular.', 'wire.')
#: The runs the brief names, and its pure functions.
RUNS = ('materialize_run', 'predict_run', 'fit_run', 'session_predict_run')
PURE = ('flatten', 'merge', 'absent')
ENDPOINTS = ('predict', 'session_create', 'session_predict', 'session_delete')
STEP_KINDS = ('run_queries', 'call_model', 'done')
#: Backends whose data is in memory: no statements.
IN_MEMORY = ('memory', 'tabular')
LIVE = ('databricks', 'lakebase', 'snowflake')

#: A request id the core makes, and how recordings write it.
REQUEST_ID = re.compile(r'sdm-[0-9a-f]{12}')
REQUEST_ID_TEXT = 'sdm-<request-id>'

#: The recording format, and its files.
FORMAT = 1
CALLS_FILE = 'calls.json'
TABLES_DIR = 'tables'

#: Whether the guard watches. Only the checks' own self-tests turn it off.
GUARD_ENABLED = True


# --- Small helpers -------------------------------------------------------------


def driver_module() -> Any:
    r"""Phase 7.5's ``_driver``: canonical forms, encodings, Arrow files."""
    import _driver

    return _driver


def rc_module() -> Any:
    r"""Phase 4's ``_rc``: cargo, copies and JSON comparison."""
    import _rc

    return _rc


def native(*names: str) -> Any:
    r"""``sdm_core._native``, built from this checkout and installed into
    this environment once per process (Phase 7's helper), holding ``names``.
    """
    from _sampling import native as build

    return build(*names)


def short(value: Any, limit: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=repr)
    return text if len(text) <= limit else text[:limit] + '...'


def qualname(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def json_diff(a: Any, b: Any, path: str = '$') -> str | None:
    r"""Where two parsed JSON values differ, by value or by JSON type."""
    return rc_module().json_diff(a, b, path)


def normalize_text(text: str) -> str:
    r"""``text`` with each request id the core makes written alike."""
    return REQUEST_ID.sub(REQUEST_ID_TEXT, text)


def normalize_json(value: Any) -> Any:
    r"""Parsed JSON with every string's request ids written alike."""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [normalize_json(item) for item in value]
    if isinstance(value, dict):
        return {normalize_text(k): normalize_json(v) for k, v in value.items()}
    return value


def without_request_id(body: Any) -> Any:
    r"""A posted body as transcripts store it: ``metadata.request_id`` set to
    ``<request-id>``."""
    body = json.loads(json.dumps(body))
    metadata = body.get('metadata') if isinstance(body, dict) else None
    if isinstance(metadata, dict) and 'request_id' in metadata:
        metadata['request_id'] = '<request-id>'
    return body


def transcripts() -> list[Path]:
    return sorted(TRANSCRIPTS.rglob('transcript.json'))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def scenario_dir(root: Path, name: str) -> Path:
    r"""Where a scenario's files go, as for transcripts: ``#`` as ``--``."""
    return root.joinpath(*name.replace('#', '--').split('/'))


def request_fixture_cases() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(FIXTURES.glob('*.json')):
        for case in load_json(path)['cases']:
            if str(case.get('operation', '')).startswith(REQUEST_PREFIXES):
                out.append(case)
    return out


def exclusions() -> dict[str, str]:
    r"""``scripts/parity/request_fixture_exclusions.json``, or nothing."""
    if not EXCLUSIONS.is_file():
        return {}
    data = load_json(EXCLUSIONS)
    assert isinstance(data, dict), f'{EXCLUSIONS.name} is not a JSON object'
    return data


# --- Plain data ------------------------------------------------------------------


def _arrow_types() -> tuple[type, ...]:
    import pyarrow as pa

    return (pa.Table, pa.RecordBatch, pa.Array, pa.ChunkedArray)


def plain_problem(value: Any, path: str = '$') -> str | None:
    r"""Why ``value`` is not plain data (``CONTRACT.md``), or ``None``."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return None
    if isinstance(value, _arrow_types()):
        return None
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            found = plain_problem(item, f'{path}[{i}]')
            if found:
                return found
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                return f'{path} has the key {key!r}, not a str'
            found = plain_problem(item, f'{path}.{key}')
            if found:
                return found
        return None
    return f'{path} is a {qualname(type(value))}'


def freeze(value: Any) -> Any:
    r"""A copy of plain data that later changes to the original don't reach."""
    if isinstance(value, list):
        return [freeze(item) for item in value]
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    return value


def tables_in(value: Any) -> Iterator[Any]:
    r"""Every ``pyarrow`` table or batch in plain data."""
    import pyarrow as pa

    if isinstance(value, (pa.Table, pa.RecordBatch)):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tables_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tables_in(item)


def _parsed(text: str) -> Any:
    stripped = text.lstrip()
    if not stripped or stripped[0] not in '[{':
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def strings_in(value: Any, depth: int = 0) -> Iterator[str]:
    r"""Every string in a value, JSON text read as well as kept whole."""
    if depth > 40:
        return
    if isinstance(value, str):
        yield value
        inner = _parsed(value)
        if inner is not None:
            yield from strings_in(inner, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings_in(item, depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from strings_in(item, depth + 1)


def warnings_in(value: Any, depth: int = 0) -> Iterator[tuple[str, str]]:
    r"""Every ``{"category", "message"}`` object in a value, JSON text read."""
    if depth > 40:
        return
    if isinstance(value, str):
        inner = _parsed(value)
        if inner is not None:
            yield from warnings_in(inner, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from warnings_in(item, depth + 1)
    elif isinstance(value, dict):
        category, message = value.get('category'), value.get('message')
        if isinstance(category, str) and isinstance(message, str):
            yield category, message
        for item in value.values():
            yield from warnings_in(item, depth + 1)


def error_parts(error: BaseException) -> dict[str, Any]:
    r"""An error's plain-data attributes: its ``args`` and its ``__dict__``."""
    out: dict[str, Any] = {}
    args = list(getattr(error, 'args', ()) or ())
    if plain_problem(args) is None:
        out['args'] = args
    for key, item in sorted(vars(error).items()):
        if key.startswith('_') or key == 'args':
            continue
        if callable(item) or plain_problem(item) is not None:
            continue
        out[key] = item
    return out


# --- The ledger and the guard -------------------------------------------------------


def harness() -> Path:
    return Path(
        os.environ.get('SDM_RC_HARNESS', '~/.local/share/sdm-dbx/rust-core')
    ).expanduser()


def current_package_path(path: str) -> str:
    for old, new in PACKAGE_RENAMES.items():
        prefix = f'packages/{old}/'
        if path.startswith(prefix):
            return f'packages/{new}/' + path[len(prefix) :]
    return path


def ledger_rules() -> list[tuple[re.Pattern[str], re.Pattern[str], str]]:
    r"""``ledger/rules.tsv``, in order, as ``tools/ledger.py`` reads it."""
    path = harness() / 'ledger' / 'rules.tsv'
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        pattern, dest = line.split('\t')[:2]
        path_glob, _, symbol_glob = pattern.partition('::')
        out.append(
            (
                re.compile(fnmatch.translate(current_package_path(path_glob))),
                re.compile(fnmatch.translate(symbol_glob or '*')),
                dest.strip(),
            )
        )
    return out


def rule_destination(
    rules: list[tuple[re.Pattern[str], re.Pattern[str], str]],
    path: str,
    symbol: str,
) -> str:
    for path_re, symbol_re, dest in rules:
        if path_re.match(path) and symbol_re.match(symbol):
            return dest
    return 'UNMAPPED'


def ported_snapshot() -> dict[tuple[str, str], str]:
    r"""``ported.tsv``: the ledger rows sent to a core crate at lock time."""
    out = {}
    for line in PORTED.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        file, symbol, dest = line.split('\t')[:3]
        out[(current_package_path(file), symbol)] = dest
    return out


def is_core(dest: str) -> bool:
    return dest.removeprefix('triage:') in CORE_CRATES


def spans_of(path: Path) -> list[tuple[str, int, int]]:
    r"""Every function and class of a module as the ledger names them
    (top-level ones and class members), with their first line, decorators
    included, and their last."""
    found: list[tuple[str, int, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            name = f'{prefix}{node.name}'
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            found.append((name, start, node.end_lineno or node.lineno))
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    walk(child, f'{name}.')

    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in tree.body:
        walk(node, '')
    return found


def source_files() -> list[tuple[str, Path]]:
    out = []
    for root in SOURCES:
        for path in sorted((REPO / root).rglob('*.py')):
            rel = path.relative_to(REPO).as_posix()
            if any(fnmatch.fnmatch(rel, glob) for glob in SKIP):
                continue
            out.append((rel, path))
    return out


@dataclass
class PortedFile:
    rel: str
    module_core: bool
    spans: list[tuple[str, int, int, bool]]

    def owner(self, line: int) -> tuple[str, bool] | None:
        inside = [s for s in self.spans if s[1] <= line <= s[2]]
        if not inside:
            return None
        name, _, _, core = min(inside, key=lambda s: s[2] - s[1])
        return name, core


def ported_files() -> dict[str, PortedFile]:
    r"""By absolute path, as given and resolved, every source file holding
    ported code."""
    rules = ledger_rules()
    snapshot = ported_snapshot()
    out: dict[str, PortedFile] = {}
    for rel, path in source_files():

        def core(symbol: str, rel: str = rel) -> bool:
            if is_core(snapshot.get((rel, symbol), '')):
                return True
            return is_core(rule_destination(rules, rel, symbol))

        spans = [(n, a, b, core(n)) for n, a, b in spans_of(path)]
        module_core = core('<module>')
        if module_core or any(s[3] for s in spans):
            found = PortedFile(rel, module_core, spans)
            out[str(path)] = found
            out[str(path.resolve())] = found
    return out


def import_everything() -> None:
    r"""Import every module of the packages, so no import runs under the
    guard; one that cannot be imported here is left alone."""
    for name in PACKAGES:
        try:
            package = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - an optional package
            continue
        for info in pkgutil.walk_packages(
            getattr(package, '__path__', []), prefix=f'{name}.'
        ):
            with contextlib.suppress(Exception), warnings.catch_warnings():
                warnings.simplefilter('ignore')
                importlib.import_module(info.name)


class Guard:
    r"""Notes every start of ported code, in any thread, while active.

    ``sys.monitoring`` reports each function, lambda and comprehension as it
    starts or resumes; code that isn't ported is never reported again. Code
    beneath ``Source._run``, which the host is meant to call, pauses it.
    """

    def __init__(self) -> None:
        self.files = ported_files()
        self.active = 0
        self.paused = 0
        self.trips: list[str] = []
        self._decided: dict[Any, str | None] = {}
        self._tool: int | None = None

    def _decide(self, code: Any) -> str | None:
        found = self.files.get(code.co_filename)
        if found is None or code.co_name == '<module>':
            return None
        owner = found.owner(code.co_firstlineno)
        if owner is None:
            return f'{found.rel}::<module>' if found.module_core else None
        name, core = owner
        return f'{found.rel}::{name}' if core else None

    def _event(self, code: Any, offset: int) -> Any:
        decided = self._decided.get(code, False)
        if decided is False:
            decided = self._decide(code)
            self._decided[code] = decided
        if decided is None:
            return sys.monitoring.DISABLE
        if self.active and not self.paused:
            self.trips.append(decided)
        return None

    def install(self) -> None:
        if not GUARD_ENABLED or self._tool is not None:
            return
        monitoring = getattr(sys, 'monitoring', None)
        assert monitoring is not None, 'the guard needs Python 3.12 or later'
        import_everything()
        for tool in (4, 3):
            if monitoring.get_tool(tool) is None:
                monitoring.use_tool_id(tool, 'sdm-p85-guard')
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
        events = monitoring.events
        for event in (events.PY_START, events.PY_RESUME):
            monitoring.register_callback(self._tool, event, None)
        monitoring.free_tool_id(self._tool)
        self._tool = None


# --- The recorder -----------------------------------------------------------------


class Breach(BaseException):
    r"""The host broke ``CONTRACT.md``. A ``BaseException``, so no
    ``except Exception`` in the host hides it."""

    def __init__(self, aspect: str, message: str) -> None:
        super().__init__(f'{aspect}: {message}')
        self.aspect = aspect
        self.message = message


@dataclass
class HostCall:
    r"""One call of the host, and what happened beneath it."""

    name: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    runs: list[RunProxy] = field(default_factory=list)
    #: Warnings the core's steps carried.
    offered: list[tuple[str, str]] = field(default_factory=list)
    #: What the core returned or raised, searched only when needed.
    values: list[Any] = field(default_factory=list)
    errors: list[BaseException] = field(default_factory=list)
    described: int = 0
    problems: list[tuple[str, str]] = field(default_factory=list)
    warn_start: int = 0

    def problem(self, aspect: str, message: str) -> None:
        self.problems.append((aspect, f'{self.name}: {message}'))

    def note_value(self, value: Any) -> None:
        self.values.append(value)

    def note_error(self, error: BaseException) -> None:
        self.errors.append(error)

    def all_offered(self) -> list[tuple[str, str]]:
        r"""Every warning the core gave: carried by a step or an error, or
        held in a value it returned."""
        out = list(self.offered)
        for value in self.values:
            out.extend(warnings_in(value))
        for error in self.errors:
            out.extend(warnings_in(getattr(error, 'warnings', None)))
        return out

    def texts(self) -> Iterator[str]:
        r"""Every text the core gave: its errors' and its values'."""
        for error in self.errors:
            yield str(error)
            yield from strings_in(error_parts(error))
        for value in self.values:
            yield from strings_in(value)


class RecordedOutcome:
    r"""An ``Outcome`` the host made, with what it was made from."""

    def __init__(self, kind: str, args: tuple[Any, ...], real: Any) -> None:
        self.kind = kind
        self.args = args
        self.real = real


class OutcomeFactory:
    r"""Stands in for ``sdm_core._native.Outcome``, keeping what each
    outcome is made from."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        found = getattr(self._real, name)
        if name.startswith('_') or not callable(found):
            return found

        def make(*args: Any) -> RecordedOutcome:
            problem = plain_problem(list(args))
            if problem is not None:
                raise Breach('plain', f'Outcome.{name}: {problem}')
            return RecordedOutcome(name, freeze(args), found(*args))

        return make


def _is_run(value: Any) -> bool:
    if value is None or isinstance(
        value, (str, bytes, list, tuple, dict, int, float, bool)
    ):
        return False
    return callable(getattr(value, 'next', None)) and callable(
        getattr(value, 'submit', None)
    )


def _session_path(endpoint: str, session_id: Any) -> tuple[str, str]:
    if endpoint == 'predict':
        return 'POST', '/v1/predictions'
    if endpoint == 'session_create':
        return 'POST', '/v1/sessions'
    if endpoint == 'session_predict':
        return 'POST', f'/v1/sessions/{session_id}/predictions'
    return 'DELETE', f'/v1/sessions/{session_id}'


class RunProxy:
    r"""A run the host was given, which keeps every step and outcome and
    holds the host to the protocol."""

    def __init__(
        self,
        recorder: Recorder,
        real: Any,
        call: dict[str, Any],
        host_call: HostCall,
    ) -> None:
        self._recorder = recorder
        self._real = real
        self._call = call
        self._host_call = host_call
        self.function = call['function']
        self.steps: list[dict[str, Any]] = []
        self.finished = False
        #: For the last run_queries step: how many queries ran, what each gave.
        self.ran = 0
        self.results: list[Any] = []
        #: For the last call_model step: whether it was sent, and the answer.
        self.sent = False
        self.answered: tuple[int, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def _breach(self, aspect: str, message: str) -> Breach:
        self._host_call.problem(aspect, message)
        return Breach(aspect, f'{self.function}: {message}')

    def _settle(self) -> None:
        if not self.steps:
            return
        last = self.steps[-1]
        if last['kind'] == 'run_queries':
            if self.ran < len(last['queries']):
                raise self._breach(
                    'statements',
                    f'next() was called with {len(last["queries"]) - self.ran} '
                    f"of the step's {len(last['queries'])} queries not run",
                )
            if any(o is None for o in last['outcomes']):
                raise self._breach(
                    'protocol',
                    "next() was called before every query's outcome was "
                    'submitted',
                )
        elif last['kind'] == 'call_model':
            if not self.sent:
                raise self._breach(
                    'requests',
                    f'next() was called without making the '
                    f'{last["endpoint"]} call the run asked for',
                )
            if last['outcome'] is None:
                raise self._breach(
                    'protocol',
                    "next() was called before the call's outcome was submitted",
                )

    def next(self) -> Any:
        if self.finished:
            return self._real.next()
        self._settle()
        try:
            step = self._real.next()
        except Exception as error:
            self.finished = True
            self.steps.append({'kind': 'raised', 'error': error})
            self._host_call.note_error(error)
            self._recorder.keep_returned(error_parts(error))
            raise
        kind = getattr(step, 'kind', None)
        if kind not in STEP_KINDS:
            raise self._breach('protocol', f'a step of kind {kind!r}')
        given = getattr(step, 'warnings', None)
        if not isinstance(given, (list, tuple)) or not all(
            isinstance(w, dict)
            and isinstance(w.get('category'), str)
            and isinstance(w.get('message'), str)
            for w in given
        ):
            raise self._breach(
                'protocol',
                f"a {kind} step's warnings are {short(repr(given))}, not a "
                f'list of {{"category", "message"}}',
            )
        found = [
            {'category': w['category'], 'message': w['message']} for w in given
        ]
        self._host_call.offered.extend(
            (w['category'], w['message']) for w in found
        )
        if kind == 'run_queries':
            queries = list(step.queries)
            encode_params = driver_module().encode_params
            record = {
                'kind': kind,
                'queries': [
                    {'sql': q.sql, 'params': encode_params(q.params)}
                    for q in queries
                ],
                'warnings': found,
                'outcomes': [None] * len(queries),
                'ids': [q.id for q in queries],
            }
            for q in queries:
                problem = plain_problem(
                    {'sql': q.sql, 'params': list(q.params or [])}
                )
                if problem or not isinstance(q.sql, str):
                    raise self._breach('plain', f'a query: {problem}')
            self.ran = 0
            self.results = [None] * len(queries)
        elif kind == 'call_model':
            endpoint = getattr(step, 'endpoint', None)
            session_id = getattr(step, 'session_id', None)
            body = getattr(step, 'body', None)
            if endpoint not in ENDPOINTS:
                raise self._breach('protocol', f'the endpoint {endpoint!r}')
            if endpoint in ('session_predict', 'session_delete') and not (
                isinstance(session_id, str) and session_id
            ):
                raise self._breach(
                    'protocol', f'a {endpoint} call without a session id'
                )
            if endpoint == 'session_delete':
                parsed = None
                if body is not None:
                    raise self._breach(
                        'protocol', 'a session_delete call with a body'
                    )
            else:
                if not isinstance(body, str):
                    raise self._breach(
                        'protocol', f'a {endpoint} call whose body is not text'
                    )
                try:
                    parsed = json.loads(body)
                except ValueError as error:
                    raise self._breach(
                        'protocol',
                        f'a {endpoint} body that is not JSON: {error}',
                    ) from None
            record = {
                'kind': kind,
                'endpoint': endpoint,
                'session_id': session_id,
                'body': parsed,
                'warnings': found,
                'outcome': None,
                'id': step.id,
            }
            self.sent = False
            self.answered = None
        else:
            value = step.value
            problem = plain_problem(value)
            if problem is not None:
                raise self._breach('plain', f'the done value: {problem}')
            record = {'kind': kind, 'value': freeze(value), 'warnings': found}
            self._host_call.note_value(value)
            self._recorder.keep_returned(value)
            self.finished = True
        self.steps.append(record)
        return step

    def submit(self, query_id: Any, outcome: Any) -> Any:
        if not isinstance(outcome, RecordedOutcome):
            raise self._breach(
                'protocol',
                'an outcome not made with sdm_core._native.Outcome',
            )
        last = self.steps[-1] if self.steps else None
        if last is None or last['kind'] not in ('run_queries', 'call_model'):
            raise self._breach('protocol', 'an outcome no step asked for')
        if last['kind'] == 'run_queries':
            at = next(
                (i for i, known in enumerate(last['ids']) if known == query_id),
                None,
            )
            if at is None:
                raise self._breach(
                    'protocol', f'an outcome for the unknown query {query_id!r}'
                )
            if at >= self.ran:
                raise self._breach(
                    'statements',
                    f'an outcome for query {at}, which the host did not run '
                    f'through Source._run',
                )
            if outcome.kind == 'data':
                if not (outcome.args and outcome.args[0] is self.results[at]):
                    raise self._breach(
                        'inputs',
                        f'Outcome.data for query {at} is not what Source._run '
                        f'returned',
                    )
            elif outcome.kind != 'error':
                raise self._breach(
                    'protocol', f'Outcome.{outcome.kind} for a query'
                )
            last['outcomes'][at] = outcome
        else:
            if query_id != last['id']:
                raise self._breach(
                    'protocol', f'an outcome for the unknown call {query_id!r}'
                )
            if not self.sent or self.answered is None:
                raise self._breach(
                    'requests',
                    'an outcome for a model call the host did not make',
                )
            self._check_answer(outcome)
            last['outcome'] = outcome
        return self._real.submit(query_id, outcome.real)

    def _check_answer(self, outcome: RecordedOutcome) -> None:
        assert self.answered is not None
        status, body = self.answered
        text = outcome.args[0] if outcome.args else None
        if not isinstance(text, str):
            raise self._breach(
                'protocol', f'Outcome.{outcome.kind} takes JSON text'
            )
        try:
            given = json.loads(text)
        except ValueError:
            raise self._breach(
                'protocol', f'Outcome.{outcome.kind} of text that is not JSON'
            ) from None
        if outcome.kind == 'response':
            if status >= 400:
                raise self._breach(
                    'requests',
                    f'Outcome.response for an answer of status {status}',
                )
            found = json_diff(body, given)
            if found:
                raise self._breach(
                    'requests',
                    f'Outcome.response is not what the model answered: {found}',
                )
        elif outcome.kind == 'model_error':
            if not isinstance(given, dict) or not {
                'class',
                'status',
                'code',
                'message',
                'details',
            } <= set(given):
                raise self._breach(
                    'protocol',
                    f'Outcome.model_error of {short(text)}, which lacks a '
                    f'key CONTRACT.md names',
                )
            if status >= 400 and given.get('status') != status:
                raise self._breach(
                    'requests',
                    f'Outcome.model_error says status {given.get("status")}, '
                    f'and the model answered {status}',
                )
        else:
            raise self._breach(
                'protocol', f'Outcome.{outcome.kind} for a model call'
            )

    # Hooks the recorder calls.

    def ask_statement(self, sql: Any, params: Any) -> int | None:
        last = self.steps[-1] if self.steps else None
        if self.finished or last is None or last['kind'] != 'run_queries':
            return None
        if self.ran >= len(last['queries']):
            return None
        want = last['queries'][self.ran]
        got = driver_module().encode_params(params)
        if sql != want['sql'] or json.dumps(got) != json.dumps(want['params']):
            raise self._breach(
                'statements',
                f'Source._run({short(sql, 120)!r}, {short(got, 80)}) ran where '
                f'query {self.ran} is {short(want["sql"], 120)!r} with '
                f'{short(want["params"], 80)}',
            )
        self.ran += 1
        return self.ran - 1

    def pending_call(self) -> dict[str, Any] | None:
        last = self.steps[-1] if self.steps else None
        if self.finished or last is None or last['kind'] != 'call_model':
            return None
        return None if self.sent else last


class ProtocolCapture:
    slots = {
        'flatten': {1: 'infer_column', 2: 'column_order', 3: 'result'},
        'merge': {1: 'equal', 2: 'truth', 3: 'metadata_entry', 4: 'result'},
        'absent': {2: 'result'},
        'materialize_run': {2: 'sequence', 3: 'sequence'},
        'request_materialized': {2: 'held'},
        'predict_run': {2: 'held'},
        'fit_run': {2: 'fit', 3: 'sequence'},
        'session_predict_run': {2: 'held'},
        'request_op': {3: 'held'},
    }
    operations = {
        'sequence': {'iter', 'list', 'len', 'bool'},
        'result': {
            'set',
            'sorted_repr',
            'dict_keys',
            'contains',
            'equal',
            'getattr_default_none',
        },
        'held': {
            'truth',
            'hashable',
            'repr',
            'set_cardinality',
            'mapping_get',
            'getitem',
            'is_instance',
            'mapping_set',
            'mapping_copy',
            'mapping_pop',
            'sorted_set_difference',
            'sorted_set_intersection',
            'dataclass_nondefault_fields',
        },
    }
    sources = {
        'sdm_client._core': 'packages/structured-data-models-client/src/sdm_client/_core.py',
        'sdm_client._request_core': 'packages/structured-data-models-client/src/sdm_client/_request_core.py',
        'sdm_core.answers': 'rust/bindings/python/python/sdm_core/answers.py',
    }

    def __init__(self, recorder, host, name):
        self.recorder = recorder
        self.host = host
        self.name = name
        self.calls = []
        self.providers = []
        self.codes = {}

    def require(self, condition, message):
        if not condition:
            self.host.problem('protocol', message)
            raise Breach('protocol', message)

    def authenticate(self, module_name, class_name, method):
        import types

        module = sys.modules.get(module_name)
        self.require(module is not None, 'callback module is not loaded')
        path = Path(module.__file__).resolve()
        canonical = REPO / self.sources[module_name]
        raw = canonical.read_bytes()
        self.require(
            module.__spec__ is not None
            and Path(module.__spec__.origin).resolve() == path
            and path.read_bytes() == raw,
            'callback source identity differs',
        )
        cls = vars(module).get(class_name)
        self.require(type(cls) is type, 'callback class identity differs')
        function = vars(cls).get(method)
        self.require(
            type(function) is types.FunctionType, 'callback method differs'
        )
        self.require(
            Path(function.__code__.co_filename).resolve() == path,
            'callback code origin differs',
        )
        key = (raw, function.__code__.co_filename)
        if key not in self.codes:
            pending = [compile(raw, key[1], 'exec', dont_inherit=True)]
            found = []
            while pending:
                code = pending.pop()
                found.append(code)
                pending.extend(
                    item
                    for item in code.co_consts
                    if isinstance(item, types.CodeType)
                )
            self.codes[key] = tuple(found)
        definitions = ast.parse(raw).body
        expected_class = next(
            node
            for node in definitions
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        members = [
            (vars(cls), node, class_name + '.') for node in expected_class.body
        ]
        members.extend((vars(module), node, '') for node in definitions)
        for namespace, node, prefix in members:
            if not isinstance(node, ast.FunctionDef):
                continue
            member = namespace.get(node.name)
            expected = [
                code
                for code in self.codes[key]
                if code.co_qualname == prefix + node.name
            ]
            self.require(
                len(expected) == 1
                and type(member) is types.FunctionType
                and member.__globals__ is vars(module)
                and member.__code__ == expected[0],
                'callback source member or defining globals differ',
            )
        return cls, function

    def check(self, value, kind):
        import types

        if kind == 'sequence':
            module, cls_name, method = (
                'sdm_client._core',
                'SequenceValue',
                '__call__',
            )
        elif kind == 'fit':
            module, cls_name, method = (
                'sdm_client._request_core',
                '_FitProtocols',
                '__call__',
            )
        else:
            module = 'sdm_core.answers'
            cls_name = 'HeldValues' if kind == 'held' else 'ResultValues'
            method = 'protocol' if kind in ('held', 'result') else kind
        cls, function = self.authenticate(module, cls_name, method)
        if kind in ('sequence', 'fit'):
            self.require(type(value) is cls, 'callback owner type differs')
        else:
            self.require(
                type(value) is types.MethodType
                and type(value.__self__) is cls
                and value.__func__ is function,
                'callback bound method differs',
            )
        if kind == 'sequence':
            self.authenticate('sdm_core.answers', 'HeldValues', 'protocol')
        if kind == 'fit':
            self.check(value.references.protocol, 'held')
            if value.sequence is not None:
                self.check(value.sequence, 'sequence')

    def wrap(self, value, kind, slot, path=()):
        self.check(value, kind)
        self.providers.append(
            {'slot': slot, 'path': list(path), 'provider': kind}
        )

        def invoke(*arguments, **keywords):
            self.require(
                self.recorder._mine() is self.host,
                'callback escaped its host call',
            )
            self.check(value, kind)
            self.require(not keywords, 'callback keyword arguments differ')
            self.require(
                plain_problem(list(arguments)) is None,
                'callback arguments are not plain',
            )
            protocol = 'held' if kind == 'fit' else kind
            if protocol in self.operations:
                self.require(
                    len(arguments) == (1 if protocol == 'sequence' else 2)
                    and type(arguments[0]) is str
                    and arguments[0] in self.operations[protocol]
                    and (protocol == 'sequence' or type(arguments[1]) is str),
                    'callback operation differs',
                )
                operation = arguments[0]
            else:
                self.require(
                    len(arguments)
                    == {
                        'infer_column': 1,
                        'column_order': 4,
                        'equal': 2,
                        'truth': 1,
                        'metadata_entry': 3,
                    }[kind],
                    'callback argument count differs',
                )
                operation = kind
            row = {
                'slot': slot,
                'path': list(path),
                'operation': operation,
                'args': freeze(list(arguments)),
            }
            self.calls.append(row)
            try:
                returned = value(*arguments)
            except BaseException as error:
                row['outcome'] = {'raised': error}
                raise
            else:
                self.require(
                    plain_problem(returned) is None,
                    'callback returned nonplain data',
                )
                row['outcome'] = {'returned': freeze(returned)}
                return returned
            finally:
                self.check(value, kind)

        if kind == 'fit':
            invoke.sequence = (
                None
                if value.sequence is None
                else self.wrap(value.sequence, 'sequence', slot, ('sequence',))
            )
        return invoke

    def arguments(self, given):
        actual, recorded = list(given), list(given)
        for slot, kind in self.slots.get(self.name, {}).items():
            if slot < len(actual) and actual[slot] is not None:
                actual[slot] = self.wrap(actual[slot], kind, slot)
                recorded[slot] = None
        return actual, recorded


class Recorder:
    r"""Sits on ``sdm_core._native``, ``sdm_core.describe.describe_frames``
    and ``Source._run`` (installed once per process), keeps every binding
    call a host call makes, and holds the host to ``CONTRACT.md``."""

    def __init__(self, module: Any, guard: Guard | None = None) -> None:
        self.module = module
        self.guard = guard or Guard()
        self.current: HostCall | None = None
        self.host_thread: int | None = None
        self.undo: list[tuple[Any, str, Any]] = []
        self.reset()

    # Scope: one transcript, or one flow.

    def reset(self, warnings_log: list[Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.host_calls: list[HostCall] = []
        self.problems: list[tuple[str, str]] = []
        self.allowed_tables: list[Any] = []
        self.warnings_log = warnings_log

    def keep_returned(self, value: Any) -> None:
        self.allowed_tables.extend(tables_in(value))

    # Installing.

    def _patch(self, owner: Any, name: str, value: Any) -> None:
        self.undo.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def install(self) -> Recorder:
        if self.undo:
            return self
        self.guard.install()
        module = self.module
        for name in sorted(dir(module)):
            if name.startswith('_'):
                continue
            found = getattr(module, name)
            if name == 'Outcome':
                self._patch(module, name, OutcomeFactory(found))
            elif callable(found) and not isinstance(found, type):
                self._patch(module, name, self._wrap(name, found))
        describe = importlib.import_module('sdm_core.describe')
        self._patch(
            describe,
            'describe_frames',
            self._described(describe.describe_frames),
        )
        for cls in _source_classes():
            self._patch(cls, '_run', self._source_run(vars(cls)['_run']))
        return self

    def uninstall(self) -> None:
        for owner, name, original in reversed(self.undo):
            setattr(owner, name, original)
        self.undo = []
        self.guard.uninstall()

    # The hooks.

    def _mine(self) -> HostCall | None:
        hc = self.current
        if hc is None or threading.get_ident() != self.host_thread:
            return None
        return hc

    def _wrap(self, name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            hc = self._mine()
            if hc is None:
                return real(*args, **kwargs)
            protocols = ProtocolCapture(self, hc, name)
            actual_args, recorded_args = protocols.arguments(args)
            problem = plain_problem([recorded_args, kwargs])
            if problem is not None:
                hc.problem('plain', f'{name}: an argument: {problem}')
                raise Breach('plain', f'{name}: an argument: {problem}')
            for table in tables_in([list(args), kwargs]):
                if not any(table is known for known in self.allowed_tables):
                    hc.problem(
                        'inputs',
                        f'{name} was given a pyarrow table that '
                        f'describe_frames did not make',
                    )
            record: dict[str, Any] = {
                'host': hc.name,
                'function': name,
                'args': freeze(recorded_args),
                'kwargs': freeze(dict(kwargs)),
            }
            if protocols.providers:
                record['protocol_slots'] = protocols.providers
                record['protocol_calls'] = protocols.calls
            hc.calls.append(record)
            self.calls.append(record)
            try:
                value = real(*actual_args, **kwargs)
            except Exception as error:
                record['raised'] = error
                hc.note_error(error)
                raise
            if _is_run(value):
                proxy = RunProxy(self, value, record, hc)
                record['steps'] = proxy.steps
                hc.runs.append(proxy)
                return proxy
            problem = plain_problem(value)
            if problem is not None:
                hc.problem('plain', f'{name} returned: {problem}')
                raise Breach('plain', f'{name} returned: {problem}')
            record['returned'] = freeze(value)
            hc.note_value(value)
            self.keep_returned(value)
            return value

        wrapper.__name__ = name
        return wrapper

    def _described(self, real: Callable[..., Any]) -> Callable[..., Any]:
        def describe_frames(frames: Any) -> Any:
            out = real(frames)
            hc = self._mine()
            if hc is not None:
                hc.described += 1
            if isinstance(out, tuple) and out and isinstance(out[0], dict):
                self.allowed_tables.extend(out[0].values())
            return out

        return describe_frames

    def _source_run(self, original: Callable[..., Any]) -> Callable[..., Any]:
        recorder = self

        def run(source: Any, sql: Any, params: Any = None) -> Any:
            hc = recorder._mine()
            asked: tuple[RunProxy, int] | None = None
            if hc is not None and not recorder.guard.paused:
                for proxy in reversed(hc.runs):
                    at = proxy.ask_statement(sql, params)
                    if at is not None:
                        asked = (proxy, at)
                        break
                if asked is None:
                    hc.problem(
                        'statements',
                        f'Source._run({short(sql, 120)!r}) ran, and no core '
                        f'run asked for it',
                    )
                    raise Breach(
                        'statements',
                        f'Source._run({short(sql, 120)!r}) ran, and no core '
                        f'run asked for it',
                    )
            recorder.guard.paused += 1
            try:
                result = original(source, sql, params)
            finally:
                recorder.guard.paused -= 1
            if asked is not None:
                asked[0].results[asked[1]] = result
            return result

        return run

    def on_request(
        self, method: str, path: str, payload: Any
    ) -> RunProxy | None:
        r"""A request reaches the model's stand-in: the run whose call it is,
        or ``None`` for a session release the host makes on its own."""
        hc = self._mine()
        if hc is None:
            return None
        for proxy in reversed(hc.runs):
            pending = proxy.pending_call()
            if pending is None:
                continue
            want = _session_path(pending['endpoint'], pending['session_id'])
            if (method, path) != want:
                raise proxy._breach(
                    'requests',
                    f'{method} {path} was sent where the run asked for '
                    f'{want[0]} {want[1]}',
                )
            if pending['body'] is not None:
                found = json_diff(pending['body'], payload)
                if found:
                    raise proxy._breach(
                        'requests',
                        f'{method} {path} was sent with another body than the '
                        f"step's: {found}",
                    )
            proxy.sent = True
            return proxy
        if method == 'DELETE' and path.startswith('/v1/sessions/'):
            return None
        hc.problem(
            'requests', f'{method} {path} was sent, and no call_model asked'
        )
        raise Breach(
            'requests', f'{method} {path} was sent, and no call_model asked'
        )

    # A host call.

    @contextlib.contextmanager
    def host_call(
        self, name: str, given_frames: bool = False
    ) -> Iterator[HostCall]:
        assert self.current is None, f'{name} inside {self.current}'
        log = self.warnings_log
        hc = HostCall(name, warn_start=len(log) if log is not None else 0)
        self.current = hc
        self.host_thread = threading.get_ident()
        before = len(self.guard.trips)
        self.guard.active += 1
        try:
            yield hc
        finally:
            self.guard.active -= 1
            self.current = None
            trips = self.guard.trips[before:]
            if trips:
                ran = sorted(set(trips))
                hc.problem(
                    'guard',
                    f'ported code ran: {", ".join(ran[:8])}'
                    + (f' and {len(ran) - 8} more' if len(ran) > 8 else ''),
                )
            for proxy in hc.runs:
                if not proxy.finished:
                    hc.problem(
                        'host', f'a {proxy.function} run was left unfinished'
                    )
            if given_frames and not hc.described:
                hc.problem(
                    'inputs',
                    'was given frames and did not call describe_frames',
                )
            if log is not None:
                pool = Counter(hc.all_offered())
                for caught in log[hc.warn_start :]:
                    if not issubclass(caught.category, UserWarning):
                        continue
                    key = (qualname(caught.category), str(caught.message))
                    if pool[key] > 0:
                        pool[key] -= 1
                    else:
                        hc.problem(
                            'warnings-source',
                            f'gave the warning {short(key[1], 160)!r}, which '
                            f'no step, error or returned value of the core '
                            f'carried',
                        )
            self.problems.extend(hc.problems)
            self.host_calls.append(hc)


def _source_classes() -> list[type]:
    import inspect

    import sdm_connectors.source as source

    return [
        cls
        for cls in vars(source).values()
        if inspect.isclass(cls)
        and issubclass(cls, source.Source)
        and '_run' in vars(cls)
    ]


_RECORDER: dict[str, Recorder] = {}


def recorder() -> Recorder:
    r"""The process's recorder, installed on the built binding."""
    if 'recorder' not in _RECORDER:
        module = native(
            *RUNS, *PURE, 'payload_size_bytes', 'request_fixture_op'
        )
        _RECORDER['recorder'] = Recorder(module).install()
    return _RECORDER['recorder']


_HOST: dict[str, Any] = {}

HOST_NAMES = (
    'schema_from_source',
    'schema_from_data',
    'predictive_query',
    'materialize',
    'request',
    'Client',
    'merge',
    'flatten',
    'frame',
    'absent',
)


def host() -> Any:
    r"""``scripts/parity/request_host.py``, imported after the recorder is
    installed, so it reaches the binding through it."""
    if 'host' not in _HOST:
        recorder()
        assert HOST.is_file(), f'{HOST.relative_to(REPO)} does not exist'
        spec = importlib.util.spec_from_file_location('request_host', HOST)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules['request_host'] = module
        spec.loader.exec_module(module)
        missing = [
            n for n in HOST_NAMES if not callable(getattr(module, n, None))
        ]
        assert not missing, f'{HOST.name} defines no {", ".join(missing)}'
        _HOST['host'] = module
    return _HOST['host']


# --- Replaying through the host ------------------------------------------------------


def checking_session(responses: list[dict[str, Any]], rec: Recorder) -> Any:
    r"""The stored responses' stand-in, which also checks each request
    against the ``call_model`` step it answers."""
    from _parity.nim import ReplayingSession

    class CheckingSession(ReplayingSession):
        def _serve(
            self, method: str, url: str, payload: dict[str, Any] | None = None
        ) -> Any:
            from _parity.nim import _path

            proxy = rec.on_request(method, _path(url), payload)
            response = super()._serve(method, url, payload)
            if proxy is not None:
                stored = self.responses[-1]
                proxy.answered = (int(stored['status']), stored.get('body'))
            return response

    return CheckingSession(json.loads(json.dumps(responses)))


_CLASSES: dict[str, type] = {}


def interpreter_class() -> type:
    r"""The interpreter that makes a transcript's calls through the host."""
    if 'core' not in _CLASSES:
        _CLASSES['core'] = _make_interpreter_class()
    return _CLASSES['core']


def _make_interpreter_class() -> type:
    from _parity.runner import _Interpreter

    class CoreInterpreter(_Interpreter):
        r"""A transcript's calls, made through the host."""

        def __init__(
            self,
            transcript: dict[str, Any],
            source: Any,
            transport: Any,
            the_host: Any,
            rec: Recorder,
        ) -> None:
            super().__init__(transcript, source, None)  # type: ignore[arg-type]
            self.transport = transport
            self.host = the_host
            self.rec = rec
            self.layout: Any = None
            self.last: HostCall | None = None

        def call(
            self,
            name: str,
            fn: Callable[..., Any],
            *args: Any,
            given_frames: bool = False,
            **kwargs: Any,
        ) -> Any:
            with self.rec.host_call(name, given_frames) as hc:
                self.last = hc
                return fn(*args, **kwargs)

        def _client(self) -> Any:
            if self.client is None:
                self.client = self.call(
                    'Client', self.host.Client, self.transport
                )
            return self.client

        def close(self) -> None:
            if self.client is not None:
                client, self.client = self.client, None
                self.call('client.close', client.close)

        def _Schema_from_source(self, **args: Any) -> None:
            model = args.pop('semantic_model', None)
            source = self.source
            if model is not None:
                source = source.semantic_model(model)
            self.schema = self.call(
                'schema_from_source',
                self.host.schema_from_source,
                source,
                **args,
            )

        def _Schema_from_data(self, tables: list[str], **args: Any) -> None:
            frames = {name: self.frames[name] for name in tables}
            self.schema = self.call(
                'schema_from_data',
                self.host.schema_from_data,
                frames,
                given_frames=True,
                **args,
            )

        def _PredictiveQuery(self, query: str, schema: Any = None) -> None:
            data = self.schema if schema is None else schema
            self.query = self.call(
                'predictive_query',
                self.host.predictive_query,
                query,
                data,
                given_frames=isinstance(data, dict),
            )

        def _PredictiveQuery_materialize(self, **args: Any) -> None:
            import pandas as pd

            framed = any(isinstance(v, pd.DataFrame) for v in args.values())
            self.request = self.call(
                'materialize',
                self.host.materialize,
                self.query,
                given_frames=framed,
                **args,
            )

        def _Request(self, **args: Any) -> None:
            self.request = self.call(
                'request', self.host.request, given_frames=True, **args
            )

        def _SDMClient_predict(self, model: str, outputs: Any = None) -> None:
            client = self._client()
            self.result = self.call(
                'predict',
                client.predict,
                self.request,
                model=model,
                outputs=outputs,
            )
            self.results.append(self.result)

        def _SDMClient_fit(
            self, model: str, outputs: Any = None, holdout: Any = None
        ) -> None:
            client = self._client()
            self.fitted = self.call(
                'fit',
                client.fit,
                self.request,
                model=model,
                outputs=outputs,
                holdout=holdout,
            )

        def _FittedModel_predict(self) -> None:
            self.result = self.call(
                'fitted.predict', self.fitted.predict, self.request
            )
            self.results.append(self.result)

        def _FittedModel_close(self) -> None:
            self.call('fitted.close', self.fitted.close)

        def _PredictionResult_merge(self) -> None:
            self.result = call_merge(self, list(self.results))

        def _PredictionResult_to_pandas(self) -> None:
            self.layout = call_flatten(self, self.result)
            self.frame = self.call('frame', self.host.frame, self.layout)

    return CoreInterpreter


def _calls_named(hc: HostCall, name: str) -> list[dict[str, Any]]:
    return [c for c in hc.calls if c['function'] == name]


def call_flatten(interp: Any, result: Any) -> Any:
    r"""``host.flatten(result)``, held to calling the binding's once."""
    layout = interp.call('flatten', interp.host.flatten, result)
    hc = interp.last
    made = _calls_named(hc, 'flatten')
    if len(made) != 1 or 'returned' not in made[0]:
        hc.problem('host', f'called sdm_core._native.flatten {len(made)} times')
        interp.rec.problems.extend(hc.problems[-1:])
    elif made[0]['returned'] is not layout:
        hc.problem(
            'host', 'did not return what sdm_core._native.flatten returned'
        )
        interp.rec.problems.extend(hc.problems[-1:])
    return layout


def _require_called(interp: Any, name: str) -> None:
    hc = interp.last
    if hc is not None and not _calls_named(hc, name):
        hc.problem('host', f'did not call sdm_core._native.{name}')
        interp.rec.problems.extend(hc.problems[-1:])


def call_merge(interp: Any, results: list[Any]) -> Any:
    r"""``host.merge(results)``, held to calling the binding's."""
    try:
        return interp.call('merge', interp.host.merge, results)
    finally:
        _require_called(interp, 'merge')


def call_absent(interp: Any, result: Any, requested: Any) -> Any:
    r"""``host.absent(result, requested)``, held to calling the binding's."""
    try:
        return interp.call('absent', interp.host.absent, result, requested)
    finally:
        _require_called(interp, 'absent')


@dataclass
class Report:
    r"""What one replay through the host gave, and where it differs."""

    name: str
    backend: str
    task_kind: str
    calls: list[str]
    problems: list[tuple[str, str]] = field(default_factory=list)
    payloads: list[Any] = field(default_factory=list)
    error: dict[str, Any] | None = None
    warnings: list[dict[str, str]] = field(default_factory=list)
    layout: Any = None
    frame: Any = None
    recorded: list[dict[str, Any]] = field(default_factory=list)

    def problem(self, aspect: str, message: str) -> None:
        self.problems.append((aspect, message))

    def about(self, *aspects: str) -> list[str]:
        return [
            f'{self.name}: [{a}] {m}' for a, m in self.problems if a in aspects
        ]


#: The aspects every check holds the host to.
HOST_ASPECTS = ('host', 'guard', 'protocol', 'plain')


def golden_payloads() -> dict[str, list[Any]]:
    return load_json(GOLDEN)


def today_frame(path: Path) -> Any:
    r"""The frame today's Python gives at the transcript's ``to_pandas``."""
    import _parity.runner as runner
    from _parity.nim import ReplayingSession
    from _parity.tape import ReplayingTape
    from _parity.transcript import load

    transcript = load(path)
    tape = ReplayingTape(
        json.loads(json.dumps(transcript['steps'])), path.parent
    )
    session = ReplayingSession(json.loads(json.dumps(transcript['responses'])))
    kept: dict[str, Any] = {}
    original = runner.frame_to_table

    def keep(frame: Any) -> Any:
        kept['frame'] = frame
        return original(frame)

    runner.frame_to_table = keep
    try:
        runner.run(transcript, tape, session=session)
    finally:
        runner.frame_to_table = original
    return kept.get('frame')


def layout_difference(expected: Any, layout: Any) -> str | None:
    r"""Where a layout differs from ``result.arrow`` (``CONTRACT.md``)."""
    import pyarrow as pa

    drv = driver_module()
    if isinstance(layout, pa.RecordBatch):
        layout = pa.Table.from_batches([layout])
    if not isinstance(layout, pa.Table):
        return f'the layout is a {qualname(type(layout))}, not a pyarrow.Table'
    return drv.difference(drv.canon_frame(expected), drv.canon_frame(layout))


def frame_difference(expected: Any, got: Any) -> str | None:
    import pandas as pd

    if not isinstance(got, pd.DataFrame):
        return f'frame() gave a {qualname(type(got))}, not a DataFrame'
    try:
        pd.testing.assert_frame_equal(expected, got, check_exact=True)
    except AssertionError as error:
        return ' '.join(str(error).split())[:600]
    return None


def replay_core(path: Path, today: Any = None) -> Report:
    r"""Replay one transcript through the host (``CONTRACT.md``, "Replaying
    a transcript"); ``today`` is today's frame, compared when given."""
    from _parity.corpus import _compare_result
    from _parity.drivers import engine_hooks, facade
    from _parity.runner import _source, beyond_the_frame, describe_error
    from _parity.tape import ReplayingTape
    from _parity.transcript import frame_to_table, load, read_arrow

    rec = recorder()
    the_host = host()
    transcript = load(path)
    root = path.parent
    calls = transcript['calls']
    report = Report(
        transcript['name'],
        transcript['backend'],
        transcript.get('task_kind', ''),
        [c['call'] for c in calls],
    )
    tape = ReplayingTape(json.loads(json.dumps(transcript['steps'])), root)
    session = checking_session(transcript.get('responses') or [], rec)
    interp: Any = None
    answer: dict[str, Any] = {}
    with engine_hooks(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rec.reset(caught)
        try:
            from sdm_client import SDMClient

            source = None
            backend = transcript['backend']
            if backend not in IN_MEMORY:
                facts = transcript['data'].get('connection', {})
                connection = facade(backend, tape, None, facts)
                source = _source(transcript, connection, None)
            client = SDMClient(url='http://localhost:8000')
            client._transport._session.close()
            client._transport._session = session
            interp = interpreter_class()(
                transcript, source, client._transport, the_host, rec
            )
            try:
                interp.run(calls)
            except Exception as error:
                if interp.at != len(calls) - 1:
                    report.problem(
                        'host',
                        f'call {interp.at} ({calls[interp.at]["call"]}) raised '
                        f'{qualname(type(error))}: {short(str(error), 400)}',
                    )
                else:
                    report.error = describe_error(error)
                    _check_error_source(report, interp.last, error)
            interp.close()
            if interp.result is not None and report.error is None:
                with rec.host_call('answer'):
                    answer = beyond_the_frame(interp.result)
        except Breach as breach:
            report.problem(breach.aspect, str(breach))
        except BaseException as error:  # noqa: BLE001 - a divergence, reported
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if tape.divergence is None and session.divergence is None:
                report.problem(
                    'host',
                    f'the replay raised {qualname(type(error))}: '
                    f'{short(str(error), 400)}',
                )
    report.problems.extend(rec.problems)
    report.recorded = list(rec.calls)
    report.warnings = [
        {'category': qualname(w.category), 'message': str(w.message)}
        for w in caught
        if issubclass(w.category, UserWarning)
    ]
    report.payloads = [without_request_id(p) for p in session.payloads]

    for divergence in (tape.divergence, tape.leftover()):
        if divergence is not None:
            report.problem('steps', str(divergence))
            break
    for divergence in (session.divergence, session.leftover()):
        if divergence is not None:
            report.problem('requests', str(divergence))
            break
    found = json_diff(transcript['payloads'], report.payloads, '$payloads')
    if found:
        report.problem('payloads', found)
    golden = golden_payloads().get(transcript['name'])
    if golden is not None:
        want = [without_request_id(p) for p in golden]
        found = json_diff(want, report.payloads, '$golden')
        if found:
            report.problem('golden', found)
    found = json_diff(transcript.get('error'), report.error, '$error')
    if found:
        report.problem('error', found)

    frame = interp.frame if interp is not None else None
    table = None
    if frame is not None:
        try:
            table = frame_to_table(frame)
        except Exception as error:  # noqa: BLE001 - reported
            report.problem(
                'result', f'the frame cannot be stored: {short(str(error))}'
            )
    if frame is None or table is not None:
        reason = _compare_result(transcript, root, table)
        if reason:
            report.problem('result', reason)
    ref = transcript.get('result')
    if isinstance(ref, str):
        layout = interp.layout if interp is not None else None
        report.layout = layout
        report.frame = frame
        if layout is None:
            report.problem('layout', 'no layout was made')
        else:
            found = layout_difference(read_arrow(root / ref), layout)
            if found:
                report.problem('layout', found)
        if today is not None:
            found = frame_difference(today, frame)
            if found:
                report.problem('frame', found)
    found = json_diff(transcript.get('answer', {}), answer, '$answer')
    if found:
        report.problem('answer', found)
    found = json_diff(
        transcript.get('warnings', []), report.warnings, '$warnings'
    )
    if found:
        report.problem('warnings', found)
    return report


def _check_error_source(
    report: Report, hc: HostCall | None, error: BaseException
) -> None:
    message = getattr(error, 'message', None)
    if not isinstance(message, str):
        message = str(error)
    texts = hc.texts() if hc is not None else iter(())
    if not any(message == t or message in t for t in texts):
        report.problem(
            'error-source',
            f'the error {short(message, 200)!r} is no text the core gave',
        )


# --- Flows over given frames ---------------------------------------------------------


@dataclass
class FlowOutcome:
    r"""What a flow gave: what it sent, the error it ended in and at which
    call, its warnings, and how the host broke the contract."""

    payloads: list[Any]
    responses: list[Any]
    error: dict[str, Any] | None
    error_at: int | None
    warnings: list[dict[str, str]]
    problems: list[tuple[str, str]] = field(default_factory=list)


def _flow_transcript(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'name': 'flow',
        'backend': 'memory',
        'data': {'generator': 'wire-snapshot', 'seed': 0},
        'calls': calls,
    }


def run_flow(
    calls: list[dict[str, Any]],
    frames: Mapping[str, Any],
    core: bool,
    stand_in: Mapping[str, Any] | None = None,
) -> FlowOutcome:
    r"""``calls`` over ``frames``, in memory, through today's Python or the
    host, with the model answered by the parity stand-in. The first call
    that raises ends the flow."""
    from _parity.nim import SyntheticSession
    from _parity.runner import _Interpreter, describe_error

    transcript = _flow_transcript(calls)
    problems: list[tuple[str, str]] = []
    error: dict[str, Any] | None = None
    error_at: int | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if core:
            rec = recorder()
            the_host = host()
            rec.reset(caught)
            session: Any = _checking_synthetic(stand_in, rec)
            from sdm_client import SDMClient

            client = SDMClient(url='http://localhost:8000')
            client._transport._session.close()
            client._transport._session = session
            interp: Any = interpreter_class()(
                transcript, None, client._transport, the_host, rec
            )
        else:
            session = SyntheticSession(stand_in)
            interp = _Interpreter(transcript, None, session)
        interp.frames = dict(frames)
        try:
            try:
                interp.run(calls)
            except Exception as raised:
                error = describe_error(raised)
                error_at = interp.at
            interp.close()
        except Breach as breach:
            problems.append((breach.aspect, str(breach)))
        if core:
            problems.extend(rec.problems)
    return FlowOutcome(
        [without_request_id(p) for p in session.payloads],
        list(session.responses),
        error,
        error_at,
        [
            {'category': qualname(w.category), 'message': str(w.message)}
            for w in caught
            if issubclass(w.category, UserWarning)
        ],
        problems,
    )


def _checking_synthetic(stand_in: Any, rec: Recorder) -> Any:
    from _parity.nim import SyntheticSession, _path

    class CheckingSynthetic(SyntheticSession):
        def _serve(
            self, method: str, url: str, payload: dict[str, Any] | None
        ) -> Any:
            proxy = rec.on_request(method, _path(url), payload)
            response = super()._serve(method, url, payload)
            if proxy is not None:
                stored = self.responses[-1]
                proxy.answered = (int(stored['status']), stored.get('body'))
            return response

    return CheckingSynthetic(stand_in)


class PureInterpreter:
    r"""Calls of the host's pure functions, each a host call."""

    def __init__(self) -> None:
        self.rec = recorder()
        self.host = host()
        self.last: HostCall | None = None

    def call(self, name: str, fn: Callable[..., Any], *args: Any) -> Any:
        with self.rec.host_call(name) as hc:
            self.last = hc
            return fn(*args)


# --- Recordings ---------------------------------------------------------------------


class Encoder:
    r"""Plain data as recording JSON. With ``root``, tables are written to
    ``root/tables``, each named by the SHA-256 of its IPC bytes; without it,
    tables are written in canonical form, for comparing."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root

    def _arrow(self, value: Any, tag: str) -> dict[str, Any]:
        import pyarrow as pa

        drv = driver_module()
        if isinstance(value, pa.RecordBatch):
            table = pa.Table.from_batches([value])
        elif isinstance(value, (pa.Array, pa.ChunkedArray)):
            table = pa.table({'values': value})
        else:
            table = value
        if self.root is None:
            return {tag: drv.canon_frame(table)}
        sink = pa.BufferOutputStream()
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
        data = sink.getvalue().to_pybytes()
        name = hashlib.sha256(data).hexdigest()
        target = self.root / TABLES_DIR / f'{name}.arrow'
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return {tag: name}

    def value(self, value: Any) -> Any:
        import pyarrow as pa

        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            return normalize_text(value)
        if isinstance(value, int):
            if -(2**63) <= value < 2**63:
                return int(value)
            return {'$int': str(value)}
        if isinstance(value, float):
            if math.isnan(value):
                return {'$float': 'nan'}
            if math.isinf(value):
                return {'$float': 'inf' if value > 0 else '-inf'}
            return float(value)
        if isinstance(value, bytes):
            return {'$bytes': base64.b64encode(value).decode('ascii')}
        if isinstance(value, pa.Table):
            return self._arrow(value, '$table')
        if isinstance(value, pa.RecordBatch):
            return self._arrow(value, '$batch')
        if isinstance(value, pa.ChunkedArray):
            return self._arrow(value, '$chunked')
        if isinstance(value, pa.Array):
            return self._arrow(value, '$array')
        if isinstance(value, tuple):
            return {'$tuple': [self.value(item) for item in value]}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, dict):
            out = {normalize_text(k): self.value(v) for k, v in value.items()}
            if any(k.startswith('$') for k in out):
                return {'$dict': out}
            return out
        raise AssertionError(f'not plain data: {qualname(type(value))}')

    def error(self, error: BaseException) -> dict[str, Any]:
        return {
            'type': type(error).__name__,
            'message': normalize_text(str(error)),
            'attributes': self.value(error_parts(error)),
        }

    def outcome(self, outcome: RecordedOutcome | None) -> Any:
        if outcome is None:
            return None
        if outcome.kind == 'data':
            return {'data': self.value(outcome.args[0])}
        if outcome.kind == 'error':
            sqlstate, vendor_code, message = (list(outcome.args) + [None] * 3)[
                :3
            ]
            return {
                'error': {
                    'sqlstate': self.value(sqlstate),
                    'vendor_code': self.value(vendor_code),
                    'message': self.value(message),
                }
            }
        return {outcome.kind: self.value(outcome.args[0])}

    def warnings(self, given: list[dict[str, str]]) -> list[dict[str, str]]:
        return [
            {
                'category': w['category'],
                'message': normalize_text(w['message']),
            }
            for w in given
        ]

    def step(self, step: dict[str, Any]) -> dict[str, Any]:
        kind = step['kind']
        if kind == 'run_queries':
            return {
                'kind': kind,
                'queries': [
                    {'sql': normalize_text(q['sql']), 'params': q['params']}
                    for q in step['queries']
                ],
                'warnings': self.warnings(step['warnings']),
                'outcomes': [self.outcome(o) for o in step['outcomes']],
            }
        if kind == 'call_model':
            return {
                'kind': kind,
                'endpoint': step['endpoint'],
                'session_id': step['session_id'],
                'body': normalize_json(step['body']),
                'warnings': self.warnings(step['warnings']),
                'outcome': self.outcome(step['outcome']),
            }
        if kind == 'done':
            return {
                'kind': kind,
                'value': self.value(step['value']),
                'warnings': self.warnings(step['warnings']),
            }
        return {'kind': 'raised', 'error': self.error(step['error'])}

    def protocol_calls(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for row in call.get('protocol_calls', []):
            outcome = row['outcome']
            out.append(
                {
                    'slot': row['slot'],
                    'path': row['path'],
                    'operation': row['operation'],
                    'args': self.value(row['args']),
                    'outcome': {'raised': self.error(outcome['raised'])}
                    if 'raised' in outcome
                    else {'returned': self.value(outcome['returned'])},
                }
            )
        return out

    def call(self, call: dict[str, Any]) -> dict[str, Any]:
        if call.get('protocol_slots'):
            raise Breach(
                'protocol',
                'callback-bearing calls require auxiliary public capture',
            )
        out: dict[str, Any] = {
            'host': call['host'],
            'function': call['function'],
            'args': [self.value(a) for a in call['args']],
            'kwargs': {k: self.value(v) for k, v in call['kwargs'].items()},
        }
        if 'steps' in call:
            out['steps'] = [self.step(s) for s in call['steps']]
        elif 'raised' in call:
            out['raised'] = self.error(call['raised'])
        else:
            out['returned'] = self.value(call.get('returned'))
        return out


def dumps(document: Any) -> str:
    return json.dumps(document, indent=1, ensure_ascii=False) + '\n'


def write_recording(
    root: Path, name: str, transcript: Path, calls: list[dict[str, Any]]
) -> Path:
    r"""One scenario's recording under ``root``, tables shared in
    ``root/tables``."""
    encoder = Encoder(root)
    document = {
        'format': FORMAT,
        'scenario': name,
        'transcript': transcript.parent.relative_to(REPO).as_posix(),
        'calls': [encoder.call(c) for c in calls],
    }
    target = scenario_dir(root, name)
    target.mkdir(parents=True, exist_ok=True)
    (target / CALLS_FILE).write_text(dumps(document), encoding='utf-8')
    return target


def recording_dirs(root: Path = RECORDINGS) -> list[Path]:
    return sorted(p.parent for p in root.rglob(CALLS_FILE))


def decode(value: Any, root: Path) -> Any:
    r"""Recording JSON as plain data, tables read from ``root/tables``."""
    if isinstance(value, list):
        return [decode(item, root) for item in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        ((tag, body),) = value.items()
        if tag in ('$table', '$batch', '$array', '$chunked'):
            table = driver_module().read_arrow(
                root / TABLES_DIR / f'{body}.arrow'
            )
            if tag == '$batch':
                batches = table.combine_chunks().to_batches()
                if batches:
                    return batches[0]
                import pyarrow as pa

                return pa.RecordBatch.from_pylist([], schema=table.schema)
            if tag == '$array':
                return table.column('values').combine_chunks()
            if tag == '$chunked':
                return table.column('values')
            return table
        if tag == '$tuple':
            return tuple(decode(item, root) for item in body)
        if tag == '$bytes':
            return base64.b64decode(body)
        if tag == '$int':
            return int(body)
        if tag == '$float':
            return float(body)
        if tag == '$dict':
            return {k: decode(v, root) for k, v in body.items()}
    return {k: decode(v, root) for k, v in value.items()}


def canonical(value: Any) -> Any:
    r"""Plain data in the form recordings are compared in."""
    return Encoder(None).value(value)


def canonical_recorded(value: Any, root: Path) -> Any:
    return canonical(decode(value, root))


def _outcome_from(module: Any, recorded: Mapping[str, Any], root: Path) -> Any:
    ((kind, body),) = recorded.items()
    if kind == 'data':
        return module.Outcome.data(decode(body, root))
    if kind == 'error':
        return module.Outcome.error(
            body.get('sqlstate'), body.get('vendor_code'), body.get('message')
        )
    return getattr(module.Outcome, kind)(body)


def redrive(directory: Path, module: Any, root: Path) -> tuple[int, list[str]]:
    r"""Replay one recording through ``module``: the number of calls, and
    every way the binding's answers differ from the recorded ones."""
    drv = driver_module()
    document = load_json(directory / CALLS_FILE)
    name = document.get('scenario', directory.name)
    problems: list[str] = []
    encoder = Encoder(None)

    def differ(where: str, want: Any, got: Any) -> None:
        found = drv.difference(want, got, where)
        if found:
            problems.append(f'{name}: {found}')

    for at, call in enumerate(document['calls']):
        where = f'call {at} ({call["function"]})'
        function = getattr(module, call['function'], None)
        if not callable(function):
            problems.append(
                f'{name}: {where}: the binding has no such function'
            )
            break
        args = [decode(a, root) for a in call['args']]
        kwargs = {k: decode(v, root) for k, v in call['kwargs'].items()}
        try:
            got = function(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - compared
            if 'raised' not in call:
                problems.append(
                    f'{name}: {where} raised {type(error).__name__}: '
                    f'{short(str(error))}'
                )
                break
            differ(f'{where}.raised', call['raised'], encoder.error(error))
            continue
        if 'raised' in call:
            problems.append(f'{name}: {where} did not raise')
            continue
        if 'steps' not in call:
            differ(
                f'{where}.returned',
                canonical_recorded(call['returned'], root),
                canonical(got),
            )
            continue
        if not _drive(name, where, module, got, call['steps'], root, problems):
            break
    return len(document['calls']), problems


def _drive(
    name: str,
    where: str,
    module: Any,
    run: Any,
    steps: list[dict[str, Any]],
    root: Path,
    problems: list[str],
) -> bool:
    drv = driver_module()
    encoder = Encoder(None)

    def differ(at: str, want: Any, got: Any) -> bool:
        found = drv.difference(want, got, at)
        if found:
            problems.append(f'{name}: {found}')
            return False
        return True

    for i, want in enumerate(steps):
        at = f'{where}.steps[{i}]'
        try:
            step = run.next()
        except Exception as error:  # noqa: BLE001 - compared
            if want['kind'] != 'raised':
                problems.append(
                    f'{name}: {at} raised {type(error).__name__}: '
                    f'{short(str(error))}, where it gave a {want["kind"]} step'
                )
                return False
            return differ(f'{at}.error', want['error'], encoder.error(error))
        kind = getattr(step, 'kind', None)
        if want['kind'] == 'raised' or kind != want['kind']:
            problems.append(f'{name}: {at} is {kind!r}, not {want["kind"]!r}')
            return False
        given = [
            {'category': w['category'], 'message': w['message']}
            for w in getattr(step, 'warnings', [])
        ]
        if not differ(
            f'{at}.warnings', want['warnings'], encoder.warnings(given)
        ):
            return False
        if kind == 'run_queries':
            queries = list(step.queries)
            got = [
                {
                    'sql': normalize_text(q.sql),
                    'params': drv.encode_params(q.params),
                }
                for q in queries
            ]
            if not differ(f'{at}.queries', want['queries'], got):
                return False
            for q, outcome in zip(queries, want['outcomes'], strict=True):
                run.submit(q.id, _outcome_from(module, outcome, root))
        elif kind == 'call_model':
            got_call = {
                'endpoint': step.endpoint,
                'session_id': step.session_id,
                'body': normalize_json(
                    None if step.body is None else json.loads(step.body)
                ),
            }
            want_call = {k: want[k] for k in ('endpoint', 'session_id', 'body')}
            if not differ(f'{at}.call', want_call, got_call):
                return False
            run.submit(step.id, _outcome_from(module, want['outcome'], root))
        else:
            return differ(
                f'{at}.value',
                canonical_recorded(want['value'], root),
                canonical(step.value),
            )
    problems.append(f'{name}: {where} gave more steps than recorded')
    return False


def capture_corpus(out: Path) -> dict[str, int]:
    r"""Replay every transcript through the host and write its recording
    under ``out``; the number of calls, by scenario. A transcript whose
    replay has any problem is an error: nothing is written for it."""
    found: dict[str, int] = {}
    failed: list[str] = []
    for path in transcripts():
        report = replay_core(path)
        if report.problems:
            failed.extend(report.about(*{a for a, _ in report.problems}))
            continue
        write_recording(out, report.name, path, report.recorded)
        found[report.name] = len(report.recorded)
    if failed:
        raise AssertionError(
            f'{len(failed)} problem(s); nothing written for those transcripts:\n'
            + '\n'.join(failed[:40])
        )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capture = sub.add_parser('capture', help='write the recordings')
    capture.add_argument('--out', type=Path, default=RECORDINGS)
    args = parser.parse_args(argv)
    if args.command == 'capture':
        out = args.out.resolve()
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        counts = capture_corpus(out)
        print(f'recorded {len(counts)} scenarios, {sum(counts.values())} calls')
    return 0


if __name__ == '__main__':
    sys.exit(main())
