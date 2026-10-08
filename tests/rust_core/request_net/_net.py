# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 8 acceptance checks share.

- Finding every warning and refusal site in the source, independently of
  ``scripts/diagnostics.py``.
- Running every parity fixture case against today's Python while noting which
  of those sites each case reaches.
- Reading transcripts, their Arrow results and the two review tables.

The interface these rely on is ``CONTRACT.md``, beside this file.
"""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, TracebackType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
CLIENT = 'packages/structured-data-models-client/src/sdm_client'
ENGINE = 'packages/structured-data-models-relational-engine/src/sdm_relational_engine'
PACKAGE_RENAMES = {
    'sdm-api': 'structured-data-models-api',
    'sdm-client': 'structured-data-models-client',
    'sdm-connectors': 'structured-data-models-connectors',
    'sdm-pql': 'structured-data-models-pql',
    'sdm-relational-engine': 'structured-data-models-relational-engine',
}
PACKAGES = (CLIENT, ENGINE)
CATALOG = (
    REPO / 'packages' / 'structured-data-models-client' / 'diagnostics.json'
)
GENERATOR = REPO / 'scripts' / 'diagnostics.py'
PARITY = REPO / 'tests' / 'parity'
FIXTURES = PARITY / 'fixtures'
TRANSCRIPTS = PARITY / 'transcripts'
FORMAT = PARITY / 'FORMAT.md'
UNREACHABLE = PARITY / 'UNREACHABLE_DIAGNOSTICS.md'
COVERAGE_GAPS = PARITY / 'COVERAGE_GAPS.md'
RUN_CORPUS = REPO / 'scripts' / 'parity' / 'run_corpus.py'
MAKE_FIXTURES = REPO / 'scripts' / 'parity' / 'make_fixtures.py'

#: The commit Phase 8 started from (``phases/phase-8/BASE``).
BASE = 'ea81f9bd7d51ee68027c08c8bd56d4448bd913b4'

LIVE_BACKENDS = frozenset({'databricks', 'lakebase', 'snowflake'})
OFFLINE_RELATIONAL = ('memory', 'duckdb', 'sqlite', 'postgres')
OFFLINE = frozenset({*OFFLINE_RELATIONAL, 'tabular'})

#: The Phase 8.5 modules, as the brief lists them.
P85_CLIENT_FILES = (
    'decompose.py',
    'request.py',
    'client.py',
    'result.py',
    'base.py',
    'models.py',
    'query.py',
    'errors.py',
)
P85_CLIENT_DIRS = ('materializers', 'adapters', 'wire')
P85_ENGINE_DIRS = ('rfm/inference',)
P85_ENGINE_FILES = ('rfm/rfm.py',)

#: The class every client refusal derives from.
ROOT_CLASS = 'StructuredDataError'
#: The engine's own error family, which ``_engine_errors`` also turns into
#: ``INVALID_REQUEST``.
ENGINE_ROOT_CLASS = 'KumoRelationalError'
#: Bare built-in raises ``_engine_errors`` turns into ``INVALID_REQUEST``.
ENGINE_BUILTINS = frozenset({'ValueError', 'KeyError', 'TypeError'})


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def p85_modules() -> list[str]:
    r"""Every source file of the Phase 8.5 modules, repository-relative."""
    out: list[str] = []
    for name in P85_CLIENT_FILES:
        out.append(f'{CLIENT}/{name}')
    for name in P85_CLIENT_DIRS:
        out += [_rel(p) for p in sorted((REPO / CLIENT / name).rglob('*.py'))]
    for name in P85_ENGINE_DIRS:
        out += [_rel(p) for p in sorted((REPO / ENGINE / name).rglob('*.py'))]
    for name in P85_ENGINE_FILES:
        out.append(f'{ENGINE}/{name}')
    missing = [m for m in out if not (REPO / m).is_file()]
    assert not missing, f'Phase 8.5 modules missing: {missing}'
    return out


def coverage_modules() -> list[str]:
    r"""The modules P8.coverage measures (CONTRACT.md, Coverage)."""
    out = [m for m in p85_modules() if not m.endswith('/rfm/rfm.py')]
    return out + [f'{ENGINE}/exceptions.py']


# --- Sites --------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    kind: str  # 'raise' or 'warning'
    file: str
    line: int
    end_line: int
    cls: str
    fragments: tuple[str, ...] = ()

    @property
    def where(self) -> str:
        return f'{self.file}:{self.line}'

    def holds(self, file: str, line: int) -> bool:
        return file == self.file and self.line <= line <= self.end_line


def _name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _fragments(node: ast.AST | None) -> tuple[str, ...]:
    if node is None:
        return ()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.JoinedStr):
        return tuple(
            v.value
            for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _fragments(node.left) + _fragments(node.right)
    return ()


def trees() -> dict[str, ast.Module]:
    out: dict[str, ast.Module] = {}
    for package in PACKAGES:
        for path in sorted((REPO / package).rglob('*.py')):
            out[_rel(path)] = ast.parse(path.read_text(encoding='utf-8'))
    return out


def family(parsed: dict[str, ast.Module], root: str) -> set[str]:
    r"""``root`` and every class deriving from it, by name."""
    classes = {root}
    grew = True
    while grew:
        grew = False
        for tree in parsed.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name not in classes:
                    if any(_name(b) in classes for b in node.bases):
                        classes.add(node.name)
                        grew = True
    return classes


def _bare_warn(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == 'warnings'
        and any(a.name == 'warn' and a.asname is None for a in node.names)
        for node in ast.walk(tree)
    )


def all_sites(
    parsed: dict[str, ast.Module],
) -> tuple[list[Site], dict[str, list[tuple[int, int]]]]:
    r"""Every ``raise C(...)``/``raise C`` with a class name ``C``, every
    ``warnings.warn``, and where each file builds a ``StructuredDataError``
    family member (as ``(line, end_line)`` spans)."""
    sde = family(parsed, ROOT_CLASS)
    sites: list[Site] = []
    builds: dict[str, list[tuple[int, int]]] = {}
    for rel, tree in parsed.items():
        bare = _bare_warn(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func) in sde:
                builds.setdefault(rel, []).append(
                    (node.lineno, node.end_lineno or node.lineno)
                )
            if isinstance(node, ast.Raise) and node.exc is not None:
                call = node.exc if isinstance(node.exc, ast.Call) else None
                cls = _name(call.func if call else node.exc)
                if cls is None:
                    continue
                message: ast.AST | None = None
                if call is not None:
                    for keyword in call.keywords:
                        if keyword.arg == 'message':
                            message = keyword.value
                    if call.args:
                        message = call.args[0]
                sites.append(
                    Site(
                        'raise',
                        rel,
                        node.lineno,
                        node.end_lineno or node.lineno,
                        cls,
                        _fragments(message),
                    )
                )
            elif isinstance(node, ast.Call):
                func = node.func
                is_warn = (
                    isinstance(func, ast.Attribute)
                    and func.attr == 'warn'
                    and isinstance(func.value, ast.Name)
                    and func.value.id == 'warnings'
                ) or (bare and isinstance(func, ast.Name) and func.id == 'warn')
                if not is_warn:
                    continue
                category: ast.AST | None = (
                    node.args[1] if len(node.args) > 1 else None
                )
                message = node.args[0] if node.args else None
                for keyword in node.keywords:
                    if keyword.arg == 'category':
                        category = keyword.value
                    if keyword.arg == 'message':
                        message = keyword.value
                cls = (
                    'UserWarning'
                    if category is None
                    else (_name(category) or ast.unparse(category))
                )
                sites.append(
                    Site(
                        'warning',
                        rel,
                        node.lineno,
                        node.end_lineno or node.lineno,
                        cls,
                        _fragments(message),
                    )
                )
    return sites, builds


def engine_sites(
    parsed: dict[str, ast.Module], sites: list[Site]
) -> list[Site]:
    r"""The engine raises that reach users as ``INVALID_REQUEST``: in
    ``rfm/inference`` and ``rfm/rfm.py``, a bare ``ValueError``, ``KeyError`` or
    ``TypeError``, or one of the engine's ``KumoRelationalError`` family."""
    reaching = ENGINE_BUILTINS | family(parsed, ENGINE_ROOT_CLASS)
    files = {m for m in p85_modules() if m.startswith(ENGINE)}
    return [
        s
        for s in sites
        if s.kind == 'raise' and s.file in files and s.cls in reaching
    ]


def required_sites(parsed: dict[str, ast.Module]) -> list[Site]:
    r"""Every catalog-worthy site in the Phase 8.5 modules (CONTRACT.md,
    Message fixtures): client refusals and warnings, engine warnings, and the
    engine raises that reach users."""
    sites, _ = all_sites(parsed)
    sde = family(parsed, ROOT_CLASS)
    modules = set(p85_modules())
    out = [
        s
        for s in sites
        if s.file in modules
        and (s.kind == 'warning' or (s.kind == 'raise' and s.cls in sde))
    ]
    out += engine_sites(parsed, sites)
    return sorted(set(out), key=lambda s: (s.file, s.line, s.kind))


def current_package_path(path: str) -> str:
    for old, new in PACKAGE_RENAMES.items():
        prefix = f'packages/{old}/'
        if path.startswith(prefix):
            return f'packages/{new}/' + path[len(prefix) :]
    return path


def catalog_entries() -> list[dict[str, Any]]:
    assert CATALOG.is_file(), f'{_rel(CATALOG)} does not exist'
    data = json.loads(CATALOG.read_text(encoding='utf-8'))
    assert isinstance(data, dict) and isinstance(
        data.get('diagnostics'), list
    ), 'the catalog is not an object with a diagnostics array'
    for entry in data['diagnostics']:
        if isinstance(entry.get('file'), str):
            entry['file'] = current_package_path(entry['file'])
    return data['diagnostics']


def entry_matches(entry: Mapping[str, Any], site: Site) -> bool:
    line = entry.get('line')
    return (
        entry.get('kind') == site.kind
        and isinstance(line, int)
        and site.holds(str(entry.get('file')), line)
    )


def template_lacks(entry: Mapping[str, Any], site: Site) -> str | None:
    r"""The first literal fragment of the site's message the entry's template
    does not hold, or ``None``."""
    template = str(entry.get('template', ''))
    unescaped = template.replace('{{', '{').replace('}}', '}')
    for fragment in site.fragments:
        if fragment.strip() and not (
            fragment in template or fragment in unescaped
        ):
            return fragment
    return None


# --- Running fixtures ---------------------------------------------------------


def qualname(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def error_code(error: BaseException) -> str:
    code = getattr(error, 'code', None)
    if not isinstance(code, str) or not code:
        code = type(error).__name__
    return code


def error_message(error: BaseException) -> str:
    message = getattr(error, 'message', None)
    return message if isinstance(message, str) else str(error)


def details_keys(error: BaseException) -> list[str]:
    details = getattr(error, 'details', None)
    if isinstance(details, Mapping):
        return sorted(str(k) for k in details)
    return []


def raise_point(error: BaseException) -> tuple[str, int] | None:
    r"""The file and line of the ``raise`` that raised ``error`` first."""
    tb: TracebackType | None = error.__traceback__
    if tb is None:
        return None
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb.tb_frame.f_code.co_filename, tb.tb_lineno


def chain(error: BaseException | None) -> Iterator[BaseException]:
    r"""``error`` and what it says it was raised from: its ``__cause__``, or
    its ``__context__`` when that is shown."""
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        if error.__cause__ is not None:
            error = error.__cause__
        elif not error.__suppress_context__:
            error = error.__context__
        else:
            error = None


def repo_file(filename: str) -> str | None:
    try:
        return Path(filename).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return None


class WarnSpy:
    r"""Notes the file and line of every ``warnings.warn`` call while active,
    with its category and text, and passes it on unchanged."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int, str, str]] = []
        self._original = warnings.warn
        self._patched: list[tuple[ModuleType, str]] = []

    def _warn(
        self,
        message: Any,
        category: type[Warning] | None = None,
        stacklevel: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        frame = sys._getframe(1)
        if isinstance(message, Warning):
            cls: type = type(message)
        else:
            cls = category or UserWarning
        self.calls.append(
            (
                repo_file(frame.f_code.co_filename),
                frame.f_lineno,
                qualname(cls),
                str(message),
            )
        )
        self._original(message, category, stacklevel + 1, *args, **kwargs)

    def __enter__(self) -> WarnSpy:
        warnings.warn = self._warn  # type: ignore[assignment]
        for module in list(sys.modules.values()):
            if getattr(module, 'warn', None) is self._original:
                module.warn = self._warn  # type: ignore[attr-defined]
                self._patched.append((module, 'warn'))
        return self

    def __exit__(self, *exc: object) -> None:
        warnings.warn = self._original
        for module, name in self._patched:
            setattr(module, name, self._original)


@dataclass
class Outcome:
    case: dict[str, Any]
    output: Any = None
    error: BaseException | None = None
    warned: list[tuple[str, str]] = field(default_factory=list)
    spied: list[tuple[str | None, int, str, str]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def fixture_ops() -> ModuleType:
    sys.path.insert(0, str(PARITY))
    import _fixture_ops  # type: ignore[import-not-found]

    return _fixture_ops


def fixture_cases() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(FIXTURES.rglob('*.json')):
        out += json.loads(path.read_text(encoding='utf-8'))['cases']
    return out


def same(a: Any, b: Any) -> bool:
    r"""Equal as JSON, type included, with NaN equal to NaN."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(map(same, a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def run_case(ops: ModuleType, case: dict[str, Any]) -> Outcome:
    r"""Run one fixture case through the Python runner and compare it with
    what it records (CONTRACT.md, Message fixtures)."""
    out = Outcome(case)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with WarnSpy() as spy:
            try:
                out.output = json.loads(
                    json.dumps(ops.run(case['operation'], case['input']))
                )
            except Exception as raised:  # noqa: BLE001 - an outcome
                error: BaseException = raised
                if (
                    type(raised).__name__ == 'FixtureError'
                    and raised.__cause__ is not None
                ):
                    error = raised.__cause__
                out.error = error
    out.spied = spy.calls
    out.warned = [
        (qualname(w.category), str(w.message))
        for w in caught
        if issubclass(w.category, UserWarning)
    ]
    cid = case.get('id')
    if out.error is None:
        if 'expected' not in case:
            out.problems.append(
                f'{cid}: returned {out.output!r:.200}, expected the error '
                f'{case.get("expected_error")!r:.200}'
            )
        elif not same(out.output, case['expected']):
            out.problems.append(
                f'{cid}: returned {out.output!r:.300}, expected '
                f'{case["expected"]!r:.300}'
            )
    else:
        want = case.get('expected_error')
        if not isinstance(want, dict):
            out.problems.append(
                f'{cid}: raised {qualname(type(out.error))}: '
                f'{error_message(out.error)!r:.300}, expected an output'
            )
        else:
            got = {
                'code': error_code(out.error),
                'message': error_message(out.error),
                'class': qualname(type(out.error)),
                'details_keys': details_keys(out.error),
            }
            for key in ('code', 'message', 'class', 'details_keys'):
                if key in want and want[key] != got[key]:
                    out.problems.append(
                        f'{cid}: expected_error.{key} is {want[key]!r:.300}, '
                        f'today it is {got[key]!r:.300}'
                    )
    if 'expected_warnings' in case:
        want_w = [
            (w.get('category'), w.get('message'))
            for w in case['expected_warnings']
        ]
        if want_w != out.warned:
            out.problems.append(
                f'{cid}: expected_warnings {want_w!r:.400}, today '
                f'{out.warned!r:.400}'
            )
    return out


def credits(outcome: Outcome, sites: list[Site]) -> set[Site]:
    r"""The sites a passing case is credited with (CONTRACT.md)."""
    if outcome.problems:
        return set()
    case = outcome.case
    got: set[Site] = set()
    want = case.get('expected_error')
    if (
        outcome.error is not None
        and isinstance(want, dict)
        and {'class', 'details_keys', 'code', 'message'} <= want.keys()
    ):
        for link in chain(outcome.error):
            point = raise_point(link)
            if point is None:
                continue
            file = repo_file(point[0])
            for site in sites:
                if (
                    site.kind == 'raise'
                    and file is not None
                    and site.holds(file, point[1])
                ):
                    got.add(site)
    if 'expected_warnings' in case:
        shown = set(outcome.warned)
        for file, line, category, message in outcome.spied:
            if file is None or (category, message) not in shown:
                continue
            for site in sites:
                if site.kind == 'warning' and site.holds(file, line):
                    got.add(site)
    return got


def check_import_location() -> None:
    r"""The packages run must be this checkout's, or no line means anything."""
    import sdm_client
    import sdm_relational_engine

    for module, rel in (
        (sdm_client, CLIENT),
        (sdm_relational_engine, ENGINE),
    ):
        where = Path(module.__file__ or '').resolve().parent
        assert where == (REPO / rel).resolve(), (
            f'{module.__name__} is imported from {where}, not from this '
            f'checkout ({REPO / rel}); install the packages editable'
        )


# --- Review tables ------------------------------------------------------------


def table_rows(path: Path, columns: int) -> list[list[str]]:
    r"""The body rows of every Markdown table in ``path`` with at least
    ``columns`` cells, the last cell holding the rest of the row."""
    rows: list[list[str]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        text = line.strip()
        if not text.startswith('|'):
            continue
        cells = [c.strip() for c in text.strip('|').split('|')]
        if len(cells) < columns:
            continue
        cells = cells[: columns - 1] + ['|'.join(cells[columns - 1 :]).strip()]
        if all(re.fullmatch(r':?-{3,}:?', c) for c in cells if c):
            continue
        rows.append(cells)
    return rows


def unquote(cell: str) -> str:
    return cell.strip().strip('`').strip()


# --- Transcripts --------------------------------------------------------------


def transcripts(root: Path = TRANSCRIPTS) -> dict[str, tuple[Path, Any]]:
    out: dict[str, tuple[Path, Any]] = {}
    for path in sorted(root.rglob('transcript.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        out[str(data['name'])] = (path.parent, data)
    return out


def base_transcripts() -> dict[str, Any]:
    r"""The transcripts at BASE, by name."""
    listed = subprocess.run(
        [
            'git',
            '-C',
            str(REPO),
            'ls-tree',
            '-r',
            '--name-only',
            BASE,
            '--',
            'tests/parity/transcripts',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    out: dict[str, Any] = {}
    for rel in listed:
        if rel.endswith('/transcript.json'):
            text = subprocess.run(
                ['git', '-C', str(REPO), 'show', f'{BASE}:{rel}'],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            data = json.loads(text)
            out[str(data['name'])] = data
    assert len(out) >= 71, f'only {len(out)} transcripts at BASE'
    return out


def read_arrow(path: Path) -> Any:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    try:
        with pa.memory_map(str(path)) as source:
            return ipc.open_file(source).read_all()
    except pa.ArrowInvalid:
        with pa.memory_map(str(path)) as source:
            return ipc.open_stream(source).read_all()


def result_rows(directory: Path, data: Mapping[str, Any]) -> list[dict] | None:
    ref = data.get('result')
    if not ref:
        return None
    return read_arrow(directory / ref).to_pylist()


def walk(value: Any) -> Iterator[tuple[str | None, Any]]:
    r"""Every ``(key, value)`` pair in a JSON value, depth first; list items
    come with key ``None``."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from walk(item)


def without(data: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in keys}


def recorded_error(data: Mapping[str, Any]) -> dict[str, Any] | None:
    error = data.get('error')
    return error if isinstance(error, dict) else None


def error_class(error: Mapping[str, Any]) -> str:
    return str(error.get('class') or error.get('type') or '')


def calls_named(data: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    return [c for c in data.get('calls', []) if c.get('call') == name]


def replay(
    names: list[str], corpus: Path | None = None, env: dict | None = None
) -> tuple[int, str]:
    args = [sys.executable, str(RUN_CORPUS), 'replay']
    if corpus is not None:
        args += ['--corpus', str(corpus)]
    if names:
        args += ['--only', ','.join(names)]
    proc = subprocess.run(
        args,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=3600,
        env=env,
    )
    return proc.returncode, proc.stdout + proc.stderr


def passed(output: str) -> set[str]:
    return {
        line.split(' ', 1)[1].strip()
        for line in output.splitlines()
        if line.startswith('PASS ')
    }
