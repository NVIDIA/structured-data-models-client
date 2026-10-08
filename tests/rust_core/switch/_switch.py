# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 9 acceptance checks share (``CONTRACT.md``, beside this).

- Building this checkout's binding into the phase venv.
- The selector: running a process on the old Python path or on the core.
- Captures: every transcript and every public fixture case run through one
  path in a fresh process, with what each did and gave.
- Telling which code is ported, from the ledger snapshot ``ported.tsv``.

    python tests/rust_core/switch/_switch.py capture --impl core|python \
        --what transcripts|fixtures --out FILE

writes one capture.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import decimal
import hashlib
import importlib
import importlib.abc
import json
import os
import subprocess
import sys
import tempfile
import traceback
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RUST_CORE = HERE.parent
REPO = HERE.parents[2]
SCRIPTS = REPO / 'scripts' / 'parity'
PARITY = REPO / 'tests' / 'parity'
TRANSCRIPTS = PARITY / 'transcripts'
FIXTURES = PARITY / 'fixtures'
SHADOW = RUST_CORE / 'shadow'
PORTED = HERE / 'ported.tsv'
BASE_FACTS = HERE / 'base.json'
#: Where the implementer lists what the checks read (CONTRACT.md).
FIX_CHANGES = SHADOW / 'fix_changes.json'
PANDAS2_DIFFERENCES = SHADOW / 'pandas2_differences.json'
CHANGED_TESTS = SHADOW / 'changed_tests.tsv'


def add_paths() -> None:
    r"""Make the harness and the earlier checks' helpers importable."""
    for path in (HERE, RUST_CORE, RUST_CORE / 'sampling', PARITY, SCRIPTS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


#: The selector (CONTRACT.md, "The selector"), and its two values.
SELECTOR = 'SDM_CLIENT_IMPLEMENTATION'
OLD = 'python'
CORE = 'core'
IMPLS = (OLD, CORE)

#: The distributions whose code stays Python, beside sdm_core's own.
STAYING = (
    'structured-data-models-client',
    'structured-data-models-connectors',
    'structured-data-models-api',
)
PACKAGE_RENAMES = {
    'sdm-api': 'structured-data-models-api',
    'sdm-client': 'structured-data-models-client',
    'sdm-connectors': 'structured-data-models-connectors',
    'sdm-pql': 'structured-data-models-pql',
    'sdm-relational-engine': 'structured-data-models-relational-engine',
}

#: The packages the switched code must not import.
PORTED_PACKAGES = ('predictive_query', 'sdm_relational_engine')

#: The fixture operations made of public calls only, which the selector
#: routes; the shadow comparison, the error check and the import check run
#: every case of these.
PUBLIC_OPERATIONS = (
    'client.run',
    'schema.from_data',
    'decompose.parts',
    'driver.materialize',
)

#: The suite-rest command's paths and marker (checks.tsv, P9.suite-rest).
SUITE_PATHS = (
    'packages/structured-data-models-api/tests',
    'packages/structured-data-models-pql/tests',
    'packages/structured-data-models-client/tests',
    'packages/structured-data-models-connectors/tests',
    'tests',
)
SUITE_MARKER = 'not live_nim and not live_warehouse'

#: The fixes of brief task 7b and 7c an answer may change for
#: (``fix_changes.json``).
FIX_IDS = frozenset(
    {
        'composite-keys-on-sql',
        'null-foreign-key',
        'sqlite-unreadable-time-bounds',
        'time-of-day-column',
        'multilabel-link-prediction',
        'empty-table-from-source',
        'table-emptied-after-read',
        'complex-period-interval-columns',
        'about-the-part-forecast-step-1',
        'column-named-like-a-derived-key',
        'reserved-name-added-after-schema',
        'dropped-target-or-entity-column',
        'int-past-uint64',
        'decimal-entity-keys',
        'uint64-entity-keys',
        'arrow-large-string-feature',
        'arrow-duration-feature',
        'arrow-dictionary-feature',
        'arrow-large-string-key',
        'arrow-time32-feature',
        'arrow-time64-feature',
    }
)

#: How the engine wraps a failure it did not foresee: never a clear refusal.
ENGINE_FAILED = 'the relational engine failed'


# --- Small helpers -------------------------------------------------------------


def run(cmd: list[Any], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(c) for c in cmd], text=True, capture_output=True, **kwargs
    )


def git(*args: str) -> str:
    return run(['git', '-C', REPO, *args], check=True).stdout.strip()


def head_sha() -> str:
    return git('rev-parse', 'HEAD')


def worktree_clean() -> bool:
    return not git('status', '--porcelain')


def base_sha() -> str:
    harness = Path(
        os.environ.get('SDM_RC_HARNESS', '~/.local/share/sdm-dbx/rust-core')
    ).expanduser()
    return (harness / 'phases' / 'phase-9' / 'BASE').read_text().strip()


def qualname(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def short(value: Any, limit: int = 400) -> str:
    text = json.dumps(value, default=repr, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + '...'


def canonical(value: Any) -> Any:
    r"""``value`` as plain JSON that compares exactly: floats by ``repr``
    (NaN and the infinities included), times by ISO text, decimals and
    other scalars by text."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return {'$float': repr(value)}
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return {'$time': value.isoformat()}
    if isinstance(value, dt.timedelta):
        return {'$delta': str(value)}
    if isinstance(value, decimal.Decimal):
        return {'$decimal': str(value)}
    if isinstance(value, bytes):
        return {'$bytes': value.hex()}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if hasattr(value, 'item') and callable(value.item):
        with contextlib.suppress(Exception):
            return canonical(value.item())
    return {'$repr': repr(value)}


def digest(value: Any) -> str:
    text = json.dumps(canonical(value), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def first_difference(a: Any, b: Any, path: str = '$') -> str | None:
    r"""Where two JSON values first differ, by value and JSON type."""
    if type(a) is not type(b):
        return f'{path}: {short(a, 200)} != {short(b, 200)}'
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f'{path}.{key}: present on one side only'
            found = first_difference(a[key], b[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(a, list):
        for index, (x, y) in enumerate(zip(a, b, strict=False)):
            found = first_difference(x, y, f'{path}[{index}]')
            if found:
                return found
        if len(a) != len(b):
            return f'{path}: {len(a)} items != {len(b)} items'
        return None
    return None if a == b else f'{path}: {short(a, 200)} != {short(b, 200)}'


def transcripts() -> list[Path]:
    return sorted(TRANSCRIPTS.rglob('transcript.json'))


def transcript_name(path: Path) -> str:
    return json.loads(path.read_text(encoding='utf-8'))['name']


def uses_engine_input(case: dict[str, Any]) -> bool:
    r"""Whether a case's input builds an engine object itself."""
    return '$inference_config' in json.dumps(case['input'])


def public_fixture_cases() -> list[dict[str, Any]]:
    r"""Every case of the public operations, in file and case order, with
    the cases whose input builds an engine object last."""
    cases = []
    for path in sorted(FIXTURES.glob('*.json')):
        for case in json.loads(path.read_text(encoding='utf-8'))['cases']:
            if case['operation'] in PUBLIC_OPERATIONS:
                cases.append(case)
    return [c for c in cases if not uses_engine_input(c)] + [
        c for c in cases if uses_engine_input(c)
    ]


def error_chain(error: BaseException | None) -> list[str]:
    r"""The classes of an error and of every error a traceback shows beneath
    it: its ``__cause__``, or its ``__context__`` unless suppressed."""
    out: list[str] = []
    seen: set[int] = set()
    while error is not None and id(error) not in seen and len(out) < 12:
        seen.add(id(error))
        out.append(qualname(type(error)))
        if error.__cause__ is not None:
            error = error.__cause__
        elif not error.__suppress_context__:
            error = error.__context__
        else:
            error = None
    return out


def is_core_error_class(name: str) -> bool:
    r"""A class of the binding or of pyo3: what must never reach a user."""
    return (
        name.startswith('sdm_core')
        or name.startswith('pyo3_runtime')
        or name.endswith('.PanicException')
        or name.endswith('.CoreError')
    )


def public_error_types() -> tuple[type, ...]:
    import sdm_client

    return tuple(
        obj
        for obj in (getattr(sdm_client, n) for n in sdm_client.__all__)
        if isinstance(obj, type) and issubclass(obj, BaseException)
    )


def ported_loaded() -> dict[str, bool]:
    return {
        name: any(m == name or m.startswith(name + '.') for m in sys.modules)
        for name in PORTED_PACKAGES
    }


# --- The binding ---------------------------------------------------------------


def install_core() -> None:
    r"""Build this checkout's binding with maturin (Phase 7's wheel cache,
    per clean commit) and install it into this environment, once per
    process; the processes the checks start import what was installed."""
    if os.environ.get('SDM_P9_CORE_INSTALLED') == '1':
        return
    add_paths()
    import _sampling

    _sampling._install()
    os.environ['SDM_P9_CORE_INSTALLED'] = '1'
    importlib.invalidate_caches()


def env_for(impl: str, **extra: str) -> dict[str, str]:
    r"""The environment of a process on path ``impl``."""
    assert impl in IMPLS, impl
    env = dict(os.environ)
    env.pop(SELECTOR, None)
    env.pop('SDM_P9_UNREACHED_OUT', None)
    if impl == OLD:
        env[SELECTOR] = OLD
    # The core is what a user gets: the selector unset.
    env.update(extra)
    return env


# --- Captures ------------------------------------------------------------------


class ImportTracker(importlib.abc.MetaPathFinder):
    r"""Notes where each ported package was first imported from.

    While ``engine_input`` is set, a first import goes to
    ``engine_input_first`` instead: a fixture case whose input builds an
    engine object imports the engine itself (``tests/parity/_fixture_ops.py``,
    ``$inference_config``), and those cases run last and are not judged, as
    in P9.imports.
    """

    def __init__(self) -> None:
        self.first: dict[str, list[str]] = {}
        self.engine_input_first: dict[str, list[str]] = {}
        self.engine_input = False

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        top = fullname.split('.')[0]
        if (
            top in PORTED_PACKAGES
            and top not in self.first
            and top not in self.engine_input_first
        ):
            frames = traceback.extract_stack()[:-1]
            into = self.engine_input_first if self.engine_input else self.first
            into[top] = [
                f'{f.filename}:{f.lineno} {f.name}'
                for f in frames
                if 'importlib' not in f.filename
            ][-12:]
        return None

    def install(self) -> None:
        sys.meta_path.insert(0, self)


class SourceRuns:
    r"""Counts and keeps every ``Source._run`` call, as (SQL, parameters)."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def install(self) -> None:
        import inspect

        import sdm_connectors.source as source

        classes = [
            cls
            for cls in vars(source).values()
            if inspect.isclass(cls)
            and issubclass(cls, source.Source)
            and '_run' in vars(cls)
        ]
        for module_name in list(sys.modules):
            if module_name.startswith('sdm_connectors'):
                module = sys.modules[module_name]
                for obj in list(vars(module).values()):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, source.Source)
                        and '_run' in vars(obj)
                        and obj not in classes
                    ):
                        classes.append(obj)
        for cls in classes:
            original = vars(cls)['_run']

            def wrapped(
                self_: Any,
                sql: Any,
                params: Any = None,
                *args: Any,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                self.calls.append([str(sql), canonical(params)])
                return _original(self_, sql, params, *args, **kwargs)

            cls._run = wrapped  # type: ignore[attr-defined]


def _lazy_engine_hooks(original: Any) -> Any:
    r"""The harness's engine adjustment, only where the engine is already
    loaded: the replay harness must not import it on the core's behalf."""

    @contextlib.contextmanager
    def hooks() -> Iterator[None]:
        if 'sdm_relational_engine' in sys.modules:
            with original():
                yield
        else:
            yield

    return hooks


def _warning_record(w: warnings.WarningMessage) -> dict[str, Any]:
    return {
        'category': qualname(w.category),
        'message': str(w.message),
        'filename': str(w.filename),
    }


def _crash(error: BaseException) -> dict[str, Any]:
    tb = traceback.extract_tb(error.__traceback__)[-1:]
    return {
        'class': qualname(type(error)),
        'message': str(error)[:2000],
        'chain': error_chain(error),
        'where': f'{tb[0].filename}:{tb[0].lineno}' if tb else '',
    }


def capture_transcripts(impl: str) -> dict[str, Any]:
    add_paths()
    from _parity import runner
    from _parity import tape as tape_mod
    from _parity.corpus import check
    from _parity.nim import ReplayingSession
    from _parity.tape import ReplayingTape
    from _parity.transcript import load
    from _parity.values import canonical as parity_canonical
    from _parity.values import encode

    tracker = ImportTracker()
    tracker.install()
    import sdm_client  # noqa: F401 - loaded before the counter wraps sources
    import sdm_connectors  # noqa: F401

    runs = SourceRuns()
    runs.install()
    public = public_error_types()
    runner.engine_hooks = _lazy_engine_hooks(runner.engine_hooks)
    original_describe = runner.describe_error
    original_warning = runner._warning
    records = []
    for path in transcripts():
        transcript = load(path)
        rec: dict[str, Any] = {
            'name': transcript['name'],
            'path': str(path.relative_to(REPO)),
            'backend': transcript['backend'],
        }
        # As run_corpus.py replay holds it to what is stored. Its own
        # Source._run calls are not kept: the run below keeps them.
        runs.calls = []
        try:
            rec['check'] = check(path)
        except BaseException as error:  # noqa: BLE001 - recorded
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            rec['check'] = f'the replay raised {qualname(type(error))}: {error}'
        # Once more, keeping what the run did and gave.
        tape = ReplayingTape(
            json.loads(json.dumps(transcript['steps'])), path.parent
        )
        statements: list[Any] = []
        execute = tape.execute

        def logged(
            real: Any,
            sql: Any,
            params: Any,
            keyword: Any,
            _execute: Any = execute,
            _statements: list[Any] = statements,
        ) -> Any:
            _statements.append(
                [tape_mod._sql_text(sql), parity_canonical(encode(params))]
            )
            return _execute(real, sql, params, keyword)

        tape.execute = logged  # type: ignore[method-assign]
        session = ReplayingSession(
            json.loads(json.dumps(transcript.get('responses') or []))
        )
        raised: list[BaseException] = []

        def describe(error: BaseException, _raised: list[Any] = raised) -> Any:
            _raised.append(error)
            return original_describe(error)

        runner.describe_error = describe
        runner._warning = _warning_record
        runs.calls = []
        outcome = None
        try:
            outcome = runner.run(transcript, tape, session=session)
        except BaseException as error:  # noqa: BLE001 - recorded
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            rec['crashed'] = _crash(error)
        finally:
            runner.describe_error = original_describe
            runner._warning = original_warning
        rec['statements'] = statements
        rec['source_runs'] = runs.calls
        rec['divergence'] = (
            str(tape.divergence or session.divergence or '') or None
        )
        if outcome is not None:
            rec['payloads'] = canonical(outcome.payloads)
            rec['error'] = canonical(outcome.error)
            rec['answer'] = canonical(outcome.answer)
            rec['warnings'] = outcome.warnings
            rec['result'] = (
                None
                if outcome.result is None
                else {
                    'schema': str(outcome.result.schema),
                    'rows': canonical(outcome.result.to_pylist()),
                }
            )
        if raised:
            rec['error_chain'] = error_chain(raised[-1])
            rec['error_public'] = isinstance(raised[-1], public)
        rec['modules'] = ported_loaded()
        records.append(rec)
    return {
        'impl': impl,
        'records': records,
        'first_imports': tracker.first,
        'native_loaded': 'sdm_core._native' in sys.modules,
    }


def driver_materialize(input: dict[str, Any]) -> Any:
    r"""``driver.materialize`` (``tests/parity/_fixture_ops.py``) through
    public calls only: the payloads ``Request.payloads`` holds, and nothing
    of the engine reset, so a process keeps its own once-only warnings."""
    from _fixture_ops import _driver_schema, _user_warnings, as_error, decode
    from sdm_client import PredictiveQuery

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            with _driver_schema(input) as schema:
                request = PredictiveQuery(input['query'], schema).materialize(
                    **decode(input.get('options') or {})
                )
        except Exception as error:  # noqa: BLE001 - raised with its warnings
            raised = as_error(error)
            raised.warnings = _user_warnings(caught)
            raise raised from error
    payloads = json.loads(json.dumps(list(request.payloads or ())))
    for payload in payloads:
        metadata = payload.get('metadata')
        if isinstance(metadata, dict) and 'request_id' in metadata:
            metadata['request_id'] = 'request'
    text = json.dumps(payloads, sort_keys=True, separators=(',', ':'))
    return {
        'payloads': len(payloads),
        'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'warnings': _user_warnings(caught),
    }


def run_case(case: dict[str, Any]) -> Any:
    import _fixture_ops

    if case['operation'] == 'driver.materialize':
        try:
            return driver_materialize(case['input'])
        except _fixture_ops.FixtureError:
            raise
        except Exception as error:  # noqa: BLE001 - every refusal is an outcome
            raise _fixture_ops.as_error(error) from error
    return _fixture_ops.run(case['operation'], case['input'])


def capture_fixtures(impl: str) -> dict[str, Any]:
    add_paths()
    import _fixture_ops

    tracker = ImportTracker()
    tracker.install()
    import sdm_client  # noqa: F401
    import sdm_connectors  # noqa: F401

    runs = SourceRuns()
    runs.install()
    public = public_error_types()
    records = []
    for case in public_fixture_cases():
        rec: dict[str, Any] = {
            'id': case['id'],
            'operation': case['operation'],
            'engine_input': uses_engine_input(case),
        }
        tracker.engine_input = rec['engine_input']
        runs.calls = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                rec['output'] = canonical(
                    json.loads(json.dumps(run_case(case), default=repr))
                )
            except _fixture_ops.FixtureError as error:
                described = error.described()
                rec['error'] = canonical(described)
                cause = error.__cause__
                rec['error_chain'] = error_chain(cause if cause else error)
                rec['error_public'] = cause is None or isinstance(cause, public)
            except BaseException as error:  # noqa: BLE001 - recorded
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                rec['crashed'] = _crash(error)
        rec['warnings'] = [
            _warning_record(w)
            for w in caught
            if issubclass(w.category, UserWarning)
        ]
        rec['source_runs'] = runs.calls
        rec['modules'] = ported_loaded()
        records.append(rec)
    return {
        'impl': impl,
        'records': records,
        'first_imports': tracker.first,
        'engine_input_imports': tracker.engine_input_first,
        'native_loaded': 'sdm_core._native' in sys.modules,
    }


def _cache_dir() -> Path | None:
    if not worktree_clean():
        return None
    root = Path(
        os.environ.get(
            'SDM_RC_P9_CACHE',
            str(Path(tempfile.gettempdir()) / 'sdm-rust-core-p9'),
        )
    ).expanduser()
    key = hashlib.sha256(
        f'{head_sha()}\n{sys.executable}'.encode()
    ).hexdigest()[:16]
    return root / key


_CAPTURES: dict[tuple[str, str, str], dict[str, Any]] = {}


def capture(impl: str, what: str, hashseed: str = '0') -> dict[str, Any]:
    r"""One capture, made in a fresh process on path ``impl``; kept per
    clean commit, so the checks of one gate run share it."""
    key = (impl, what, hashseed)
    if key in _CAPTURES:
        return _CAPTURES[key]
    install_core()
    cache = _cache_dir()
    target = None
    if cache is not None:
        target = cache / f'{impl}-{what}-{hashseed}.json'
        if target.is_file():
            _CAPTURES[key] = json.loads(target.read_text())
            return _CAPTURES[key]
    with tempfile.TemporaryDirectory(prefix='sdm-p9-capture-') as tmp:
        out = Path(tmp) / 'capture.json'
        proc = run(
            [
                sys.executable,
                HERE / '_switch.py',
                'capture',
                '--impl',
                impl,
                '--what',
                what,
                '--out',
                out,
            ],
            cwd=REPO,
            env=env_for(impl, PYTHONHASHSEED=hashseed),
            timeout=3600,
        )
        assert proc.returncode == 0 and out.is_file(), (
            f'the {impl} capture of the {what} failed (exit {proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}'
        )
        data = json.loads(out.read_text())
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))
    _CAPTURES[key] = data
    return data


def by_key(data: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {rec[key]: rec for rec in data['records']}


def core_in_use(data: dict[str, Any]) -> list[str]:
    r"""Why a capture on the core path did not run on the core, if it did
    not: the binding was never loaded, or a ported package was (not counting
    what the fixture cases whose input builds an engine object import)."""
    problems = []
    if not data['native_loaded']:
        problems.append('sdm_core._native was never loaded')
    for name, stack in sorted(data['first_imports'].items()):
        problems.append(
            f'{name} was imported, first from:\n    ' + '\n    '.join(stack)
        )
    return problems


def fix_changes() -> dict[str, dict[str, str]]:
    r"""``fix_changes.json``: each transcript name or fixture case id whose
    answer a brief task 7b or 7c fix changes, with the fix and why."""
    if not FIX_CHANGES.is_file():
        return {}
    data = json.loads(FIX_CHANGES.read_text(encoding='utf-8'))
    assert isinstance(data, dict), 'fix_changes.json is not a JSON object'
    return data


# --- Ported code ----------------------------------------------------------------


def current_package_path(path: str) -> str:
    for old, new in PACKAGE_RENAMES.items():
        prefix = f'packages/{old}/'
        if path.startswith(prefix):
            return f'packages/{new}/' + path[len(prefix) :]
    return path


def ported_rows() -> dict[str, set[str]]:
    r"""``ported.tsv``: by repository path, the symbols that count as
    ported (``<module>`` for a module's own top-level code)."""
    out: dict[str, set[str]] = {}
    for line in PORTED.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        file, symbol = line.split('\t')[:2]
        out.setdefault(file, set()).add(symbol)
    return out


def spans_of(path: Path) -> list[tuple[str, int, int]]:
    import ast

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


class PortedCode:
    r"""Decides, per code object, whether it is ported code, first-party
    code that is not, test or harness code, or anything else."""

    PORTED = 'ported'
    LIBRARY = 'library'
    TEST = 'test'
    OTHER = 'other'

    def __init__(self) -> None:
        rows = ported_rows()
        self.files: dict[
            str, tuple[bool, list[tuple[str, int, int, bool]]]
        ] = {}
        for rel, symbols in rows.items():
            path = REPO / current_package_path(rel)
            if not path.is_file():
                continue
            spans = [(n, a, b, n in symbols) for n, a, b in spans_of(path)]
            entry = ('<module>' in symbols, spans)
            self.files[str(path)] = entry
            self.files[str(path.resolve())] = entry
        # The packages that stay. The engine and sdm-pql are the old
        # implementation as a whole: their code that is not ported is
        # neither, and the walk passes through it.
        self.src_roots = [
            str((REPO / 'packages' / p / 'src').resolve()) + os.sep
            for p in STAYING
        ]
        self.src_roots.append(
            str((REPO / 'rust' / 'bindings' / 'python' / 'python').resolve())
            + os.sep
        )
        self.test_roots = [
            str((REPO / 'tests').resolve()) + os.sep,
            str((REPO / 'scripts').resolve()) + os.sep,
        ] + [
            str((REPO / 'packages' / p / 'tests').resolve()) + os.sep
            for p in os.listdir(REPO / 'packages')
            if (REPO / 'packages' / p / 'tests').is_dir()
        ]
        self.installed_names = (
            'sdm_client',
            'sdm_connectors',
            'sdm_api',
            'sdm_core',
        )
        self._cache: dict[Any, tuple[str, str]] = {}

    def symbol(self, code: Any) -> str | None:
        r"""The ported symbol ``code`` belongs to, or ``None``."""
        found = self.files.get(code.co_filename)
        if found is None:
            with contextlib.suppress(OSError):
                found = self.files.get(str(Path(code.co_filename).resolve()))
        if found is None:
            return None
        module_ported, spans = found
        rel = str(Path(code.co_filename).resolve().relative_to(REPO.resolve()))
        if code.co_name == '<module>':
            return f'{rel}::<module>' if module_ported else None
        inside = [s for s in spans if s[1] <= code.co_firstlineno <= s[2]]
        if not inside:
            return f'{rel}::<module>' if module_ported else None
        name, _, _, ported = min(inside, key=lambda s: s[2] - s[1])
        return f'{rel}::{name}' if ported else None

    def symbol_resolved(self, code: Any, filename: str) -> str | None:
        r"""Use a freshly resolved filename supplied by the vetted catalog."""
        found = self.files.get(code.co_filename)
        if found is None:
            found = self.files.get(filename)
        if found is None:
            return None
        module_ported, spans = found
        rel = str(Path(filename).relative_to(REPO.resolve()))
        if code.co_name == '<module>':
            return f'{rel}::<module>' if module_ported else None
        inside = [s for s in spans if s[1] <= code.co_firstlineno <= s[2]]
        if not inside:
            return f'{rel}::<module>' if module_ported else None
        name, _, _, ported = min(inside, key=lambda s: s[2] - s[1])
        return f'{rel}::{name}' if ported else None

    def kind(self, code: Any) -> tuple[str, str]:
        cached = self._cache.get(code)
        if cached is not None:
            return cached
        symbol = self.symbol(code)
        if symbol is not None:
            out = (self.PORTED, symbol)
        else:
            filename = code.co_filename
            with contextlib.suppress(OSError, ValueError):
                filename = str(Path(filename).resolve())
            where = f'{filename}:{code.co_firstlineno} {code.co_qualname}'
            if any(filename.startswith(r) for r in self.test_roots) or (
                os.path.basename(filename) == 'conftest.py'
            ):
                out = (self.TEST, where)
            elif any(filename.startswith(r) for r in self.src_roots) or any(
                f'{os.sep}{name}{os.sep}' in filename
                and 'site-packages' in filename
                for name in self.installed_names
            ):
                out = (self.LIBRARY, where)
            else:
                out = (self.OTHER, where)
        self._cache[code] = out
        return out


# --- The suite and the replay under the guard -----------------------------------


def _cached(name: str, make: Any) -> dict[str, Any]:
    r"""``make()``, kept per clean commit under the cache."""
    cache = _cache_dir()
    if cache is not None and (cache / name).is_file():
        return json.loads((cache / name).read_text())
    data = make()
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_text(json.dumps(data))
    return data


def suite_report() -> dict[str, Any]:
    r"""The suite-rest command (checks.tsv, P9.suite-rest) run on the core
    under ``_unreached.py``'s guard: the guard's report and the outcomes."""

    def make() -> dict[str, Any]:
        install_core()
        with tempfile.TemporaryDirectory(prefix='sdm-p9-suite-') as tmp:
            out = Path(tmp) / 'report.json'
            env = env_for(CORE)
            env['SDM_P9_UNREACHED_OUT'] = str(out)
            env['PYTHONPATH'] = os.pathsep.join(
                [str(HERE), *filter(None, [env.get('PYTHONPATH')])]
            )
            proc = run(
                [
                    sys.executable,
                    '-m',
                    'pytest',
                    *SUITE_PATHS,
                    '-q',
                    '-m',
                    SUITE_MARKER,
                    '-p',
                    '_unreached',
                    '-p',
                    'no:cacheprovider',
                ],
                cwd=REPO,
                env=env,
                timeout=5400,
            )
            assert out.is_file(), (
                f'the suite wrote no guard report (exit {proc.returncode}):\n'
                f'{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}'
            )
            report = json.loads(out.read_text())
            report['tail'] = proc.stdout[-3000:]
            return report

    return _cached('suite-report.json', make)


def replay_report() -> dict[str, Any]:
    r"""``run_corpus.py replay`` on the core under the guard."""

    def make() -> dict[str, Any]:
        install_core()
        with tempfile.TemporaryDirectory(prefix='sdm-p9-replay-') as tmp:
            out = Path(tmp) / 'report.json'
            proc = run(
                [
                    sys.executable,
                    HERE / '_unreached.py',
                    'replay',
                    '--out',
                    out,
                ],
                cwd=REPO,
                env=env_for(CORE),
                timeout=3600,
            )
            assert proc.returncode == 0 and out.is_file(), (
                f'the guarded replay failed (exit {proc.returncode}):\n'
                f'{proc.stdout[-3000:]}\n{proc.stderr[-4000:]}'
            )
            report = json.loads(out.read_text())
            report['tail'] = proc.stdout[-3000:] + proc.stderr[-1000:]
            return report

    return _cached('replay-report.json', make)


def collected(paths: tuple[str, ...] = SUITE_PATHS) -> list[str]:
    r"""The test ids the suite-rest command collects, the implementer's
    shadow tests left out."""
    proc = run(
        [
            sys.executable,
            '-m',
            'pytest',
            *paths,
            '-q',
            '-m',
            SUITE_MARKER,
            '--collect-only',
            '-p',
            'no:cacheprovider',
            '--ignore',
            str(SHADOW.relative_to(REPO)),
        ],
        cwd=REPO,
        env=env_for(CORE),
        timeout=1800,
    )
    ids = [
        line.strip()
        for line in proc.stdout.splitlines()
        if '::' in line and not line.startswith(' ')
    ]
    assert ids, (
        f'nothing collected:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
    )
    return ids


def suite_test_files() -> list[Path]:
    r"""The suite-rest test code: every ``.py`` file under its paths, the
    shadow harness and these checks left out."""
    out = []
    for root in SUITE_PATHS:
        for path in sorted((REPO / root).rglob('*.py')):
            rel = path.relative_to(REPO).as_posix()
            if (
                rel.startswith(
                    ('tests/rust_core/shadow/', 'tests/rust_core/switch/')
                )
                or '__pycache__' in rel
            ):
                continue
            out.append(path)
    return out


def assertion_count(paths: list[Path]) -> int:
    r"""``assert`` statements, and calls of ``pytest.raises``,
    ``pytest.warns``, ``pytest.approx`` or any ``assert_*`` function."""
    import ast

    total = 0
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                total += 1
            elif isinstance(node, ast.Call):
                fn = node.func
                name = (
                    fn.attr
                    if isinstance(fn, ast.Attribute)
                    else fn.id
                    if isinstance(fn, ast.Name)
                    else ''
                )
                if name in ('raises', 'warns', 'approx') or name.startswith(
                    'assert_'
                ):
                    total += 1
    return total


def changed_test_files() -> list[str]:
    r"""Every test-code file under the suite-rest paths added, changed,
    deleted or renamed since BASE (both names of a rename)."""
    out = set()
    for line in git(
        'diff', '--name-status', '-M', f'{base_sha()}...HEAD'
    ).splitlines():
        parts = line.split('\t')
        for rel in parts[1:]:
            if not rel.endswith('.py'):
                continue
            if not current_package_path(rel).startswith(
                tuple(p + '/' for p in SUITE_PATHS)
            ):
                continue
            if rel.startswith(
                ('tests/rust_core/shadow/', 'tests/rust_core/switch/')
            ):
                continue
            out.add(rel)
    return sorted(out)


# --- CPU time of the replay ----------------------------------------------------


def replay_cpu_seconds(skip: frozenset[str] = frozenset()) -> dict[str, Any]:
    r"""In this process: the CPU time replaying every transcript but those
    named in ``skip`` takes, imports included, and whether each passed."""
    import time

    add_paths()
    from _parity import runner
    from _parity.corpus import check

    tracker = ImportTracker()
    tracker.install()
    runner.engine_hooks = _lazy_engine_hooks(runner.engine_hooks)
    named = {path: transcript_name(path) for path in transcripts()}
    paths = [path for path, name in named.items() if name not in skip]
    skipped = sorted(name for name in named.values() if name in skip)
    failed = []
    started = time.process_time()
    for path in paths:
        reason = check(path)
        if reason is not None:
            failed.append(
                f'{path.parent.relative_to(TRANSCRIPTS)}: {reason[:200]}'
            )
    spent = time.process_time() - started
    return {
        'cpu': spent,
        'transcripts': len(paths),
        'skipped': skipped,
        'failed': failed,
        'first_imports': tracker.first,
        'native_loaded': 'sdm_core._native' in sys.modules,
    }


# --- The command line ---------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    cap = commands.add_parser('capture')
    cap.add_argument('--impl', choices=IMPLS, required=True)
    cap.add_argument(
        '--what', choices=('transcripts', 'fixtures'), required=True
    )
    cap.add_argument('--out', required=True)
    cpu = commands.add_parser('cpu')
    cpu.add_argument('--impl', choices=IMPLS, required=True)
    cpu.add_argument('--out', required=True)
    cpu.add_argument('--skip', action='append', default=[])
    args = parser.parse_args(argv)
    if args.command == 'cpu':
        assert os.environ.get(SELECTOR, CORE) == args.impl
        Path(args.out).write_text(
            json.dumps(replay_cpu_seconds(frozenset(args.skip)))
        )
        return 0
    assert os.environ.get(SELECTOR, CORE) == args.impl, (
        f'{SELECTOR} is {os.environ.get(SELECTOR)!r}, not {args.impl!r}'
    )
    if args.what == 'transcripts':
        data = capture_transcripts(args.impl)
    else:
        data = capture_fixtures(args.impl)
    Path(args.out).write_text(json.dumps(data))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
