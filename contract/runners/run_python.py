#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Python runner of the contract cases, in three modes.

Run with ``python -I``, from the interpreter of the environment under test:

    python -I contract/runners/run_python.py --mode MODE [--reference DIR]
        --out FILE CASE_OR_DIRECTORY...

- ``python-ref``: the reference, the oracle. ``--reference`` names the
  reference checkout, whose environment must be the interpreter's. Its guard
  (``scripts/reference_guard.py``) is loaded by path, and its
  ``require_isolation()`` runs before the first case and after the last;
  either failing stops the run, which writes nothing. ``sdm_client`` must be
  the reference checkout's.
- ``python-core``: the candidate, with ``SDM_CORE_ROUTES`` set for each case
  to its ``routes``. The core operations the case starts, with their inputs,
  and the core's answer are recorded at ``sdm_client._native``'s boundary,
  changing nothing that crosses it.
- ``python-old``: the candidate on its default routes, a smoke runner that
  must complete every case (decision H10).

Each case builds its DuckDB base on a fresh in-memory connection, then makes
the public calls: ``Schema.from_source``, then ``materialize`` (the S0 call),
as far as the case's ``until``. An exception a public call raises is the
case's outcome, recorded by class, ``StructuredDataError.code`` and stage, a
``RuntimeError`` included; the comparator decides whether it is the expected
one. A runner failure is anything else: the guard failing, a missing
``sdm_client._native`` or a refused forced route (``RouteError``), an error
in the runner itself, an observation that does not validate, or the process
dying. It exits non-zero and writes nothing; observations are written only
when every case was recorded. No message text is recorded (decision H17).

This file also holds the case loader and the Phase 1b inventory loader,
which maps each inventory case into the case format and back without loss.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent
REPO = CONTRACT.parent
BASES = CONTRACT / 'bases.yaml'
INVENTORY = REPO / 'docs/design/rust-core/phase-0/phase-1b-inventory.yaml'
SCHEMAS = {
    'case': CONTRACT / 'case.schema.json',
    'inventory': CONTRACT / 'inventory.schema.json',
    'observation': CONTRACT / 'observation.schema.json',
}
MODES = ('python-ref', 'python-core', 'python-old')
#: The runtime cells (design, "Runtime cells"): Python, pandas, pyarrow and
#: NumPy versions. An environment with exactly one's versions is that cell.
CELLS = {
    'ref-pandas3': ('3.12', '3.0.6', '25.0.1', '2.5.3'),
    'dbr17-pandas2': ('3.12', '2.2.3', '19.0.1', '2.1.3'),
    'floor-pandas2.2': ('3.10', '2.2.0', '14.0.1', '1.26.4'),
}


class RunnerFailure(Exception):
    r"""The runner, not the case, failed; its text never quotes an input."""


# --- schemas and cases ---------------------------------------------------------


def validator(name: str) -> Any:
    r"""A JSON Schema validator for ``name`` (``case``, ``inventory`` or
    ``observation``), or for a ``$ref`` into one, resolving the references
    between the three."""
    import jsonschema
    from referencing import Registry, Resource

    schemas = {
        key: json.loads(path.read_text(encoding='utf-8'))
        for key, path in SCHEMAS.items()
    }
    registry = Registry().with_resources(
        (schema['$id'], Resource.from_contents(schema))
        for schema in schemas.values()
    )
    schema = schemas[name] if name in schemas else {'$ref': name}
    cls = jsonschema.validators.validator_for(schemas['case'])
    cls.check_schema(schema)
    return cls(schema, registry=registry)


def _invalid(validate: Any, document: Any) -> str | None:
    r"""The first schema error's location and rule, or None; never the value."""
    error = next(iter(validate.iter_errors(document)), None)
    if error is None:
        return None
    where = '/'.join(str(part) for part in error.absolute_path) or '(root)'
    return f'{where} fails {error.validator!r}'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def case_files(paths: Iterable[Path]) -> list[Path]:
    r"""The case files named, a directory standing for its ``*.json``."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob('*.json')))
        else:
            found.append(path)
    return found


def load_cases(paths: Iterable[Path]) -> list[tuple[dict[str, Any], str]]:
    r"""Each case with the SHA-256 of its file, validated, ids unique."""
    validate = validator('case')
    cases: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for path in case_files(paths):
        data = path.read_bytes()
        case = json.loads(data)
        problem = _invalid(validate, case)
        if problem is not None:
            raise RunnerFailure(f'{path.name}: not a case: {problem}')
        if case['id'] in seen:
            raise RunnerFailure(f'{path.name}: case id {case["id"]} repeats')
        seen.add(case['id'])
        cases.append((case, digest(data)))
    return cases


def load_bases(path: Path = BASES) -> dict[str, list[str]]:
    r"""The named DuckDB bases of ``contract/bases.yaml``."""
    import yaml

    document = yaml.safe_load(path.read_text(encoding='utf-8'))
    if (
        not isinstance(document, dict)
        or set(document) != {'format', 'bases'}
        or document['format'] != 'sdm-bases/1'
    ):
        raise RunnerFailure(f'{path.name}: not sdm-bases/1')
    problem = _invalid(
        validator('urn:sdm:case:1#/$defs/bases'), document['bases']
    )
    if problem is not None:
        raise RunnerFailure(f'{path.name}: not sdm-bases/1: {problem}')
    bases: dict[str, list[str]] = document['bases']
    return bases


# --- the Phase 1b inventory ------------------------------------------------------

#: What every case taken from the inventory has, which the inventory does
#: not say: diagnostic until its task admits it, S1, no paths or routes yet
#: (coverage and the routes arrive with the cases' tasks), run to predict.
_FROM_INVENTORY = {
    'format': 'sdm-case/1',
    'status': 'diagnostic',
    'slice': 'S1',
    'paths': [],
    'routes': [],
    'until': 'predict',
}
_INVENTORY_INPUT = ('base', 'setup', 'from_source')
_INVENTORY_KEPT = ('id', 'seam', 'cells', 'expect', 'register', 'why', 'source')


def load_inventory(path: Path = INVENTORY) -> dict[str, Any]:
    r"""The Phase 1b inventory, validated, with unique case ids."""
    import yaml

    inventory = yaml.safe_load(path.read_text(encoding='utf-8'))
    problem = _invalid(validator('inventory'), inventory)
    if problem is not None:
        raise RunnerFailure(f'{path.name}: not the inventory: {problem}')
    ids = [entry['id'] for entry in inventory['cases']]
    if len(ids) != len(set(ids)):
        raise RunnerFailure(f'{path.name}: a case id repeats')
    unknown = {entry['base'] for entry in inventory['cases']} - set(
        inventory['bases']
    )
    if unknown:
        raise RunnerFailure(f'{path.name}: a case names an unknown base')
    loaded: dict[str, Any] = inventory
    return loaded


def inventory_case(
    entry: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    r"""An inventory case in the case format; :func:`inventory_entry` is
    its inverse. The inventory's ``pql`` and ``materialize`` default from
    its ``defaults``."""
    entry = copy.deepcopy(entry)
    case: dict[str, Any] = copy.deepcopy(_FROM_INVENTORY)
    case['id'] = entry['id']
    case['input'] = {
        key: entry[key] for key in _INVENTORY_INPUT if key in entry
    }
    case['input']['pql'] = entry.get('pql', defaults['pql'])
    case['input']['materialize'] = entry.get(
        'materialize', copy.deepcopy(defaults['materialize'])
    )
    for key in _INVENTORY_KEPT:
        if key in entry:
            case[key] = entry[key]
    return case


def inventory_entry(
    case: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    r"""A case taken from the inventory, as the inventory wrote it."""
    case = copy.deepcopy(case)
    for key, value in _FROM_INVENTORY.items():
        if case.pop(key) != value:
            raise RunnerFailure(f'{case["id"]}: not a case of the inventory')
    entry = {key: case.pop(key) for key in _INVENTORY_KEPT if key in case}
    given = case.pop('input')
    for key in _INVENTORY_INPUT:
        if key in given:
            entry[key] = given.pop(key)
    for key in ('pql', 'materialize'):
        value = given.pop(key)
        if value != defaults[key]:
            entry[key] = value
    if given or case:
        raise RunnerFailure(
            f'{entry["id"]}: keys the inventory has no place for'
        )
    return entry


# --- identity --------------------------------------------------------------------


def _git(directory: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        done = subprocess.run(
            ['git', '-C', str(directory), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode != 0:
            raise RunnerFailure(f'git {arguments[0]} failed in {directory}')
        return done.stdout

    return {
        'sha': run('rev-parse', 'HEAD').strip(),
        'clean': run('status', '--porcelain').strip() == '',
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions() -> dict[str, str]:
    found = {
        'python': platform.python_version(),
        'pandas': _version('pandas'),
        'pyarrow': _version('pyarrow'),
        'numpy': _version('numpy'),
        'duckdb': _version('duckdb'),
    }
    return {key: value for key, value in found.items() if value is not None}


def _cell(versions: dict[str, str]) -> str | None:
    have = (
        '.'.join(versions['python'].split('.')[:2]),
        versions.get('pandas'),
        versions.get('pyarrow'),
        versions.get('numpy'),
    )
    return next((name for name, want in CELLS.items() if want == have), None)


def _environment() -> str:
    r"""The SHA-256 of the environment's installed distributions."""
    names = sorted(
        f'{(dist.metadata["Name"] or "").lower()}=={dist.version}'
        for dist in importlib.metadata.distributions()
    )
    return digest('\n'.join(names).encode())


# --- the reference guard -----------------------------------------------------------


class _Guard:
    r"""The reference checkout's guard, loaded by path."""

    def __init__(self, reference: Path) -> None:
        path = reference / 'scripts/reference_guard.py'
        spec = importlib.util.spec_from_file_location('reference_guard', path)
        if spec is None or spec.loader is None or not path.is_file():
            raise RunnerFailure('the reference checkout has no guard')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module

    def check(self, when: str) -> None:
        try:
            self._module.require_isolation()
        except Exception:  # noqa: BLE001 - any failure of the guard stops
            raise RunnerFailure(
                f'the reference guard failed {when}; the reference '
                'environment can load the Rust core'
            ) from None


# --- the core's boundary, for python-core --------------------------------------------


class _Capture:
    r"""What crossed ``sdm_client._native`` during one case."""

    def __init__(self) -> None:
        self.operations: list[str] = []
        self.inputs: list[dict[str, Any]] = []
        self.accepted: bool | None = None
        self.error: dict[str, str] | None = None

    def started(self, message: bytes, attachments: Any) -> None:
        operation = json.loads(message)['operation']
        self.operations.append(operation['op'])
        if operation['op'] == 'validate_query':
            self.inputs.append(
                {
                    'op': 'validate_query',
                    'query': json.loads(attachments[operation['query']]),
                    'profile': operation['profile'],
                    'schema': json.loads(attachments[operation['schema']]),
                }
            )

    def failed(self, error: Any) -> None:
        self.error = {'kind': str(error.kind), 'code': str(error.code)}

    def done(self, result: dict[str, Any]) -> None:
        if 'accepted' in result:
            self.accepted = bool(result['accepted'])

    def core(self) -> dict[str, Any]:
        return {
            'inputs': self.inputs,
            'accepted': self.accepted,
            'error': self.error,
        }


class _Handle:
    def __init__(self, handle: Any, native: Any, capture: _Capture) -> None:
        self._handle = handle
        self._native = native
        self._capture = capture

    def next(self) -> Any:
        from sdm_client._rust import codec

        try:
            control, sent = self._handle.next()
        except self._native.CoreError as error:
            self._capture.failed(error)
            raise
        message = codec.read(control, sent)
        if message.get('message') == 'step' and message.get('step') == 'done':
            self._capture.done(message['result'])
        return control, sent

    def reply(self, message: bytes, attachments: Any) -> None:
        try:
            self._handle.reply(message, attachments)
        except self._native.CoreError as error:
            self._capture.failed(error)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


class _Run:
    def __init__(self, native: Any, capture: _Capture) -> None:
        self._native = native
        self._capture = capture

    def start(self, message: bytes, attachments: Any) -> _Handle:
        self._capture.started(message, attachments)
        try:
            handle = self._native.Run.start(message, attachments)
        except self._native.CoreError as error:
            self._capture.failed(error)
            raise
        return _Handle(handle, self._native, self._capture)


class _Native:
    r"""``sdm_client._native`` as the host sees it, its runs recorded."""

    def __init__(self, native: Any, capture: _Capture) -> None:
        self._native = native
        self.Run = _Run(native, capture)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)


class _Boundary:
    r"""Records each case's core runs, by wrapping the one function through
    which the Python host imports ``sdm_client._native``."""

    def __init__(self) -> None:
        from sdm_client._rust import run

        self.capture = _Capture()
        imported = run.native

        def native() -> Any:
            return _Native(imported(), self.capture)

        run.native = native  # type: ignore[assignment]

    def reset(self) -> _Capture:
        self.capture = _Capture()
        return self.capture


# --- one case --------------------------------------------------------------------


def _connection(case: dict[str, Any], bases: dict[str, list[str]]) -> Any:
    import duckdb

    connection = duckdb.connect()
    given = case['input']
    if given['base'] not in bases:
        raise RunnerFailure(f'{case["id"]}: unknown base')
    for statement in [*bases[given['base']], *given.get('setup', [])]:
        try:
            connection.execute(statement)
        except Exception:  # noqa: BLE001 - building the input is not the call
            raise RunnerFailure(
                f'{case["id"]}: its data could not be built'
            ) from None
    return connection


def _materialize_arguments(spec: dict[str, Any]) -> dict[str, Any]:
    from decimal import Decimal

    import pandas as pd

    indices = spec['indices']
    if spec.get('indices_as') == 'decimal':
        indices = [Decimal(value) for value in indices]
    arguments: dict[str, Any] = {'indices': list(indices)}
    if 'anchor_time' in spec:
        arguments['anchor_time'] = pd.Timestamp(spec['anchor_time'])
    return arguments


def _route_error(error: BaseException) -> bool:
    kind = type(error)
    return kind.__module__ == 'sdm_client._routing' and kind.__name__ == (
        'RouteError'
    )


def _outcome(
    result: str, stage: str, error: BaseException | None = None
) -> dict[str, Any]:
    code = getattr(error, 'code', None)
    return {
        'result': result,
        'stage': stage,
        'error': None if error is None else type(error).__name__,
        'code': code if isinstance(code, str) else None,
    }


def _call(
    case: dict[str, Any], bases: dict[str, list[str]]
) -> tuple[dict[str, Any], list[str]]:
    r"""Makes the case's public calls: its outcome and warning classes."""
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource

    connection = _connection(case, bases)
    given = case['input']
    stage = 'from_source'
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        try:
            schema = Schema.from_source(
                DuckDBSource(connection), **given.get('from_source', {})
            )
            if case['until'] != 'from_source':
                stage = 'materialize'
                PredictiveQuery(given['pql'], schema).materialize(
                    **_materialize_arguments(given['materialize'])
                )
            if case['until'] == 'predict':
                raise RunnerFailure(
                    f'{case["id"]}: this runner does not make the predict '
                    'call yet'
                )
            outcome = _outcome('success', stage)
        except RunnerFailure:
            raise
        except Exception as error:  # noqa: BLE001 - the call's outcome
            if _route_error(error):
                raise RunnerFailure(
                    f'{case["id"]}: the route was refused '
                    f'({type(error).__name__})'
                ) from None
            outcome = _outcome('refusal', stage, error)
        finally:
            connection.close()
    return outcome, [type(warning.message).__name__ for warning in shown]


# --- the run ---------------------------------------------------------------------


def _native_sha256() -> str | None:
    module = sys.modules.get('sdm_client._native')
    location = getattr(module, '__file__', None)
    if module is None or location is None:
        return None
    return digest(Path(location).read_bytes())


def run(
    mode: str,
    cases: list[tuple[dict[str, Any], str]],
    reference: Path | None,
) -> list[dict[str, Any]]:
    r"""Every case's observation, or :class:`RunnerFailure`."""
    os.environ.pop('SDM_CORE_ROUTES', None)
    guard = _Guard(reference) if reference is not None else None
    if guard is not None:
        guard.check('before the first case')
        import sdm_client

        location = Path(sdm_client.__file__).resolve()
        if not location.is_relative_to(reference.resolve()):  # type: ignore[union-attr]
            raise RunnerFailure("sdm_client is not the reference checkout's")
    bases_path = BASES
    bases = load_bases(bases_path)
    bases_digest = digest(bases_path.read_bytes())
    boundary = _Boundary() if mode == 'python-core' else None
    versions = _versions()
    if 'duckdb' not in versions:
        raise RunnerFailure('duckdb is not installed')
    recorded = []
    for case, case_digest in cases:
        capture = boundary.reset() if boundary is not None else None
        if mode == 'python-core':
            os.environ['SDM_CORE_ROUTES'] = ','.join(case['routes'])
        try:
            outcome, shown = _call(case, bases)
        finally:
            os.environ.pop('SDM_CORE_ROUTES', None)
        recorded.append(
            (
                case,
                case_digest,
                outcome,
                shown,
                capture,
                'sdm_client._native' in sys.modules,
            )
        )
    reference_identity = None
    if guard is not None and reference is not None:
        guard.check('after the last case')
        reference_identity = {
            'commit': _git(reference),
            'environment': _environment(),
            'guard': {'before': 'passed', 'after': 'passed'},
        }
    native_sha256 = _native_sha256() if mode == 'python-core' else None
    contract = None
    if mode == 'python-core' and 'sdm_client._native' in sys.modules:
        contract = sys.modules['sdm_client._native'].CONTRACT_VERSION
    commit = _git(REPO)
    validate = validator('observation')
    observations = []
    for case, case_digest, outcome, shown, capture, loaded in recorded:
        observation = {
            'format': 'sdm-observation/1',
            'case': case['id'],
            'identity': {
                'runner': mode,
                'cell': _cell(versions),
                'platform': f'{sys.platform}-{platform.machine()}',
                'commit': commit,
                'reference': reference_identity,
                'native': (
                    {'sha256': native_sha256} if native_sha256 else None
                ),
                'native_loaded': loaded,
                'operations': capture.operations if capture else [],
                'contract': contract,
                'versions': versions,
                'inputs': {'case': case_digest, 'bases': bases_digest},
            },
            'outcome': outcome,
            'host': {'warnings': shown},
            'core': capture.core() if capture else None,
        }
        problem = _invalid(validate, observation)
        if problem is not None:
            raise RunnerFailure(
                f'{case["id"]}: its observation is not valid: {problem}'
            )
        observations.append(observation)
    return observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('cases', type=Path, nargs='+')
    arguments = parser.parse_args(argv)
    if (arguments.mode == 'python-ref') != (arguments.reference is not None):
        parser.error('--reference is given in python-ref mode, and only there')
    try:
        cases = load_cases(arguments.cases)
        if not cases:
            raise RunnerFailure('no cases')
        observations = run(arguments.mode, cases, arguments.reference)
    except RunnerFailure as failure:
        print(f'runner failure: {failure}', file=sys.stderr)
        return 1
    partial = arguments.out.with_name(arguments.out.name + '.partial')
    partial.write_text(
        ''.join(
            json.dumps(item, sort_keys=True) + '\n' for item in observations
        ),
        encoding='utf-8',
    )
    partial.replace(arguments.out)
    print(f'{arguments.mode}: {len(observations)} cases recorded')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
