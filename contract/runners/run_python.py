#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Python runner of the contract cases, in three modes.

Run with ``python -I``, from the interpreter of the environment under test:

    python -I contract/runners/run_python.py --mode MODE [--reference DIR]
        [--no-capture] [--inventory ID]... --out FILE [CASE_OR_DIRECTORY...]

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
then ``SDMClient.predict`` against the scripted endpoint
(``scripted_endpoint.py``), served on localhost and reached through the
client's real transport, as far as the case's ``until``.

The source is handed the connection through the test-only driver capture
(design, "Driver capture"; decision Q2), which records every statement the
driver runs, its typed parameters, each result read and every relation
bound by name, derived cursors included, and changes nothing that crosses
it; ``--no-capture`` hands it the connection itself, the conservation
control. Setup, the statements before the connection reaches the source,
is outside it. DuckDB's own query log is read after each case either way,
so ``capture_check.py`` can hold the captured counts to it and the two runs
to the same public projection (``public``: the schema, each request the
scripted endpoint received, the result). Cases of the Phase 1b inventory are
named with ``--inventory ID``. An exception a public call raises is the
case's outcome, recorded by class, ``StructuredDataError.code`` and stage, a
``RuntimeError`` included; the comparator decides whether it is the expected
one. A runner failure is anything else: the guard failing, a missing
``sdm_client._native`` or a refused forced route (``RouteError``), an error
in the runner itself, an observation that does not validate, or the process
dying. It exits non-zero and leaves no output file, an earlier run's
included; observations are written only when every case was recorded. No message text is recorded (decision H17).

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
import struct
import subprocess
import sys
import warnings
from collections.abc import Iterable, Mapping
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


def _indices(given: dict[str, Any]) -> str | None:
    r"""An entity key is a JSON integer literal of any size, or a string.

    JSON Schema counts ``1.0`` as an integer; ``json`` reads it as a float,
    and a bool is no index either, so the type read is checked too, as
    sdm-testkit's loader refuses any number with a fraction or exponent.
    """
    for position, value in enumerate(given['materialize']['indices']):
        if type(value) not in (int, str):
            return f'input/materialize/indices/{position} is not an integer'
    return None


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
        problem = _invalid(validate, case) or _indices(case['input'])
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
    # The case-level rules beyond the schema hold for the defaults a case
    # inherits as for what it states; the indices rule is the only one.
    problem = _indices(inventory['defaults'])
    if problem is not None:
        raise RunnerFailure(f'{path.name}: defaults: {problem}')
    for entry in inventory['cases']:
        problem = _indices(entry) if 'materialize' in entry else None
        if problem is not None:
            raise RunnerFailure(f'{path.name}: {entry["id"]}: {problem}')
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


def inventory_cases(ids: Iterable[str]) -> list[tuple[dict[str, Any], str]]:
    r"""The inventory cases named, in the case format, each with the
    SHA-256 of its canonical JSON."""
    wanted = list(ids)
    if not wanted:
        return []
    inventory = load_inventory()
    entries = {entry['id']: entry for entry in inventory['cases']}
    validate = validator('case')
    found = []
    for case_id in wanted:
        if case_id not in entries:
            raise RunnerFailure(f'{case_id}: not an inventory case')
        case = inventory_case(entries[case_id], inventory['defaults'])
        problem = _invalid(validate, case)
        if problem is not None:
            raise RunnerFailure(f'{case_id}: not a case: {problem}')
        text = json.dumps(case, sort_keys=True).encode()
        found.append((case, digest(text)))
    return found


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
    r"""What crossed ``sdm_client._native`` during one case: each operation
    started with its inputs, each fetch the core asked for (its request and
    plan, the step trace), and the core's answers: ``validate_query``'s
    acceptance, ``describe_source``'s description, ``materialize``'s
    request state (its version, the query and the payload, each number as
    the text the core wrote), ``predict``'s call (its body, each number as
    its text) and answer (the endpoint's model, call id and metadata, and
    the result's rows), or an error.

    Attachment ids are a run's own, so what this remembers of one run's ids
    is forgotten when the next run starts. The request id the host starts
    ``predict`` with is recorded where it reaches the endpoint, in the
    call's ``metadata.request_id``, and a call that does not carry it is a
    runner failure."""

    def __init__(self) -> None:
        self.operations: list[str] = []
        self.inputs: list[dict[str, Any]] = []
        self.fetches: list[dict[str, Any]] = []
        self.accepted: bool | None = None
        self.described: dict[str, Any] | None = None
        self.materialized: dict[str, Any] | None = None
        self.predicted: dict[str, Any] | None = None
        self.error: dict[str, str] | None = None
        self._description: int | None = None
        self._state: dict[str, Any] | None = None
        self._request_id: str | None = None
        self._call: int | None = None
        self._body: dict[str, Any] | None = None
        self._answer: dict[str, Any] | None = None

    def started(self, message: bytes, attachments: Any) -> None:
        operation = json.loads(message)['operation']
        self.operations.append(operation['op'])
        self._description = None
        self._state = None
        self._call = None
        self._answer = None
        if operation['op'] == 'predict':
            state = operation['state']
            self._request_id = operation['request_id']
            self.inputs.append(
                {
                    'op': 'predict',
                    'model': operation['model'],
                    'state': {
                        'version': state['version'],
                        'query': json.loads(attachments[state['query']]),
                        'payload': json.loads(
                            attachments[state['payload']],
                            parse_int=_number,
                            parse_float=_number,
                        ),
                    },
                }
            )
        if operation['op'] == 'validate_query':
            self.inputs.append(
                {
                    'op': 'validate_query',
                    'query': json.loads(attachments[operation['query']]),
                    'profile': operation['profile'],
                    'schema': json.loads(attachments[operation['schema']]),
                }
            )
        elif operation['op'] == 'describe_source':
            self.inputs.append(
                {
                    'op': 'describe_source',
                    'provider': operation['provider'],
                    'source': json.loads(attachments[operation['source']]),
                    'check_keys': operation['check_keys'],
                }
            )
        elif operation['op'] == 'materialize':
            indices = operation['indices']
            self.inputs.append(
                {
                    'op': 'materialize',
                    'provider': operation['provider'],
                    'schema': json.loads(attachments[operation['schema']]),
                    'query': json.loads(attachments[operation['query']]),
                    'indices': (
                        None
                        if indices is None
                        else _scalars(attachments[indices])
                    ),
                    'anchor_time': operation['anchor_time'],
                    'seed': operation['seed'],
                }
            )

    def stepped(self, step: dict[str, Any]) -> None:
        if step.get('step') == 'fetch':
            self.fetches.extend(
                {'request': asked['request'], 'plan': asked['plan']}
                for asked in step['requests']
            )
        if step.get('step') == 'call_model':
            self._call = step['body']

    def failed(self, error: Any) -> None:
        self.error = {'kind': str(error.kind), 'code': str(error.code)}

    def done(self, result: dict[str, Any]) -> None:
        if 'accepted' in result:
            self.accepted = bool(result['accepted'])
        if result.get('op') == 'describe_source':
            self._description = result['description']
        if result.get('op') == 'materialize':
            self._state = result['state']
            self.materialized = {'version': result['state']['version']}
        if result.get('op') == 'predict':
            self._answer = result
            self.predicted = {
                'body': self._body,
                'model': result['model'],
                'request_id': result['request_id'],
                'metadata': None,
                'result': None,
            }

    def taken(self, attachment: int, value: Any) -> None:
        if attachment == self._call and isinstance(value, bytes):
            self._body = json.loads(
                value, parse_int=_number, parse_float=_number
            )
            if self._body.get('metadata', {}).get('request_id') != (
                self._request_id
            ):
                raise RunnerFailure(
                    'the model call does not carry the request id predict '
                    'was started with'
                )
        answer = self._answer
        if answer is not None and self.predicted is not None:
            if attachment == answer['metadata'] and isinstance(value, bytes):
                self.predicted['metadata'] = json.loads(value)
            if attachment == answer['result'] and not isinstance(value, bytes):
                self.predicted['result'] = _result_rows(value)
        if attachment == self._description and isinstance(value, bytes):
            self.described = json.loads(value)
        state = self._state
        if state is None or self.materialized is None:
            return
        if attachment == state['payload'] and isinstance(value, bytes):
            self.materialized['payload'] = json.loads(
                value, parse_int=_number, parse_float=_number
            )
        if attachment == state['query'] and isinstance(value, bytes):
            self.materialized['query'] = json.loads(value)

    def core(self) -> dict[str, Any]:
        return {
            'inputs': self.inputs,
            'fetches': self.fetches,
            'accepted': self.accepted,
            'described': self.described,
            'materialized': self.materialized,
            'predicted': self.predicted,
            'error': self.error,
        }


def _result_rows(stream: Any) -> dict[str, Any]:
    r"""An Arrow result the core gave, as the pure-Rust host records one:
    its columns' names and types, and its rows in order, each cell an
    integer's digits (``i64``), a float's IEEE-754 bits (``f64_bits``, 16
    hex digits, so no text form of a float is compared) or text
    (``utf8``)."""
    import pyarrow as pa

    table = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
    kinds = {pa.int64(): 'int64', pa.float64(): 'float64', pa.utf8(): 'utf8'}
    columns = [
        {'name': field.name, 'type': kinds.get(field.type, 'other')}
        for field in table.schema
    ]

    def cell(kind: str, value: Any) -> Any:
        if value is None:
            return None
        if kind == 'int64':
            return {'i64': str(value)}
        if kind == 'float64':
            return {'f64_bits': struct.pack('>d', value).hex()}
        if kind == 'utf8':
            return {'utf8': value}
        raise RunnerFailure('a result column of a type this runner does not record')

    rows = [
        [cell(column['type'], value) for column, value in zip(columns, row, strict=True)]
        for row in zip(*(table.column(i).to_pylist() for i in range(table.num_columns)), strict=True)
    ]
    return {'columns': columns, 'rows': rows}


#: A pyarrow integer type's contract scalar tag.
_INTEGER_TAGS = {
    'int8': 'i8',
    'int16': 'i16',
    'int32': 'i32',
    'int64': 'i64',
    'uint8': 'u8',
    'uint16': 'u16',
    'uint32': 'u32',
    'uint64': 'u64',
}


def _scalars(table: Any) -> list[dict[str, Any]]:
    r"""The one column of an Arrow table the host sent, each value as the
    contract's tagged scalar: an integer by its width, text as ``utf8``."""
    from sdm_client._rust import codec

    column = table.column(0)
    kind = str(column.type)
    values = column.to_pylist()
    if kind in _INTEGER_TAGS:
        return [codec.integer(_INTEGER_TAGS[kind], v) for v in values]
    if kind in ('string', 'large_string'):
        return [{'t': 'utf8', 'v': v} for v in values]
    raise RunnerFailure('the entities sent are neither integers nor text')


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
        if message.get('message') == 'step':
            self._capture.stepped(message)
            if message.get('step') == 'done':
                self._capture.done(message['result'])
        return control, sent

    def take_attachment(self, attachment: int) -> Any:
        value = self._handle.take_attachment(attachment)
        self._capture.taken(attachment, value)
        return value

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


def _sibling(name: str) -> Any:
    r"""A module beside this file, loaded by path: ``python -I`` puts no
    script directory on ``sys.path``."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    if spec is None or spec.loader is None:
        raise RunnerFailure(f'{name}.py is missing beside the runner')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- the driver capture (Q2) ---------------------------------------------------------

#: The methods that read a statement's result, by what they give back.
_ROWS = ('fetchall', 'fetchmany')
_ROW = ('fetchone',)
_ARROW = ('to_arrow_table', 'fetch_arrow_table')
#: Results the capture cannot record without consuming them or changing
#: what the caller gets; reading one makes the capture incomplete, which
#: fails the case rather than drop a result.
_UNRECORDED = (
    # A connection's arrow() gives a RecordBatchReader, which the capture
    # cannot read without consuming what the caller reads.
    'arrow',
    'fetch_record_batch',
    'to_arrow_reader',
    'fetchdf',
    'fetch_df',
    'df',
    'fetch_df_chunk',
    'fetchnumpy',
    'pl',
    'torch',
    'tf',
)
#: Methods that start a statement the capture does not see as one (a lazy
#: relation runs when it is read, outside the capture).
_UNCAPTURED = (
    'executemany',
    'sql',
    'query',
    'from_query',
    'table',
    'view',
    'values',
    'read_csv',
    'read_parquet',
    'read_json',
    'from_df',
    'from_arrow',
    'append',
    'load_extension',
    'install_extension',
)


def _typed(value: Any) -> dict[str, Any]:
    r"""A parameter or a fetched value as its type and its text: the
    capture's own record, which no value passes through to change."""
    kind = type(value)
    return {
        'type': f'{kind.__module__}.{kind.__qualname__}',
        'value': repr(value),
    }


def _parameters(parameters: Any) -> list[dict[str, Any]] | None:
    r"""A statement's parameters as recorded: in order, each typed, and a
    named one with its name."""
    if parameters is None:
        return None
    if isinstance(parameters, Mapping):
        return [
            {'name': str(name), **_typed(value)}
            for name, value in parameters.items()
        ]
    return [_typed(value) for value in parameters]


def _rows_digest(rows: list[Any]) -> str:
    text = json.dumps(
        [[_typed(value) for value in row] for row in rows], sort_keys=True
    )
    return digest(text.encode())


def _arrow_digest(table: Any) -> str:
    r"""The SHA-256 of an Arrow table's contents, its batch boundaries left
    out (decision H9): the IPC stream of one combined copy."""
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    combined = table.combine_chunks()
    with pa.ipc.new_stream(sink, combined.schema) as writer:
        writer.write_table(combined)
    return digest(sink.getvalue().to_pybytes())


class DriverLog:
    r"""What the capture recorded of one case: each statement, in the order
    the driver ran them, with its handle (0 the connection the source was
    given, then each derived cursor in the order made), typed parameters
    and every result read from it; each relation bound by name; and what
    it could not record."""

    def __init__(self) -> None:
        self.statements: list[dict[str, Any]] = []
        self.bindings: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.incomplete: list[str] = []
        self.passed: list[str] = []
        self.handles = 0

    def handle(self) -> int:
        number = self.handles
        self.handles += 1
        return number

    def record(self) -> dict[str, Any]:
        return {
            'statements': self.statements,
            'bindings': self.bindings,
            'events': self.events,
            'incomplete': sorted(set(self.incomplete)),
            'passed_through': sorted(set(self.passed)),
            'cursors': self.handles - 1,
        }


class _Captured:
    r"""A DuckDB connection or derived cursor, as the capture hands it to
    the source: every call goes to the driver unchanged, with the same
    arguments, once, and its answer comes back unchanged; what was run and
    read is recorded on the side.

    It passes the reference's dispatch as the connection does (design,
    "Driver capture"): its class's module is under ``duckdb``, which
    ``sdm_connectors.source._backend_of`` reads over the MRO; a derived
    cursor is the same class, so ``reader._query_handle``'s
    ``isinstance(cursor, type(connection))`` holds as it does for the
    driver; and ``__class__`` answers the driver's class, so
    ``Graph.from_duckdb``'s ``isinstance(connection, DuckDBPyConnection)``
    holds too.
    """

    __slots__ = ('_driver', '_log', '_handle', '_statement')

    def __init__(self, driver: Any, log: DriverLog) -> None:
        object.__setattr__(self, '_driver', driver)
        object.__setattr__(self, '_log', log)
        object.__setattr__(self, '_handle', log.handle())
        object.__setattr__(self, '_statement', None)

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        return type(self._driver)

    def _same(self, answer: Any) -> Any:
        # The driver gives itself back from execute and register; the
        # caller gets this proxy in its place, so what it reads next is
        # recorded too.
        return self if answer is self._driver else answer

    def cursor(self) -> Any:
        derived = self._driver.cursor()
        self._log.events.append({'event': 'cursor', 'handle': self._handle})
        return _Captured(derived, self._log)

    def execute(self, *arguments: Any, **keywords: Any) -> Any:
        # Bound as the driver binds them: execute(query, parameters=None).
        query = keywords.get('query', arguments[0] if arguments else None)
        parameters = keywords.get(
            'parameters', arguments[1] if len(arguments) > 1 else None
        )
        statement: dict[str, Any] = {
            'handle': self._handle,
            'sql': query if isinstance(query, str) else None,
            'parameters': _parameters(parameters),
            'results': [],
            'error': None,
        }
        if not isinstance(query, str):
            self._log.incomplete.append('execute:not_text')
        self._log.statements.append(statement)
        object.__setattr__(self, '_statement', statement)
        try:
            answer = self._driver.execute(*arguments, **keywords)
        except BaseException as error:
            statement['error'] = type(error).__name__
            raise
        return self._same(answer)

    def _read(self, method: str, *arguments: Any, **keywords: Any) -> Any:
        answer = getattr(self._driver, method)(*arguments, **keywords)
        statement = self._statement
        if statement is None:
            self._log.incomplete.append(f'{method}:no_statement')
            return answer
        if method in _ROWS:
            rows = list(answer)
            result = {'method': method, 'rows': len(rows)}
            result['digest'] = _rows_digest(rows)
        elif method in _ROW:
            result = {'method': method, 'rows': 0 if answer is None else 1}
            result['digest'] = _rows_digest([] if answer is None else [answer])
        else:
            result = {
                'method': method,
                'rows': answer.num_rows,
                'schema': [
                    {'name': field.name, 'type': str(field.type)}
                    for field in answer.schema
                ],
                'digest': _arrow_digest(answer),
            }
        statement['results'].append(result)
        return answer

    def fetchall(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('fetchall', *arguments, **keywords)

    def fetchmany(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('fetchmany', *arguments, **keywords)

    def fetchone(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('fetchone', *arguments, **keywords)

    def to_arrow_table(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('to_arrow_table', *arguments, **keywords)

    def fetch_arrow_table(self, *arguments: Any, **keywords: Any) -> Any:
        return self._read('fetch_arrow_table', *arguments, **keywords)

    def register(self, name: Any, value: Any) -> Any:
        binding: dict[str, Any] = {
            'handle': self._handle,
            'name': name,
            'type': f'{type(value).__module__}.{type(value).__qualname__}',
        }
        if type(value).__module__.startswith('pyarrow') and hasattr(
            value, 'combine_chunks'
        ):
            binding['rows'] = value.num_rows
            binding['schema'] = [
                {'name': field.name, 'type': str(field.type)}
                for field in value.schema
            ]
            binding['digest'] = _arrow_digest(value)
        else:
            self._log.incomplete.append('register:not_arrow')
        self._log.bindings.append(binding)
        return self._same(self._driver.register(name, value))

    def unregister(self, name: Any) -> Any:
        self._log.events.append(
            {'event': 'unregister', 'handle': self._handle, 'name': name}
        )
        return self._same(self._driver.unregister(name))

    def _event(self, method: str, *arguments: Any, **keywords: Any) -> Any:
        self._log.events.append({'event': method, 'handle': self._handle})
        return self._same(getattr(self._driver, method)(*arguments, **keywords))

    def close(self) -> Any:
        return self._event('close')

    def commit(self) -> Any:
        return self._event('commit')

    def rollback(self) -> Any:
        return self._event('rollback')

    def begin(self) -> Any:
        return self._event('begin')

    def __enter__(self) -> Any:
        self._driver.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        self._log.events.append({'event': 'exit', 'handle': self._handle})
        return self._driver.__exit__(*exc)

    def __repr__(self) -> str:
        return repr(self._driver)

    def __getattr__(self, name: str) -> Any:
        if name in _UNRECORDED or name in _UNCAPTURED:
            self._log.incomplete.append(name)
        else:
            self._log.passed.append(name)
        return getattr(self._driver, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._driver, name, value)


# Under the driver's own package, which the reference's backend dispatch
# reads (sdm_connectors.source._DRIVER_MODULES); a test-only name no import
# resolves.
_Captured.__module__ = 'duckdb.sdm_capture'


def capture(connection: Any, log: DriverLog) -> Any:
    r"""``connection`` as the capture hands it to a source."""
    return _Captured(connection, log)


def _logged_texts(
    log: DriverLog | None, logged: list[str]
) -> tuple[int, list[int]]:
    r"""Each captured statement's text against the text DuckDB logged for
    it, in order: how many were compared, and the positions that differ.
    A ``PRAGMA`` is not compared, since DuckDB logs it as the table
    function it runs; binding names are the same in both, being one run's.
    With no capture, or counts that differ (which completeness reports),
    nothing is compared."""
    ran = [text for text in logged if text]
    if log is None or len(ran) != len(log.statements):
        return 0, []
    compared, differing = 0, []
    for position, (statement, text) in enumerate(
        zip(log.statements, ran, strict=True)
    ):
        sql = statement['sql']
        if sql is None or sql.lstrip().upper().startswith('PRAGMA'):
            continue
        compared += 1
        if sql != text:
            differing.append(position)
    return compared, differing


def enable_query_log(connection: Any) -> None:
    r"""Turns on DuckDB's own query log, which every connection and
    derived cursor of the database writes to, after the case's setup."""
    connection.execute("CALL enable_logging('QueryLog')")


def query_log(connection: Any) -> list[str]:
    r"""Each query DuckDB logged since :func:`enable_query_log`, in order:
    a statement's text, or ``''`` for a relation bound through the API
    (``register``)."""
    rows = connection.execute(
        'SELECT message FROM duckdb_logs '
        "WHERE type = 'QueryLog' ORDER BY context_id"
    ).fetchall()
    return [row[0] for row in rows]


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


def _number(token: str) -> dict[str, str]:
    return {'number': token}


def _payload(body: bytes) -> dict[str, Any]:
    r"""A request body the endpoint received: its SHA-256, and its parsed
    form with every number as the decimal text sent, so no value passes
    through a float."""
    try:
        parsed = json.loads(body, parse_int=_number, parse_float=_number)
    except ValueError:
        parsed = None
    return {'sha256': digest(body), 'parsed': parsed}


def _schema_projection(schema: Any) -> dict[str, Any]:
    r"""The public ``Schema``: each table's columns with their dtype and
    stype, its key, time column and duplicate-key count, and its
    relationships, each as source table and column, then target table and
    column."""
    tables = []
    for name, table in schema.tables.items():
        key = table.primary_key
        tables.append(
            {
                'name': name,
                'columns': [
                    {'name': column, 'dtype': c.dtype, 'stype': c.stype}
                    for column, c in table.columns.items()
                ],
                'primary_key': (
                    key if key is None or isinstance(key, str) else list(key)
                ),
                'time_column': table.time_column,
                'duplicate_keys': table.duplicate_keys,
            }
        )
    links = [
        [
            link.source_table,
            link.source_column,
            link.target_table,
            link.target_column,
        ]
        for link in schema.relationships
    ]
    return {'tables': tables, 'links': links}


def _result_projection(result: Any) -> dict[str, Any]:
    r"""The public result, through ``to_pandas()``: its columns with their
    dtypes, and its rows as typed values with their digest."""
    frame = result.to_pandas()
    rows = [list(row) for row in frame.itertuples(index=False, name=None)]
    return {
        'columns': [
            {'name': str(name), 'dtype': str(dtype)}
            for name, dtype in frame.dtypes.items()
        ],
        'rows': len(rows),
        'digest': _rows_digest(rows),
    }


def _call(
    case: dict[str, Any],
    bases: dict[str, list[str]],
    driver_capture: bool = True,
) -> tuple[dict[str, Any], list[str], dict[str, Any], dict[str, Any]]:
    r"""Makes the case's public calls, with the driver capture on or off:
    its outcome, warning classes, driver record and public projection."""
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
    from sdm_connectors import DuckDBSource

    serve = _sibling('scripted_endpoint').serve

    given = case['input']
    # The call's arguments are the harness's to build: an input it cannot
    # build is a runner failure, never the case's outcome.
    try:
        arguments = _materialize_arguments(given['materialize'])
    except Exception:  # noqa: BLE001 - building the input is not the call
        raise RunnerFailure(
            f'{case["id"]}: its materialize arguments could not be built'
        ) from None
    connection = _connection(case, bases)
    # Setup is everything run before the connection reaches the source; the
    # query log and the capture start after it.
    enable_query_log(connection)
    log = DriverLog() if driver_capture else None
    given_to_source = (
        capture(connection, log) if log is not None else connection
    )
    public: dict[str, Any] = {'schema': None, 'payloads': [], 'result': None}
    stage = 'from_source'
    with (
        serve() as (url, received),
        warnings.catch_warnings(record=True) as shown,
    ):
        warnings.simplefilter('always')
        try:
            schema = Schema.from_source(
                DuckDBSource(given_to_source), **given.get('from_source', {})
            )
            public['schema'] = _schema_projection(schema)
            if case['until'] != 'from_source':
                stage = 'materialize'
                request = PredictiveQuery(given['pql'], schema).materialize(
                    **arguments
                )
            if case['until'] == 'predict':
                stage = 'predict'
                with SDMClient(url) as client:
                    result = client.predict(request, model=KumoRelational())
                public['result'] = _result_projection(result)
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
            try:
                logged = query_log(connection)
            finally:
                connection.close()
        public['payloads'] = [
            {'method': method, 'path': path, **_payload(body)}
            for method, path, body in received.requests
        ]
    compared, differing = _logged_texts(log, logged)
    driver = {
        'capture': log.record() if log is not None else None,
        'query_log': {
            'statements': sum(1 for text in logged if text),
            'bindings': sum(1 for text in logged if not text),
            'compared': compared,
            'differing': differing,
        },
    }
    return (
        outcome,
        [type(warning.message).__name__ for warning in shown],
        driver,
        public,
    )


# --- the run ---------------------------------------------------------------------


def _native_sha256() -> str | None:
    module = sys.modules.get('sdm_client._native')
    location = getattr(module, '__file__', None)
    if module is None or location is None:
        return None
    return digest(Path(location).read_bytes())


def _require_reference_client(reference: Path) -> None:
    r"""``sdm_client`` is the reference checkout's own."""
    import sdm_client

    location = Path(sdm_client.__file__).resolve()
    if not location.is_relative_to(reference.resolve()):
        raise RunnerFailure("sdm_client is not the reference checkout's")


def run(
    mode: str,
    cases: list[tuple[dict[str, Any], str]],
    reference: Path | None,
    driver_capture: bool = True,
) -> list[dict[str, Any]]:
    r"""Every case's observation, or :class:`RunnerFailure`."""
    os.environ.pop('SDM_CORE_ROUTES', None)
    guard = _Guard(reference) if reference is not None else None
    if guard is not None:
        guard.check('before the first case')
        _require_reference_client(reference)  # type: ignore[arg-type]
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
            outcome, shown, driver, public = _call(case, bases, driver_capture)
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
                driver,
                public,
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
    for (
        case,
        case_digest,
        outcome,
        shown,
        capture,
        loaded,
        driver,
        public,
    ) in recorded:
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
            'driver': driver,
            'public': public,
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
    parser.add_argument(
        '--inventory',
        action='append',
        default=[],
        metavar='ID',
        help='a case of the Phase 1b inventory, by id, in the case format',
    )
    parser.add_argument(
        '--no-capture',
        action='store_true',
        help='hand the source the driver connection itself (the capture '
        "conservation control); DuckDB's query log is still read",
    )
    parser.add_argument('cases', type=Path, nargs='*')
    arguments = parser.parse_args(argv)
    if (arguments.mode == 'python-ref') != (arguments.reference is not None):
        parser.error('--reference is given in python-ref mode, and only there')
    # A failed run leaves no output, not an earlier run's.
    partial = arguments.out.with_name(arguments.out.name + '.partial')
    arguments.out.unlink(missing_ok=True)
    partial.unlink(missing_ok=True)
    try:
        cases = load_cases(arguments.cases) if arguments.cases else []
        cases.extend(inventory_cases(arguments.inventory))
        if not cases:
            raise RunnerFailure('no cases')
        observations = run(
            arguments.mode,
            cases,
            arguments.reference,
            driver_capture=not arguments.no_capture,
        )
    except RunnerFailure as failure:
        print(f'runner failure: {failure}', file=sys.stderr)
        return 1
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
