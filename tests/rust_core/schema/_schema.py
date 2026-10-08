# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 6.5 acceptance checks share: today's Python schema and its
warnings in the form ``CONTRACT.md`` names, the Rust core's through
``sdm_core._native``, the run loop's host, the corpus replay, and the
comparison.

The interface these rely on is ``CONTRACT.md``, beside this file.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import os
import sys
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRANSCRIPTS = REPO / 'tests' / 'parity' / 'transcripts'
FIXTURES = REPO / 'tests' / 'parity' / 'fixtures'

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / 'tests' / 'parity'))
sys.path.insert(0, str(REPO / 'scripts' / 'parity'))
sys.path.insert(0, str(HERE))

from _rc import json_diff  # noqa: E402

#: The dialect, as Phase 6 names a backend, of each ``Source.backend``.
DIALECT = {
    'databricks': 'databricks',
    'duckdb': 'duckdb',
    'postgres': 'postgres',
    'snowflake': 'snow',
    'sqlite': 'sqlite',
}

#: Modules the Python describer must work without: it describes columns, and
#: never asks today's inference what they are.
INFERENCE_MODULES = ('sdm_client', 'sdm_relational_engine', 'predictive_query')


# --- The Rust core, through the binding --------------------------------------

_NATIVE: dict[str, Any] = {}
#: Set to ``1`` in a process started after the wheel was built and installed.
INSTALLED = 'SDM_P65_WHEEL_INSTALLED'


def native(*names: str) -> Any:
    r"""``sdm_core._native``, freshly built from this checkout and installed
    into this environment, once per process, holding every one of ``names``.

    A build that fails fails every later call the same way, without building
    again.
    """
    if 'error' in _NATIVE:
        raise AssertionError(_NATIVE['error'])
    if 'module' not in _NATIVE:
        try:
            # A process the checks start after building the wheel themselves
            # imports what they installed rather than building it again.
            if os.environ.get(INSTALLED) != '1':
                from _rc import install_local_wheel

                install_local_wheel()
            importlib.invalidate_caches()
            _NATIVE['module'] = importlib.import_module('sdm_core._native')
        except Exception as error:  # noqa: BLE001 - kept for every later call
            _NATIVE['error'] = f'the binding is not usable: {error}'
            raise AssertionError(_NATIVE['error']) from error
    module = _NATIVE['module']
    missing = [name for name in names if not hasattr(module, name)]
    assert not missing, f'sdm_core._native has no {", ".join(missing)}'
    return module


def describer() -> Any:
    r"""``sdm_core.describe``, from the wheel :func:`native` installed."""
    native()
    module = importlib.import_module('sdm_core.describe')
    assert hasattr(module, 'describe_frames'), (
        'sdm_core.describe has no describe_frames'
    )
    return module


def loads(text: Any, what: str) -> Any:
    assert isinstance(text, str), f'{what} gave {type(text).__name__}, not str'
    try:
        return json.loads(text)
    except ValueError as error:
        raise AssertionError(
            f'{what} gave text that is not JSON: {error}'
        ) from error


# --- Today's Python ------------------------------------------------------------


def error_of(error: BaseException) -> dict[str, str]:
    r"""A refusal as ``{"code", "message"}``: the error's ``code`` where it
    has one, otherwise its class name; its ``message`` where it has one,
    otherwise its text."""
    code = getattr(error, 'code', None)
    if not isinstance(code, str) or not code:
        code = type(error).__name__
    message = getattr(error, 'message', None)
    if not isinstance(message, str):
        message = str(error)
    return {'code': code, 'message': message}


def warning_of(caught: warnings.WarningMessage) -> dict[str, str]:
    category = caught.category
    return {
        'category': f'{category.__module__}.{category.__qualname__}',
        'message': str(caught.message),
    }


def _user_warnings(
    caught: list[warnings.WarningMessage],
) -> list[dict[str, str]]:
    return [
        warning_of(w) for w in caught if issubclass(w.category, UserWarning)
    ]


def describe_schema(schema: Any) -> dict[str, Any]:
    r"""A schema as ``CONTRACT.md``'s description: the ``_describe_schema``
    shape of ``tests/parity/_fixture_ops.py``, with key quality, each
    column's record of how it was typed, and ``to_schema_definition``."""
    tables: dict[str, Any] = {}
    for name, table in schema.tables.items():
        key = table.primary_key
        described: dict[str, Any] = {
            'primary_key': list(key) if isinstance(key, tuple) else key,
            'time_column': table.time_column,
            'duplicate_keys': table.duplicate_keys,
            'columns': {
                column: {
                    'dtype': str(spec.dtype),
                    'stype': str(spec.stype),
                    'untyped': bool(spec.untyped),
                    'stype_declared': bool(spec.stype_declared),
                }
                for column, spec in table.columns.items()
            },
        }
        if hasattr(table, 'source_name'):
            described['source_name'] = table.source_name
        tables[name] = described
    return {
        'tables': tables,
        'relationships': [
            [r.source_table, r.source_column, r.target_table, r.target_column]
            for r in schema.relationships
        ],
    }


def describe_definition(definition: Any) -> dict[str, Any]:
    r"""A ``predictive_query`` ``SchemaDefinition`` as JSON."""

    def value(v: Any) -> Any:
        return getattr(v, 'value', v)

    return {
        'tables': {
            name: {
                'cols': [
                    [c.name, value(c.stype), value(c.dtype), c.timestamp_format]
                    for c in table.cols
                ],
                'pkey': table.pkey,
                'time_col': table.time_col,
                'end_time_col': table.end_time_col,
            }
            for name, table in definition.tables.items()
        },
        'col_groups': [
            [[key.table_name, key.col_name] for key in group.columns]
            for group in definition.col_groups
        ],
    }


def python_outcome(
    make: Callable[[], Any], edits: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    r"""What today's Python gives: ``{"schema", "warnings"}`` or ``{"error",
    "warnings"}``.

    ``make`` builds the schema; each edit is then applied as
    ``setattr(schema[table][column], set, value)``. The schema's
    ``to_schema_definition`` is part of its description, with the warnings it
    gives.
    """
    from sdm_client.schema import to_schema_definition

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            schema = make()
            for edit in edits:
                column = schema[edit['table']][edit['column']]
                setattr(column, edit['set'], edit['value'])
            described = describe_schema(schema)
        except Exception as error:  # noqa: BLE001 - every refusal is an outcome
            return {
                'error': error_of(error),
                'warnings': _user_warnings(caught),
            }
    with warnings.catch_warnings(record=True) as defining:
        warnings.simplefilter('always')
        definition = describe_definition(to_schema_definition(schema))
    definition['warnings'] = _user_warnings(defining)
    described['definition'] = definition
    return {'schema': described, 'warnings': _user_warnings(caught)}


def relationships_json(given: Any) -> Any:
    if given is None:
        return None
    return [
        [r.source_table, r.source_column, r.target_table, r.target_column]
        for r in given
    ]


def keys_json(given: Any) -> Any:
    if given is None:
        return None
    return {
        table: list(key) if isinstance(key, (tuple, list)) else key
        for table, key in given.items()
    }


def relationships_of(given: Any) -> Any:
    from sdm_client import Relationship

    if given is None:
        return None
    return [Relationship(*link) for link in given]


def keys_of(given: Any) -> Any:
    if given is None:
        return None
    return {
        table: tuple(key) if isinstance(key, list) else key
        for table, key in given.items()
    }


# --- The Rust core's answer -------------------------------------------------------


def outcome_of(pair: Any, what: str) -> dict[str, Any]:
    r"""``(schema_json, warnings_json)`` as an outcome like
    :func:`python_outcome`'s."""
    assert isinstance(pair, tuple) and len(pair) == 2, (
        f'{what} gave {pair!r}, not a 2-tuple'
    )
    schema = loads(pair[0], what)
    found = loads(pair[1], what)
    assert isinstance(found, list), f'{what}: the warnings are not a list'
    if isinstance(schema, dict) and set(schema) == {'error'}:
        return {'error': schema['error'], 'warnings': found}
    return {'schema': schema, 'warnings': found}


def options_json(
    primary_keys: Any = None,
    time_columns: Any = None,
    relationships: Any = None,
    edits: Sequence[Mapping[str, Any]] = (),
) -> str:
    return json.dumps(
        {
            'primary_keys': keys_json(primary_keys),
            'time_columns': None
            if time_columns is None
            else dict(time_columns),
            'relationships': relationships_json(relationships),
            'edits': [dict(edit) for edit in edits],
        }
    )


def rust_from_data(
    frames: Mapping[str, Any],
    primary_keys: Any = None,
    time_columns: Any = None,
    relationships: Any = None,
    edits: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    r"""What the Rust core gives for ``Schema.from_data``, through the
    describer and ``schema_from_data``."""
    module = native('schema_from_data')
    tables, tags = describer().describe_frames(dict(frames))
    assert isinstance(tables, dict), 'describe_frames gave no dict of tables'
    assert list(tables) == list(frames), (
        f'describe_frames gave the tables {list(tables)}, not {list(frames)}'
    )
    loads(tags, 'describe_frames')
    got = module.schema_from_data(
        tables,
        tags,
        options_json(primary_keys, time_columns, relationships, edits),
    )
    return outcome_of(got, 'schema_from_data')


def _order(described: Any) -> Any:
    r"""The order of a description's tables and of each table's columns."""
    if not isinstance(described, dict):
        return None
    tables = described.get('tables')
    definition = described.get('definition')
    return {
        'tables': [
            [name, list((table or {}).get('columns') or {})]
            for name, table in (tables or {}).items()
        ]
        if isinstance(tables, dict)
        else None,
        'definition': list((definition or {}).get('tables') or {})
        if isinstance(definition, dict)
        else None,
    }


def difference(want: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""How the Rust outcome differs from Python's: by value and JSON type,
    and in the order of tables and columns."""
    found = json_diff(json.loads(json.dumps(want)), json.loads(json.dumps(got)))
    if found:
        return found
    if 'schema' in want and 'schema' in got:
        a, b = _order(want['schema']), _order(got['schema'])
        if a != b:
            return f'tables or columns in another order: {b} where {a}'
    return None


# --- The run loop ----------------------------------------------------------------


def outcome_error(error: BaseException) -> Any:
    r"""What the host submits for a statement that raised."""
    from sdm_core._native import Outcome

    sqlstate = getattr(error, 'sqlstate', None)
    vendor = getattr(error, 'errno', None)
    return Outcome.error(
        sqlstate if isinstance(sqlstate, str) else None,
        vendor
        if isinstance(vendor, int) and not isinstance(vendor, bool)
        else None,
        str(error),
    )


@dataclass
class Driven:
    r"""A run driven to its end: its value, and every query it asked."""

    value: Any
    queries: list[tuple[str, Any]] = field(default_factory=list)


def drive(run: Any, source: Any, limit: int = 10_000) -> Driven:
    r"""Drive ``run`` to its end, running each query as the core's host does:
    ``source._run(query.sql, query.params or None)``."""
    from sdm_core._native import Outcome

    asked: list[tuple[str, Any]] = []
    for _ in range(limit):
        step = run.next()
        if step.kind == 'done':
            return Driven(step.value, asked)
        assert step.kind == 'run_queries', f'a step of kind {step.kind!r}'
        assert step.queries, 'a run_queries step with no query'
        for query in step.queries:
            params = list(query.params) if query.params else None
            asked.append((query.sql, params))
            try:
                table = source._run(query.sql, params)
            except Exception as error:  # noqa: BLE001 - the run decides
                outcome = outcome_error(error)
            else:
                outcome = Outcome.data(table)
            run.submit(query.id, outcome)
    raise AssertionError(f'the run did not end within {limit} steps')


def source_json(source: Any) -> str:
    r"""What ``schema_from_source`` is told about the source: its namespace
    and, for a semantic model, its name and trust."""
    model = None
    trust = False
    if hasattr(source, 'trust_expressions') and hasattr(source, 'source'):
        model = source.name
        trust = bool(source.trust_expressions)
    return json.dumps(
        {
            'namespace': dict(source.namespace),
            'semantic_model': model,
            'trust_expressions': trust,
        }
    )


def from_source_args_json(args: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            'tables': None
            if args.get('tables') is None
            else dict(args['tables']),
            'primary_keys': keys_json(args.get('primary_keys')),
            'time_columns': None
            if args.get('time_columns') is None
            else dict(args['time_columns']),
            'relationships': relationships_json(args.get('relationships')),
            'check_keys': bool(args.get('check_keys', True)),
        }
    )


def rust_from_source(
    source: Any, args: Mapping[str, Any]
) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    r"""What the Rust core gives for ``Schema.from_source(source, **args)``,
    with every query answered through ``source``; and the queries asked."""
    module = native('schema_from_source')
    run = module.schema_from_source(
        DIALECT[source.backend],
        source_json(source),
        from_source_args_json(args),
    )
    driven = drive(run, source)
    return outcome_of(driven.value, 'schema_from_source'), driven.queries


# --- The corpus -------------------------------------------------------------------


@dataclass
class SchemaCall:
    r"""One ``Schema.from_data`` or ``Schema.from_source`` a scenario made."""

    scenario: str
    kind: str
    args: dict[str, Any]
    python: dict[str, Any]
    #: For ``from_source``: the recorded steps it took, ``[start, end)``.
    steps: tuple[int, int] | None = None
    source: Any = None


def transcript_paths() -> list[Path]:
    return sorted(TRANSCRIPTS.glob('**/transcript.json'))


@contextlib.contextmanager
def _capturing(
    tape: Any, calls: list[SchemaCall], scenario: str
) -> Iterator[None]:
    r"""Wrap ``Schema.from_data`` and ``Schema.from_source`` so each call the
    scenario makes, from any depth of the public API, is kept with what it
    gave."""
    from sdm_client import Schema

    originals = {
        name: vars(Schema)[name] for name in ('from_data', 'from_source')
    }
    depth = [0]

    def wrap(kind: str) -> Any:
        function = originals[kind].__func__

        @functools.wraps(function)
        def captured(cls: Any, *args: Any, **kwargs: Any) -> Any:
            if depth[0]:
                return function(cls, *args, **kwargs)
            depth[0] += 1
            start = getattr(tape, 'position', 0)
            result: list[Any] = []

            def make() -> Any:
                result.append(function(cls, *args, **kwargs))
                return result[0]

            try:
                outcome = python_outcome(make)
            finally:
                depth[0] -= 1
            end = getattr(tape, 'position', 0)
            if kind == 'from_data':
                given = dict(zip(('tables',), args, strict=False), **kwargs)
                call = SchemaCall(scenario, kind, given, outcome)
            else:
                given = dict(
                    zip(('source', 'tables'), args, strict=False), **kwargs
                )
                call = SchemaCall(
                    scenario,
                    kind,
                    {k: v for k, v in given.items() if k != 'source'},
                    outcome,
                    steps=(start, end),
                    source=given['source'],
                )
            calls.append(call)
            if 'error' in outcome:
                raise AssertionError(
                    f'{scenario}: Schema.{kind} raised {outcome["error"]}'
                )
            return result[0]

        return classmethod(captured)

    for kind in originals:
        setattr(Schema, kind, wrap(kind))
    try:
        yield
    finally:
        for kind, original in originals.items():
            setattr(Schema, kind, original)


def scenario_calls(path: Path) -> tuple[dict[str, Any], list[SchemaCall]]:
    r"""Replay a transcript through today's Python, keeping every schema it
    made, what each gave, and which recorded steps each ``from_source``
    took."""
    from _parity.drivers import engine_hooks, facade
    from _parity.runner import IN_MEMORY, _Interpreter, _source
    from _parity.tape import ReplayingTape

    transcript = json.loads(path.read_text())
    tape = ReplayingTape(transcript['steps'], path.parent)
    calls: list[SchemaCall] = []
    with engine_hooks(), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        source = None
        if transcript['backend'] not in IN_MEMORY:
            connection = facade(
                transcript['backend'],
                tape,
                None,
                transcript['data'].get('connection', {}),
            )
            source = _source(transcript, connection, None)
        interpreter = _Interpreter(transcript, source)
        with _capturing(tape, calls, transcript['name']):
            interpreter.run(transcript['calls'])
    return transcript, calls


def matching_tape(
    transcript: dict[str, Any], path: Path, window: tuple[int, int]
) -> Any:
    r"""A tape answering each statement from the recorded ``execute`` steps
    in ``window``, in whatever order the run asks: the step with the same
    SQL, byte for byte, and parameters of the same encoding, each step at
    most once."""
    from _parity.tape import ReplayingTape, Statement, _sql_text
    from _parity.values import canonical, encode

    class MatchingTape(ReplayingTape):
        def __init__(self) -> None:
            super().__init__(transcript['steps'], path.parent)
            self.used: set[int] = set()
            self.unmatched: list[tuple[str, Any]] = []

        def execute(
            self, real: Any, sql: Any, params: Any, keyword: Any
        ) -> Any:
            text = _sql_text(sql)
            ran = canonical(encode(params))
            for index in range(*window):
                step = self.steps[index]
                if (
                    index not in self.used
                    and step.get('op') == 'execute'
                    and step.get('sql') == text
                    and canonical(step.get('params')) == ran
                ):
                    self.used.add(index)
                    self._raise_recorded(step)
                    return Statement(self, index)
            self.unmatched.append((text, params))
            raise AssertionError(
                f'no recorded step of Schema.from_source matches '
                f'{text[:300]!r} with parameters {ran[:200]}'
            )

        def paramstyle(self, real: Any, style: Any) -> None:
            return None

    return MatchingTape()


def rust_scenario_from_source(
    transcript: dict[str, Any], path: Path, call: SchemaCall
) -> tuple[dict[str, Any], Any]:
    r"""The Rust core's ``from_source`` for a recorded call, every query
    answered by the recorded step it matches; and the tape, which knows the
    steps used."""
    from _parity.drivers import engine_hooks

    assert call.steps is not None
    tape = matching_tape(transcript, path, call.steps)
    with engine_hooks():
        source = _matching_source(transcript, tape, call)
        outcome, _ = rust_from_source(source, call.args)
    return outcome, tape


def _matching_source(
    transcript: dict[str, Any], tape: Any, call: SchemaCall
) -> Any:
    r"""The source the call was given, over a connection reading ``tape``."""
    from _parity.drivers import facade
    from _parity.runner import _source

    connection = facade(
        transcript['backend'],
        tape,
        None,
        transcript['data'].get('connection', {}),
    )
    source = _source(transcript, connection, None)
    given = call.source
    if hasattr(given, 'trust_expressions') and hasattr(given, 'source'):
        source = source.semantic_model(
            given.name, trust_expressions=given.trust_expressions
        )
    return source


def python_over_matching_tape(
    transcript: dict[str, Any], path: Path, call: SchemaCall
) -> tuple[dict[str, Any], Any]:
    r"""Today's Python's ``from_source`` again, answered by the host the core
    is answered by: the check that the host serves what the recording
    holds."""
    from _parity.drivers import engine_hooks
    from sdm_client import Schema

    assert call.steps is not None
    tape = matching_tape(transcript, path, call.steps)
    with engine_hooks():
        source = _matching_source(transcript, tape, call)
        outcome = python_outcome(
            lambda: Schema.from_source(source, **call.args)
        )
    return outcome, tape


def rust_scenario_from_data(call: SchemaCall) -> dict[str, Any]:
    args = call.args
    return rust_from_data(
        args['tables'],
        args.get('primary_keys'),
        args.get('time_columns'),
        args.get('relationships'),
    )


def recorded_executes(
    transcript: dict[str, Any], window: tuple[int, int]
) -> list[int]:
    return [
        i
        for i in range(*window)
        if transcript['steps'][i].get('op') == 'execute'
    ]


# --- Fixtures --------------------------------------------------------------------


def fixture_cases(name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURES / f'{name}.json').read_text())['cases']


def fixture_frames(input: Mapping[str, Any]) -> dict[str, Any]:
    from _fixture_ops import build_frame

    return {name: build_frame(f) for name, f in input['tables'].items()}


def python_fixture_from_data(input: Mapping[str, Any]) -> dict[str, Any]:
    from sdm_client import Schema

    return python_outcome(
        lambda: Schema.from_data(
            fixture_frames(input),
            primary_keys=keys_of(input.get('primary_keys')),
            time_columns=input.get('time_columns'),
            relationships=relationships_of(input.get('relationships')),
        ),
        input.get('edits', []),
    )


def rust_fixture_from_data(input: Mapping[str, Any]) -> dict[str, Any]:
    return rust_from_data(
        fixture_frames(input),
        keys_of(input.get('primary_keys')),
        input.get('time_columns'),
        relationships_of(input.get('relationships')),
        input.get('edits', []),
    )


def op_result(text: Any, what: str) -> dict[str, Any]:
    r"""``{"output": X}`` or ``{"error": {"code", "message"}}``, read."""
    found = loads(text, what)
    assert isinstance(found, dict) and (
        set(found) == {'output'}
        or (
            set(found) == {'error'}
            and isinstance(found['error'], dict)
            and set(found['error']) == {'code', 'message'}
        )
    ), f'{what} gave {str(found)[:200]}, not {{"output"}} or {{"error"}}'
    return found


def python_op(operation: str, input: Mapping[str, Any]) -> dict[str, Any]:
    r"""What today's Python gives for a fixture operation, in
    :func:`op_result`'s form."""
    from _fixture_ops import FixtureError, run

    try:
        output = run(operation, dict(input))
    except FixtureError as error:
        return {'error': {'code': error.code, 'message': error.message}}
    return {'output': json.loads(json.dumps(output))}


def expected_of(case: Mapping[str, Any]) -> dict[str, Any]:
    if 'expected' in case:
        return {'output': case['expected']}
    return {'error': dict(case['expected_error'])}


def same_as_fixture(want: dict[str, Any], got: dict[str, Any]) -> str | None:
    r"""How ``got`` differs from a fixture's expectation; a fixture error
    without a message matches on its code alone."""
    if 'error' in want and 'error' in got and 'message' not in want['error']:
        if want['error']['code'] != got['error']['code']:
            return (
                f'code {got["error"]["code"]!r}, not {want["error"]["code"]!r}'
            )
        return None
    return difference(want, got)


def rust_dates_op(operation: str, input: Mapping[str, Any]) -> dict[str, Any]:
    module = native('dates_op')
    options = {k: v for k, v in input.items() if k != 'values'}
    got = module.dates_op(
        operation, json.dumps(input.get('values')), json.dumps(options)
    )
    return op_result(got, f'dates_op({operation!r})')


def rust_composite_op(
    operation: str, input: Mapping[str, Any]
) -> dict[str, Any]:
    module = native('composite_op')
    got = module.composite_op(operation, json.dumps(dict(input)))
    return op_result(got, f'composite_op({operation!r})')


def python_contains_timestamp(
    values: Sequence[Any], column: str
) -> dict[str, Any]:
    r"""``metadata_inference.contains_timestamp`` over a text column."""
    import pandas as pd
    from predictive_query.schema.typing import Dtype
    from sdm_relational_engine.rfm.metadata_inference import contains_timestamp

    cells = pd.Series(list(values), dtype=object)
    try:
        return {'output': bool(contains_timestamp(cells, column, Dtype.string))}
    except Exception as error:  # noqa: BLE001 - every refusal is an outcome
        return {'error': error_of(error)}
