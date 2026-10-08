# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A thin Python host of the core (``tests/rust_core/request/CONTRACT.md``,
"The host"): what Phase 9's wrappers will be, in miniature.

The core does the work. This module turns what a caller passes into plain
data (frames through ``sdm_core.describe.describe_frames``, everything else
as tagged JSON), runs each statement a run asks for through the source's
``_run``, sends each call a run asks for through the client's transport,
gives the warnings the core gave, and raises the errors it refused with, as
today's classes.

A handle (a schema, a query, a request, a fitted model) is what the core
gave back: JSON text and the Arrow tables it names, passed back unchanged.
"""

from __future__ import annotations

import ast
import copy
import importlib
import json
import warnings
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from sdm_core import _native
from sdm_core.answers import Answer, HeldValues, result_json
from sdm_core.answers import decode as _decode_value
from sdm_core.answers import encode as _encode_value
from sdm_core.answers import frame as _frame
from sdm_core.describe import describe_frames

# --- Talking to the core ------------------------------------------------------------


def _category(qualified: str) -> type[Warning]:
    r"""The warning class a core warning names."""
    module, _, name = qualified.rpartition('.')
    found = getattr(importlib.import_module(module or 'builtins'), name, None)
    if isinstance(found, type) and issubclass(found, Warning):
        return found
    return UserWarning


def _give(given: Any) -> None:
    r"""Give the warnings a step, an error or a value carried, in the core's
    order."""
    for warning in given or ():
        warnings.warn(
            warning['message'], _category(warning['category']), stacklevel=3
        )


def _error(
    error: BaseException, references: dict[int, Any] | None = None
) -> BaseException:
    r"""What the core refused with, as today's class."""
    qualified = getattr(error, 'error_class', None)
    if not isinstance(qualified, str):
        return error
    module, _, name = qualified.rpartition('.')
    cls = getattr(importlib.import_module(module or 'builtins'), name, None)
    message = getattr(error, 'message', str(error))
    details = _decode_value(
        dict(getattr(error, 'details', None) or {}), references
    )
    code = getattr(error, 'code', None)
    if module == 'sdm_client.errors':
        from sdm_client import errors

        if cls is errors.UnsupportedRequestError:
            return cls(message, details=details)
        if cls is errors.NimRequestError:
            return cls(
                getattr(error, 'status_code', None) or 0,
                code=code,
                message=message,
                details=details,
            )
        if cls is errors.UnknownModelError:
            return cls(
                details.get('model'), list(details.get('known_models', ()))
            )
        if cls is errors.MissingExtraError:
            return cls(details.get('extra'), details.get('package'))
        built = cls(message, code=code, details=details)
        status = getattr(error, 'status_code', None)
        if isinstance(status, int):
            built.status_code = status
        return built
    if cls is KeyError:
        # The core's message is today's `str(error)`, which for a `KeyError`
        # is the repr of what it was raised with.
        try:
            raised = ast.literal_eval(message)
        except (ValueError, SyntaxError):
            raised = message
        return KeyError(raised)
    return (cls or ValueError)(message)


def _raise(
    error: BaseException, references: dict[int, Any] | None = None
) -> None:
    _give(getattr(error, 'warnings', None))
    if isinstance(references, HeldValues) and references.error is not None:
        raise references.error from None
    raise _error(error, references) from None


def _plain(value: Any, references: dict[int, Any] | None = None) -> Any:
    r"""A value a caller passed, as tagged JSON; what has no plain form is
    described by its type, so the core can refuse it as today does."""
    try:
        return _encode_value(value, references=references)
    except TypeError:
        import pandas as pd

        if isinstance(value, pd.Series):
            return {'$object': {'type': 'Series', 'repr': ''}}
        if isinstance(value, Mapping):
            return {'$mapping': type(value).__name__}
        if hasattr(value, '__iter__') and not isinstance(value, str):
            return {'$container': type(value).__name__}
        return {'$object': {'type': type(value).__name__, 'repr': repr(value)}}


def _model(model: Any) -> Any:
    r"""A model, as its id or ``{"$model": CLASS}`` for a handle."""
    from sdm_client.models import Model

    if isinstance(model, Model):
        return {'$model': type(model).__name__}
    return _plain(model)


def _drive(
    run: Any,
    source: Any = None,
    transport: Any = None,
    references: dict[int, Any] | None = None,
) -> Any:
    r"""A run, driven to its end: each statement through ``source._run``,
    each model call through ``transport``; its ``done`` value."""
    try:
        while True:
            step = run.next()
            _give(step.warnings)
            if step.kind == 'done':
                return step.value
            if step.kind == 'run_queries':
                for query in step.queries:
                    run.submit(query.id, _ran(source, query))
                continue
            run.submit(step.id, _called(transport, step, references))
    except Exception as error:  # noqa: BLE001 - re-raised as today's class
        if getattr(error, 'error_class', None) is not None:
            _raise(error, references)
        raise


def _ran(source: Any, query: Any) -> Any:
    try:
        table = source._run(query.sql, query.params or None)
    except Exception as error:  # noqa: BLE001 - handed to the core
        sqlstate = getattr(error, 'sqlstate', None)
        errno = getattr(error, 'errno', None)
        return _native.Outcome.error(
            sqlstate if isinstance(sqlstate, str) else None,
            errno if isinstance(errno, int) else None,
            str(error),
        )
    return _native.Outcome.data(table)


def _called(
    transport: Any, step: Any, references: dict[int, Any] | None = None
) -> Any:
    try:
        if step.endpoint == 'predict':
            answer = transport.predict(
                _decode_value(json.loads(step.body), references)
            )
        elif step.endpoint == 'session_create':
            answer = transport.create_session(
                _decode_value(json.loads(step.body), references)
            )
        elif step.endpoint == 'session_predict':
            answer = transport.session_predict(
                step.session_id,
                _decode_value(json.loads(step.body), references),
            )
        else:
            answer = transport.delete_session(step.session_id)
    except Exception as error:  # noqa: BLE001 - handed to the core
        status = getattr(error, 'status_code', None)
        code = getattr(error, 'code', None)
        message = getattr(error, 'message', None)
        details = getattr(error, 'details', None)
        cls = type(error)
        return _native.Outcome.model_error(
            json.dumps(
                {
                    'class': f'{cls.__module__}.{cls.__qualname__}',
                    'status': status if isinstance(status, int) else None,
                    'code': code if isinstance(code, str) and code else None,
                    'message': message
                    if isinstance(message, str)
                    else str(error),
                    'details': _plain(dict(details))
                    if isinstance(details, Mapping)
                    else {},
                }
            )
        )
    return _native.Outcome.response(json.dumps(answer))


def _call(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except Exception as error:  # noqa: BLE001 - re-raised as today's class
        if getattr(error, 'error_class', None) is not None:
            _raise(error)
        raise


def _owned_call(
    function: Any, state: Any, tables: Any, references: HeldValues
) -> Any:
    if references:
        references.error = None
        try:
            return function(state, tables, references.protocol)
        except Exception as error:
            if getattr(error, 'error_class', None) is not None:
                _raise(error, references)
            raise
    return _call(function, state, tables)


# --- Handles -----------------------------------------------------------------------


class _Frames:
    r"""Arrow tables by name, and their tags."""

    def __init__(
        self, tables: Mapping[str, Any] | None = None, tags: Any = None
    ) -> None:
        self.tables: dict[str, Any] = dict(tables or {})
        self.tags: dict[str, Any] = dict(tags or {})

    def add(self, other: _Frames) -> _Frames:
        return _Frames(
            {**self.tables, **other.tables}, {**self.tags, **other.tags}
        )

    @staticmethod
    def described(frames: Mapping[str, Any]) -> _Frames:
        tables, tags = describe_frames(frames)
        return _Frames(tables, json.loads(tags))


class Schema:
    r"""A schema the core described: what it says, and, over frames, the
    caller's frames, held as today's ``Schema`` holds them and described
    afresh at each call that reads them."""

    def __init__(
        self, state: dict[str, Any], frames: Mapping[str, Any] | None = None
    ) -> None:
        self.state = state
        self.data: dict[str, Any] = dict(frames or {})
        self.source: Any = None

    @property
    def frames(self) -> _Frames:
        r"""The frames as they stand now, named ``schema/NAME``."""
        if not self.data:
            return _Frames()
        described = _Frames.described(self.data)
        return _Frames(
            {f'schema/{n}': t for n, t in described.tables.items()},
            {f'schema/{n}': described.tags[n] for n in described.tables},
        )


class Query:
    r"""A ``PredictiveQuery``: its text, and the schema it is over."""

    def __init__(self, text: str, schema: Schema) -> None:
        self.text = text
        self.schema = schema
        #: The relational model the core keeps between materializes, and the
        #: tables of the graph it built, as the core gave them.
        self.model: Any = None
        self.graph = _Frames()


class RequestHandle:
    r"""A request the core built or gathered."""

    def __init__(
        self,
        state: dict[str, Any],
        frames: _Frames,
        references: dict[int, Any] | None = None,
    ) -> None:
        self.state = _request_mappings(
            state, lambda value: _decode_value(value, references)
        )
        self.frames = frames


def _request_mappings(state: dict[str, Any], convert: Any) -> dict[str, Any]:
    found = {
        **state,
        'task': {
            **state['task'],
            'kind': convert(state['task']['kind']),
            'entity_table': convert(state['task']['entity_table']),
            'target': {
                **state['task']['target'],
                **{
                    key: convert(state['task']['target'][key])
                    for key in ('column_name', 'dtype')
                },
            },
            'options': convert(state['task']['options']),
        },
        'inference': convert(state['inference']),
        'requested_inference': convert(state['requested_inference']),
        **{
            key: convert(state[key])
            for key in ('query', 'entity_table', 'windowed', 'indices')
        },
    }
    if state.get('payloads') is not None:
        found['payloads'] = [convert(payload) for payload in state['payloads']]
    if state.get('parts') is not None:
        found['parts'] = [
            _request_mappings(part, convert) for part in state['parts']
        ]
    return found


def _relationship(link: Any) -> list[str]:
    if isinstance(link, (list, tuple)):
        return [str(part) for part in link]
    return [
        link.source_table,
        link.source_column,
        link.target_table,
        link.target_column,
    ]


def _keys(given: Any) -> Any:
    if given is None:
        return None
    return {
        str(table): key if isinstance(key, str) else list(key)
        for table, key in given.items()
    }


def schema_from_data(frames: Mapping[str, Any], **args: Any) -> Schema:
    r"""``Schema.from_data(frames, **args)``, then ``edits``, each an
    assignment to a column's or a table's attribute as the fixtures write
    one (``{'table', 'column', 'set', 'value'}`` or ``{'table',
    'set_table', 'value'}``)."""
    described = _Frames.described(frames)
    options: dict[str, Any] = {
        'primary_keys': _keys(args.get('primary_keys')),
        'time_columns': args.get('time_columns'),
        'relationships': None
        if args.get('relationships') is None
        else [_relationship(link) for link in args['relationships']],
        'edits': list(args.get('edits') or []),
    }
    found, said = _call(
        _native.data_schema,
        described.tables,
        json.dumps(described.tags),
        json.dumps(options),
    )
    _give(json.loads(said))
    description = json.loads(found)
    refused = description.get('error')
    if refused is not None:
        from sdm_client.errors import StructuredDataError

        code = refused['code']
        if code.replace('_', '').isupper():
            raise StructuredDataError(refused['message'], code=code)
        raise getattr(importlib.import_module('builtins'), code, ValueError)(
            refused['message']
        )
    return Schema(description, frames)


def _column_json(column: Any) -> dict[str, Any]:
    r"""A ``Column`` as a schema's description holds one."""
    stype = column.stype
    return {
        'name': str(column.name),
        'dtype': str(column.dtype),
        'stype': str(getattr(stype, 'value', stype)),
        'untyped': bool(column.untyped),
        'stype_declared': bool(column.stype_declared),
        'before_key': getattr(column, '_stype_before_key', None),
    }


def _described(schema: Schema, name: str) -> dict[str, Any]:
    for table in schema.state['schema']['tables']:
        if table['name'] == name:
            return table
    raise KeyError(name)


def put_column(
    schema: Schema, table: str, column: Any, key: Any = None
) -> None:
    r"""``schema[table].columns[key] = column``, ``key`` by default the
    column's name: the schema says the column from now on, under ``key``,
    and its frame is as it was. A key that is not text is described by its
    ``str`` and its ``repr``, as a frame's label is."""
    columns = _described(schema, table)['columns']
    described = _column_json(column)
    if key is not None:
        described['name'] = str(key)
        if not isinstance(key, str):
            described['name_repr'] = repr(key)
    for at, held in enumerate(columns):
        if (held['name'], held.get('name_repr')) == (
            described['name'],
            described.get('name_repr'),
        ):
            columns[at] = described
            return
    columns.append(described)


def put_relationship(schema: Schema, link: Any) -> None:
    r"""``schema.relationships.append(link)``."""
    schema.state['schema']['relationships'].append(_relationship(link))


def put_table(schema: Schema, name: str, other: Schema) -> None:
    r"""``schema.tables[name] = other.tables[name]``: a table another schema
    over the same source described, which this one has not read."""
    tables = schema.state['schema']['tables']
    tables[:] = [table for table in tables if table['name'] != name]
    tables.append(copy.deepcopy(_described(other, name)))


def drop_table(schema: Schema, name: str) -> None:
    r"""``del schema.tables[name]``: the schema says no more of the table, and
    holds no frame for it, its relationships kept as they were."""
    schema.state['tables'] = [n for n in schema.state['tables'] if n != name]
    described = schema.state['schema']['tables']
    described[:] = [table for table in described if table['name'] != name]
    schema.data.pop(name, None)


def schema_of_tables(
    tables: Mapping[str, Any], relationships: Sequence[Any] = ()
) -> Schema:
    r"""``Schema(tables, relationships)`` of hand-built ``Table``\ s over
    frames: the schema says what each table says, and reads its frame."""
    names = list(tables)
    state = {
        'kind': 'data',
        'tables': names,
        'options': {
            'primary_keys': None,
            'time_columns': None,
            'relationships': None,
            'edits': [],
        },
        'schema': {
            'tables': [
                {
                    'name': name,
                    'source_name': None,
                    'columns': [
                        _column_json(column)
                        for column in tables[name].columns.values()
                    ],
                    'primary_key': _keys({name: tables[name].primary_key})[name]
                    if tables[name].primary_key is not None
                    else None,
                    'time_column': tables[name].time_column,
                    'duplicate_keys': None,
                }
                for name in names
            ],
            'relationships': [_relationship(link) for link in relationships],
        },
    }
    return Schema(state, {name: tables[name].data for name in names})


#: The engine's name for each source's backend.
_DIALECT = {
    'databricks': 'databricks',
    'duckdb': 'duckdb',
    'postgres': 'postgres',
    'snowflake': 'snow',
    'sqlite': 'sqlite',
}


def _source_json(source: Any) -> str:
    r"""What the core is told about a source: its namespace and, for a
    semantic model, its name and trust."""
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


def schema_from_source(source: Any, **args: Any) -> Schema:
    r"""``Schema.from_source(source, **args)``."""
    relationships = args.get('relationships')
    given = {
        'tables': None if args.get('tables') is None else dict(args['tables']),
        'primary_keys': _keys(args.get('primary_keys')),
        'time_columns': None
        if args.get('time_columns') is None
        else dict(args['time_columns']),
        'relationships': None
        if relationships is None
        else [_relationship(link) for link in relationships],
        'check_keys': bool(args.get('check_keys', True)),
    }
    run = _call(
        _native.schema_source_run,
        _DIALECT[source.backend],
        _source_json(source),
        json.dumps(given),
    )
    found, said = _drive(run, source)
    _give(json.loads(said))
    description = json.loads(found)
    refused = description.get('error')
    if refused is not None:
        from sdm_client.errors import StructuredDataError

        code = refused['code']
        if code.replace('_', '').isupper():
            raise StructuredDataError(refused['message'], code=code)
        raise getattr(importlib.import_module('builtins'), code, ValueError)(
            refused['message']
        )
    schema = Schema(description)
    schema.source = source
    return schema


def predictive_query(query: Any, data: Any) -> Query:
    r"""``PredictiveQuery(query, data)``."""
    text = _call(_native.query_text, _plain(query))
    schema = data if isinstance(data, Schema) else schema_from_data(dict(data))
    return Query(text, schema)


def _arguments(args: Mapping[str, Any]) -> tuple[dict[str, Any], _Frames]:
    r"""Arguments as tagged JSON, frames among them described and named
    ``arg/NAME``."""
    import pandas as pd

    frames = {
        f'arg/{name}': value
        for name, value in args.items()
        if isinstance(value, pd.DataFrame)
    }
    described = _Frames.described(frames) if frames else _Frames()
    encoded: dict[str, Any] = {}
    for name, value in args.items():
        if isinstance(value, pd.DataFrame):
            encoded[name] = {'$frame': f'arg/{name}'}
        elif name == 'model':
            encoded[name] = _model(value)
        elif isinstance(value, Mapping):
            encoded[name] = {str(k): _plain(v) for k, v in value.items()}
        else:
            encoded[name] = _plain(value)
    return encoded, described


def materialize(query: Query, **args: Any) -> RequestHandle:
    r"""``query.materialize(**args)``: over the schema's frames as they stand,
    and the graph the query's model built at its first relational gather."""
    encoded, described = _arguments(args)
    schema_frames = query.schema.frames
    frames = schema_frames.add(described).add(query.graph)
    state = {
        'schema': query.schema.state,
        'query': query.text,
        'args': encoded,
        'tags': frames.tags,
        'model': query.model,
    }
    run = _call(_native.materialize_run, json.dumps(state), frames.tables)
    answer, tables = _drive(run, getattr(query.schema, 'source', None))
    found = json.loads(answer)
    query.model = found.get('model')
    graph = {n: t for n, t in tables.items() if n.startswith('model/')}
    query.graph = _Frames(graph, {n: found['tags'].get(n, {}) for n in graph})
    refused = found.get('refused')
    if refused is not None:
        # Refused after the query's model was built, which the query keeps.
        raise _error(
            SimpleNamespace(
                error_class=refused['class'],
                code=refused['code'],
                message=refused['message'],
                details=refused['details'],
                status_code=refused['status'],
            )
        ) from None
    gathered = {n: t for n, t in tables.items() if n not in graph}
    return RequestHandle(
        found['request'],
        schema_frames.add(
            _Frames(gathered, {n: found['tags'].get(n, {}) for n in gathered})
        ),
    )


def request(**args: Any) -> RequestHandle:
    r"""``Request(**args)``."""
    import pandas as pd

    given = dict(args)
    schema = given.pop('schema', None)
    related = given.pop('related_tables', None) or {}
    frames = {}
    for half in ('context', 'predict'):
        if isinstance(given.get(half), pd.DataFrame):
            frames[f'arg/{half}'] = given[half]
    for name, value in related.items():
        if isinstance(value, pd.DataFrame):
            frames[f'arg/related/{name}'] = value
    described = _Frames.described(frames)
    encoded: dict[str, Any] = {}
    for name, value in given.items():
        if name in ('context', 'predict') and isinstance(value, pd.DataFrame):
            encoded[name] = {'$frame': f'arg/{name}'}
        elif isinstance(value, Mapping):
            encoded[name] = {str(k): _plain(v) for k, v in value.items()}
        else:
            encoded[name] = _plain(value)
    encoded['related_tables'] = {
        str(name): {'$frame': f'arg/related/{name}'}
        if isinstance(value, pd.DataFrame)
        else _plain(value)
        for name, value in related.items()
    }
    if isinstance(schema, Schema):
        encoded['schema'] = schema.state
        described = described.add(schema.frames)
    state = {'args': encoded, 'tags': described.tags}
    answer, tables = _call(
        _native.request_build, json.dumps(state), described.tables
    )
    found = json.loads(answer)
    _give(found.get('warnings'))
    return RequestHandle(found['request'], _Frames(tables, found['tags']))


def request_materialized(
    *,
    payloads: Sequence[Any],
    schema: Schema,
    query: str | None = None,
    indices: Sequence[Any] = (),
    entity_table: str | None = None,
    windowed: bool = False,
    requested_inference: Mapping[str, Any] | None = None,
) -> RequestHandle:
    r"""``Request.materialized(payloads=..., schema=..., ...)``: a request of
    the envelopes a caller holds, over ``schema``. Complete envelopes use
    the shared tagged codec until the host hands them to the transport."""
    frames = schema.frames
    references = HeldValues()
    state = {
        'payloads': [_plain(payload, references) for payload in payloads],
        'schema': schema.state,
        'query': _plain(query, references),
        'indices': _plain(indices, references),
        'entity_table': _plain(entity_table, references),
        'windowed': _plain(windowed, references),
        'requested_inference': None
        if requested_inference is None
        else _plain(requested_inference, references),
        'tags': frames.tags,
    }
    answer, tables = _owned_call(
        _native.request_materialized,
        json.dumps(state),
        frames.tables,
        references,
    )
    found = json.loads(answer)
    request = RequestHandle(
        found['request'], _Frames(tables, found['tags']), references
    )
    request.state['payloads'] = list(payloads)
    return request


# --- Asking ------------------------------------------------------------------------


class Client:
    r"""An ``SDMClient`` that sends through ``transport``."""

    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self.pinned: list[Fitted] = []

    def predict(
        self,
        request: RequestHandle,
        *,
        model: Any,
        outputs: Sequence[str] | None = None,
    ) -> Answer:
        r"""``SDMClient.predict``."""
        references = HeldValues()
        state = {
            'request': _request_mappings(
                request.state, lambda value: _plain(value, references)
            ),
            'model': _model(model),
            'outputs': _plain(outputs),
            'tags': request.frames.tags,
        }
        run = _owned_call(
            _native.predict_run,
            json.dumps(state),
            request.frames.tables,
            references,
        )
        return Answer(_drive(run, None, self.transport, references))

    def fit(
        self,
        request: RequestHandle,
        *,
        model: Any,
        outputs: Sequence[str] | None = None,
        holdout: Any = None,
    ) -> Fitted:
        r"""``SDMClient.fit``."""
        references = HeldValues()
        state = {
            'request': _request_mappings(
                request.state, lambda value: _plain(value, references)
            ),
            'model': _model(model),
            'outputs': _plain(outputs),
            'holdout': None if holdout is None else _plain(holdout),
            'tags': request.frames.tags,
        }
        run = _owned_call(
            _native.fit_run,
            json.dumps(state),
            request.frames.tables,
            references,
        )
        fitted = Fitted(
            self,
            json.loads(_drive(run, None, self.transport, references)),
            references,
        )
        self.pinned.append(fitted)
        return fitted

    def close(self) -> None:
        r"""``SDMClient.close``: every context it pinned released first."""
        for fitted in list(self.pinned):
            fitted.close()


class Fitted:
    r"""A ``FittedModel``: what ``fit_run`` gave, kept as it gave it."""

    def __init__(
        self,
        client: Client,
        state: dict[str, Any],
        references: dict[int, Any] | None = None,
    ) -> None:
        self.client = client
        self.state = state
        self.closed = False
        self.references = HeldValues(references)

    def predict(self, request: RequestHandle) -> Answer:
        r"""``FittedModel.predict``."""
        references = HeldValues(self.references)
        state = {
            'request': _request_mappings(
                request.state, lambda value: _plain(value, references)
            ),
            'fitted': {**self.state, 'closed': self.closed},
            'tags': request.frames.tags,
        }
        run = _owned_call(
            _native.session_predict_run,
            json.dumps(state),
            request.frames.tables,
            references,
        )
        return Answer(_drive(run, None, self.client.transport, references))

    def close(self) -> None:
        r"""``FittedModel.close``: the one call the host makes on its own."""
        if self.closed:
            return
        self.client.transport.delete_session(self.state['session_id'])
        self.closed = True
        self.references.clear()
        self.references.error = None
        if self in self.client.pinned:
            self.client.pinned.remove(self)


# --- What a result's frame is built from -------------------------------------------


def flatten(result: Any) -> Any:
    r"""The layout ``to_pandas()`` is built from: what
    ``sdm_core._native.flatten`` returned, unchanged."""
    return _native.flatten(result_json(result))


def frame(layout: Any) -> Any:
    r"""``to_pandas()``, built from the layout alone."""
    return _frame(layout)


def merge(results: Sequence[Any]) -> Answer:
    r"""``PredictionResult.merge(results)``."""
    merged = _call(_native.merge, [result_json(result) for result in results])
    return Answer(merged)


def absent(result: Any, requested: Sequence[str]) -> tuple[str, ...]:
    r"""``result.absent(requested)``."""
    return tuple(
        _native.absent(result_json(result), [str(n) for n in requested])
    )
