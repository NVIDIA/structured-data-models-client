# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What questions, requests, clients and answers ask the core.

`PredictiveQuery`, `Request`, `SDMClient`, `FittedModel` and
`PredictionResult` stay the objects a caller holds, reads, copies and pickles,
with the attributes they always had. What they do is the core's: building a
request, gathering one for a query, sending one and reading the answer, and
merging and laying out answers. Each call hands the core what the objects
say, as plain data, and turns what it gives back into those objects again.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sdm_core import _native
from sdm_core.answers import (
    HeldValues,
    ResultValues,
    absence_json,
    decode,
    result_json,
)
from sdm_core.describe import numpy_text_prefix, timestamp_series_units

from sdm_client import _core, _schema_core

if TYPE_CHECKING:
    from sdm_client.client import FittedModel, SDMClient
    from sdm_client.query import PredictiveQuery
    from sdm_client.request import Request
    from sdm_client.result import PredictionResult
    from sdm_client.schema import Schema


# --- Requests -----------------------------------------------------------------------


def _plain_mapping(value: Any) -> dict[str, Any]:
    return {str(k): _core.plain(v) for k, v in dict(value or {}).items()}


def _frame_or_plain(value: Any, name: str, frames: dict[str, Any]) -> Any:
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        frames[name] = value
        return {'$frame': name}
    return _core.plain(value)


def init_request(
    request: Request,
    context: Any,
    predict: Any,
    *,
    target: Any,
    task: Any,
    related_tables: Any,
    schema: Any,
    entity_table: Any,
    inference: Any,
    task_options: Any,
) -> None:
    r"""``Request(context, predict, ...)``: the core checks and builds it;
    the request holds the caller's frames."""
    from sdm_client.request import Split, Target, Task
    from sdm_client.schema import Schema

    frames: dict[str, Any] = {}
    given: dict[str, Any] = {
        'context': _frame_or_plain(context, 'arg/context', frames),
        'predict': _frame_or_plain(predict, 'arg/predict', frames),
        'target': _core.plain(target),
        'task': _core.plain(task),
        'related_tables': {
            str(name): _frame_or_plain(frame, f'arg/related/{name}', frames)
            for name, frame in dict(related_tables or {}).items()
        },
        'entity_table': entity_table,
        'inference': _plain_mapping(inference),
        'task_options': _plain_mapping(task_options),
    }
    described = _core.Frames.described(frames)
    if isinstance(schema, Schema):
        state, schema_frames = _schema_core.schema_state(schema)
        given['schema'] = state
        described = described.add(schema_frames)
    answer, _tables = _core.call(
        _native.request_build,
        json.dumps({'args': given, 'tags': described.tags}),
        described.tables,
    )
    found = json.loads(answer)
    _core.give(found.get('warnings'))
    built = found['request']
    if not isinstance(schema, Schema):
        said = built['schema']
        schema = _schema_core.schema_from(
            Schema, said['schema'], {entity_table: context}
        )
    related = dict(related_tables or {})
    request.schema = schema
    request.task = Task(
        kind=built['task']['kind'],
        target=Target(
            column_name=built['task']['target']['column_name'],
            dtype=built['task']['target']['dtype'],
            table_name=built['task']['target'].get('table_name'),
        ),
        entity_table=built['task'].get('entity_table'),
        options=dict(task_options or {}),
    )
    request._context = Split(instance_table=context, related_tables=related)
    request._predict = Split(instance_table=predict, related_tables=related)
    request.inference = dict(inference or {})
    request.requested_inference = dict(inference or {})
    request.entity_table = entity_table


def _owned_call(function: Any, *args: Any, references: HeldValues) -> Any:
    references.error = None
    try:
        return _core.call(function, *args, references=references)
    except Exception:
        if references.error is not None:
            raise references.error from references.error.__cause__
        raise


def init_materialized(
    cls: type[Request],
    *,
    payloads: Any,
    schema: Schema,
    query: Any,
    indices: Any,
    entity_table: Any,
    windowed: Any,
    requested_inference: Any,
) -> Request:
    references = HeldValues()
    schema_said, frames = _schema_core.schema_state(schema)
    state = {
        'payloads': [
            _core.plain(payload, references=references) for payload in payloads
        ],
        'schema': schema_said,
        'query': _core.plain(query, references=references),
        'indices': _core.plain(indices, references=references),
        'entity_table': _core.plain(entity_table, references=references),
        'windowed': _core.plain(windowed, references=references),
        'requested_inference': _core.plain(
            requested_inference, references=references
        ),
        'tags': frames.tags,
    }
    answer, _tables = _owned_call(
        _native.request_materialized,
        json.dumps(state),
        frames.tables,
        references.protocol,
        references=references,
    )
    found = json.loads(answer)
    _core.give(found.get('warnings'))
    made = request_from(
        found['request'],
        schema=schema,
        frames={},
        indices=None,
        requested=None,
        references=references,
        cls=cls,
    )
    made.indices = indices
    made._payloads = payloads
    return made


def _task_json(
    task: Any, references: HeldValues | None = None
) -> dict[str, Any]:
    target = task.target
    return {
        'kind': _core.plain(task.kind, references=references),
        'target': {
            'column_name': _core.plain(
                target.column_name, references=references
            ),
            'dtype': _core.plain(target.dtype, references=references),
            'table_name': getattr(target, 'table_name', None),
        },
        'entity_table': _core.plain(task.entity_table, references=references),
        'options': _core.plain(task.options, references=references),
    }


def _decomposition_json(found: Any) -> Any:
    if found is None:
        return None
    return {
        'kind': found.kind,
        'parts': [[text, _core.plain(about)] for text, about in found.parts],
        'top': found.top,
    }


def request_state(
    request: Request,
    frames: dict[str, Any] | None = None,
    references: HeldValues | None = None,
) -> tuple[dict[str, Any], _core.Frames]:
    r"""What a request says, as the core reads one, and the frames it
    names: its halves of frames, and its schema's."""
    import pandas as pd

    held: dict[str, Any] = {} if frames is None else frames
    context = getattr(request, '_context', None)
    predict = getattr(request, '_predict', None)
    payloads = getattr(request, '_payloads', None)
    halves = (
        not payloads
        and context is not None
        and predict is not None
        and isinstance(getattr(context, 'instance_table', None), pd.DataFrame)
        and isinstance(getattr(predict, 'instance_table', None), pd.DataFrame)
    )
    if halves:
        held['context'] = getattr(context, 'instance_table', None)
        held['predict'] = getattr(predict, 'instance_table', None)
    parts = getattr(request, '_parts', None)
    schema_said, schema_frames = _schema_core.schema_state(request.schema)
    state = {
        'task': _task_json(request.task, references),
        'schema': schema_said,
        'entity_table': _core.plain(
            getattr(request, 'entity_table', None), references=references
        ),
        'query': _core.plain(
            getattr(request, 'query', None), references=references
        ),
        'inference': _core.plain(
            getattr(request, 'inference', None), references=references
        ),
        'requested_inference': _core.plain(
            getattr(request, 'requested_inference', None), references=references
        ),
        'frames': bool(halves),
        'payloads': None
        if not payloads
        else [
            _core.plain(payload, references=references) for payload in payloads
        ],
        'indices': _core.plain(
            getattr(request, 'indices', ()), references=references
        ),
        'windowed': _core.plain(
            getattr(request, '_windowed', False), references=references
        ),
        'parts': None
        if parts is None
        else [request_state(part, held, references)[0] for part in parts],
        'decomposition': _decomposition_json(
            getattr(request, '_decomposition', None)
        ),
        'asked': getattr(request, '_asked', None),
    }
    if frames is not None:
        return state, _core.Frames()
    described = _core.Frames.described(held).add(schema_frames)
    return state, described


def request_from(
    found: Mapping[str, Any],
    *,
    schema: Schema,
    frames: Mapping[str, Any],
    indices: Any,
    requested: Mapping[str, Any] | None,
    references: HeldValues | None = None,
    cls: type[Request] | None = None,
) -> Request:
    r"""Restore a native request using the owner of its tagged values."""
    from sdm_client.request import Request, Split, Target, Task

    cls = Request if cls is None else cls
    made: Any = cls.__new__(cls)
    task = found['task']
    target = task['target']
    made.schema = schema
    made.task = Task(
        kind=decode(task['kind'], references),
        target=Target(
            column_name=decode(target['column_name'], references),
            dtype=decode(target['dtype'], references),
            table_name=target.get('table_name')
            if found.get('frames')
            else None,
        ),
        entity_table=decode(task.get('entity_table'), references),
        options=decode(task.get('options', {}), references),
    )
    made.inference = decode(found.get('inference', {}), references)
    made.entity_table = decode(found.get('entity_table'), references)
    made.query = decode(found.get('query'), references)
    made.requested_inference = (
        dict(requested)
        if requested is not None
        else decode(found.get('requested_inference', {}), references)
    )
    if found.get('frames'):
        rows = found.get('rows') or {}
        made._context = Split(
            instance_table=_half(
                rows.get('context'), frames, made.entity_table, 'context'
            )
        )
        made._predict = Split(
            instance_table=_half(
                rows.get('predict'), frames, made.entity_table, 'predict'
            )
        )
    else:
        made._windowed = decode(found.get('windowed'), references)
        made._context = None
        made._predict = None
        made.indices = list(indices) if indices is not None else []
        made._payloads = tuple(
            decode(payload, references)
            for payload in found.get('payloads') or ()
        )
    return made


def _half(rows: Any, frames: Mapping[str, Any], entity: Any, half: str) -> Any:
    r"""A half of a request of frames: the rows of the entity table's frame
    it was taken from, or the frame the caller gave for it."""
    if rows is None:
        return frames[f'arg/{half}']
    taken = frames[f'schema/{entity}'].iloc[list(rows)]
    held = frames.get(f'core/{half}')
    if held is not None and list(held.column_names) != list(
        map(str, taken.columns)
    ):
        # The columns the core kept, such as all but the target.
        taken = taken[list(held.column_names)]
    return taken


# --- Questions ----------------------------------------------------------------------

#: Whether the sampler has said, in this process, that it found too few
#: labelled examples: it says so once a process, as today's does.
_coverage_warned = False


def init_query(query: Any, text: Any, data: Any) -> None:
    r"""``PredictiveQuery(query, data)``."""
    from sdm_client.schema import Schema

    _core.call(_native.query_text, _core.plain(text))
    query.text = text
    query.schema = (
        data if isinstance(data, Schema) else Schema.from_data(dict(data))
    )
    #: The relational model the query's first gather built, as the core
    #: gave it, and the tables of its graph.
    query._core_model = None
    query._core_graph = _core.Frames()


def _given(
    args: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    _core.SequenceValue | None,
    _core.SequenceValue | None,
]:
    r"""Arguments as tagged JSON, frames among them named ``arg/NAME``."""
    import pandas as pd

    frames: dict[str, Any] = {}
    encoded: dict[str, Any] = {}
    sequence = None
    candidates = None
    for name, value in args.items():
        if name == 'candidates':
            candidates = _core.SequenceValue(value)
            encoded[name] = {'$container': type(value).__name__}
        elif isinstance(value, pd.DataFrame):
            frames[f'arg/{name}'] = value
            encoded[name] = {'$frame': f'arg/{name}'}
        elif name == 'model':
            encoded[name] = _core.model_arg(value)
        elif name == 'indices':
            if isinstance(value, Sequence) and not isinstance(
                value, (list, tuple, str, bytes)
            ):
                sequence = _core.SequenceValue(value)
                encoded[name] = {'$container': type(value).__name__}
            else:
                encoded[name] = _core.plain(value, max_depth=100)
        elif isinstance(value, Mapping):
            encoded[name] = {str(k): _core.plain(v) for k, v in value.items()}
        else:
            encoded[name] = _core.plain(value)
    return encoded, frames, sequence, candidates


def materialize(query: Any, args: dict[str, Any]) -> Request:
    r"""``query.materialize(**args)``: gathered by the core over the schema's
    frames as they stand, or its source, and the graph the query's model
    built at its first relational gather."""
    schema = query.schema
    encoded, arg_frames, sequence, candidates = _given(args)
    state_said, schema_frames = _schema_core.schema_state(schema)
    graph = getattr(query, '_core_graph', None) or _core.Frames()
    frames = schema_frames.add(_core.Frames.described(arg_frames)).add(graph)
    state = {
        'schema': state_said,
        'query': query.text,
        'args': encoded,
        'tags': frames.tags,
        'model': getattr(query, '_core_model', None),
        'coverage_warned': _coverage_warned,
    }
    units = timestamp_series_units()
    if units is not None:
        state['timestamp_series_units'] = units
    if numpy_text_prefix():
        state['numpy_text_prefix'] = True
    callbacks = (
        (sequence, candidates)
        if candidates is not None
        else ()
        if sequence is None
        else (sequence,)
    )
    run = _core.call(
        _native.materialize_run, json.dumps(state), frames.tables, *callbacks
    )
    try:
        answer, tables = _core.Driver(schema.source).drive(run)
    except Exception:
        for callback in (sequence, candidates):
            if callback is not None and callback.error is not None:
                raise callback.error from callback.error.__cause__
        raise
    found = json.loads(answer)
    _said_coverage(found)
    query._core_model = found.get('model')
    graph_names = [n for n in tables if n.startswith('model/')]
    query._core_graph = _core.Frames(
        {n: tables[n] for n in graph_names},
        {n: found['tags'].get(n, {}) for n in graph_names},
    )
    refused = found.get('refused')
    if refused is not None:
        for callback in (sequence, candidates):
            if callback is not None and callback.error is not None:
                raise callback.error
        # Refused after the query's model was built, which the query keeps.
        made = _core.error_of(_core.refusal(refused))
        raise made from made.__cause__
    by_name = {
        f'schema/{name}': t.data
        for name, t in schema._tables.items()
        if hasattr(t, 'data')
    }
    by_name.update(arg_frames)
    by_name.update(
        {f'core/{n}': tables[n] for n in ('context', 'predict') if n in tables}
    )
    return _gathered(found['request'], query, by_name, args)


def _said_coverage(found: Mapping[str, Any]) -> None:
    global _coverage_warned
    _coverage_warned = bool(found.get('coverage_warned', _coverage_warned))


def _gathered(
    found: Mapping[str, Any],
    query: PredictiveQuery,
    frames: Mapping[str, Any],
    args: Mapping[str, Any],
) -> Request:
    from sdm_client._decomposition import Decomposition

    requested = None
    if not found.get('frames'):
        options = args
        requested = {'inference_config': options.get('inference_config') or {}}
    parts = found.get('parts')
    if parts is None:
        return request_from(
            found,
            schema=query.schema.snapshot(),
            frames=frames,
            indices=args.get('indices'),
            requested=requested,
        )
    gathered = tuple(
        request_from(
            part,
            schema=query.schema.snapshot(),
            frames=frames,
            indices=args.get('indices'),
            requested=None
            if part.get('frames')
            else {'inference_config': args.get('inference_config') or {}},
        )
        for part in parts
    )
    import copy

    whole: Any = copy.copy(gathered[0])
    whole._parts = gathered
    rewritten = found.get('decomposition') or {}
    whole._decomposition = Decomposition(
        rewritten.get('kind', ''),
        tuple(
            (text, decode(about))
            for text, about in rewritten.get('parts') or ()
        ),
        rewritten.get('top'),
    )
    whole._asked = found.get('asked')
    return whole


# --- Asking -------------------------------------------------------------------------


def _result(
    plain: str,
    cls: type | None = None,
    references: ResultValues | None = None,
) -> PredictionResult:
    r"""A result the core gave, as a ``PredictionResult`` (or ``cls``)."""
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.result import PredictionResult

    cls = cls or PredictionResult
    found = json.loads(plain)
    items = []
    for item in found.get('items') or ():
        if references is not None:
            items.append(decode(item['metadata'], references))
            continue
        fields = {
            name: decode(value, references) for name, value in item.items()
        }
        items.append(PredictionItem(**fields))
    constructor = found.get('host_values') or found
    made = cls(
        model=decode(constructor['model'], references),
        request_id=decode(constructor['request_id'], references),
        items=items,
        metadata=decode(found.get('metadata') or {}, references),
        explanation=decode(found.get('explanation'), references),
        outputs=decode(constructor.get('outputs') or (), references),
        unscored_candidates=tuple(
            decode(c, references)
            for c in found.get('unscored_candidates') or ()
        ),
    )
    made._names_its_rows = bool(found.get('names_its_rows', True))
    made._names_its_step = bool(found.get('names_its_step', False))
    return made


def predict(
    client: SDMClient, request: Request, model: Any, outputs: Any
) -> PredictionResult:
    r"""``client.predict(request, model=, outputs=)``."""
    references = HeldValues()
    said, frames = request_state(request, references=references)
    state = {
        'request': said,
        'model': _core.model_arg(model),
        'outputs': _core.plain(outputs),
        'tags': frames.tags,
    }
    run = _owned_call(
        _native.predict_run,
        json.dumps(state),
        frames.tables,
        references.protocol,
        references=references,
    )
    return _result(
        _core.Driver(transport=client._transport, references=references).drive(
            run
        )
    )


class _FitProtocols:
    def __init__(
        self, references: HeldValues, sequence: _core.SequenceValue | None
    ) -> None:
        self.references = references
        self.sequence = sequence

    def __call__(self, operation: str, operands: str) -> str:
        return self.references.protocol(operation, operands)


def fit(
    client: SDMClient,
    request: Request,
    model: Any,
    outputs: Any,
    holdout: Any,
) -> FittedModel:
    r"""``client.fit(request, model=, outputs=, holdout=)``."""
    from sdm_client.client import FittedModel

    sequence = None if holdout is None else _core.SequenceValue(holdout)
    references = HeldValues()
    said, frames = request_state(request, references=references)
    state = {
        'request': said,
        'model': _core.model_arg(model),
        'outputs': _core.plain(outputs),
        'holdout': _core.plain(holdout)
        if holdout is None or isinstance(holdout, (str, bytes))
        else {'$container': type(holdout).__name__},
        'tags': frames.tags,
    }
    run = _owned_call(
        _native.fit_run,
        json.dumps(state),
        frames.tables,
        _FitProtocols(references, sequence),
        references=references,
    )
    try:
        answer = _core.Driver(
            transport=client._transport, references=references
        ).drive(run)
    except Exception:
        if sequence is not None and sequence.error is not None:
            raise sequence.error from sequence.error.__cause__
        raise
    pinned = json.loads(answer)
    fitted = FittedModel.__new__(FittedModel)
    fitted._client = client
    fitted._core = pinned
    fitted._core_references = references
    # What a fitted model held before the core did, read from what it keeps.
    fitted._session_id = str(pinned['session_id'])
    fitted._outputs = tuple(pinned.get('outputs') or ())
    named = pinned.get('named_outputs')
    fitted._named_outputs = None if named is None else tuple(named)
    fitted._pinned_keys = frozenset(
        decode(pinned.get('pinned_keys') or (), references)
    )
    fitted._pinned_over_a_window = bool(pinned.get('pinned_over_a_window'))
    fitted._held_back = int(pinned.get('held_back') or 0)
    fitted._pinned = decode(pinned.get('pinned') or {}, references)
    fitted._closed = False
    client._pinned.add(fitted)
    return fitted


def session_predict(fitted: FittedModel, request: Request) -> PredictionResult:
    r"""``fitted.predict(request)``."""
    references = HeldValues(getattr(fitted, '_core_references', None))
    said, frames = request_state(request, references=references)
    state = {
        'request': said,
        'fitted': {**fitted._core, 'closed': fitted._closed},
        'tags': frames.tags,
    }
    run = _owned_call(
        _native.session_predict_run,
        json.dumps(state),
        frames.tables,
        references.protocol,
        references=references,
    )
    return _result(
        _core.Driver(transport=_Through(fitted), references=references).drive(
            run
        )
    )


class _Through:
    r"""The transport of a fitted model's client, looked up when a call is
    made: a session that is refused before it sends needs none."""

    def __init__(self, fitted: FittedModel) -> None:
        self._fitted = fitted

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fitted._client._transport, name)


# --- Answers ------------------------------------------------------------------------


def to_pandas(result: PredictionResult) -> Any:
    from sdm_core.answers import frame

    references = ResultValues()
    return frame(
        _core.call(
            _native.flatten,
            result_json(result, references=references, layout=True),
            references.infer_column,
            references.column_order,
            references.protocol,
        ),
        references,
    )


def merge(cls: type, parts: Sequence[PredictionResult]) -> PredictionResult:
    r"""``PredictionResult.merge(parts)``: the core merges; one part is
    itself."""
    if len(parts) == 1:
        return parts[0]
    references = ResultValues()
    merged = _core.call(
        _native.merge,
        [result_json(part, references=references) for part in parts],
        references.equal,
        references.truth,
        references.metadata_entry,
        references.protocol,
    )
    return _result(merged, cls, references)


def absent(
    result: PredictionResult, requested: Sequence[str]
) -> tuple[str, ...]:
    references = ResultValues()
    return tuple(
        decode(
            json.loads(
                _core.call(
                    _native.absent,
                    absence_json(result, requested, references),
                    [],
                    references.protocol,
                )
            ),
            references,
        )
    )


def spec(model: Any) -> Any:
    r"""The registered spec ``model`` names, as the core reads it; today's
    refusal of anything else."""
    from sdm_client.models import MODELS

    found = json.loads(
        _core.call(_native.model_spec, json.dumps(_core.model_arg(model)))
    )
    return MODELS[found['model_id']]


# --- What a request's wrapper asks ---------------------------------------------------


def _gathered_state(request: Request, references: HeldValues) -> dict[str, Any]:
    r"""As much of a request as says whether it is gathered: its envelopes,
    and its parts'."""
    parts = getattr(request, '_parts', None)
    return {
        'task': {'kind': '', 'target': {'column_name': '', 'dtype': ''}},
        'schema': {'kind': 'data', 'tables': []},
        'payloads': [
            _core.plain(payload, references=references)
            for payload in getattr(request, '_payloads', None) or ()
        ]
        or None,
        'parts': None
        if parts is None
        else [_gathered_state(part, references) for part in parts],
    }


def refuse_if_gathered(request: Request, half: str) -> None:
    r"""Refuse to replace a half of a request already gathered."""
    references = HeldValues()
    _owned_call(
        _native.request_op,
        'refuse_if_gathered',
        json.dumps(
            {'request': _gathered_state(request, references), 'half': half}
        ),
        {},
        references.protocol,
        references=references,
    )


def entity_table_in(payloads: Sequence[Any]) -> str:
    r"""The table the query asks about, as the envelopes record it."""
    return str(
        _core.call(
            _native.request_op,
            'entity_table_in',
            json.dumps({'payloads': list(payloads)}, default=repr),
        )
    )


def question(request: Request) -> Any:
    r"""``request._question``: what it asks, the core's way of telling one
    context from another included."""
    from sdm_client.request import Question

    references = HeldValues()
    said, frames = request_state(request, references=references)
    found = decode(
        json.loads(
            _owned_call(
                _native.request_op,
                'question',
                json.dumps({'request': said, 'tags': frames.tags}),
                frames.tables,
                references.protocol,
                references=references,
            )
        ),
        references,
    )
    return Question(
        query=found.get('query'),
        kind=found.get('kind', ''),
        target=found.get('target', ''),
        dtype=found.get('dtype', ''),
        entity_table=found.get('entity_table'),
        context=found.get('context'),
    )


def question_json(asked: Any, references: HeldValues | None = None) -> Any:
    r"""A question a session was given, as the core holds one."""
    if asked is None:
        return None
    if isinstance(asked, Mapping):
        return _core.plain(asked, references=references)
    fields = ('query', 'kind', 'target', 'dtype', 'entity_table', 'context')
    if not all(hasattr(asked, name) for name in fields):
        return {'query': str(asked), 'kind': '', 'target': '', 'dtype': ''}
    found = {name: getattr(asked, name) for name in fields}
    if found['context'] is not None:
        found['context'] = str(found['context'])
    return {
        name: _core.plain(value, references=references)
        for name, value in found.items()
    }


def fitted_state(fitted: FittedModel) -> dict[str, Any]:
    r"""What the core reads a session from, of a ``FittedModel`` built with
    its constructor."""
    spec = getattr(getattr(fitted, '_adapter', None), 'spec', None)
    named = fitted._named_outputs
    references = HeldValues()
    fitted._core_references = references
    return {
        'session_id': str(fitted._session_id),
        'model': str(getattr(spec, 'model_id', '') or ''),
        'outputs': [str(name) for name in fitted._outputs],
        'named_outputs': None if named is None else [str(n) for n in named],
        'pinned': _core.plain(fitted._pinned, references=references),
        'pinned_keys': [
            _core.plain(key, references=references)
            for key in fitted._pinned_keys
        ],
        'pinned_over_a_window': bool(fitted._pinned_over_a_window),
        'held_back': int(fitted._held_back),
        'question': question_json(fitted._question, references),
    }


def gather_graph(schema: Schema) -> dict[str, Any]:
    r"""The graph a gather of ``schema``, over a source, samples: its backend,
    its tables as the core keeps them, and its edges. For looking at, as
    tests do; a gather builds its own."""
    said, frames = _schema_core.schema_state(schema)
    run = _core.call(
        _native.gather_graph_run,
        json.dumps({'schema': said, 'tags': frames.tags}),
        frames.tables,
    )
    return json.loads(_core.Driver(schema.source).drive(run))
