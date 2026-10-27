# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ownership around the core's fixed frame data operations.

Every operation consumes useful immutable graph, task or context data and
finishes through the ordinary run boundary.
The host retains actual native dtype and exception owners between those
operations; all selection, graph, label and request policy stays in Rust.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import secrets
from typing import Any, NoReturn

import pandas as pd

from sdm_client._routing import RouteError
from sdm_client._rust import codec, errors, frame_materialize, results
from sdm_client._rust.frame_native import (
    NativeColumn,
    NativeFrameColumns,
    NativeFrameFailure,
    convert_text,
    guard_failure,
)
from sdm_client._rust.run import Finished, run
from sdm_client.errors import StructuredDataError


@dataclass(frozen=True)
class _Product:
    """One closed core value and exactly the owners its typed fields name."""

    kind: str
    data: dict[str, Any]
    owners: dict[int, Any]


class _Bindings:
    """Call-local attachment bindings, with no persistent ids or registry."""

    def __init__(self) -> None:
        self.attachments: dict[int, Any] = {}

    def add(self, owner: Any) -> int:
        key = len(self.attachments) + 2
        self.attachments[key] = owner
        return key

    def json(self, value: Any, *, options: bool = False) -> int:
        return self.add(
            codec.attachment(value, named=options, option_values=options)
        )


def _ids(kind: str, value: dict[str, Any]) -> list[int]:
    """Attachment fields of the fixed contract, never an arbitrary JSON scan."""
    if kind in ('column_meaning', 'integer_column'):
        return []
    if kind == 'source_column':
        return [value['values']]
    if kind == 'question':
        return [value['owner']]
    if kind == 'questions':
        return [] if value['call_values'] is None else [value['call_values']]
    if kind == 'graph':
        return [value['owner']]
    if kind == 'labels':
        return _ids('graph', value['graph']) + (
            [] if value['context'] is None else [value['context']]
        )
    if kind == 'task':
        return _ids('labels', value['labelled']) + [
            value[name]
            for name in ('prediction', 'owner')
            if value[name] is not None
        ]
    if kind == 'plan':
        return _ids('task', value['task']) + [value['sampling_values']]
    if kind == 'references':
        return _ids('plan', value['plan']) + [value['owner']]
    if kind == 'context':
        return (
            _ids('plan', value['plan'])
            + [value['anchors']]
            + [
                table[name]
                for table in value['tables']
                for name in ('rows', 'source_rows')
            ]
            + [link['pairs'] for link in value['links']]
        )
    if kind == 'part':
        return [value['payload']]
    raise codec.CodecError('unknown retained frame value')


def _adopt(finished: Finished, expected: str) -> _Product:
    answer = results.expect(finished.result, results.FramePrepared)
    value = json.loads(finished.take(answer.value))
    if value.get('value') != expected or not isinstance(
        value.get('data'), dict
    ):
        raise codec.CodecError('frame operation returned another value')
    data = value['data']
    return _Product(
        expected,
        data,
        {
            key: finished.take(key)
            for key in dict.fromkeys(_ids(expected, data))
        },
    )


def _bind(product: _Product, into: _Bindings) -> dict[str, Any]:
    """Rebind the explicitly named fields of one closed domain value."""
    bound = {key: into.add(owner) for key, owner in product.owners.items()}

    def graph(value: dict[str, Any]) -> dict[str, Any]:
        return {**value, 'owner': bound[value['owner']]}

    def labels(value: dict[str, Any]) -> dict[str, Any]:
        return {
            **value,
            'graph': graph(value['graph']),
            'context': None
            if value['context'] is None
            else bound[value['context']],
        }

    def task(value: dict[str, Any]) -> dict[str, Any]:
        return {
            **value,
            'labelled': labels(value['labelled']),
            **{
                name: None if value[name] is None else bound[value[name]]
                for name in ('prediction', 'owner')
            },
        }

    def plan(value: dict[str, Any]) -> dict[str, Any]:
        return {
            **value,
            'task': task(value['task']),
            'sampling_values': bound[value['sampling_values']],
        }

    value = product.data
    if product.kind == 'question':
        return {**value, 'owner': bound[value['owner']]}
    if product.kind == 'graph':
        return graph(value)
    if product.kind == 'labels':
        return labels(value)
    if product.kind == 'task':
        return task(value)
    if product.kind == 'plan':
        return plan(value)
    if product.kind == 'references':
        return {
            **value,
            'plan': plan(value['plan']),
            'owner': bound[value['owner']],
        }
    if product.kind == 'context':
        return {
            **value,
            'plan': plan(value['plan']),
            'anchors': bound[value['anchors']],
            'tables': [
                {
                    **table,
                    'rows': bound[table['rows']],
                    'source_rows': bound[table['source_rows']],
                }
                for table in value['tables']
            ],
            'links': [
                {**link, 'pairs': bound[link['pairs']]}
                for link in value['links']
            ],
        }
    if product.kind == 'part':
        return {**value, 'payload': bound[value['payload']]}
    raise codec.CodecError('unknown retained frame value')


def _prepare(
    value: dict[str, Any], bindings: _Bindings, state: dict[str, Any]
) -> Finished:
    attachments = {1: codec.attachment(value), **bindings.attachments}
    try:
        operation: dict[str, Any] = {'op': 'prepare_frames', 'input': 1}
        if state.get('frame_part') is not None:
            operation['part'] = state['frame_part']
        return run(operation, attachments, state=state)
    finally:
        attachments.clear()
        bindings.attachments.clear()


def _options(
    options: dict[str, Any], names: tuple[str, ...], **extra: Any
) -> dict[str, Any]:
    return {
        **extra,
        **{name: options[name] for name in names if name in options},
    }


def _table(stream: Any) -> Any:
    import pyarrow as pa

    return pa.table(stream)


def _string_encoding(
    graph: dict[str, Any], native: dict[str, Any]
) -> str | None:
    for table in graph['native_tables']:
        if table['source'] == native['source']:
            for column in table['columns']:
                if column['name'] == native['column']:
                    return column['strings']
    raise codec.CodecError('core-selected native column has no source binding')


def _native_series(
    values: Any,
    columns: NativeFrameColumns,
    binding: dict[str, Any],
    encoding: str | None,
    name: str,
) -> pd.Series:
    items = values.to_pylist()
    if encoding == 'utf8_surrogatepass':
        items = [
            None if value is None else value.decode('utf-8', 'surrogatepass')
            for value in items
        ]
    elif encoding not in ('utf8', None):
        raise codec.CodecError('unknown native frame string encoding')
    dtype = columns.dtype(binding)
    if type(dtype) in (pd.Float32Dtype, pd.Float64Dtype):
        import numpy as np

        data = np.array(
            [np.nan if item is None else item for item in items],
            dtype=dtype.numpy_dtype,
        )
        missing = np.array([item is None for item in items], dtype=bool)
        return pd.Series(pd.arrays.FloatingArray(data, missing), name=name)
    return pd.Series(items, dtype=dtype, name=name)


def _strings(column: NativeColumn, bindings: _Bindings) -> dict[str, Any]:
    rows, encoding = column.arrow()
    return {'values': bindings.add(rows), 'encoding': encoding}


def _references(
    task: _Product,
    columns: NativeFrameColumns,
    bindings: _Bindings,
    start: int,
    stop: int,
) -> tuple[dict[str, Any], NativeFrameFailure | None]:
    kind = task.data['labelled']['prediction_type']
    if kind['kind'] == 'core_key':
        return {'kind': 'core_keys', 'values': None}, None
    owner = task.owners[task.data['owner']]
    context = _table(owner.context()).column('entity')
    predicted = _table(owner.prediction()).column(0).slice(start, stop - start)
    native = kind['native']
    encoding = _string_encoding(task.data['labelled']['graph'], native)
    selected = pd.concat(
        [
            _native_series(context, columns, native, encoding, 'ENTITY'),
            _native_series(
                predicted,
                columns,
                native,
                task.data['prediction_encoding'],
                'ENTITY',
            ),
        ],
        axis=0,
        ignore_index=True,
    )
    try:
        converted: NativeColumn | NativeFrameFailure = NativeColumn(
            selected.astype(str)
        )
    except Exception as error:
        converted = NativeFrameFailure.capture(error)
    if isinstance(converted, NativeFrameFailure):
        return {
            'kind': 'native_string',
            'converted': {
                'outcome': 'failed',
                'failure': converted.facts.as_wire(),
            },
        }, converted
    return {
        'kind': 'native_string',
        'converted': {
            'outcome': 'value',
            'values': _strings(converted, bindings),
        },
    }, None


def _guard(
    failure: NativeFrameFailure,
    graph: dict[str, Any],
    rows_per_call: int | None = None,
    part: dict[str, Any] | None = None,
) -> NoReturn:
    guard_failure(
        failure,
        origin='engine',
        dropped_summary=graph.get('dropped_summary'),
        rows_per_call=rows_per_call,
        part=part,
    )


def _native_rejected(
    error: StructuredDataError,
    failure: NativeFrameFailure | None,
    graph: dict[str, Any],
    rows_per_call: int | None = None,
    part: dict[str, Any] | None = None,
) -> None:
    """Present only the reached native failure identified by the core."""
    details = error.details or {}
    if failure is None or 'failure' not in details or 'stage' not in details:
        return
    constructor = details.get('native_constructor_kind')
    if constructor is not None:
        if constructor != 'value_error':
            raise codec.CodecError('unknown native constructor wrapper')
        wrapped = ValueError(details['native_constructor_message'])
        wrapped.__cause__ = failure.error
        failure = NativeFrameFailure.capture(wrapped)
    _guard(failure, graph, rows_per_call, part)


def _single(
    materializer: Any,
    query: str,
    indices: Sequence[Any],
    num_context_samples: int | None,
    options: dict[str, Any],
    limits: dict[str, Any],
    *,
    cache: dict[str, Any],
    call_values: bytes,
    part: dict[str, Any] | None = None,
) -> tuple[tuple[dict[str, Any], ...], results.QueryFacts, results.CoreRequest]:
    from sdm_client._rust.materialize import (
        TAKEN,
        _anchor,
        _context_anchor,
        _indices,
    )

    state = {**cache, 'frame_cache': cache, 'frame_part': part}
    products: list[_Product] = []
    attached: list[tuple[dict[str, Any], Any]] = []

    def adopt(finished: Finished, expected: str) -> _Product:
        product = _adopt(finished, expected)
        products.append(product)
        return product

    bindings = _Bindings()
    try:
        try:
            attached, schema, columns, native_tables = (
                frame_materialize.native_inputs(materializer.schema, state)
            )
        except RouteError:
            raise
        except Exception as error:
            guard_failure(
                NativeFrameFailure.capture(error), origin='engine', part=part
            )
        state.update(
            frame_columns=columns,
            frame_schema=schema,
            frame_native_tables=native_tables,
        )
        bindings = _Bindings()
        frames = {'tables': [], 'entropy': secrets.randbits(64)}
        for source, rows in attached:
            frames['tables'].append(
                {'source': source, 'rows': bindings.add(rows)}
            )
        checkpoint = state.get('held', {}).get('frame_checkpoint')
        retained = None
        if checkpoint is not None:
            retained = bindings.add(checkpoint['owner'])
        graph = adopt(
            _prepare(
                {
                    'stage': 'graph',
                    'schema': bindings.json(schema),
                    'frames': bindings.json(frames),
                    'checkpoint': None,
                    'retained': retained,
                    'native_tables': native_tables,
                    'values': bindings.add(call_values),
                    'options': sorted(set(options) - TAKEN),
                    'prediction_rows': len(indices),
                    'limits': limits,
                },
                bindings,
                state,
            ),
            'graph',
        )
        attached.clear()
        rows = None
        # Native iteration follows graph-drop warning presentation, as in the
        # public materializer. Keep this call's list independent of reentry.
        prediction_indices = list(indices)
        bindings = _Bindings()
        labelled = adopt(
            _prepare(
                {
                    'stage': 'labels',
                    'graph': _bind(graph, bindings),
                    'query': bindings.json(query),
                    'values': bindings.json(
                        {
                            **_options(
                                options,
                                (
                                    'random_seed',
                                    'prediction_statistic',
                                    'lag_timesteps',
                                    'max_pq_iterations',
                                ),
                                num_context_samples=num_context_samples,
                            ),
                            **_context_anchor(options),
                        },
                        options=True,
                    ),
                    'seed': codec.seed(options.get('random_seed')),
                    'entropy': secrets.randbits(64),
                    'anchor_time': _anchor(options.get('anchor_time')),
                    'anchor_entity': isinstance(options.get('anchor_time'), str)
                    and options['anchor_time'] == 'entity',
                    'num_context_samples': None,
                    'limits': limits,
                },
                bindings,
                state,
            ),
            'labels',
        )
        bindings = _Bindings()
        prediction_type = labelled.data['prediction_type']
        native_failure = None
        if prediction_type['kind'] == 'core_key':
            prediction = {
                'kind': 'core_keys',
                'values': bindings.add(_indices(prediction_indices)),
            }
        else:
            constructed = columns.prediction(
                prediction_type['native'], prediction_indices
            )
            if isinstance(constructed, NativeFrameFailure):
                diagnostic, native_failure = columns.prediction_diagnostic(
                    prediction_type['native'], prediction_indices, constructed
                )
                prediction = {
                    'kind': 'native_string',
                    'constructed': {
                        'outcome': 'failed',
                        'failure': constructed.facts.as_wire(),
                        'diagnostic': diagnostic,
                    },
                }
            else:
                prediction = {
                    'kind': 'native_string',
                    'constructed': {
                        'outcome': 'value',
                        'values': _strings(constructed, bindings),
                    },
                }
        try:
            task = adopt(
                _prepare(
                    {
                        'stage': 'task',
                        'labelled': _bind(labelled, bindings),
                        'prediction': prediction,
                        'values': bindings.json(
                            _options(
                                options,
                                (
                                    'rows_per_call',
                                    'lag_timesteps',
                                    'num_hops',
                                    'num_neighbors',
                                ),
                            ),
                            options=True,
                        ),
                    },
                    bindings,
                    state,
                ),
                'task',
            )
        except StructuredDataError as error:
            _native_rejected(error, native_failure, graph.data, part=part)
            raise
        finally:
            # Task owns its narrowed context. The pre-narrowing native
            # export is not needed by request planning or later batches.
            labelled.owners.clear()
        bindings = _Bindings()
        plan = adopt(
            _prepare(
                {
                    'stage': 'plan',
                    'task': _bind(task, bindings),
                    'values': bindings.json(
                        _options(
                            options,
                            (
                                'top_k',
                                'inference_config',
                                'exclude_cols_dict',
                                'num_hops',
                                'num_neighbors',
                                'lag_timesteps',
                            ),
                        ),
                        options=True,
                    ),
                },
                bindings,
                state,
            ),
            'plan',
        )
        bindings = _Bindings()
        converted, native_failure = _references(
            task, columns, bindings, 0, task.data['prediction_rows']
        )
        try:
            references = adopt(
                _prepare(
                    {
                        'stage': 'references',
                        'plan': _bind(plan, bindings),
                        'converted': converted,
                    },
                    bindings,
                    state,
                ),
                'references',
            )
        except StructuredDataError as error:
            _native_rejected(
                error,
                native_failure,
                graph.data,
                task.data['rows_per_call'],
                part,
            )
            raise
        parts = []
        for start in range(
            0, task.data['prediction_rows'], task.data['rows_per_call']
        ):
            stop = min(
                start + task.data['rows_per_call'], task.data['prediction_rows']
            )
            bindings = _Bindings()
            converted, native_failure = _references(
                task, columns, bindings, start, stop
            )
            try:
                context = adopt(
                    _prepare(
                        {
                            'stage': 'context',
                            'references': _bind(references, bindings),
                            'range': {'start': start, 'stop': stop},
                            'converted': converted,
                        },
                        bindings,
                        state,
                    ),
                    'context',
                )
            except StructuredDataError as error:
                _native_rejected(
                    error,
                    native_failure,
                    graph.data,
                    task.data['rows_per_call'],
                    part,
                )
                raise
            bindings = _Bindings()
            converted_text = []
            native_failure = None
            for selected in context.data['text']:
                table = next(
                    table
                    for table in context.data['tables']
                    if table['name'] == selected['column']['table']
                )
                rows = _table(context.owners[table['rows']])
                native = selected['native']
                selected_series = _native_series(
                    rows.column(selected['column']['column']),
                    columns,
                    native,
                    _string_encoding(graph.data, native),
                    selected['column']['column'],
                )
                converted_column = convert_text(selected_series)
                if isinstance(converted_column, NativeFrameFailure):
                    native_failure = converted_column
                    text = {
                        'outcome': 'failed',
                        'completed': converted_text,
                        'column': selected['column'],
                        'failure': native_failure.facts.as_wire(),
                    }
                    break
                converted_text.append(
                    {
                        'column': selected['column'],
                        'values': _strings(converted_column, bindings),
                        'facts': converted_column.text_facts(
                            context.data['text_identity_input']
                        ),
                    }
                )
            else:
                text = {'outcome': 'converted', 'columns': converted_text}
            try:
                encoded_part = adopt(
                    _prepare(
                        {
                            'stage': 'payload',
                            'context': _bind(context, bindings),
                            'text': text,
                            'values': bindings.json(
                                _options(
                                    options,
                                    (
                                        'return_embeddings',
                                        'use_prediction_time',
                                    ),
                                ),
                                options=True,
                            ),
                        },
                        bindings,
                        state,
                    ),
                    'part',
                )
            except StructuredDataError as error:
                context.owners.clear()
                _native_rejected(
                    error,
                    native_failure,
                    graph.data,
                    task.data['rows_per_call'],
                    part,
                )
                raise
            finally:
                # Complete payloads retain no raw sampled context. Do not
                # accumulate source-row/link buffers across later batches.
                context.owners.clear()
                rows = None
                selected_series = None
                converted_column = None
            parts.append(encoded_part)
        bindings = _Bindings()
        finished = _prepare(
            {
                'stage': 'request',
                'plan': _bind(plan, bindings),
                'parts': {
                    'version': plan.data['version'],
                    'plan_identity': plan.data['identity'],
                    'task_identity': task.data['identity'],
                    'parts': [_bind(part, bindings) for part in parts],
                },
            },
            bindings,
            state,
        )
        answer = results.expect(finished.result, results.FramePrepared)
        prepared = json.loads(finished.take(answer.value))
        if prepared.get('value') != 'request':
            raise codec.CodecError(
                'frame request assembly returned another value'
            )
        value = prepared['data']
        gathered = results.expect(
            {'op': 'materialize', **value['request']}, results.Gathered
        )
        payloads = []
        for companion in value['payloads']:
            owner = finished.take(companion['value'])
            payloads.append(owner.value())
        if gathered.query is None:
            raise codec.CodecError('frame request has no resolved query facts')
        return (
            tuple(payloads),
            gathered.query,
            results.CoreRequest(
                version=gathered.state.version,
                batches=None
                if gathered.state.batches is None
                else tuple(batch.rows for batch in gathered.state.batches),
            ),
        )
    finally:
        # Tracebacks may retain this frame; release only call-owned exports
        # and typed products. The materializer cache retains its own graph.
        for product in products:
            product.owners.clear()
        products.clear()
        attached.clear()
        state.clear()
        bindings.attachments.clear()
        owner = None
        finished = None


def gather(
    materializer: Any,
    query: Any,
    indices: Sequence[Any],
    num_context_samples: int | None,
    options: dict[str, Any],
    candidates: Sequence[Any] | None,
) -> tuple[tuple[dict[str, Any], ...], results.QueryFacts, results.CoreRequest]:
    from sdm_client import decompose
    from sdm_client.materializers import relational

    limits = {
        'entities': max(0, min(int(relational.MOST_ENTITIES), 2**64 - 1)),
        'calls': max(0, min(int(relational.MOST_CALLS), 2**32 - 1)),
        'parts': max(0, min(int(decompose.MOST_PARTS), 2**32 - 1)),
    }
    with errors.holding(
        inference_option=lambda fields: options['inference_config'][
            fields['key']
        ],
    ):
        questions = _questions(
            materializer,
            query.to_string(),
            indices,
            num_context_samples,
            options,
            candidates,
            limits,
        )
        decomposition = questions.data['decomposition']
        if decomposition is None:
            return _single(
                materializer,
                query.to_string(),
                indices,
                num_context_samples,
                options,
                limits,
                cache=materializer._core_sampler_state,
                call_values=questions.owners[questions.data['call_values']],
            )
        child_options = {
            name: options[name] for name in decomposition['child_options']
        }
        prepared = []
        for question in decomposition['parts']:
            scope = {
                'kind': decomposition['kind'],
                'position': question['position'],
                'parts': len(decomposition['parts']),
            }
            child = _questions(
                materializer,
                question['query'],
                indices,
                num_context_samples,
                child_options,
                None,
                limits,
                part=scope,
            )
            if child.data['decomposition'] is not None:
                raise codec.CodecError(
                    'a core-selected frame part decomposed again'
                )
            # Each rewritten question owns a new native graph, as the old
            # public PredictiveQuery constructor did for every reached part.
            # Neither this parent nor a preceding part supplies a graph/RNG.
            prepared.append(
                _single(
                    materializer,
                    question['query'],
                    indices,
                    num_context_samples,
                    child_options,
                    limits,
                    cache={},
                    call_values=child.owners[child.data['call_values']],
                    part=scope,
                )
            )
        first_payloads, _, first_state = prepared[0]
        first_question = decomposition['parts'][0]
        parts = results.CoreParts(
            kind=decomposition['kind'],
            query=first_question['query'],
            position=first_question['position'],
            name=first_question['name'],
            top_k=decomposition['top_k'],
            later=tuple(
                results.CorePart(
                    position=question['position'],
                    name=question['name'],
                    payload=payloads[0],
                    query=question['query'],
                )
                for question, (payloads, _, _) in zip(
                    decomposition['parts'][1:], prepared[1:], strict=True
                )
            ),
        )
        return (
            first_payloads,
            results.QueryFacts.from_contract(questions.data['query']),
            results.CoreRequest(
                version=first_state.version,
                parts=parts,
            ),
        )


def _questions(
    materializer: Any,
    query: str,
    indices: Sequence[Any],
    num_context_samples: int | None,
    options: dict[str, Any],
    candidates: Sequence[Any] | None,
    limits: dict[str, Any],
    *,
    part: dict[str, Any] | None = None,
) -> _Product:
    from sdm_client._rust.materialize import TAKEN, _candidates
    from sdm_client._rust.spec import held_schema

    bindings = _Bindings()
    question = _adopt(
        _prepare(
            {
                'stage': 'question',
                'schema': bindings.json(held_schema(materializer.schema)),
                'query': bindings.json(query),
                'candidates_given': candidates is not None,
                'statistic_given': 'prediction_statistic' in options,
                'top_k_given': 'top_k' in options,
            },
            bindings,
            {'frame_part': part},
        ),
        'question',
    )
    # No option/candidate object has been encoded before successful query
    # resolution. Each later observation belongs to its reached domain.
    bindings = _Bindings()
    kind = question.data['kind']
    names = {
        None: ('prediction_statistic', 'num_hops', 'num_neighbors'),
        'ranking': ('top_k', 'rows_per_call'),
        'forecast': ('rows_per_call',),
    }[kind]
    extra = {'num_context_samples': num_context_samples} if kind is None else {}
    value = {
        'stage': 'questions',
        'input': {
            'question': _bind(question, bindings),
            'candidates': None
            if candidates is None
            else bindings.json(_candidates(candidates)),
            'values': bindings.json(
                _options(options, names, **extra),
                options=True,
            ),
            'options': sorted(set(options) - TAKEN),
            'option_facts': [
                {'name': name, 'is_none': value is None}
                for name, value in options.items()
            ],
            'prediction_rows': len(indices),
            'limits': limits,
        },
    }
    try:
        return _adopt(
            _prepare(value, bindings, {'frame_part': part}), 'questions'
        )
    finally:
        question.owners.clear()
