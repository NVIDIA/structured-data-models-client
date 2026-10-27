# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ownership between useful gathered-prediction core operations.

Rust owns model rules, attribution, numbering, merging and decomposition.
Python retains live public containers until their actual native observations,
encodes each reached call, performs I/O and presents the selected owners.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

from sdm_client._rust import codec, errors, results
from sdm_client._rust import relational_native as native
from sdm_client._rust.prepare import _call
from sdm_client._rust.relational_response import CompletedResponse, present
from sdm_client._rust.run import Finished, run
from sdm_client.decompose import RANKING
from sdm_client.errors import StructuredDataError


@dataclass
class CallResult:
    """One ordinary public prediction's complete call datasets."""

    bound: dict[str, Any]
    parts: list[CompletedResponse]
    counted: list[int | None]
    known_empty: bool = False
    native_outputs: tuple[Any, ...] | None = None
    requested_outputs: Any = None
    presented: Any = None

    def attached(self, first: int) -> tuple[dict[str, Any], dict[int, Any]]:
        parts = []
        attached = {}
        for response in self.parts:
            part, buffers = response.attached(first + len(attached))
            parts.append(part)
            attached.update(buffers)
        return {
            'bound': self.bound,
            'parts': parts,
            'counted': self.counted,
            'known_empty': self.known_empty,
        }, attached


def _owners(children: list[CallResult]) -> dict[str, CompletedResponse]:
    return {
        response.call_request_id: response
        for child in children
        for response in child.parts
    }


def _present(
    finished: Finished,
    owners: dict[str, CompletedResponse],
    *,
    by_step: bool = False,
    given: dict[int, Any] | None = None,
    native_outputs: tuple[Any, ...] | None = None,
) -> tuple[Any, dict[str, int]]:
    from sdm_client._rust import predict as prediction

    prepared = results.expect(finished.result, results.Prepared)
    data = json.loads(finished.take(prepared.value))
    answered = results.expect(
        {'op': 'predict', **data['predicted']}, results.Predicted
    )
    table = prediction._table(finished.take(answered.result))
    rows = table.to_pylist()
    lineage = data['lineage']
    # Lineage is in endpoint answer order, independent of the Arrow table's
    # row-index sorting. The core selected every call and source ordinal.
    for row in rows:
        source = lineage[row['answer_index']]
        if source['source_row'] is not None:
            row['id'] = owners[source['call_request_id']].entity(source)
            row.pop('id_as_written', None)
    metadata = codec.restored_json(finished.take(answered.metadata))
    if given is not None:
        result = prediction._ranked(
            answered,
            sorted(rows, key=lambda row: row['answer_index']),
            metadata,
            given,
            native_outputs=native_outputs,
        )
    else:
        explained = (table.schema.metadata or {}).get(b'explanation')
        result = prediction._result(
            rows,
            model=answered.model,
            request_id=answered.request_id,
            metadata=metadata,
            outputs=answered.outputs,
            by_step=by_step,
            explanation=None if explained is None else explained.decode(),
        )
    named = (table.schema.metadata or {}).get(b'names_its_rows')
    if named is not None:
        result._names_its_rows = named == b'true'
    return result, data['answer_details']


def _finish(
    bound: dict[str, Any],
    aggregation: dict[str, Any],
    attached: dict[int, Any],
    owners: dict[str, CompletedResponse],
    *,
    report_absent: bool = True,
    by_step: bool = False,
    given: dict[int, Any] | None = None,
    native_outputs: tuple[Any, ...] | None = None,
) -> tuple[Any, dict[str, int]]:
    finished = run(
        {'op': 'prepare_request', 'input': 1},
        {
            1: codec.attachment(
                {
                    'stage': 'finish_relational_response',
                    'input': {
                        'bound': bound,
                        'aggregation': aggregation,
                        'report_absent': report_absent,
                    },
                }
            ),
            **attached,
        },
    )
    return _present(
        finished,
        owners,
        by_step=by_step,
        given=given,
        native_outputs=native_outputs,
    )


def _finish_calls(
    child: CallResult, *, report_absent: bool = True
) -> tuple[Any, dict[str, int]]:
    value, attached = child.attached(2)
    value.pop('bound')
    answer, details = _finish(
        child.bound,
        {'kind': 'calls', **value},
        attached,
        _owners([child]),
        report_absent=report_absent and child.native_outputs is None,
    )
    if child.native_outputs is not None:
        if len(child.parts) > 1:
            # Native merge reaches this set only after model/metadata layout,
            # before final absence. Repeated equal tuples must still hash.
            shapes, shape_facts = native.output_shapes(
                child.native_outputs, len(child.parts)
            )
            with errors.holding(
                relational_output_shapes=lambda fields: repr(sorted(shapes))
            ):
                native._prepared(
                    {
                        'data': 'output_shapes',
                        'call': child.bound,
                        **shape_facts,
                    }
                )
        answer.outputs = child.native_outputs
        if report_absent:
            native.absent(child.bound, answer, child.requested_outputs)
    return answer, details


def _carry(error: Exception, result: Any, details: dict[str, int]) -> None:
    held = getattr(error, 'details', None)
    error.details = {  # type: ignore[attr-defined]
        **(held if isinstance(held, dict) else {}),
        **details,
        'answered': result,
    }


def _one(client: Any, request: Any, model: Any, outputs: Any) -> CallResult:
    from sdm_client._rust.predict import _send

    prepared = native.prepare(request, model, outputs)
    bound = prepared.bound
    if native.nothing_to_score(request.payloads):
        return CallResult(
            bound,
            [],
            [],
            known_empty=True,
            native_outputs=prepared.outputs,
            requested_outputs=outputs,
        )
    payloads = request.payloads
    native._prepared(
        {'data': 'payloads', 'call': bound, 'present': bool(payloads)}
    )
    native.preflight(bound, payloads, request.query, prepared.outputs)
    planned = native.envelopes(bound, request, payloads, prepared.outputs)
    counted = native.count_calls(planned)
    completed = []
    for envelope in planned:
        try:
            # Neither depth traversal nor JSON encoding observes future calls.
            body = native.encoded(client._transport, envelope)
            finished = run(
                {'op': 'prepare_request', 'input': 1},
                {
                    1: codec.attachment(
                        {
                            'stage': 'call_relational',
                            'input': {
                                'call': bound,
                                'body': 2,
                                'request_id': envelope['metadata'][
                                    'request_id'
                                ],
                                'model_endpoint': getattr(
                                    client._transport, 'model_endpoint', None
                                ),
                            },
                        }
                    ),
                    2: body,
                },
                model=lambda sent: _send(client, bound['model'], sent),
            )
            completed.append(present(finished, envelope))
        except Exception as error:
            if completed:
                result, details = _finish_calls(
                    CallResult(
                        bound,
                        completed,
                        counted[: len(completed)],
                        native_outputs=prepared.outputs,
                    ),
                    report_absent=False,
                )
                _carry(error, result, details)
            raise
    return CallResult(
        bound,
        completed,
        counted,
        native_outputs=prepared.outputs,
        requested_outputs=outputs,
    )


def _tag(
    error: StructuredDataError, scope: dict[str, Any], candidate: Any
) -> StructuredDataError:
    description = _call({'stage': 'describe_relational_part', 'scope': scope})
    # Reached native presentation, with its sentence grammar selected by Rust.
    # Complete the prefix before formatting the candidate or reading the
    # forecast remedy, and retain extra native exception arguments.
    text = str(error.args[0]).rstrip()
    ending = description['ending']
    first = (
        text
        if not text or text[-1] in ending['terminal_characters']
        else f'{text}{ending["period"]}'
    )
    if scope['kind'] == 'steps':
        outran = bool(
            (getattr(error, 'details', None) or {}).get(
                'window_outran_the_data'
            )
        )
        if outran:
            description = _call(
                {
                    'stage': 'describe_relational_part',
                    'scope': {**scope, 'window_outran': outran},
                }
            )
    fields = errors.decoded(description['fields'])
    suffix = errors.render(
        description['sentence'], fields, {'part_candidate': candidate}
    )
    said = f'{first} {suffix}'
    described = error.__class__.__new__(error.__class__)
    described.__dict__.update(error.__dict__)
    described.message = said
    described.args = (said, *error.args[1:])
    described.details = {**error.details, **fields}
    # The public client's outer wrapper recognizes an already attributed core
    # error. Preserve that attribution even when its original cause was native.
    described._sentence = description['sentence']
    return described


def predict(client: Any, request: Any, model: Any, outputs: Any) -> Any:
    """Predict from the currently held native request, including edits."""
    from sdm_client._rust.predict import positions

    if request.parts is None:
        child = _one(client, request, model, outputs)
        return _finish_calls(child)[0]
    rewritten = request._decomposition
    assert rewritten is not None
    ranking = rewritten.kind == RANKING
    positions_ = (
        positions(request) if ranking else tuple(range(len(request.parts)))
    )
    given = {
        position: about
        for position, (_, about) in zip(
            positions_, rewritten.parts, strict=True
        )
    }
    with errors.holding(
        candidates=lambda fields: [given[at] for at in fields['candidates']],
        candidate=lambda fields: given[fields['candidate']],
        part_candidate=lambda fields: given[fields['ranking_candidate']],
    ):
        return _parts(client, request, model, outputs, positions_, given)


def _parts(
    client: Any,
    request: Any,
    model: Any,
    outputs: Any,
    positions_: tuple[int, ...],
    given: dict[int, Any],
) -> Any:
    rewritten = request._decomposition
    ranking = rewritten.kind == RANKING
    total = len(request.parts)
    answered = []

    def scope(position: int, about: Any) -> dict[str, Any]:
        return (
            {
                'kind': 'ranking',
                'position': position,
                'total': total,
                'completed': [at for at, _ in answered],
            }
            if ranking
            else {'kind': 'steps', 'step': about, 'total': total}
        )

    cold = []
    wanted = outputs
    ranking_outputs = None
    if ranking:
        selected = native.ranking_outputs(outputs, total)
        wanted = selected['part_outputs']
        ranking_outputs = selected
        for position, part, (_, about) in zip(
            positions_, request.parts, rewritten.parts, strict=True
        ):
            try:
                prepared = native.prepare(part, model, wanted)
                bound = prepared.bound
                native.preflight(
                    bound, part.payloads, part.query, prepared.outputs
                )
                native.envelopes(bound, part, part.payloads, prepared.outputs)
            except StructuredDataError as error:
                if error.details.get('classes_in_context') != 1:
                    raise _tag(error, scope(position, about), about) from error
                cold.append(position)
        _call(
            {
                'stage': 'validate_relational_ranking',
                'outputs': None,
                'selected': selected,
                'total_parts': total,
                'cold': cold,
                'model': bound['model'],
            }
        )
    children = []
    for position, part, (_, about) in zip(
        positions_, request.parts, rewritten.parts, strict=True
    ):
        if position in cold:
            continue
        try:
            # Original decomposition enters the public predict method for
            # each scored child, before its separate preparation guard.
            from sdm_client.client import (
                _require_a_request,
                _require_output_names,
            )

            _require_a_request(part)
            _require_output_names(wanted)
            child = _one(client, part, model, wanted)
            # Each ordinary child reports its absent outputs before the next
            # part's model checks, matching separate public predict calls.
            child.presented, _ = _finish_calls(child)
            children.append(child)
            answered.append((position, about))
        except Exception as error:
            if children:
                if ranking:
                    held = getattr(error, 'details', None)
                    paid = _call(
                        {
                            'stage': 'describe_relational_part',
                            'scope': scope(position, about),
                        }
                    )
                    paid = errors.decoded(paid['fields'])
                    error.details = {  # type: ignore[attr-defined]
                        **(held if isinstance(held, dict) else {}),
                        'parts_answered': paid['parts_answered'],
                    }
                else:
                    result, details = _finish_parts(children, answered, None)
                    _carry(error, result, details)
            if isinstance(error, StructuredDataError):
                raise _tag(error, scope(position, about), about) from error
            raise
    result, _ = _finish_parts(
        children,
        answered,
        (given, cold, rewritten.top, ranking_outputs, outputs)
        if ranking
        else None,
    )
    if not ranking:
        native.forecast_explanations(model, outputs, children, answered)
    return result


def _finish_parts(
    children: list[CallResult],
    answered: list[tuple[int, Any]],
    ranking: Any,
) -> tuple[Any, dict[str, int]]:
    values = []
    attached = {}
    for child in children:
        value, buffers = child.attached(2 + len(attached))
        values.append(value)
        attached.update(buffers)
    owners = _owners(children)
    given = None
    native_outputs = None
    if ranking is None:
        aggregation = {
            'kind': 'steps',
            'children': values,
            'steps': [about for _, about in answered],
        }
    else:
        from sdm_client._rust import relational_rank

        given, cold, top_k, selected, requested = ranking
        positions_ = [position for position, _ in answered]
        population = relational_rank.population(
            {'children': values, 'positions': positions_}, attached
        )
        grouped = relational_rank.groups(population, owners, given)
        words = _call(
            {
                'stage': 'ranking_output_vocabulary',
                'named': False,
                'nonempty': None,
            }
        )
        native_outputs = tuple(
            requested or tuple(sys.intern(name) for name in words['outputs'])
        )
        group_id = 2 + len(attached)
        attached[group_id] = codec.attachment(grouped)
        aggregation = {
            'kind': 'ranking',
            'children': values,
            'positions': positions_,
            'candidates': ['null' for _ in positions_],
            'top_k': top_k,
            'outputs': selected['outputs'],
            'selected': selected,
            'unscored_candidates': cold,
            'groups': group_id,
        }
    result, details = _finish(
        children[0].bound,
        aggregation,
        attached,
        owners,
        # Native post-child containment chooses forecast explanation advice.
        # The public caller still performs its own final absent-output read.
        report_absent=False,
        by_step=ranking is None,
        given=given,
        native_outputs=native_outputs,
    )
    if ranking is None:
        result.outputs = (
            tuple(children[0].presented.outputs) if children else ()
        )
    elif native_outputs is not None:
        result.outputs = native_outputs
    return result, details
