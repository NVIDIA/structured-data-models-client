# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Sending a relational request to ``kumo-relational``.

Unlike the tabular model, this one does heavy client-side work: building the
graph, resolving the query, sampling each entity's neighbourhood and encoding
the result. All of that happens in :meth:`PredictiveQuery.materialize`, which
runs in the engine, so by the time a request reaches this adapter its payloads
are already finished.

That split is why this adapter is short. It sends what was gathered, and reads
the answer back. It does not reimplement the gathering, which would duplicate
point-in-time correctness logic that already lives -- and is tested -- in the
engine.
"""

from __future__ import annotations

import dataclasses
import re
import uuid
import warnings
from collections.abc import Iterator, Mapping
from typing import Any

from sdm_api.generated.tfm_api import PredictionResponse, ResponseContractError

from sdm_client.base import ModelSpec
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import (
    WIDEN_THE_CONTEXT,
    NimRequestError,
    StructuredDataError,
    UnsupportedRequestError,
)
from sdm_client.models import KUMO_RELATIONAL
from sdm_client.request import Request
from sdm_client.result import PredictionResult
from sdm_client.wire import as_declared
from sdm_client.wire.answers import as_declared_target

# What to ask for when the caller named nothing, by task kind. Asking for
# everything would make a regression request carry probabilities it cannot
# have, which the endpoint rejects.
_DEFAULT_OUTPUTS = {
    'classification': ('prediction', 'probabilities'),
    'binary_classification': ('prediction', 'probabilities'),
    'multiclass_classification': ('prediction', 'probabilities'),
    'regression': ('prediction',),
    'forecasting': ('prediction',),
    'temporal_link_prediction': ('rankings',),
}

_SESSION_PINNED = ('model', 'task', 'schema', 'context')
#: Which halves of a request a call puts on the wire. Every judgement made
#: before sending asks only about a half that travels.
_SENDS_EVERYTHING = frozenset({'context', 'predict'})
_SENDS_A_CONTEXT = frozenset({'context'})
_SENDS_ROWS_TO_SCORE = frozenset({'predict'})

_SESSION_SCORING = ('predict', 'output', 'inference', 'metadata')


def _request_id() -> str:
    return f'sdm-{uuid.uuid4().hex[:12]}'


def _keys_the_caller_set(requested: Any) -> set[str]:
    r"""The inference-config keys that were asked for rather than defaulted.

    The engine takes ``InferenceConfig | dict | None`` and this client
    publishes nothing narrower, so a caller may hand over the dataclass. It is
    not iterable, and reading it as a set of keys destroyed a request whose
    payload was already gathered and correct.

    A dataclass carries every field whether or not it was set, so only the
    ones differing from their default are a request for anything. Counting the
    rest would refuse a bare ``RegressionInferenceConfig()``, which asks for
    exactly what passing no config at all asks for.
    """
    if isinstance(requested, Mapping):
        return set(requested)
    if dataclasses.is_dataclass(requested) and not isinstance(requested, type):
        return {
            field.name
            for field in dataclasses.fields(requested)
            if getattr(requested, field.name) != _default_of(field)
        }
    return set()


def _default_of(field: dataclasses.Field) -> Any:
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return field.default


class RelationalAdapter:
    r"""Shapes and reads ``kumo-relational`` requests."""

    spec: ModelSpec = KUMO_RELATIONAL

    def default_output(self, request: Request) -> tuple[str, ...]:
        return _DEFAULT_OUTPUTS.get(request.task.kind, ('prediction',))

    def prediction_plan(
        self,
        request: Request,
        outputs: tuple[str, ...],
        sends: frozenset[str] = _SENDS_EVERYTHING,
    ) -> Iterator[dict[str, Any]]:
        payloads = self._payloads(request)
        self._preflight(payloads, outputs, request.query, sends)
        for payload in payloads:
            envelope = dict(payload)
            envelope['model'] = self.spec.model_id
            envelope['output'] = {'fields': list(outputs)}
            envelope.setdefault('metadata', {})['request_id'] = _request_id()
            inference = self._inference(envelope.get('inference'), request)
            if inference is None:
                envelope.pop('inference', None)
            else:
                envelope['inference'] = inference
            yield envelope

    def _inference(
        self, inference: Any, request: Request
    ) -> dict[str, Any] | None:
        """Send only the inference options this model reads.

        The engine was written against a different backend and fills in an
        inference config on every request, including keys this one refuses
        outright. Dropping them is safe because they are the engine's defaults
        rather than anything asked for -- but if the *caller* asked for one, it
        is named rather than silently discarded, because a dropped request for
        four estimators is a different prediction than the one that was wanted.

        Forwarding a key is not a promise that the endpoint serves it. This is
        a name-level filter over what the engine sends; whether a deployment
        accepts the *value* is its own answer, and it may still refuse.
        """
        if not isinstance(inference, dict):
            return None
        config = inference.get('inference_config')
        if not isinstance(config, dict):
            return dict(inference)

        forwarded = self.spec.runtime.inference_options
        unsupported = sorted(set(config) - forwarded)
        asked_for = _keys_the_caller_set(
            request.requested_inference.get('inference_config')
        )
        refused = sorted(set(unsupported) & asked_for)
        if refused:
            raise UnsupportedRequestError(
                f'{self.spec.model_id} does not read inference option(s) '
                f'{refused}; drop them from inference_config to send the '
                f'request without them. The keys this client forwards are '
                f'{sorted(forwarded)}, and the endpoint decides for itself '
                f'whether it serves the value given for one',
                details={
                    'model': self.spec.model_id,
                    'unsupported': refused,
                },
            )

        trimmed = dict(inference)
        trimmed['inference_config'] = {
            key: value for key, value in config.items() if key in forwarded
        }
        return trimmed

    def session_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> dict[str, Any]:
        payloads = self._payloads(request)
        if len(payloads) > 1:
            # A session pins one context. Several gathered parts each carry
            # their own, so pinning the first would score later batches
            # against the wrong one.
            raise UnsupportedRequestError(
                f'this request gathered {len(payloads)} batches, and a session '
                f'pins one context; use predict() instead',
                details={'batches': len(payloads)},
            )
        envelope = next(
            iter(self.prediction_plan(request, outputs, _SENDS_A_CONTEXT))
        )
        pinned = {
            key: envelope[key] for key in _SESSION_PINNED if key in envelope
        }
        pinned['metadata'] = envelope['metadata']
        return pinned

    def session_prediction_plan(
        self, request: Request, outputs: tuple[str, ...]
    ) -> Iterator[dict[str, Any]]:
        for envelope in self.prediction_plan(
            request, outputs, _SENDS_ROWS_TO_SCORE
        ):
            yield {
                key: envelope[key]
                for key in _SESSION_SCORING
                if key in envelope
            }

    def normalize(
        self,
        response: dict[str, Any],
        outputs: tuple[str, ...],
        payload: dict[str, Any] | None = None,
    ) -> PredictionResult:
        try:
            parsed = PredictionResponse.from_dict(response)
        except (ResponseContractError, TypeError, ValueError) as error:
            raise StructuredDataError(
                f'{self.spec.model_id} returned a response that does not '
                f'match the contract: {error}',
                code='INVALID_RESPONSE',
            ) from error

        explanation = next(
            (
                item.explanation
                for item in parsed.predictions
                if item.explanation
            ),
            None,
        )
        return PredictionResult(
            model=parsed.model,
            request_id=parsed.id,
            items=as_declared_target(
                _named_entities(parsed.predictions, payload),
                payload,
            ),
            metadata=parsed.metadata,
            outputs=outputs,
            explanation=explanation,
        )

    def explain_error(
        self, error: Exception, payload: dict[str, Any]
    ) -> Exception:
        r"""Name the columns behind a semantic-type refusal.

        ``semantic type 'text' is unsupported`` is true and unactionable:
        ``Schema.from_data`` marks any string column with more than 64 distinct
        values as ``text``, so a real dataset acquires them in bulk and the
        endpoint names none of them. The payload it just refused does.

        The deployments word this two ways and only one of them was matched, so
        on the one that says ``categorical cardinality 13812 exceeds limit
        10000`` the hook never ran and the caller got a number and a limit
        against a graph of twenty-two string columns. Both forms are read now,
        and the second names the columns by counting the payload rather than by
        reading a declared type, because a cardinality complaint is about the
        values.

        A different complaint needs a different remedy. Where the columns are
        already categorical, telling their owner to make them categorical
        would be advice that changes nothing.
        """
        columns, complaint = _columns_the_endpoint_means(error, payload)
        if not columns:
            return error
        if complaint == 'cardinality':
            message = (
                f'{getattr(error, "message", error)}. In this request that is '
                f'{", ".join(columns)}. A column with that many distinct '
                f'values carries no signal the model can use; leave it out '
                f'with materialize(..., exclude_cols_dict=...), or replace it '
                f'with something coarser.'
            )
            return _reworded(error, message)
        # The payload does not carry the entity table's real name --
        # `instance_table` is what the wire calls it -- so an example naming
        # it would render a line the caller cannot run. Prefer a related
        # table, whose name did survive, and fall back to naming no table at
        # all rather than a made-up one.
        named = next(
            (c for c in columns if not c.startswith('instance_table.')), None
        )
        if named is not None:
            table, _, column = named.partition('.')
            suggestion = f"schema[{table!r}][{column!r}].stype = 'categorical'"
        else:
            _, _, column = columns[0].partition('.')
            suggestion = (
                f"schema[<the entity table>][{column!r}].stype = 'categorical'"
            )
        message = (
            f'{getattr(error, "message", error)}. In this request that is '
            f'{", ".join(columns)}. Set a supported type on each, e.g. '
            f'{suggestion}'
        )
        return _reworded(error, message)

    def close(self) -> None:
        r"""Nothing to release: gathering happens in the query, which owns
        whatever it opened, and sending goes through the client's transport."""

    def _preflight(
        self,
        payloads: tuple[dict[str, Any], ...],
        outputs: tuple[str, ...],
        asked: str | None = None,
        sends: frozenset[str] = _SENDS_EVERYTHING,
    ) -> None:
        """Refuse what the endpoint enforces, before spending the round trip.

        An explanation is the narrow case: the model explains one entity at a
        time, and only in the mode the smaller context budget selects. Sending
        more gets a 500 that names neither limit, so both are checked here.

        `sends` names the halves this call puts on the wire. A session pins its
        context at `fit()` and a scoring call carries only rows, so judging the
        context there refuses a request over rows that are thrown away -- and
        judging the rows when pinning refuses over rows that are not sent at
        all. Every check below asks only about a half that travels.
        """
        limits = self.spec.runtime.limits
        if 'context' in sends:
            self._check_context_is_not_empty(payloads)
            self._check_class_count(payloads, limits)
            self._check_target_varies(payloads, asked)
        if 'explanation' not in outputs:
            return

        scored = sum(
            len(
                payload.get('predict', {})
                .get('instance_table', {})
                .get('rows', [])
            )
            for payload in payloads
        )
        allowed = limits['explanation_predict_rows']
        if 'predict' in sends and scored != allowed:
            raise UnsupportedRequestError(
                f'an explanation covers exactly {allowed} entity, and this '
                f'request scores {scored}; ask for one at a time',
                details={'model': self.spec.model_id, 'scored': scored},
            )

        cap = limits['context_rows_fast']
        if 'context' in sends and _asks_for_the_larger_context(payloads, cap):
            raise UnsupportedRequestError(
                f'an explanation is only produced in the mode the smaller '
                f'context budget selects, and this request was gathered with '
                f'num_context_samples above {cap:,}. Re-materialize with '
                f'num_context_samples={cap} or fewer',
                details={
                    'model': self.spec.model_id,
                    'context_rows_fast': cap,
                },
            )

    def _check_class_count(
        self, payloads: tuple[dict[str, Any], ...], limits: Mapping[str, int]
    ) -> None:
        r"""A classification target the model cannot represent is a size too.

        The endpoint answers ``422 UNSUPPORTED_FEATURE`` with both numbers in
        it, which is a decent message -- but it arrives after the context has
        been gathered, encoded and sent, and the count is knowable here.
        """
        ceiling = limits.get('classes_max')
        if not ceiling:
            return
        for payload in payloads:
            kind = str(payload.get('task', {}).get('kind', ''))
            if 'classification' not in kind:
                continue
            column = (
                payload.get('task', {}).get('target', {}).get('column_name')
            )
            labels = _distinct_labels(payload, column)
            if labels is not None and labels > ceiling:
                raise UnsupportedRequestError(
                    f'{self.spec.model_id} classifies at most {ceiling} '
                    f'classes, and the gathered context carries {labels} '
                    f'distinct values of the target; predict a coarser '
                    f'target, or narrow the context to the classes you care '
                    f'about',
                    details={
                        'model': self.spec.model_id,
                        'classes': labels,
                        'limit': ceiling,
                    },
                )

    def _check_context_is_not_empty(
        self, payloads: tuple[dict[str, Any], ...]
    ) -> None:
        r"""Refuse a context this client emptied, rather than sending it.

        Scoring every row of the entity table takes every row out of the
        context. One row left is refused; none fell through the same check,
        because zero distinct answers is not one.
        """
        for payload in payloads:
            table = payload.get('context', {}).get('instance_table')
            if not isinstance(table, dict) or table.get('rows'):
                continue
            raise UnsupportedRequestError(
                'no context rows are left: every row that could be context is '
                'also being scored, and a row carrying the answer being asked '
                'for is not sent as context. Score fewer rows, so that the '
                'rest are there to learn from',
                details={'model': self.spec.model_id, 'context_rows': 0},
            )

    def _check_target_varies(
        self, payloads: tuple[dict[str, Any], ...], asked: str | None = None
    ) -> None:
        r"""Refuse, or remark on, a context whose every row says the same thing.

        The tabular route judges this before it sends and the relational one
        said nothing, so the same unlearnable question was refused by one model
        and answered by the other. The gathered payload carries the evidence
        either way: the target column of the context it is about to send.

        A classification over one class is refused by the endpoint, so it is
        refused here. A regression over a constant is accepted there, so it is
        only warned about. A forecast and a link prediction are neither: their
        targets are a horizon and a candidate list, and one value in the column
        does not mean the same thing.

        The wire calls this column `TARGET` whatever the question was, and a
        message naming it describes something the caller never wrote. Where
        the target is a column of theirs the query says so and it is named;
        where it is an aggregation or a predicate there is no column to name,
        and the question is quoted instead.
        """
        for payload in payloads:
            kind = str(payload.get('task', {}).get('kind', ''))
            if 'classification' not in kind and kind != 'regression':
                continue
            column = (
                payload.get('task', {}).get('target', {}).get('column_name')
            )
            values = _answered_target_values(payload, column)
            if values is None or len(values) != 1:
                continue
            said = _how_to_name_the_target(column, asked)
            if 'classification' in kind:
                raise UnsupportedRequestError(
                    f'every context row has the same {said}, and a '
                    f'classification needs at least two classes to learn '
                    f'from. {WIDEN_THE_CONTEXT}',
                    details={
                        'model': self.spec.model_id,
                        'target': column,
                        'classes_in_context': 1,
                    },
                )
            warnings.warn(
                f'every context row has the same {said}, so nothing in the '
                f'context distinguishes one answer from another. The endpoint '
                f'will answer, but the answer cannot say much. Gather a wider '
                f'context with num_context_samples, or ask about entities '
                f'whose answers differ.',
                category=SDMWarning,
                stacklevel=caller_stacklevel(),
            )
            return

    @staticmethod
    def _payloads(request: Request) -> tuple[dict[str, Any], ...]:
        payloads = request.payloads
        if not payloads:
            raise UnsupportedRequestError(
                'a relational request has to be gathered first, and this '
                'one was not. Build it with '
                'PredictiveQuery(...).materialize(indices=[...], '
                "model='kumo-relational') -- a schema of one table is "
                'gathered for the tabular model unless the relational one is '
                'named -- rather than with Request(context=..., '
                'predict=...), which is the single-table form',
                details={'model': KUMO_RELATIONAL.model_id},
            )
        return payloads


_UNSUPPORTED_STYPE = re.compile(r"semantic type '([^']+)' is unsupported")

#: The other wording, from the deployment that counts rather than names.
_TOO_MANY_LABELS = re.compile(
    r'cardinality (\d+) exceeds limit (\d+)', re.IGNORECASE
)


def _columns_the_endpoint_means(
    error: Exception, payload: Mapping[str, Any]
) -> tuple[list[str], str]:
    r"""The columns a refusal is about, and which complaint it was."""
    said = str(error)
    named = _UNSUPPORTED_STYPE.search(said)
    if named is not None:
        return _columns_with_stype(payload, named.group(1)), 'stype'
    counted = _TOO_MANY_LABELS.search(said)
    if counted is None:
        return [], ''
    return _columns_over(payload, int(counted.group(2))), 'cardinality'


def _reworded(error: Exception, message: str) -> Exception:
    r"""The same refusal, saying more."""
    if isinstance(error, NimRequestError):
        return NimRequestError(
            error.status_code,
            code=error.code,
            message=message,
            details=error.details,
        )
    rebuilt = StructuredDataError(
        message,
        code=getattr(error, 'code', None),
        details=getattr(error, 'details', None),
    )
    status = getattr(error, 'status_code', None)
    if status is not None:
        # `fit()` reads this to decide whether an endpoint simply does not
        # serve sessions; losing it turns a clean fallback into a failure.
        rebuilt.status_code = status
    return rebuilt


def _columns_over(payload: Mapping[str, Any], limit: int) -> list[str]:
    r"""Every ``table.column`` holding more distinct values than ``limit``.

    The endpoint reports one number and the limit it broke, which says nothing
    about where to look. The rows it just refused do.

    Counted by the engine's own helper rather than here. What the guard counts
    is not every cell: it is string cells only, across the predict half as well
    as the context, reduced to the larger of the two -- and a text cell travels
    as a list of tokens, which it counts individually. Counting it a second way
    here would name columns the endpoint was not talking about, a numeric key
    among them.
    """
    try:
        from sdm_relational_engine.rfm.inference.payload import (
            high_cardinality_columns,
        )
    except ImportError:  # pragma: no cover - the extra is optional
        return []
    return [
        f'{table}.{column} ({count:,} values)'
        for table, column, count in high_cardinality_columns(
            dict(payload), limit
        )
    ]


def _asks_for_the_larger_context(
    payloads: tuple[dict[str, Any], ...], cap: int
) -> bool:
    r"""Whether this request selected the mode that will not explain.

    The endpoint gates explaining on ``inference.run_mode``, which the client
    derives from the *requested* ``num_context_samples``. Counting the gathered
    rows instead misses every graph that supplies less context than was asked
    for: 120 rows gathered under a request for 1,500 passed this guard and the
    endpoint's refusal escaped, naming a knob the caller never set.

    A payload the engine did not gather carries no ``run_mode``, and for that
    one the rows are all there is to read.
    """
    modes = {
        str(payload['inference']['run_mode'])
        for payload in payloads
        if isinstance(payload.get('inference'), Mapping)
        and payload['inference'].get('run_mode') is not None
    }
    if modes:
        return modes != {'fast'}
    return (
        sum(
            len(
                payload.get('context', {})
                .get('instance_table', {})
                .get('rows', [])
            )
            for payload in payloads
        )
        > cap
    )


_WIRE_TARGET = 'TARGET'
_A_PREDICTED_COLUMN = re.compile(
    r'^\s*PREDICT\s+([A-Za-z_]\w*\.[A-Za-z_]\w*)\s', re.IGNORECASE
)


def _how_to_name_the_target(column: str | None, asked: str | None) -> str:
    """What to call the thing every context row agrees about.

    `TARGET` is the name the relational wire gives the label column whatever
    the question was, so printing it describes a column of nobody's data.
    """
    if column != _WIRE_TARGET:
        return str(column)
    named = _A_PREDICTED_COLUMN.match(asked or '')
    if named is not None:
        return named.group(1)
    return f'answer to {asked!r}' if asked else 'answer'


def _answered_target_values(
    payload: Mapping[str, Any], column: Any
) -> set[Any] | None:
    r"""The distinct answers the context holds for ``column``, nulls aside.

    ``None`` when the context does not carry the target at all, or carries no
    answered row: neither is a context that teaches one thing.
    """
    table = payload.get('context', {}).get('instance_table', {})
    columns = table.get('columns') or []
    rows = table.get('rows') or []
    if not columns or not rows or column not in columns:
        return None
    position = columns.index(column)
    answered = {
        row[position]
        for row in rows
        if len(row) > position and row[position] is not None
    }
    return answered or None


def _distinct_labels(payload: Mapping[str, Any], column: Any) -> int | None:
    r"""How many distinct values the context holds for ``column``."""
    table = payload.get('context', {}).get('instance_table', {})
    columns = table.get('columns') or []
    rows = table.get('rows') or []
    if not columns or not rows or column not in columns:
        return None
    position = columns.index(column)
    return len({row[position] for row in rows if len(row) > position})


def _columns_with_stype(payload: Mapping[str, Any], stype: str) -> list[str]:
    r"""Every ``table.column`` in the payload declared as ``stype``.

    The endpoint refuses an unsupported semantic type without saying which
    column carries it, and a real dataset acquires them in bulk -- Northwind
    produces fifteen. The payload that was just refused knows.

    A relational schema is ``instance_table`` plus ``related_tables``, so both
    shapes are walked; the instance table's own name is not in the payload, and
    ``instance_table`` is what the wire calls it.
    """
    found = []
    schema = payload.get('schema') or {}
    tables = {'instance_table': schema.get('instance_table') or {}}
    tables.update(schema.get('related_tables') or {})
    for table_name, table in tables.items():
        if not isinstance(table, Mapping):
            continue
        for column_name, spec in (table.get('columns') or {}).items():
            if isinstance(spec, Mapping) and spec.get('stype') == stype:
                found.append(f'{table_name}.{column_name}')
    return sorted(found)


def _named_entities(
    items: tuple[Any, ...], payload: dict[str, Any] | None
) -> list[Any]:
    """Put the caller's entity back on each row, and check it belongs there.

    Entity values never travel: the engine sends an opaque reference and an
    internal instance id, so a response's ``id`` is the transport key rather
    than the user's key. The reference column carries the entity and the
    instance-id column carries the key the endpoint answers under, so the two
    together restore the mapping. It is read from the payload that produced
    this response, so a run split into several calls maps each batch against
    its own rows.

    **A mismatch is an error, not a silent pass.** Rewriting an id is exactly
    the operation that can misattribute an answer, so every row is checked
    against the batch it claims to belong to before its id is replaced.

    Forecasting is the one task that answers a row more than once, and only
    then: a row may repeat across steps, but not within one.
    """
    if payload is None:
        return list(items)
    table = payload.get('predict', {}).get('instance_table', {})
    columns = table.get('columns') or []
    rows = table.get('rows') or []
    reference = next(
        (name for name in columns if name.startswith('__kumo_entity_ref')),
        None,
    )
    if reference is None:
        return list(items)

    entities = [
        as_declared(
            row[columns.index(reference)], _key_dtype(payload, reference)
        )
        for row in rows
    ]
    instances = (
        [row[columns.index('instance_id')] for row in rows]
        if 'instance_id' in columns
        else None
    )
    forecasting = _forecast_task(payload)

    named: list[Any] = []
    seen: set[tuple[int, int | None]] = set()
    for item in items:
        index = item.row_index
        if index is None:
            raise StructuredDataError(
                'the response is missing a row_index, so its answer cannot be '
                'attributed to an entity',
                code='INVALID_RESPONSE',
            )
        if not 0 <= index < len(entities):
            raise StructuredDataError(
                f'the response names row_index {index}, and this batch asked '
                f'about {len(entities)} rows',
                code='INVALID_RESPONSE',
            )

        step = item.forecast_step
        if forecasting:
            if step is None:
                raise StructuredDataError(
                    f'a forecast answer for row {index} names no step, so its '
                    f'place in the horizon is unknown',
                    code='INVALID_RESPONSE',
                )
        elif step is not None:
            raise StructuredDataError(
                f'a {payload.get("task", {}).get("kind")!r} answer carries a '
                f'forecast step, which only a forecast has',
                code='INVALID_RESPONSE',
            )

        key = (index, step)
        if key in seen:
            where = (
                f'row {index} step {step}' if forecasting else f'row {index}'
            )
            raise StructuredDataError(
                f'the response answers {where} more than once',
                code='INVALID_RESPONSE',
            )
        seen.add(key)

        if instances is not None:
            if item.id is None:
                raise StructuredDataError(
                    f'the answer for row {index} carries no id, and this batch '
                    f'asked under {instances[index]!r}, so the two cannot be '
                    f'matched',
                    code='INVALID_RESPONSE',
                )
            if str(item.id) != str(instances[index]):
                raise StructuredDataError(
                    f'the response pairs row_index {index} with id '
                    f'{item.id!r}, and this batch asked about '
                    f'{instances[index]!r} there',
                    code='INVALID_RESPONSE',
                )
        named.append(dataclasses.replace(item, id=entities[index]))

    answered = {index for index, _ in seen}
    if answered != set(range(len(entities))):
        missing = sorted(set(range(len(entities))) - answered)
        raise StructuredDataError(
            f'the response left row(s) {missing} unanswered',
            code='INVALID_RESPONSE',
        )
    return named


def _forecast_task(payload: dict[str, Any]) -> bool:
    """Whether this batch asked a forecast.

    Read from the request rather than from the shape of the answer. Deciding
    it from the answer let a single item carrying a step switch off the
    duplicate guard for every other row in the response.
    """
    return payload.get('task', {}).get('kind') == 'forecasting'


def _key_dtype(payload: dict[str, Any], reference: str) -> str | None:
    r"""The dtype of the column the entity reference stands for.

    An integer wider than a double travels as a base-10 string by contract, so
    a key past 2**53 comes back as text unless what it was is looked up. The
    reference column itself is declared `string` -- it is a transport key --
    so the answer is on the table it points at.
    """
    schema = payload.get('schema', {})
    for link in schema.get('relationships', []):
        if list(link.get('source_columns') or []) != [reference]:
            continue
        target = link.get('target_table')
        columns = link.get('target_columns') or []
        if not target or len(columns) != 1:
            return None
        return (
            schema.get('related_tables', {})
            .get(target, {})
            .get('columns', {})
            .get(columns[0], {})
            .get('dtype')
        )
    return None
