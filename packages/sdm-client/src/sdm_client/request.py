# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""One request type, for every model.

A :class:`Request` is the wire envelope minus the two things that belong to the
call rather than the question: which model answers it, and which output fields
you want back. Both are arguments to
:meth:`~sdm_client.SDMClient.predict`.

It is not a new abstraction. The Universal TFM contract already unified the
single-table and relational cases under one envelope, and a tabular request is
simply the one where the schema has one table and no relationships.

Two builders produce it. This module holds the direct one:

>>> request = Request(
...     context=labelled_rows, predict=rows_to_score,
...     target='churn', task='classification',
... )

The other is a predictive query, which arrives in a later phase.

``Request`` is **singular**. A run that exceeds the endpoint's per-call cap is
split into several round trips by the adapter, because batching is transport
and does not belong in a type you read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sdm_client.errors import StructuredDataError
from sdm_client.schema import Schema, supports_dtype
from sdm_client.wire import UNTYPED, infer_tfm_dtype, is_container

# Task kinds the contract carries. A model narrows this further -- a tabular
# model takes only the coarse 'classification' -- and its adapter says so.
TASK_KINDS = (
    'classification',
    'binary_classification',
    'multiclass_classification',
    'regression',
    'forecasting',
    'temporal_link_prediction',
    'ranking',
)

_CLASSIFICATION_KINDS = frozenset(
    {'classification', 'binary_classification', 'multiclass_classification'}
)


@dataclass(frozen=True)
class Target:
    r"""The column being predicted, and the dtype it is declared as.

    The dtype has to match the schema's dtype for the same column exactly: the
    endpoint compares them and answers 422 when they differ, so this is taken
    from the schema rather than guessed at separately.
    """

    column_name: str
    dtype: str
    table_name: str | None = None


@dataclass(frozen=True)
class Task:
    r"""What is being asked.

    Attributes:
        kind: One of :data:`TASK_KINDS`.
        target: The column being predicted.
        entity_table: The table whose rows are the entities, when the question
            is relational. ``None`` for a single-table question.
        options: Task options the contract carries but this client does not
            model individually, such as ``anchor_time_column``.
    """

    kind: str
    target: Target
    entity_table: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def is_classification(self) -> bool:
        return self.kind in _CLASSIFICATION_KINDS


@dataclass(frozen=True)
class Split:
    r"""One half of a request: the instance rows, plus any related tables.

    ``context`` is the half where the answer is known; ``predict`` is the half
    being asked about. Both carry the same shape, which is what lets one type
    describe a single-table and a relational request alike.
    """

    instance_table: pd.DataFrame
    related_tables: dict[str, pd.DataFrame] = field(default_factory=dict)


class Request:
    r"""What to ask, independent of which model answers it.

    Attributes:
        task: The question.
        schema: The tables and how they relate.
        context: Rows where the answer is known.
        predict: Rows being asked about.
        inference: Gathering knobs, e.g. ``num_context_samples``. The adapter
            turns these into whatever the endpoint expects; ``run_mode`` is
            derived there rather than being something you set.
    """

    def __init__(
        self,
        context: pd.DataFrame,
        predict: pd.DataFrame,
        *,
        target: str,
        task: str,
        related_tables: Mapping[str, pd.DataFrame] | None = None,
        schema: Schema | None = None,
        entity_table: str = 'instance',
        inference: Mapping[str, Any] | None = None,
        task_options: Mapping[str, Any] | None = None,
    ) -> None:
        r"""Build a request from frames you already have.

        Args:
            context: Rows where the answer is known. Must contain ``target``.
            predict: Rows to score. Must not contain ``target``; these are the
                rows whose answer you do not have, including rows the model
                has never seen.
            target: The column being predicted.
            task: One of :data:`TASK_KINDS`. Stated, never inferred from the
                target's cardinality -- guessing there is what turns a
                two-class target into a multiclass one.
            related_tables: Tables the instance rows join to. Supplying these
                makes the request relational, so a single-table model will
                refuse it.
            schema: Override the inferred schema. Rarely needed.
            entity_table: The name the instance table is known by. Only visible
                in errors and in a query, so the default is fine.
            inference: Gathering knobs passed to the adapter.
            task_options: Task fields the contract carries that this client
                does not model individually.

        Raises:
            StructuredDataError: If a frame is not a DataFrame, ``task`` is not
                a known kind, ``context`` has no rows, ``target`` is missing
                from ``context``, or ``predict`` carries the target column.
        """
        context = _require_frame(context, 'context')
        predict = _require_frame(predict, 'predict')
        related = {
            name: _require_frame(frame, f'related_tables[{name!r}]')
            for name, frame in (related_tables or {}).items()
        }

        if task not in TASK_KINDS:
            raise StructuredDataError(
                f'task must be one of {list(TASK_KINDS)}, got {task!r}',
                code='INVALID_REQUEST',
            )
        if target in context.columns and _has_container_cells(context[target]):
            # Said here because this is the only place that knows which column
            # is the target. SDM's own `Task.classification` covers "binary,
            # multi-class, or multi-label", but the wire does not: `CellValue`
            # is documented as excluding array-valued prediction scalars on
            # purpose, so a multi-label answer has no shape to come back in.
            raise StructuredDataError(
                f'{target!r} holds a list per row, which is a multi-label '
                f'target. These models predict one value per row, and the '
                f'contract has no form for a set of labels, so this cannot be '
                f'asked or answered',
                code='UNSUPPORTED_REQUEST',
            )
        if len(context) == 0:
            # A zero-row `predict` is legal -- the contract sets no minimum on
            # rows, and an empty batch inside a loop is a reasonable no-op.
            # A zero-row context is not: a model that predicts from labelled
            # examples cannot be given none, and the endpoint answers a 500.
            raise StructuredDataError(
                'context has no rows, so there is nothing for the model to '
                'learn the answer from',
                code='INVALID_REQUEST',
            )
        if target not in context.columns:
            raise StructuredDataError(
                f'context has no target column {target!r}; it has '
                f'{sorted(map(str, context.columns))}',
                code='INVALID_REQUEST',
            )
        if target in predict.columns:
            raise StructuredDataError(
                f'predict carries the target column {target!r}. Those are the '
                f'rows whose answer is being asked for, so the column has to '
                f'be absent; drop it with predict.drop(columns=[{target!r}])',
                code='INVALID_REQUEST',
            )
        if context[target].notna().sum() == 0:
            raise StructuredDataError(
                'context must contain at least one row with a known target',
                code='INVALID_REQUEST',
            )

        missing = [
            column
            for column in predict.columns
            if column not in context.columns
        ]
        if missing:
            raise StructuredDataError(
                f'predict has column(s) {missing} that context does not; the '
                f'model is shown context first, so it cannot score a feature '
                f'it has never been given',
                code='INVALID_REQUEST',
            )

        if schema is None:
            schema = Schema.from_data(
                {entity_table: context, **related},
                relationships=None if related else (),
            )

        _check_scored_values_mean_what_the_schema_says(
            predict, schema, entity_table
        )

        # And the other direction: a feature the model was shown has to be
        # supplied for the rows being scored. The endpoint rejects a predict
        # half that is missing one, so name it here instead. A primary key is
        # identity rather than a feature, so scoring rows may omit it.
        identity = set(_as_key_tuple(schema[entity_table].primary_key))
        absent = [
            column
            for column in context.columns
            if column != target
            and column not in identity
            and column not in predict.columns
        ]
        if absent:
            raise StructuredDataError(
                f'predict is missing column(s) {absent} that context carries. '
                f'Every feature the model is shown has to be supplied for the '
                f'rows being scored; drop them from context, or supply them',
                code='INVALID_REQUEST',
            )

        self.schema = schema
        self.task = Task(
            kind=task,
            target=Target(
                column_name=target,
                dtype=schema[entity_table][target].dtype,
                table_name=entity_table,
            ),
            entity_table=entity_table if related else None,
            options=dict(task_options or {}),
        )
        self.context = Split(instance_table=context, related_tables=related)
        self.predict = Split(instance_table=predict, related_tables=related)
        self.inference = dict(inference or {})
        self.requested_inference = dict(inference or {})
        self.entity_table = entity_table

    @classmethod
    def materialized(
        cls,
        *,
        payloads: tuple[dict[str, Any], ...],
        schema: Schema,
        query: str,
        indices: list[Any],
        requested_inference: Mapping[str, Any] | None = None,
    ) -> Request:
        r"""A request whose data has already been gathered.

        The predictive-query path does its gathering in ``materialize()``, so
        what arrives here is one or more finished wire envelopes rather than
        frames to encode. More than one when the run exceeds the endpoint's
        per-call cap; the adapter sends each and the results are merged.

        This is still one request. Batching is transport, and a caller reads
        this object to see one question.
        """
        request = cls.__new__(cls)
        request.schema = schema
        first = payloads[0]
        target = first.get('task', {}).get('target', {})
        request.task = Task(
            kind=first.get('task', {}).get('kind', 'classification'),
            target=Target(
                column_name=target.get('column_name', ''),
                dtype=target.get('dtype', ''),
            ),
            entity_table=None,
            options=dict(first.get('task', {}).get('options', {})),
        )
        empty = pd.DataFrame()
        request.context = Split(instance_table=empty)
        request.predict = Split(instance_table=empty)
        request.inference = dict(first.get('inference', {}))
        request.entity_table = next(iter(schema), 'instance')
        request.query = query
        request.indices = indices
        request._payloads = payloads
        # What the caller asked for, as distinct from what the engine filled
        # in. An adapter may drop the engine's defaults silently; it must not
        # drop something that was asked for.
        request.requested_inference = dict(requested_inference or {})
        return request

    @property
    def payloads(self) -> tuple[dict[str, Any], ...] | None:
        r"""The gathered wire envelopes, when this came from a query."""
        return getattr(self, '_payloads', None)

    @property
    def is_relational(self) -> bool:
        r"""Whether this request needs a model that can read related tables."""
        return bool(
            self.context.related_tables
            or self.schema.relationships
            or len(self.schema) > 1
        )

    @property
    def context_rows(self) -> int:
        r"""How many rows the model will learn the answer from, in total.

        A relational request keeps its rows in the gathered envelopes rather
        than in ``context``, which stays an empty frame. Reading the frame
        reported nought for every graph request -- and these two counts are the
        only description of a request there is before it is sent, so the one
        shape whose size is worth checking was the one that would not say.

        Summed across envelopes, because a request gathered in several parts
        gives each part its own context: that is why a session refuses a split
        request rather than pinning the first one's.
        """
        return self._rows('context')

    @property
    def predict_rows(self) -> int:
        r"""How many rows are being asked about, across every call."""
        return self._rows('predict')

    def _rows(self, half: str) -> int:
        if self.payloads:
            return sum(
                _envelope_rows(payload, half) or 0 for payload in self.payloads
            )
        return len(getattr(self, half).instance_table)

    @property
    def nothing_to_score(self) -> bool:
        r"""Whether this request is known to ask about no rows at all.

        Distinct from ``predict_rows == 0``, and the distinction matters:
        `predict()` answers a request with nothing in it without sending, so a
        payload whose rows simply cannot be read must not be mistaken for one
        that is empty. An unreadable envelope is a request that should reach
        the endpoint and be refused, not one that quietly succeeds with an
        empty answer.
        """
        if self.payloads:
            counts = [
                _envelope_rows(payload, 'predict') for payload in self.payloads
            ]
            return all(count is not None for count in counts) and not any(
                counts
            )
        return len(self.predict.instance_table) == 0

    def __repr__(self) -> str:
        shape = 'relational' if self.is_relational else 'single-table'
        return (
            f'Request({self.task.kind}, target={self.task.target.column_name!r}, '
            f'{self.context_rows} context rows, {self.predict_rows} to score, '
            f'{shape})'
        )


def _as_key_tuple(key: Any) -> tuple[str, ...]:
    if key is None:
        return ()
    if isinstance(key, str):
        return (key,)
    return tuple(key)


def _has_container_cells(series: pd.Series) -> bool:
    r"""The same question the wire asks, so the two cannot disagree."""
    return any(is_container(value) for value in series)


def _require_frame(value: Any, name: str) -> pd.DataFrame:
    r"""Reject a table argument that is not a frame, naming the parameter.

    Every table here is used as a frame immediately, so a dict, a list of
    records or a Series -- the shapes a pandas user reaches for before building
    a frame -- would otherwise surface as an ``AttributeError`` from inside an
    adapter, which ``except StructuredDataError`` does not catch and which
    names neither the argument nor the type it should have been.
    """
    if not isinstance(value, pd.DataFrame):
        hint = (
            '; wrap a single row with pd.DataFrame([row])'
            if isinstance(value, (dict, pd.Series))
            else '; build one with pd.DataFrame(...)'
        )
        raise StructuredDataError(
            f'{name} must be a pandas DataFrame, got '
            f'{type(value).__name__}{hint}',
            code='INVALID_REQUEST',
        )
    return value


def _envelope_rows(payload: Mapping[str, Any], half: str) -> int | None:
    r"""How many instance rows one wire envelope carries, or `None`.

    Read defensively, and report not knowing as not knowing. These counts are
    what `repr` shows, so a request that cannot describe itself is worse than
    one that describes itself roughly -- but `nothing_to_score` decides whether
    to answer without sending, and there `None` and `0` are opposite answers.
    """
    if not isinstance(payload, Mapping):
        return None
    half_of = payload.get(half)
    if not isinstance(half_of, Mapping):
        return None
    instance_table = half_of.get('instance_table')
    if not isinstance(instance_table, Mapping):
        return None
    rows = instance_table.get('rows')
    # A list, specifically. `Sized` admits `{}` and `''`, which would be read
    # as an envelope carrying no rows rather than as one that cannot be read,
    # and those are the two answers this has to keep apart.
    return len(rows) if isinstance(rows, (list, tuple)) else None


def _check_scored_values_mean_what_the_schema_says(
    predict: pd.DataFrame, schema: Schema, entity_table: str | None
) -> None:
    r"""Refuse rows whose values contradict what their column means.

    The columns of the two halves are already checked against each other, by
    name, each with a message naming the fix. The values were not, and the
    adapter widens a column's dtype to cover both halves -- rightly, for the
    case it was written for: an all-null context column carries no type, and
    taking the context's answer alone would send the scored half's integers as
    text.

    Nothing bounded it. A column the schema declares `float64` meaning
    `numerical` could be widened to `string` by scored rows holding text, and
    `stype` is read from the schema on another path and is not widened with it,
    so the request went out declaring a column that is `string` and
    `numerical` at once -- which is not a thing a column can be. It was
    answered rather than refused: measured against the live endpoint on a
    target that is exactly three times a feature, the scored rows came back
    300.0, 329.9 and 299.7 where the truth is 300, 330 and 360. Plausible
    numbers, one wrong by a sixth, and nothing in the answer to say so.

    Only a contradiction is refused. Integers in a `float64` column and unseen
    labels in a `categorical` one still mean what the column means, and pass.

    Runs after the schema is settled rather than beside the column checks it
    belongs with: a request built by hand may carry no schema at all, and one
    is inferred from the context a few lines below. Checking before that ran
    left the hand-built request -- the one with no query to have validated it
    -- as the only one not covered.
    """
    if predict is None or entity_table is None or entity_table not in schema:
        # Not skipped merely for being empty: `pd.Series([], dtype='int64')`
        # still says int64, and a zero-row batch that declares a type the
        # column cannot mean would build the same contradictory envelope as a
        # full one. A column that genuinely carries no type reports UNTYPED
        # below and is skipped there.
        return
    table = schema[entity_table]
    for name in predict.columns:
        column = table.columns.get(name)
        if column is None:
            continue
        if column.untyped and not column.stype_declared:
            # No claim to contradict. An all-null context column is what
            # widening exists for, and the `categorical` it was given is a
            # placeholder rather than a finding, so the scored rows decide both
            # its type and its meaning. Once a caller has said what the column
            # means, that is a claim like any other.
            continue
        arriving = infer_tfm_dtype(predict[name])
        if arriving == UNTYPED or supports_dtype(column.stype, arriving):
            continue
        raise StructuredDataError(
            f'the rows being scored hold {arriving!r} in column {name!r}, and '
            f'the schema declares it {column.dtype!r} meaning '
            f'{column.stype!r}, which {arriving!r} cannot mean. Sent, the '
            f'column would be described as both at once and answered with '
            f'numbers that are wrong without saying so. Convert it before '
            f'scoring -- predict[{name!r}] = pd.to_numeric(predict[{name!r}]) '
            f'-- or say what the column really is with '
            f'schema[{entity_table!r}][{name!r}].stype = ...',
            code='INVALID_REQUEST',
        )
