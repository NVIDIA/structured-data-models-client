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

import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.decompose import Decomposition
from sdm_client.errors import StructuredDataError, UnsupportedRequestError
from sdm_client.schema import Schema, supports_dtype
from sdm_client.wire import UNTYPED, infer_tfm_dtype, is_container
from sdm_client.wire.keys import key_dtypes, keys_from_a_frame

if TYPE_CHECKING:
    from sdm_client._rust.results import CoreRequest

# Task kinds the contract carries. A model narrows this further: a tabular
# model takes only the coarse 'classification', and its adapter says so.
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


_A_QUOTED_LITERAL = re.compile(r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'""")


def _one_spelling_of_each_literal(text: str) -> str:
    r"""Rewrite a query's string literals to one spelling of the quote.

    ``channel="web"`` and ``channel='web'`` are one question.
    """

    def canonical(match: re.Match[str]) -> str:
        literal = match.group(0)
        inner = literal[1:-1]
        if literal[0] == "'":
            inner = inner.replace("\\'", "'").replace('"', '\\"')
        try:
            return json.dumps(json.loads(f'"{inner}"'))
        except json.JSONDecodeError:
            return literal

    return _A_QUOTED_LITERAL.sub(canonical, text)


def _what_the_context_holds(context: Split | None) -> int | None:
    r"""A value standing for the labelled rows, or ``None`` for gathered ones.

    Frames carry their own labels, so two requests over different ones ask
    different questions. A gathered context is sampled from the schema, so two
    materializations of one query differ in rows and not in question.
    """
    if context is None:
        return None
    frames = [context.instance_table, *context.related_tables.values()]
    total = hash(tuple(tuple(map(str, frame.columns)) for frame in frames))
    for frame in frames:
        try:
            total ^= int(pd.util.hash_pandas_object(frame, index=False).sum())
        except TypeError:
            total ^= hash((len(frame), len(frame.columns)))
    return total


@dataclass(frozen=True)
class Question:
    r"""What a request asks, apart from the rows it asks about.

    Assembled from the request rather than recorded by one constructor, so a
    request built from frames carries one too.
    """

    query: str | None
    kind: str
    target: str
    dtype: str
    entity_table: str | None
    context: int | None


def _judge_a_context_that_teaches_nothing(
    column: str, target: pd.Series, kind: str
) -> None:
    r"""Remark on a context whose every row says the same thing.

    Said rather than refused: this endpoint answers a single-class
    classification, returning that class at probability 1.0, so refusing it
    here turned a route that gathers into one that rejects what the route that
    does not gather is answered for. A context with no answered rows is left to
    ``Request``.

    Only of the two kinds one value in a column means something for. A
    forecast's target is a horizon and a ranking's is a candidate list, and one
    value there does not mean what it means here.
    """
    answered = target.dropna()
    if len(answered) == 0 or answered.nunique() >= 2:
        return
    if kind not in _CLASSIFICATION_KINDS and kind != 'regression':
        return

    said = (
        'a classification needs at least two classes to learn from, and the '
        'answer will be that one class at probability 1.0'
        if kind in _CLASSIFICATION_KINDS
        else 'nothing in the context distinguishes one answer from another, '
        'so the answer cannot say much'
    )
    warnings.warn(
        f'every context row has the same {column}, so {said}. If the rows '
        f'being scored are the ones that differ, score fewer of them; if '
        f'{column} is constant in the whole table, there is no context that '
        f'answers this question.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


def warn_of_rows_scored_from_their_own_answers(request: Request) -> None:
    r"""Say when the rows being scored sit in the context with their answers.

    Both gathered routes take such rows out of the context they send, and a
    session says so when it cannot. A request built from frames is handed both
    halves and did neither, so the model was shown the answer to the question
    it was being asked, which every other route here prevents.

    Asked where both halves travel together rather than where the request is
    built, because that is the only call at which the pairing is a leak. A
    request built to score against a *session's* context sends only its rows,
    and warning about a context that never travels made a correct call fail
    under warnings-as-errors.
    """
    if request.payloads:
        return
    schema, entity_table = request.schema, request.entity_table
    target = request.task.target.column_name
    if schema is None or entity_table is None or not target:
        return
    try:
        context = request.context.instance_table
        predict = request.predict.instance_table
    except Exception:  # noqa: BLE001 - a warning is never worth a failure
        return
    if target not in context.columns:
        return
    _warn_of_rows_scored_from_their_own_answers(
        context, predict, schema, entity_table, target
    )


def _warn_of_rows_scored_from_their_own_answers(
    context: pd.DataFrame,
    predict: pd.DataFrame,
    schema: Schema,
    entity_table: str,
    target: str,
) -> None:
    r"""The comparison itself, given both halves as frames."""
    key = _as_key_tuple(schema[entity_table].primary_key)
    # Every component or none. Reducing a composite key to the columns both
    # frames happen to carry compares a part of an identity, and two rows that
    # share a part are not the same row: one tenant's entity 1 was reported as
    # another tenant's, scored from an answer that is not its own.
    if not key or any(
        column not in context.columns or column not in predict.columns
        for column in key
    ):
        return
    key = list(key)
    answered = context[context[target].notna()]
    if answered.empty:
        return
    # Compared as the wire spells them, through the encoder that writes them:
    # a key column whose two halves hold different python types is one column
    # written one way, so values that differ here are the same value by the
    # time the model reads them. One dtype for the whole request, not one per
    # half. Settled per half, the same entity came out `1.0` on the side
    # read from the context and `1` on the side being scored, and the
    # comparison reported overlaps the encoded request does not have.
    table = schema[entity_table]
    dtypes = key_dtypes(table, key, predict)
    answers = set(keys_from_a_frame(answered, key, dtypes))
    if not answers:
        return
    # Rows, not keys: the sentence counts rows being scored, and two rows
    # asking about one entity are two rows whose answer is being shown.
    shown = sum(
        1
        for named in keys_from_a_frame(predict, key, dtypes)
        if named in answers
    )
    if not shown:
        return
    warnings.warn(
        f'{shown} of the {len(predict)} rows being scored are also in '
        f'context, carrying the {target!r} being asked for, so the model is '
        f'being shown the answers to rows it is being asked about. Drop them '
        f'from context, or gather the request with materialize(), which '
        f'leaves the rows it scores out of the context it sends.',
        category=SDMWarning,
        stacklevel=caller_stacklevel(),
    )


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
        query: The predictive query this was built from, when it was built
            from one, and ``None`` when it was built from frames.
    """

    query: str | None = None

    # How this question was rewritten, when it had to be: the kind of question
    # that was asked, what each part asks after (a candidate or a step) as the
    # caller gave it, and how many candidates a ranking keeps. Re-deriving any
    # of this from the query text would read a key as a string, and would call
    # a forecast over a table named `rank_events` a ranking. Internal, because
    # it is how the client puts the parts' answers together, not something a
    # caller reads: `parts` and `asked` say what a caller needs to see.
    _decomposition: Decomposition | None = None

    # What a request materialized on the core keeps for predict on the core
    # (``sdm_client._rust.results.CoreRequest``, T3.16): ``None`` for one
    # the core did not materialize, which predict on the core predicts from
    # what it holds, as the old client does (T3.26).
    _core: CoreRequest | None = None
    _direct: Any = None

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
                target's cardinality. Guessing there is what turns a
                two-class target into a multiclass one.
            related_tables: Refused. A relational request is gathered by
                ``PredictiveQuery(...).materialize(...)``; no model can be
                sent one built from frames here.
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
            UnsupportedRequestError: If ``related_tables`` is given.
        """
        from sdm_client import _routing

        if _routing.direct_constructor_on_core():
            from sdm_client._rust.prepare import construct

            construct(
                self,
                context,
                predict,
                target=target,
                task=task,
                related_tables=related_tables,
                schema=schema,
                entity_table=entity_table,
                inference=inference,
                task_options=task_options,
            )
            return
        context = _require_frame(context, 'context')
        predict = _require_frame(predict, 'predict')
        related = {
            name: _require_frame(frame, f'related_tables[{name!r}]')
            for name, frame in (related_tables or {}).items()
        }
        _refuse_tables_no_model_can_be_given(related)

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
            # A zero-row `predict` is legal. The contract sets no minimum on
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

        _judge_a_context_that_teaches_nothing(target, context[target], task)

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
        self._context = Split(instance_table=context, related_tables=related)
        self._predict = Split(instance_table=predict, related_tables=related)
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
        entity_table: str | None = None,
        windowed: bool = False,
        requested_inference: Mapping[str, Any] | None = None,
    ) -> Request:
        r"""A request whose data has already been gathered.

        The predictive-query path does its gathering in ``materialize()``, so
        what arrives here is one or more finished wire envelopes rather than
        frames to encode. More than one when the run exceeds the endpoint's
        per-call cap; the adapter sends each and the results are merged.

        This is still one request. Batching is transport, and a caller reads
        this object to see one question.

        `entity_table` is named by the query rather than taken from whichever
        table the caller's dict happened to hold first. Every pre-send
        description of a relational request read it from insertion order, so
        it answered correctly or incorrectly depending on how the schema was
        built, and a programmatic caller could not use it as a guard.

        It is optional because it was not there to pass before, and a caller
        who does not name it is not asking for insertion order back: the
        envelopes carry the query's own answer, which is read instead.
        """
        request = cls.__new__(cls)
        request._windowed = windowed
        if entity_table is None:
            entity_table = _entity_table_in(payloads)
        request.schema = schema
        first = payloads[0]
        target = first.get('task', {}).get('target', {})
        request.task = Task(
            kind=first.get('task', {}).get('kind', 'classification'),
            target=Target(
                column_name=target.get('column_name', ''),
                dtype=target.get('dtype', ''),
            ),
            entity_table=entity_table,
            options=dict(first.get('task', {}).get('options', {})),
        )
        request._context = None
        request._predict = None
        request.inference = dict(first.get('inference', {}))
        request.entity_table = entity_table
        request.query = query
        request.indices = indices
        request._payloads = payloads
        # What the caller asked for, as distinct from what the engine filled
        # in. An adapter may drop the engine's defaults silently; it must not
        # drop something that was asked for.
        request.requested_inference = dict(requested_inference or {})
        return request

    @property
    def _question(self) -> Question:
        r"""What this request asks. See :class:`Question`."""
        target = self.task.target
        query = self.query
        return Question(
            query=None
            if query is None
            else _one_spelling_of_each_literal(query),
            kind=self.task.kind,
            target=target.column_name,
            dtype=target.dtype,
            entity_table=self.entity_table,
            context=_what_the_context_holds(self._context),
        )

    @property
    def context(self) -> Split:
        r"""The rows where the answer is known.

        Read from the wire envelopes on the route that keeps its data as
        envelopes, and on first access rather than while materializing: the
        adapter sends those envelopes as they are, so building the frames
        eagerly would copy every context row of every request on a path where
        nothing reads them.
        """
        return self._half('context')

    @context.setter
    def context(self, value: Split) -> None:
        self._refuse_if_gathered('context')
        self._context = value

    @property
    def predict(self) -> Split:
        r"""The rows being asked about. Built the way `context` is."""
        return self._half('predict')

    @predict.setter
    def predict(self, value: Split) -> None:
        self._refuse_if_gathered('predict')
        self._predict = value

    def _half(self, half: str) -> Split:
        r"""One half, read from the envelopes that carry it.

        Read on access rather than while materializing, because the adapter
        sends those envelopes as they are and building the frames eagerly
        would copy every row of every request on the path where nothing looks
        at them.

        Not held between reads. The envelopes are edited in place when leaked
        context rows are taken out of them, so a kept frame would go on
        describing rows the request no longer sends, and a cache correct only
        for the edits made today is a trap for the ones made later. Reading is
        a few milliseconds per thousand rows, against a call that spends
        hundreds on the wire.

        Where there are envelopes they are the request, and they are what is
        read, in step with the counts beside them and with what the adapter
        sends. A half stored alongside them cannot have been put there through
        the setter, which refuses exactly that, but an object restored from an
        older pickle carries one; reading it would describe rows that are
        neither counted nor sent.

        Which makes these a description of the request rather than a handle
        into it. Editing a frame read from here changes what is sent only on
        the route that happens to keep its data that way; on the other it is
        read from the envelopes again and the edit is not there. Assigning the
        half is what changes the request, and does so on both.
        """
        payloads = self.payloads
        if payloads:
            return _split_from(payloads, half)
        held = getattr(self, f'_{half}', None)
        return held if held is not None else Split(pd.DataFrame())

    def _refuse_if_gathered(self, half: str) -> None:
        r"""Refuse to replace a half of a request that is already gathered.

        A gathered request holds finished wire envelopes, and the adapter
        sends those. An envelope carries what a frame cannot: the entity
        references the rows are addressed by, and the anchor each is asked
        at. A `Split` put here could not be turned back into one. It
        would change what the request reports and not what it sends, which is
        worse than not being allowed: the object would describe rows the model
        never sees.

        Gathered is read the same way here as everywhere else on this object,
        so that the route a request is on is one answer rather than four.
        """
        if not self.payloads:
            return
        raise StructuredDataError(
            f'{half} cannot be replaced on a request gathered by a predictive '
            f'query: its rows are already encoded, and the encoding carries '
            f'what a frame does not. Materialize the query again for the rows '
            f'you want instead.',
            code='INVALID_REQUEST',
        )

    @property
    def parts(self) -> tuple[Request, ...] | None:
        r"""The requests this one is asked as, when it is asked as several.

        A ranking and a forecast have no task on the endpoint, so each is
        asked as a set of questions it does serve, one per candidate and one
        per step. They are kept together here because the answer that was
        asked for is the one they make between them, not any one of them.
        """
        return getattr(self, '_parts', None)

    @property
    def asked(self) -> str | None:
        r"""The query a caller wrote, when this was gathered as several."""
        return getattr(self, '_asked', None)

    @property
    def target_moves_with_time(self) -> bool:
        r"""Whether the answer is a window rather than a standing value.

        An aggregation is a window: the same entity asked about at a later
        moment is a different question, and the answer to one is not the
        answer to the other. A column is not: it has one value, and whoever
        has seen it has the answer whenever they are asked.

        Which is the difference between a context that has answered the
        question and one that merely mentions the same entity, so anything
        deciding whether a row was already given away has to know it.
        """
        return getattr(self, '_windowed', False)

    @property
    def payloads(self) -> tuple[dict[str, Any], ...] | None:
        r"""The gathered wire envelopes, when this came from a query.

        No envelopes and never having been gathered are the same request, and
        are reported the same way. They were told apart once, so an empty
        tuple read as gathered here and as not gathered where the rows are,
        and an object could belong to neither route: the halves it reported
        came from its frames while the adapters refused it for carrying
        envelopes.
        """
        mine = getattr(self, '_payloads', None) or None
        parts = self.parts
        if parts is None:
            return mine
        # A question asked as several carries all of their envelopes: this is
        # the object the caller holds, and its size is the size of what will
        # be sent. Reporting one part's would undercount by as many parts as
        # there are, which for a ranking is one per candidate.
        gathered = tuple(
            envelope for part in parts for envelope in part.payloads or ()
        )
        return gathered or mine

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
        reported nought for every graph request. These two counts are the
        only description of a request there is before it is sent, so the one
        shape whose size is worth checking was the one that would not say.

        Summed across envelopes, because a request gathered in several parts
        gives each part its own context: that is why a session refuses a split
        request rather than pinning the first one's.

        A sum is not what any one call sends, and the endpoint's cap is per
        call. A request batched for size gathers a context for each batch,
        around that batch's own rows, so this counts every one of them;
        `_context_rows_per_call` is the largest, which is the figure the cap
        is held against, and `__repr__` says both.
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

    def _context_rows_per_call(self) -> int:
        r"""The most context rows any one call of this request sends.

        The endpoint caps the context of each call, and a request gathered in
        several calls sends a context with each, so ``context_rows``, their
        sum, is not what any call carries.
        """
        if self.payloads:
            return max(
                _envelope_rows(payload, 'context') or 0
                for payload in self.payloads
            )
        return len(self.context.instance_table)

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
        r"""What this request asks, and how much of it travels per call.

        A request gathered in several calls carries a context in each, so the
        total alone reads as a context far wider than any call carries. The
        number of calls is said where there is more than one, and the context
        is the widest one call's, since that is what the cap is held to.
        """
        shape = 'relational' if self.is_relational else 'single-table'
        calls = len(self.payloads or ())
        context = (
            f'up to {self._context_rows_per_call()} context rows per call '
            f'across {calls} calls'
            if calls > 1
            else f'{self.context_rows} context rows'
        )
        return (
            f'Request({self.task.kind}, target={self.task.target.column_name!r}, '
            f'{context}, {self.predict_rows} to score, '
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
    records or a Series, the shapes a pandas user reaches for before building
    a frame, would otherwise surface as an ``AttributeError`` from inside an
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


def _refuse_tables_no_model_can_be_given(
    related: Mapping[str, pd.DataFrame],
) -> None:
    r"""Refuse related tables handed to the builder that cannot carry them.

    Neither model can be sent a request built this way. The relational one is
    sent gathered envelopes, which contain neighbourhoods sampled around each
    entity with columns folded onto the instance table, and refuses frames. The
    tabular one reads one table and would drop these without saying so. So the
    argument took frames, built an object, and every path that could have sent
    it said no, one call later than the mistake.

    Said here, where the tables are offered, and naming the builder that does
    gather them.
    """
    if not related:
        return
    raise UnsupportedRequestError(
        f'related_tables {sorted(related)} makes this a relational request, '
        f'and no model can be sent one built from frames: the relational '
        f'model is sent tables gathered around each entity, and the tabular '
        f'model reads a single table. Gather it with '
        f'PredictiveQuery(...).materialize(indices=[...], '
        f"model='kumo-relational'), whose schema holds the related tables and "
        f'the relationships that reach them.',
        details={'related_tables': sorted(related)},
    )


def _entity_table_in(payloads: Sequence[Mapping[str, Any]]) -> str:
    r"""The table the query asks about, as the envelopes record it.

    `task.entity_table_names` is the query's own answer and is what the
    caller would otherwise have passed. Read here so that a caller who
    predates the argument gets that rather than the insertion order this
    replaced.
    """
    for payload in payloads:
        task = payload.get('task') if isinstance(payload, Mapping) else None
        named = (task or {}).get('entity_table_names')
        if isinstance(named, (list, tuple)) and named:
            return str(named[0])
    return ''


def _wire_frame(table: Any) -> pd.DataFrame | None:
    r"""One wire table as a frame, or nothing if it cannot be read as one."""
    if not isinstance(table, Mapping):
        return None
    columns = table.get('columns')
    rows = table.get('rows')
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None
    try:
        return pd.DataFrame(rows, columns=columns)
    except (TypeError, ValueError, OverflowError):
        # A row wider than its header raises rather than padding, and a
        # number too large for pandas to hold raises as well. Reading a
        # request is a description of it, and a description that cannot be
        # made is not worth failing the call for.
        return None


#: What ties a related row to the instance it hangs off.
INSTANCE_ID = 'instance_id'


def _is_a_number(value: Any) -> bool:
    r"""Whether ``value`` is a number that can be moved along."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not pd.isna(value)
        # An infinity is a number that no other can be moved past, and
        # rounding one has no answer at all. An integer too large to ask the
        # question of is the same case, and asking raises.
        and _is_finite(value)
    )


def _is_finite(value: int | float) -> bool:
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _instance_bounds(
    frame: pd.DataFrame | None,
) -> tuple[int, int | None]:
    r"""How far an ``instance_id`` column reaches, rounded outwards.

    Outwards because these bound a shift: truncating toward zero would round
    a floor of `-0.5` up to `0` and leave the shift half a step short, which
    is enough for two envelopes to land on the same id.

    Only the values a shift can move are measured. Reading `'7'` as seven
    would let it decide how far the next envelope moves while staying where
    it is itself, so it would set the distance for a journey it does not
    make.
    """
    if frame is None or INSTANCE_ID not in frame or frame.empty:
        return -1, None
    movable = [value for value in frame[INSTANCE_ID] if _is_a_number(value)]
    if not movable:
        return -1, None
    return math.ceil(max(movable)), math.floor(min(movable))


def _past(frame: pd.DataFrame | None, shift: int) -> pd.DataFrame | None:
    r"""``frame`` with its numeric ``instance_id`` moved along by ``shift``.

    Only the numbers move. An id that is not one cannot be moved past
    anything, and replacing it with the nothing it coerces to would lose the
    value rather than relocate it. It is left where it is, still tying
    its related rows to the instance it always named.
    """
    if frame is None or not shift or INSTANCE_ID not in frame:
        return frame
    moved = frame.copy()
    moved[INSTANCE_ID] = moved[INSTANCE_ID].map(
        lambda value: value + shift if _is_a_number(value) else value
    )
    return moved


def _split_from(payloads: Sequence[Mapping[str, Any]], half: str) -> Split:
    r"""The rows of one half, gathered from the envelopes that carry them.

    A relational request keeps its data as finished wire envelopes, and these
    two attributes are how a caller reads what it holds. Leaving them empty
    described the shape of the storage rather than the request: a relational
    `context` reported no rows while `context_rows` on the same object
    reported several hundred, so the object answered its own question twice
    and differently.

    Concatenated across envelopes, because more than one is batching and
    batching is transport. A caller reads this object to see one question, so
    the halves it exposes are the whole of each half.

    An `instance_id` that is not a number is left exactly as it is. There is
    no arithmetic that moves it past another, so two envelopes using the same
    one stay indistinguishable here, as they already were. Two envelopes
    sharing an `instance_id` are a malformed envelope, not something this
    can repair.
    """
    instances: list[pd.DataFrame] = []
    related: dict[str, list[pd.DataFrame]] = {}
    already = 0
    first = True
    for payload in payloads:
        half_of = payload.get(half) if isinstance(payload, Mapping) else None
        if not isinstance(half_of, Mapping):
            continue
        frame = _wire_frame(half_of.get('instance_table'))
        tables = half_of.get('related_tables')
        parts = [
            (name, _wire_frame(table))
            for name, table in (
                tables.items() if isinstance(tables, Mapping) else ()
            )
        ]
        # Each envelope numbers its rows from zero, and `instance_id` is what
        # ties a related row to the instance it hangs off. Stacking them
        # unchanged puts the second envelope's row 0 beside the first's, so a
        # join that was exact within one envelope silently crosses between
        # them. The first envelope is left as it arrived; each one after has
        # its lowest id put where the numbers so far ran out, which an offset
        # alone did not do: an envelope numbering from below zero landed
        # back among ids already taken however far the offset had advanced.
        numbered = [frame, *(table for _, table in parts)]
        bounds = [_instance_bounds(t) for t in numbered]
        highest = max((top for top, _ in bounds), default=-1)
        lowest = min(
            (floor for _, floor in bounds if floor is not None), default=None
        )
        shift = 0 if first or lowest is None else already - lowest
        if (moved := _past(frame, shift)) is not None:
            instances.append(moved)
        for name, table in parts:
            if (shifted := _past(table, shift)) is not None:
                related.setdefault(name, []).append(shifted)
        # An envelope that carried no numbered rows has not taken a place in
        # the sequence: it reserves none of it, and the next one that does is
        # still the first. Counting it moved the first readable envelope off
        # the numbers it arrived with, which is the one thing this promises
        # not to do.
        if lowest is not None:
            seen = 0 if frame is None else len(frame)
            already = max(already + seen, highest + shift + 1)
            first = False

    def joined(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)

    return Split(
        instance_table=joined(instances),
        related_tables={name: joined(f) for name, f in related.items()},
    )


def _envelope_rows(payload: Mapping[str, Any], half: str) -> int | None:
    r"""How many instance rows one wire envelope carries, or `None`.

    Read defensively, and report not knowing as not knowing. These counts are
    what `repr` shows, so a request that cannot describe itself is worse than
    one that describes itself roughly. However, `nothing_to_score` decides
    whether
    to answer without sending, and there `None` and `0` are opposite answers.

    A list, specifically: `Sized` admits `{}` and `''`, which would read as an
    envelope carrying no rows rather than as one that cannot be read, and
    those are the two answers this has to keep apart.
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
    return len(rows) if isinstance(rows, (list, tuple)) else None


def _check_scored_values_mean_what_the_schema_says(
    predict: pd.DataFrame, schema: Schema, entity_table: str | None
) -> None:
    r"""Refuse rows whose values contradict what their column means.

    The columns of the two halves are already checked against each other, by
    name, each with a message naming the fix. The values were not, and the
    adapter rightly widens a column's dtype to cover both halves for the
    case it was written for: an all-null context column carries no type, and
    taking the context's answer alone would send the scored half's integers as
    text.

    Nothing bounded it. A column the schema declares `float64` meaning
    `numerical` could be widened to `string` by scored rows holding text, and
    `stype` is read from the schema on another path and is not widened with it,
    so the request went out declaring a column that is `string` and
    `numerical` at once, which is not a thing a column can be. It was
    answered rather than refused: measured against the live endpoint on a
    target that is exactly three times a feature, the scored rows came back
    300.0, 329.9 and 299.7 where the truth is 300, 330 and 360. Plausible
    numbers, one wrong by a sixth, and nothing in the answer to say so.

    Only a contradiction is refused. Integers in a `float64` column and unseen
    labels in a `categorical` one still mean what the column means, and pass.

    Runs after the schema is settled rather than beside the column checks it
    belongs with: a request built by hand may carry no schema at all, and one
    is inferred from the context a few lines below. Checking before that ran
    left the hand-built request, which has no query to validate it, as the only
    one not covered.

    A column is not skipped merely for holding nothing. `pd.Series([],
    dtype='int64')` still says int64, and a zero-row batch declaring a type
    the column cannot mean builds the same contradictory envelope as a full
    one. A column that genuinely carries no type reports UNTYPED and is
    skipped for that reason instead. An all-null context column, by contrast,
    has no claim to contradict: the `categorical` it was given is a
    placeholder rather than a finding, so the scored rows decide both its type
    and its meaning. Once a caller has said what the column means, that is a
    claim like any other.
    """
    if predict is None or entity_table is None or entity_table not in schema:
        return
    table = schema[entity_table]
    for name in predict.columns:
        column = table.columns.get(name)
        if column is None:
            continue
        if column.untyped and not column.stype_declared:
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
            f'scoring: predict[{name!r}] = pd.to_numeric(predict[{name!r}]) '
            f'or say what the column really is with '
            f'schema[{entity_table!r}][{name!r}].stype = ...',
            code='INVALID_REQUEST',
        )
