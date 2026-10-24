# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core's results, as this host reads them (T3.16, the structure
design's fix D and rule 10).

Every result a core run finishes with (the contract's ``OperationResult``)
is decoded once, by :func:`decode`, into the frozen dataclasses here; the
public objects are built from these, never from objects shaped like the
old engine's, and a result of a shape the contract does not hold is
refused by name (:class:`~sdm_client._rust.codec.CodecError`), never read as
it is. What the host keeps for the core between calls is one typed value it
hands back whole: a column's :class:`SourceFacts`, and a request's
:class:`CoreRequest`.

Java has records with the same fields; both read the samples in
``rust/bindings/sdm-py/tests/messages.py``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, TypeVar

from sdm_client._rust.codec import CodecError
from sdm_client.errors import StructuredDataError

__all__ = [
    'Batch',
    'CorePart',
    'CoreParts',
    'CoreQuery',
    'CoreRanking',
    'CoreRequest',
    'Described',
    'Diagnostic',
    'Edited',
    'Gathered',
    'GatheredState',
    'Part',
    'Predicted',
    'QueryFacts',
    'Ranking',
    'SourceFacts',
    'Step',
    'Validated',
    'ViewNote',
    'decode',
    'expect',
]


def _fields(
    value: Any,
    where: str,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    r"""``value``, an object of exactly ``required`` and some of
    ``optional``; refused naming ``where`` otherwise."""
    if not isinstance(value, Mapping):
        raise CodecError(f'{where} is not an object')
    keys = set(value)
    if not set(required) <= keys or not keys <= set(required) | set(optional):
        raise CodecError(f'{where} does not hold the fields the contract does')
    return {key: value.get(key) for key in (*required, *optional)}


def _int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CodecError(f'{where} is not a count')
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise CodecError(f'{where} is not text')
    return value


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise CodecError(f'{where} is not a boolean')
    return value


def _maybe(value: Any, read: Any, where: str) -> Any:
    return None if value is None else read(value, where)


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise CodecError(f'{where} is not a list')
    return value


# --- What the source said of a column ------------------------------------------


@dataclasses.dataclass(frozen=True)
class SourceFacts:
    r"""What the source said of one column (the contract's
    ``SourceFacts``), which ``Column._source`` keeps and the host hands back
    whole: the relation it was read from (the contract's ``Relation``, as
    JSON; the core reads afresh a column of a table pointed at another
    since), the data type the source's catalog and values give it, the type
    the provider handed over (the contract's ``ColumnType``, as JSON),
    whether it may hold no value, whether a key of it alone keeps it
    unique, a big integer's declared type, and whether it is a column of
    exact numbers."""

    relation: Any
    dtype: str
    received: Any
    nullable: bool
    unique: bool
    declared_type: str | None = None
    exact_number: str | None = None

    @classmethod
    def from_contract(cls, value: Any) -> SourceFacts:
        r"""The facts as the contract writes them."""
        read = _fields(
            value,
            "a column's source facts",
            ('relation', 'dtype', 'received', 'nullable', 'unique'),
            ('declared_type', 'exact_number'),
        )
        return cls(
            relation=read['relation'],
            dtype=_text(read['dtype'], 'dtype'),
            received=read['received'],
            nullable=_flag(read['nullable'], 'nullable'),
            unique=_flag(read['unique'], 'unique'),
            declared_type=_maybe(read['declared_type'], _text, 'declared_type'),
            exact_number=_maybe(read['exact_number'], _text, 'exact_number'),
        )

    @classmethod
    def of_described(
        cls, column: Mapping[str, Any], relation: Any
    ) -> SourceFacts:
        r"""The facts of a column of the contract's ``SourceDescription``,
        of a table read from ``relation``."""
        return cls.from_contract(
            {
                'relation': relation,
                **{
                    key: column[key]
                    for key in (
                        'dtype',
                        'received',
                        'nullable',
                        'unique',
                        'declared_type',
                        'exact_number',
                    )
                    if key in column
                },
            }
        )

    def to_contract(self) -> dict[str, Any]:
        r"""The facts as the contract writes them, a field left out where
        none."""
        written: dict[str, Any] = {
            'relation': self.relation,
            'dtype': self.dtype,
            'received': self.received,
            'nullable': self.nullable,
            'unique': self.unique,
        }
        if self.declared_type is not None:
            written['declared_type'] = self.declared_type
        if self.exact_number is not None:
            written['exact_number'] = self.exact_number
        return written


# --- validate_query --------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QueryFacts:
    r"""What the core resolved of a query (the contract's ``QueryFacts``)."""

    entity_table: str
    problem_type: str | None
    windowed: bool
    num_forecasts: int
    text: str | None

    @classmethod
    def from_contract(cls, value: Any) -> QueryFacts:
        read = _fields(
            value,
            "a query's facts",
            ('entity_table', 'windowed', 'num_forecasts'),
            ('problem_type', 'text'),
        )
        return cls(
            entity_table=_text(read['entity_table'], 'entity_table'),
            problem_type=_maybe(read['problem_type'], _text, 'problem_type'),
            windowed=_flag(read['windowed'], 'windowed'),
            num_forecasts=_int(read['num_forecasts'], 'num_forecasts'),
            text=_maybe(read['text'], _text, 'text'),
        )


@dataclasses.dataclass(frozen=True)
class Diagnostic:
    r"""One of the core's diagnostics of a query: its code, message and
    severity (``error`` or ``warning``)."""

    code: str
    message: str
    severity: str


@dataclasses.dataclass(frozen=True)
class Validated:
    r"""``validate_query``'s result."""

    accepted: bool
    diagnostics: tuple[Diagnostic, ...]
    query: QueryFacts | None


def _validated(read: dict[str, Any]) -> Validated:
    diagnostics = []
    for each in _list(read['diagnostics'], 'diagnostics'):
        found = _fields(
            each, 'a diagnostic', ('code', 'message', 'severity'), ('span',)
        )
        diagnostics.append(
            Diagnostic(
                code=_text(found['code'], 'code'),
                message=_text(found['message'], 'message'),
                severity=_text(found['severity'], 'severity'),
            )
        )
    return Validated(
        accepted=_flag(read['accepted'], 'accepted'),
        diagnostics=tuple(diagnostics),
        query=(
            None
            if read['query'] is None
            else QueryFacts.from_contract(read['query'])
        ),
    )


# --- describe_source and edit_schema -------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Described:
    r"""``describe_source``'s result: the attachment of the
    ``SourceDescription``, and a semantic model's conversion notes
    (contract v28, T3.27), each a :class:`ViewNote`, in order; none for any
    other source."""

    description: int
    notes: tuple[ViewNote, ...] = ()


@dataclasses.dataclass(frozen=True)
class ViewNote:
    r"""A note of a semantic model's conversion (the contract's
    ``ViewNote``): its sentence and fields, the schema's table it is about
    (``None`` for one of the whole model) and the time column it says was
    taken, so the host shows it only while the schema still holds that."""

    sentence: str
    fields: dict[str, Any]
    table: str | None = None
    time_column: str | None = None


def _note(value: Any) -> ViewNote:
    read = _fields(
        value, 'a view note', ('sentence',), ('fields', 'table', 'time_column')
    )
    fields = read.get('fields', {})
    if not isinstance(fields, Mapping):
        raise CodecError("a view note's fields are not an object")
    return ViewNote(
        sentence=_text(read['sentence'], "a view note's sentence"),
        fields=dict(fields),
        table=_maybe(read.get('table'), _text, "a view note's table"),
        time_column=_maybe(
            read.get('time_column'), _text, "a view note's time column"
        ),
    )


def _described(read: dict[str, Any]) -> Described:
    return Described(
        description=_int(read['description'], 'description'),
        # Left out where there is none, which the reading gives as None.
        notes=tuple(
            _note(note) for note in _list(read.get('notes') or [], 'notes')
        ),
    )


@dataclasses.dataclass(frozen=True)
class Edited:
    r"""``edit_schema``'s result: the attachment of the schema with the
    edit made (a ``HeldSchema``), and the column whose meaning the caller
    chose, as ``(table, column)``."""

    schema: int
    declared: tuple[str, str] | None


def _edited(read: dict[str, Any]) -> Edited:
    declared = read['declared']
    if declared is not None:
        found = _fields(declared, 'the column declared', ('table', 'column'))
        declared = (
            _text(found['table'], 'table'),
            _text(found['column'], 'column'),
        )
    return Edited(schema=_int(read['schema'], 'schema'), declared=declared)


# --- materialize -----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Step:
    r"""A forecast's later step: its number, and the attachments of its
    payload and query."""

    step: int
    payload: int
    query: int


@dataclasses.dataclass(frozen=True)
class Batch:
    r"""A later call of a request asked in several: its payload's
    attachment, and the rows it scores."""

    payload: int
    rows: int


@dataclasses.dataclass(frozen=True)
class Part:
    r"""A ranking's later question: its candidate's position among those
    given and name (the JSON text of its value), and the attachments of its
    payload and query."""

    candidate: int
    value: str
    payload: int
    query: int


@dataclasses.dataclass(frozen=True)
class Ranking:
    r"""A ranking asked as one question per candidate: how many a row
    keeps, the first question's candidate and name, and the later ones."""

    top_k: int | None
    candidate: int
    value: str
    parts: tuple[Part, ...]


@dataclasses.dataclass(frozen=True)
class GatheredState:
    r"""The request state ``materialize`` wrote (the contract's
    ``RequestState``), its attachments by id."""

    version: int
    payload: int
    query: int
    forecast: tuple[Step, ...] | None
    batches: tuple[Batch, ...] | None
    ranking: Ranking | None


@dataclasses.dataclass(frozen=True)
class Gathered:
    r"""``materialize``'s result: the request state, and what the core
    resolved of the query."""

    state: GatheredState
    query: QueryFacts | None


def _state(value: Any) -> GatheredState:
    read = _fields(
        value,
        'the request state',
        ('version', 'payload', 'query'),
        ('forecast', 'batches', 'ranking'),
    )
    forecast = None
    if read['forecast'] is not None:
        steps = []
        for each in _list(read['forecast'], 'forecast'):
            found = _fields(
                each, 'a step', ('step', 'payload', 'query'), ('request_id',)
            )
            steps.append(
                Step(
                    step=_int(found['step'], 'step'),
                    payload=_int(found['payload'], 'payload'),
                    query=_int(found['query'], 'query'),
                )
            )
        forecast = tuple(steps)
    batches = None
    if read['batches'] is not None:
        calls = []
        for each in _list(read['batches'], 'batches'):
            found = _fields(
                each, 'a call', ('payload', 'rows'), ('request_id',)
            )
            calls.append(
                Batch(
                    payload=_int(found['payload'], 'payload'),
                    rows=_int(found['rows'], 'rows'),
                )
            )
        batches = tuple(calls)
    ranking = None
    if read['ranking'] is not None:
        found = _fields(
            read['ranking'],
            'the ranking',
            ('candidate', 'value', 'parts'),
            ('top_k',),
        )
        parts = []
        for each in _list(found['parts'], 'parts'):
            part = _fields(
                each,
                "a ranking's question",
                ('candidate', 'value', 'payload', 'query'),
                ('request_id',),
            )
            parts.append(
                Part(
                    candidate=_int(part['candidate'], 'candidate'),
                    value=_text(part['value'], 'value'),
                    payload=_int(part['payload'], 'payload'),
                    query=_int(part['query'], 'query'),
                )
            )
        ranking = Ranking(
            top_k=_maybe(found['top_k'], _int, 'top_k'),
            candidate=_int(found['candidate'], 'candidate'),
            value=_text(found['value'], 'value'),
            parts=tuple(parts),
        )
    return GatheredState(
        version=_int(read['version'], 'version'),
        payload=_int(read['payload'], 'payload'),
        query=_int(read['query'], 'query'),
        forecast=forecast,
        batches=batches,
        ranking=ranking,
    )


# --- predict ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Predicted:
    r"""``predict``'s result: the attachments of the result and of the
    endpoint's metadata, the model and request id, the outputs, and the
    ranking's candidates no answer scored, by position."""

    result: int
    model: str
    request_id: str
    metadata: int
    outputs: tuple[str, ...]
    unscored_candidates: tuple[int, ...]


def _predicted(read: dict[str, Any]) -> Predicted:
    return Predicted(
        result=_int(read['result'], 'result'),
        model=_text(read['model'], 'model'),
        request_id=_text(read['request_id'], 'request_id'),
        metadata=_int(read['metadata'], 'metadata'),
        outputs=tuple(
            _text(each, 'an output')
            for each in _list(read['outputs'], 'outputs')
        ),
        unscored_candidates=tuple(
            _int(each, 'a candidate')
            for each in _list(
                read['unscored_candidates'] or [], 'unscored_candidates'
            )
        ),
    )


_RESULTS: dict[str, tuple[tuple[str, ...], tuple[str, ...], Any]] = {
    'validate_query': (('accepted', 'diagnostics'), ('query',), _validated),
    'describe_source': (('description',), ('notes',), _described),
    'edit_schema': (('schema',), ('declared',), _edited),
    'materialize': (
        ('state',),
        ('query',),
        lambda read: Gathered(
            state=_state(read['state']),
            query=(
                None
                if read['query'] is None
                else QueryFacts.from_contract(read['query'])
            ),
        ),
    ),
    'predict': (
        ('result', 'model', 'request_id', 'metadata', 'outputs'),
        ('unscored_candidates',),
        _predicted,
    ),
}


def decode(
    result: Any,
) -> Validated | Described | Edited | Gathered | Predicted:
    r"""The result a ``done`` step carries, decoded once.

    Raises:
        CodecError: A result of an operation this host does not know, or
            of another shape than the contract's.
    """
    if not isinstance(result, Mapping) or not isinstance(result.get('op'), str):
        raise CodecError('a result names no operation')
    known = _RESULTS.get(result['op'])
    if known is None:
        raise CodecError('a result of an operation this host does not know')
    required, optional, build = known
    read = _fields(
        {key: value for key, value in result.items() if key != 'op'},
        f'the result of {result["op"]}',
        required,
        optional,
    )
    built: Validated | Described | Edited | Gathered | Predicted = build(read)
    return built


_T = TypeVar('_T', Validated, Described, Edited, Gathered, Predicted)


def expect(result: Any, kind: type[_T]) -> _T:
    r"""``result`` decoded (:func:`decode`), the result of the operation
    ``kind`` is of; the core's answering another, or of another shape than
    the contract's, is its defect, ``INTERNAL_ERROR``."""
    try:
        decoded = decode(result)
    except CodecError as unread:
        raise StructuredDataError(
            f'the core answered a result this host does not read: {unread}',
            code='INTERNAL_ERROR',
        ) from None
    if not isinstance(decoded, kind):
        raise StructuredDataError(
            'the core answered another operation than the one started',
            code='INTERNAL_ERROR',
        )
    return decoded


# --- What a host keeps of the core's results -----------------------------------------


@dataclasses.dataclass(frozen=True)
class CoreRanking:
    r"""What ``Request._core`` keeps of a ranking the core gathered: how many
    candidates a row keeps, each question's candidate by its position among
    those given, and by its name in the result."""

    top_k: int | None
    positions: tuple[int, ...]
    values: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CorePart:
    r"""A later part of a request the core gathered in parts: a forecast's
    step by its number, or a ranking's question by its candidate's position
    among those given and its name (the JSON text of its value); and its
    payload and query, as the core wrote them."""

    position: int
    name: str | None
    payload: Any
    query: Any


@dataclasses.dataclass(frozen=True)
class CoreParts:
    r"""A request the core gathered in parts, as one materialize answered
    it, which ``PredictiveQuery.materialize`` keeps as one request per part
    (T1.36 fix 21, T2.8): ``ranking`` or ``forecast``; the first part's
    query, position and name; how many candidates a ranking's row keeps;
    and the later parts."""

    kind: str
    query: Any
    position: int
    name: str | None
    top_k: int | None
    later: tuple[CorePart, ...]


@dataclasses.dataclass(frozen=True)
class CoreRequest:
    r"""What a request materialized on the core keeps for predict on the
    core (``Request._core``), the one value it hands back whole: the
    request state's version; how many rows each later call of a request
    asked in several calls scores; a ranking's candidates; and, on the
    request one materialize answered before it is kept in parts, its parts
    (:class:`CoreParts`)."""

    version: int
    batches: tuple[int, ...] | None = None
    ranking: CoreRanking | None = None
    parts: CoreParts | None = None


@dataclasses.dataclass(frozen=True)
class CoreQuery:
    r"""What the core route reads of a query it gathers, and nothing of the
    old validator's object (T3.8): a query the core validated (S0 on the
    core, P2-MID2-1), from the core's own validation (``validate_query``'s
    ``query``, contract v28); or, on the old validator's route, one naming a
    column whose name holds a dot, a backtick or a line break, which the old
    validator refuses and the core holds apart from its table (KO2-36,
    T2.18).

    ``text`` is what materialize sends and the request keeps as its query:
    the query as the core writes it back where the core validated it, as the
    reference's ``ValidatedPredictiveQuery.to_string()`` writes it, else as
    the caller wrote it. ``num_forecasts`` is how many steps a forecast asks
    (1 for any other query), which a refusal of one step names; ``None``
    where the core did not validate it. ``refusal`` is the old validator's,
    said of a query the core does not resolve either; ``None`` where the old
    validator was not asked.
    """

    text: str
    num_forecasts: int | None
    refusal: StructuredDataError | None

    def to_string(self) -> str:
        r"""The query's text, which materialize sends."""
        return self.text
