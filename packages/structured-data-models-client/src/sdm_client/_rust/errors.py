# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core's errors and warnings as this client's exceptions and warnings.

The core says what each error and warning says: one sentence of a closed
list (``sentence``), whose one template every host fills, which the binding
exports as data (``_native.templates()``, T3.17; invariant 8 of the Phase 3
structure design). This module chooses no sentence: it fills the core's
with the values the error or warning names, as Python writes them (the one
value formatter, :class:`_Words`), with the advice the template names as
Python is written (:data:`ADVICE`), and, for a sentence that relays a
library of this host's own, in that library's words (:data:`HOST_OWNED`).
A sentence with no template, or one its fields cannot fill, is the core's
defect, said as one.

One reviewed table, keyed on the error's code alone (decisions H6, E22),
maps each code to one :class:`~sdm_client.errors.StructuredDataError`
class (:data:`ERRORS`), and a test iterates every code the core has. One
narrow exception (:data:`AS_THE_BASE_CLASS`, T2.8c fix 2): the refusals the
reference raises as ``StructuredDataError`` under their code, by the rule
the core names, keep the reference's class. A warning's class is chosen by
its code (:data:`WARNINGS`), except frame schema and graph notices retain
the reference's SDMWarning presentation without a code prefix.
Warnings given one after another that the table groups are shown as one.

``details`` holds the error's ``fields``, decoded, with its ``kind`` and
``retryable``; a provider error also names the driver's exception class
the host saw (``driver_class``) beside the SQLSTATE and vendor code the
core keeps. The exception is raised ``from None``: no driver exception is
chained, and nothing the driver wrote reaches a traceback (decision H17).
"""

from __future__ import annotations

import builtins
import contextlib
import datetime
import functools
import json
import string
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from sdm_client._rust import codec
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import (
    MissingExtraError,
    NimRequestError,
    StructuredDataError,
    UnknownModelError,
    UnsupportedRequestError,
)

#: Each code the core gives, with the class it raises as.
ERRORS: dict[str, type[StructuredDataError]] = {
    'INVALID_REQUEST': StructuredDataError,
    'UNSUPPORTED_REQUEST': UnsupportedRequestError,
    'INVALID_RESPONSE': StructuredDataError,
    'INVALID_CONFIGURATION': StructuredDataError,
    'TRANSPORT_ERROR': StructuredDataError,
    'INTERNAL_ERROR': StructuredDataError,
    'UNKNOWN_MODEL': UnknownModelError,
    'MISSING_EXTRA': MissingExtraError,
    'INVALID_SCHEMA': StructuredDataError,
    'SESSIONS_UNSUPPORTED': StructuredDataError,
    'UNSUPPORTED_TYPE': StructuredDataError,
    'VALUE_OUT_OF_RANGE': StructuredDataError,
    'PROVIDER_SCHEMA_MISMATCH': StructuredDataError,
    'PROVIDER_ERROR': StructuredDataError,
    'UNKNOWN_REPLY': StructuredDataError,
    'REPLY_MISSING': StructuredDataError,
    'WRONG_REPLY': StructuredDataError,
    'RUN_FINISHED': StructuredDataError,
    'RUN_CANCELLED': StructuredDataError,
    'RUN_POISONED': StructuredDataError,
    'RUN_BUSY': StructuredDataError,
    'INVALID_MESSAGE': StructuredDataError,
    # A serving endpoint's refusal, raised as the old client raises it:
    # under the endpoint's status and its own code (T3.27).
    'ENDPOINT_REFUSED': NimRequestError,
}

#: The refusals the reference raises as its base class under their code,
#: by code and rule: ``main``'s 1fc5a686 ceilings on what one
#: ``materialize`` gathers and its refusal of a batched forecast, each a
#: ``StructuredDataError(code='UNSUPPORTED_REQUEST')`` in the reference's
#: client, so a caller's ``except`` sees the class it would there (T2.8c fix
#: 2). Every other refusal takes its code's class (:data:`ERRORS`).
AS_THE_BASE_CLASS: frozenset[tuple[str, str]] = frozenset(
    {
        ('UNSUPPORTED_REQUEST', 'most_entities'),
        ('UNSUPPORTED_REQUEST', 'most_calls'),
        ('UNSUPPORTED_REQUEST', 'batched_part'),
    }
)


class RuntimeWarning(builtins.RuntimeWarning):  # noqa: A001 - the reference's name
    r"""The class NumPy's warning of a cast that overflows is shown as
    (FX-1): the reference's ``RuntimeWarning``, owned by this client, as
    :class:`UserWarning` is."""


class UserWarning(builtins.UserWarning):  # noqa: A001 - the reference's name
    r"""The class the core's sampler warnings are shown as: the reference's
    ``UserWarning`` (a subclass of it, named so, so a filter, a
    ``pytest.warns`` and an observation's class name see the reference's),
    owned by this client, so where a warning came from can be told by its
    category (T1.11d review round 4)."""


#: Warning codes the core gives, with the class each is shown as; any other
#: code is shown as :class:`~sdm_client.core.warnings.SDMWarning`. From
#: T1.11c ``materialize`` gives the reference's sampler warnings, which the
#: reference shows as ``UserWarning``: indices missing for its reads, a
#: context reaching back before the data, an anchor after it, and from
#: T1.26b a context collected short, once in a process; this client's own
#: :class:`UserWarning`, so their provenance shows.
WARNINGS: dict[str, type[Warning]] = {
    'MISSING_INDICES': UserWarning,
    'CONTEXT_BEFORE_DATA': UserWarning,
    'ANCHOR_AFTER_DATA': UserWarning,
    'CONTEXT_COVERAGE': UserWarning,
    'FLOAT32_OVERFLOW': RuntimeWarning,
    # From T2.11, the engine's warnings of the sampling options: a context
    # anchored after the predictions, or whose window reaches past them,
    # and num_hops ignored for num_neighbors.
    'CONTEXT_AFTER_ANCHOR': UserWarning,
    'CONTEXT_LEAKS': UserWarning,
    'HOPS_IGNORED': UserWarning,
    'TIME_VALUES_UNREAD': UserWarning,
    'DATES_READ_DAY_FIRST': UserWarning,
    'DATE_LITERAL_DAY_FIRST': UserWarning,
    'KEY_CANDIDATES_TIED': UserWarning,
}

#: Each advice a template names, as Python is written, from the fields the
#: template gives it, in order (the core's ``wording::Advice``).
ADVICE: dict[str, Callable[..., str]] = {
    'append_relationship': lambda tables: (
        f'schema.relationships.append(Relationship({tables[0]!r}, '
        '<foreign key>, <parent table>, <its primary key>))'
    ),
    'booleans': lambda: 'True and False',
    'build_relationship': lambda: (
        'Relationship(source_table, source_column, target_table, target_column)'
    ),
    'choose_statistic': lambda: 'materialize(..., prediction_statistic=...)',
    'clear_time_column': lambda table: f'time_columns={{{table!r}: None}}',
    'context_samples': lambda most: f'num_context_samples={most}',
    'empty_request': lambda: 'Request(predict=<no rows>)',
    'give_a_hop': lambda: 'num_neighbors=[8, 8]',
    'keep_one_link': lambda source, target, others: (
        'schema.relationships = [link for link in schema.relationships if '
        f'not (link.source_table == {source!r} and link.target_table == '
        f'{target!r} and link.source_column in {others!r})]'
    ),
    'materialize_call': lambda: 'materialize()',
    'name_candidates': lambda column: (
        f'materialize(..., candidates=frame[{column!r}])'
    ),
    'name_primary_key': lambda table: f'primary_keys={{{table!r}: ...}}',
    'name_time_column': lambda table, column: (
        f'time_columns={{{table!r}: {column!r}}}'
    ),
    'nothing': lambda: 'None',
    'optimize_indices': lambda: 'KumoRelational(graph, optimize=True)',
    'predict_call': lambda: 'predict()',
    'raise_most_parts': lambda: 'raise sdm_client.decompose.MOST_PARTS',
    'semantic_model': lambda name: f'source.semantic_model({name!r})',
    'set_primary_key': lambda table, column: (
        f'primary_keys={{{table!r}: {column!r}}}'
    ),
    'skip_key_check': lambda: (
        'Name another key with primary_keys=, or pass check_keys=False to '
        'skip (one full scan per table).'
    ),
    'missing_time': lambda: "'NaT'",
    'result_frame': lambda: 'to_pandas()',
    'smaller_calls': lambda: 'materialize(rows_per_call=N)',
    'trust_expressions': lambda: 'trust_expressions=True',
}

#: The sentences that relay a library of this host's own, in that
#: library's words, which this host says in place of the shared sentence:
#: pandas' day-first dates (as the reference's ``pd.to_datetime`` gives
#: them), NumPy's cast that overflows (FX-1), and Python's comparison of a
#: count with a value of another type (the engine's ``TypeError``).
_DAY_FIRST = (
    'Parsing dates in {format} format when dayfirst=False (the default) was '
    'specified. Pass `dayfirst=True` or specify a format to silence this '
    'warning.'
)
HOST_OWNED: dict[str, str] = {
    'dates_read_day_first': _DAY_FIRST,
    'date_literal_day_first': _DAY_FIRST,
    'float32_overflow': 'overflow encountered in cast',
    'max_pq_iterations_not_a_whole_number': (
        "'<' not supported between instances of '{given!N}' and 'int'"
    ),
}


@dataclass(frozen=True)
class _Table:
    r"""The core's table of sentences (``_native.templates()``)."""

    #: Each sentence's entry, by name: its ``kind``, ``code``, ``rule``,
    #: ``template``, ``owner`` and ``group``.
    sentences: Mapping[str, Mapping[str, Any]]
    #: Each group's words, by name: ``code``, ``agree``, ``opening``,
    #: ``between`` and ``closing``.
    groups: Mapping[str, Mapping[str, Any]]
    #: Each advice's arguments, by name.
    advice: Mapping[str, tuple[str, ...]]
    #: Each database's names, by the name the core gives it: ``backend``
    #: and ``place``.
    databases: Mapping[str, Mapping[str, str]]


@functools.cache
def table() -> _Table:
    r"""The core's table of sentences, read once."""
    from sdm_client._rust.run import native

    read = json.loads(native().templates())
    return _Table(
        sentences={entry['name']: entry for entry in read['sentences']},
        groups={entry['name']: entry for entry in read['groups']},
        advice={
            entry['name']: tuple(entry['arguments']) for entry in read['advice']
        },
        databases={entry['name']: entry for entry in read['databases']},
    )


class _Advice:
    r"""What ``{advice:...}`` reads: its spec names the advice and fields."""


class _Words(string.Formatter):
    r"""``str.format`` over a sentence's fields, with the conversions the
    core's templates name (``sdm_contract::wording``): this host's one value
    formatter."""

    def __init__(
        self, fields: Mapping[str, Any], host: Mapping[str, Any]
    ) -> None:
        super().__init__()
        self._fields = fields
        self._host = host

    def get_value(
        self, key: int | str, args: Any, kwargs: Mapping[str, Any]
    ) -> Any:
        if key == 'advice':
            return _Advice()
        if key == 'host':
            return _Held(self._host)
        return self._fields[str(key)]

    def format_field(self, value: Any, format_spec: str) -> Any:
        if isinstance(value, _Advice):
            name, _, rest = format_spec.partition('(')
            given = [a.strip() for a in rest.rstrip(')').split(',')]
            return ADVICE[name](*(self._fields[a] for a in given if a))
        if format_spec.startswith('to(') and format_spec.endswith(')'):
            # A list paired with another field's: each pair as its two
            # literals joined by ` to `, the pairs by commas.
            others = self._fields[format_spec[3:-1]]
            return ', '.join(
                f'{one!r} to {other!r}'
                for one, other in zip(value, others, strict=True)
            )
        if format_spec in ('place', 'backend'):
            names = table().databases.get(str(value))
            if format_spec == 'backend':
                return str(value) if names is None else names['backend']
            return f'{value} database' if names is None else names['place']
        return super().format_field(value, format_spec)

    def convert_field(self, value: Any, conversion: str | None) -> Any:
        if conversion == 'l':
            quoted = [repr(name) for name in value]
            if len(quoted) < 3:
                return ' and '.join(quoted)
            return f'{", ".join(quoted[:-1])} and {quoted[-1]}'
        if conversion == 'a':
            plain = [str(name) for name in value]
            if len(plain) < 3:
                return ' and '.join(plain)
            return f'{", ".join(plain[:-1])} and {plain[-1]}'
        if conversion == 'j':
            return ', '.join(repr(name) for name in value)
        if conversion == 'c':
            return ', '.join(str(name) for name in value)
        if conversion == 'R':
            return '; '.join(str(reason) for reason in value)
        if conversion == 'd':
            return '.'.join(str(name) for name in value)
        if conversion == 'p':
            # A row of values as the reference shows one: a tuple.
            return repr(tuple(value) if isinstance(value, list) else value)
        if conversion == 'I':
            # A whole number the caller gave, as the JSON text it was
            # written in, so a spec can group its thousands.
            return int(json.loads(value))
        if conversion in ('G', 'S', 'N'):
            # A value the caller gave, as the JSON text the host wrote it
            # in: its ``repr``, its ``str`` or its type's name, as Python
            # holds the value JSON reads back.
            given = json.loads(value)
            if conversion == 'G':
                return repr(given)
            if conversion == 'S':
                return str(given)
            return type(given).__name__
        if conversion == 't':
            native = _held().get('frame_timestamp')
            if native is not None and isinstance(value, codec.Temporal):
                return native({**self._fields, 'value': value})
            if isinstance(value, str):
                return str(_as_timestamp(value))
            return _shown_timestamp(value)
        if conversion == 'T':
            return repr(
                [
                    item if isinstance(item, str) else _time_of_day(item)
                    for item in value
                ]
            )
        return super().convert_field(value, conversion)


class _Held:
    r"""The values only this host holds, which ``{host.<name>}`` reads."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = values

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None


#: How the call in progress reads each value only this host holds that a
#: sentence names (``{host.<name>}``), from the error's or warning's
#: decoded fields: a candidate as the caller gave it, by its position, the
#: query as written, the source described (:func:`holding`).
_HELD: ContextVar[Mapping[str, Callable[[dict[str, Any]], Any]] | None] = (
    ContextVar('sdm_client_held_values', default=None)
)


def _held() -> Mapping[str, Callable[[dict[str, Any]], Any]]:
    return _HELD.get() or {}


@contextlib.contextmanager
def holding(**values: Callable[[dict[str, Any]], Any]) -> Iterator[None]:
    r"""Within the block, each error and warning the core gives fills a
    sentence's ``{host.<name>}`` with ``values[name]`` of its decoded fields:
    what only the caller holds, which the core names by position or not at
    all (T3.17). Blocks nest; an inner one adds to and overrides the outer."""
    token = _HELD.set({**_held(), **values})
    try:
        yield
    finally:
        _HELD.reset(token)


class _HostValues(Mapping[str, Any]):
    r"""The values only this host holds, each read from a sentence's fields
    when the sentence names it, and only then."""

    def __init__(
        self,
        read: Mapping[str, Callable[[dict[str, Any]], Any]],
        fields: dict[str, Any],
    ) -> None:
        self._read = read
        self._fields = fields

    def __getitem__(self, name: str) -> Any:
        return self._read[name](self._fields)

    def __iter__(self) -> Any:
        return iter(self._read)

    def __len__(self) -> int:
        return len(self._read)


def render(
    sentence: str,
    fields: Mapping[str, Any],
    host: Mapping[str, Any] | None = None,
) -> str:
    r"""What ``sentence`` says, filled with ``fields`` (decoded) and the
    values only this host holds (``host``), as Python writes them; a
    sentence of a library of this host's own in that library's words.

    Raises:
        KeyError: The table has no such sentence, or its template names a
            field or a host value it is not given.
    """
    entry = table().sentences[sentence]
    template = entry['template']
    if entry['owner'] != 'shared':
        template = HOST_OWNED[sentence]
    return _Words(fields, host or {}).format(template)


#: The words of a sentence the core gave without what its template names:
#: the defect, named, which tests hold the core's producers to never give.
DEFECT = (
    'the core said {sentence!r} without what it names; this is a defect, not '
    'a problem with the request'
)


def _said(sentence: str, fields: dict[str, Any]) -> str:
    r""":func:`render` with the values the call in progress holds
    (:func:`holding`), or, where the core gave a sentence its fields cannot
    fill, the defect named: never a sentence the core did not choose."""
    try:
        return render(sentence, fields, _HostValues(_held(), fields))
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return DEFECT.format(sentence=sentence)


def decoded(given: Mapping[str, Any]) -> dict[str, Any]:
    r"""A sentence's fields as the core wrote them, decoded as
    :func:`render` takes them."""
    return _fields(given)


def source_values(source: Any) -> dict[str, Callable[[dict[str, Any]], Any]]:
    r"""What a sentence names of the source of the call in progress
    (``{host.source}``), and, for a semantic model, the model
    (``{host.semantic_model}``, whose ``repr`` the old client wrote:
    ``SemanticModel('name' in DatabricksSource(...))``), for
    :func:`holding`."""
    values: dict[str, Callable[[dict[str, Any]], Any]] = {
        'source': lambda fields: source
    }
    if getattr(source, 'kind', None) in (
        'databricks-metric-view',
        'snowflake-semantic-view',
    ):
        values['semantic_model'] = lambda fields: source
    return values


def _fields(given: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _detail(value) for name, value in given.items()}


def _timestamp(value: codec.Temporal) -> Any:
    r"""A date or timestamp scalar as pandas holds it, as the reference's
    rows read it."""
    import pandas as pd

    if value.tag == 'date32':
        return pd.Timestamp(datetime.date(1970, 1, 1)) + pd.Timedelta(
            days=value.value
        )
    unit = {'s': 's', 'ms': 'ms', 'us': 'us', 'ns': 'ns'}[str(value.unit)]
    if value.zone is None:
        return pd.Timestamp(value.value, unit=unit)
    return pd.Timestamp(value.value, unit=unit, tz='UTC').tz_convert(value.zone)


def _shown_timestamp(value: codec.Temporal) -> str:
    r"""A date or timestamp scalar as the reference shows it in a warning.

    A zoned timestamp as the reference's rows hand it over
    (``ArrowExtensionArray.__getitem__``): pyarrow's ``as_py``, a
    ``datetime`` at its wall clock in the zone through ``zoneinfo`` (local
    mean time before the zone's first transition), boxed as a pandas
    ``Timestamp`` of the column's unit where that unit is not ``ns``. pandas'
    own ``tz_convert`` labels such a value with the zone's offset at that
    instant but moves it by another, and refuses some near year 1
    (T1.36 fix 11). One ``as_py`` cannot hold, past the years 1 to 9999
    in UTC or in the zone, reaches a warning only as a whole table's
    smallest value, which the reference reads through DuckDB's
    ``fetchall`` as DuckDB's own text (:func:`_duckdb_text`) and shows as it
    is (T1.36 fix 13); every other value as pandas holds it."""
    if value.tag != 'date32' and value.zone is not None:
        import pandas as pd
        import pyarrow as pa

        unit = {'s': 's', 'ms': 'ms', 'us': 'us', 'ns': 'ns'}[str(value.unit)]
        try:
            read = pa.scalar(value.value, pa.timestamp(unit, tz=value.zone))
            held = read.as_py()
        except OverflowError:
            return _duckdb_text(value.value, unit)
        if unit != 'ns':
            held = pd.Timestamp(held).as_unit(unit)
        return str(held)
    return str(_timestamp(value))


def _duckdb_text(count: int, unit: str) -> str:
    r"""A ``TIMESTAMPTZ`` of ``count`` ``unit`` since the epoch as DuckDB
    writes one Python's ``datetime`` cannot hold: its UTC wall clock, the
    year at least four digits, a year before 1 as ``1 - year`` followed by
    `` (BC)``, the fraction of a second without its trailing zeros, then
    ``+00`` (``10000-06-01 12:34:56.5+00``, ``0001-12-31 (BC)
    15:00:00+00``)."""
    micros = (
        count // 1_000
        if unit == 'ns'
        else count * {'s': 1_000_000, 'ms': 1_000, 'us': 1}[unit]
    )
    days, of_day = divmod(micros, 86_400_000_000)
    # Days since 1970-01-01 to a proleptic Gregorian date (civil_from_days).
    shifted = days + 719_468
    era = shifted // 146_097
    of_era = shifted - era * 146_097
    year_of_era = (
        of_era - of_era // 1_460 + of_era // 36_524 - of_era // 146_096
    ) // 365
    of_year = of_era - (
        365 * year_of_era + year_of_era // 4 - year_of_era // 100
    )
    month_index = (5 * of_year + 2) // 153
    day = of_year - (153 * month_index + 2) // 5 + 1
    month = month_index + 3 if month_index < 10 else month_index - 9
    year = year_of_era + era * 400 + (1 if month <= 2 else 0)
    seconds, fraction = divmod(of_day, 1_000_000)
    clock = (
        f'{seconds // 3_600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'
    )
    if fraction:
        clock += f'.{fraction:06d}'.rstrip('0')
    date = (
        f'{year:04d}-{month:02d}-{day:02d}'
        if year > 0
        else f'{1 - year:04d}-{month:02d}-{day:02d} (BC)'
    )
    return f'{date} {clock}+00'


def _as_timestamp(value: str) -> Any:
    r"""Text a time column holds, as the reference names it in a warning
    (``time_columns._as_timestamp``): a timestamp where pandas reads it as
    one, else the text."""
    import pandas as pd

    try:
        return pd.Timestamp(value)
    except ValueError:
        return value


def _time_of_day(value: codec.Temporal) -> datetime.time:
    micros = value.value // 1_000 if value.unit == 'ns' else value.value
    seconds, micro = divmod(micros, 1_000_000)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return datetime.time(hour, minute, second, micro)


def said(warning: Mapping[str, Any]) -> str:
    r"""The words of one warning: its sentence, filled with its fields and
    the values the call in progress holds (:func:`holding`)."""
    return _said(warning['sentence'], _fields(warning.get('fields', {})))


def _instance(
    cls: type[StructuredDataError],
    message: str,
    code: str,
    details: dict[str, Any],
) -> StructuredDataError:
    r"""An instance of ``cls`` with ``message``, ``code`` and ``details``.

    Some classes take their own arguments to build their message; the
    sentence replaces that message, so the base initializer is the one
    that runs.
    """
    error = cls.__new__(cls)
    if isinstance(error, NimRequestError):
        # The core's fields name the status and the code the refusal is
        # raised under (`REFUSED_STATUS`, `REFUSED_CODE`), never details.
        details = dict(details)
        error.status_code = details.pop('status')
        refused: str | None = details.pop('refused_code', None)
        StructuredDataError.__init__(
            error, message, code=refused, details=details
        )
        return error
    StructuredDataError.__init__(error, message, code=code, details=details)
    if isinstance(error, UnknownModelError):
        error.model = details.get('model')  # type: ignore[assignment]
    return error


def error_from_core(
    control: bytes,
    attachments: dict[int, Any],
    *,
    driver_class: str | None = None,
) -> StructuredDataError:
    r"""The exception an ``error`` control message raises as, saying the
    core's sentence; the sentence's name is kept on it (``_sentence``), so
    a caller holding values the core does not can say it again with them
    (:func:`render`).

    ``driver_class`` names the class of the driver exception behind a
    ``PROVIDER_ERROR``, which the run loop saw and the core never does.
    """
    message = codec.read(control, attachments)
    error = message['error']
    code = error['code']
    cls = ERRORS[code]
    fields = _fields(error.get('fields', {}))
    details: dict[str, Any] = {
        **fields,
        'kind': error['kind'],
        'retryable': bool(error.get('retryable', False)),
    }
    response_cause: Exception | None = None
    if code == 'PROVIDER_ERROR' and driver_class is not None:
        details['driver_class'] = driver_class
    if (code, details.get('rule')) in AS_THE_BASE_CLASS:
        cls = StructuredDataError
    sentence = error['sentence']
    if (
        sentence == 'no_table'
        and (present := _held().get('direct_task_tables')) is not None
    ):
        fields['tables'] = present(fields)
        details['tables'] = fields['tables']
    if (
        sentence == 'no_column'
        and (present := _held().get('direct_task_column')) is not None
    ):
        shown = present(fields)
        fields.update(shown)
        details.update(shown)
    if (
        sentence == 'direct_related'
        and (present := _held().get('direct_related_names')) is not None
    ):
        # The core has selected the related-table refusal. Native key sorting
        # belongs to its reached diagnostic and can itself raise TypeError.
        # Names need no eager UTF-8 conversion or generic formatting wrapper.
        fields['related_tables'] = present(fields)
        details['related_tables'] = fields['related_tables']
    if (
        sentence == 'tabular_row_bounds'
        and (present := _held().get('direct_row_index')) is not None
    ):
        # Bounds are already selected by the core. Native integer display can
        # itself raise under active runtime limits, as at the reference f-string.
        # Keep it outside _said's formatting-defect catch.
        fields['index'] = present(fields)
    if (
        sentence == 'direct_schema_missing'
        and (present := _held().get('direct_schema_missing')) is not None
    ):
        present(fields)
    if sentence == 'direct_response_contract':
        from sdm_api.generated.tfm_api import ResponseContractError

        if 'offending' in fields:
            value = codec.restored_json(fields['offending'])
            try:
                if fields.get('native') == 'dict':
                    # The core selected the failed conversion. Preserve its
                    # actual native wording and cause inside the SDK wrapper.
                    dict(value)
                elif fields.get('native') == 'iter':
                    iter(value)
            except (TypeError, ValueError) as failed:
                fields['diagnostic'] = str(failed)
                response_cause = failed
            if fields.get('before') or fields.get('after'):
                fields['diagnostic'] = (
                    fields['before'] + repr(value) + fields['after']
                )
        details['diagnostic'] = fields['diagnostic']
    said = _said(sentence, fields)
    part = error.get('part')
    if part is not None:
        # The part of a question asked in parts it happened in, the core's
        # sentence too, after the error's, which ends in a full stop.
        said = said.rstrip()
        if said and said[-1] not in '.!?':
            said += '.'
        said = f'{said} {_said(part, fields)}'
    made = _instance(cls, said, code, details)
    made._sentence = sentence
    if sentence == 'direct_response_contract':
        made.__cause__ = (
            response_cause
            if response_cause is not None
            else ResponseContractError(fields['diagnostic'])
        )
    return made


def _detail(value: Any) -> Any:
    if isinstance(value, list):
        return [codec.decode_scalar(item) for item in value]
    return codec.decode_scalar(value)


def raise_from_core(
    control: bytes,
    attachments: dict[int, Any],
    *,
    driver_class: str | None = None,
    internal_context: BaseException | None = None,
) -> NoReturn:
    r"""Raise the public exception, retaining only its selected explicit cause."""
    error = error_from_core(control, attachments, driver_class=driver_class)
    raise_error(error, internal_context=internal_context)


def raise_error(
    error: StructuredDataError,
    *,
    internal_context: BaseException | None = None,
) -> NoReturn:
    """Keep the selected cause and its caller context at the public boundary."""
    cause = error.__cause__
    if cause is None:
        raise error from None
    try:
        # A normal decoder raises the wrapper inside its cause's handler.
        # Reuse that same exception, including its original native traceback.
        raise cause
    except BaseException as selected:
        if selected is not cause:
            raise
        if (
            internal_context is not None
            and cause.__context__ is internal_context
        ):
            # Remove only our caught core exception, retaining any original
            # caller exception behind it and its native suppression flags.
            cause.__context__ = internal_context.__context__
        raise error from cause


def numpy_warns_on_cast_overflow() -> bool:
    r"""Whether the installed NumPy warns when a cast to float32 overflows
    (``RuntimeWarning: overflow encountered in cast``), as it does from
    NumPy 1.24 on, which first warns on a cast that overflows (its 1.24.0
    release notes); before it the cast is silent. The reference's warning
    (FX-1) is NumPy's, so this client shows the core's ``FLOAT32_OVERFLOW``
    only where the reference's NumPy would show it (T2.14b fix 5)."""
    import re

    import numpy as np

    found = re.match(r'(\d+)\.(\d+)', np.__version__)
    if found is None:
        return True
    return (int(found.group(1)), int(found.group(2))) >= (1, 24)


# The frame reference presents schema notices and graph sanitization as
# SDMWarning without a code prefix; other warnings keep their code policy.
WarningPresentation = Literal['coded', 'frame_schema', 'frame_graph']


def warn_from_core(
    control: bytes,
    attachments: dict[int, Any],
    quiet: frozenset[str] = frozenset(),
    given: set[str] | None = None,
    held: list[dict[str, Any]] | None = None,
    *,
    presentation: WarningPresentation = 'coded',
) -> None:
    r"""Shows each warning of a ``warnings`` control message, in order, but
    those whose code ``quiet`` names: one the caller has shown already, or
    may show itself, kept in ``held`` when it is given (:func:`warn_held`).
    Each code given is added to ``given``, shown or not. A sentence naming
    a value only this host holds takes it from the call in progress
    (:func:`holding`). Warnings of one group of the core's table,
    given one after another with the same values of the fields it names,
    are shown as one."""
    message = codec.read(control, attachments)
    groups = table().groups
    shown: list[tuple[Any, list[dict[str, Any]]]] = []
    for warning in message['warnings']:
        code = warning['code']
        if given is not None:
            given.add(code)
        if code in quiet:
            if held is not None:
                held.append(warning)
            continue
        if code == 'FLOAT32_OVERFLOW' and not numpy_warns_on_cast_overflow():
            # The core reports the cast's overflow on every runtime; this
            # client shows it where the reference's NumPy warns of it.
            continue
        group = table().sentences[warning['sentence']]['group']
        key = (
            None
            if group is None
            else (
                group,
                tuple(
                    repr(warning.get('fields', {}).get(name))
                    for name in groups[group]['agree']
                ),
            )
        )
        if key is not None and shown and shown[-1][0] == key:
            shown[-1][1].append(warning)
        else:
            shown.append((key, [warning]))

    for key, together in shown:
        first = together[0]
        if key is None:
            text = said(first)
        else:
            words = groups[key[0]]
            text = (
                _opening(words['opening'], _fields(first.get('fields', {})))
                + words['between'].join(said(w) for w in together)
                + words['closing']
            )
        frame_notice = presentation == 'frame_schema' or (
            presentation == 'frame_graph'
            and first['code'] == 'FRAME_ROWS_DROPPED'
        )
        warnings.warn(
            text if frame_notice else f'[{first["code"]}] {text}',
            category=SDMWarning
            if frame_notice
            else WARNINGS.get(first['code'], SDMWarning),
            stacklevel=caller_stacklevel(),
        )


def _opening(template: str, fields: dict[str, Any]) -> str:
    r"""A group's opening, over its first warning's fields and the values
    the call in progress holds (``{host.semantic_model!r}``, T3.5); the
    defect named where they cannot fill it."""
    try:
        return _Words(fields, _HostValues(_held(), fields)).format(template)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return DEFECT.format(sentence=template)


def warn_held(held: list[dict[str, Any]]) -> None:
    r"""Shows warnings :func:`warn_from_core` held back, in order, each as it
    would have been shown (none of them of a group)."""
    for warning in held:
        assert table().sentences[warning['sentence']]['group'] is None
        warnings.warn(
            f'[{warning["code"]}] {said(warning)}',
            category=WARNINGS.get(warning['code'], SDMWarning),
            stacklevel=caller_stacklevel(),
        )
