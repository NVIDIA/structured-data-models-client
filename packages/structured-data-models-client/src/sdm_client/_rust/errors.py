# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core's errors and warnings as this client's exceptions and warnings.

One reviewed table, keyed on the error's code alone (decisions H6, E22):
each code maps to one :class:`~sdm_client.errors.StructuredDataError` class
and a sentence written for people, and a test iterates every code the core
has. The kind is fixed by the code in ``sdm-contract``, so it decides
nothing here, and the core's message text is never read: the exception's
text is the table's sentence, and what a caller branches on is ``code`` and
``details``. A warning's class is chosen by its code alone.

A warning the core gives as a code and fields is worded here, from
:data:`SAID`'s templates, in the reference's words (decision E59): the
core says what happened and names what it happened to, and this client
says it as the reference did, its Python remedies included.

``details`` holds the error's ``fields``, decoded, with its ``kind`` and
``retryable``; a provider error also names the driver's exception class
the host saw (``driver_class``) beside the SQLSTATE and vendor code the
core keeps. The exception is raised ``from None``: no driver exception is
chained, and nothing the driver wrote reaches a traceback (decision H17).
"""

from __future__ import annotations

import builtins
import datetime
import string
import warnings
from typing import Any, NoReturn

from sdm_client._rust import codec
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import (
    MissingExtraError,
    StructuredDataError,
    UnknownModelError,
    UnsupportedRequestError,
)

#: Each code the core gives: the class it raises as, and what it says.
ERRORS: dict[str, tuple[type[StructuredDataError], str]] = {
    'INVALID_REQUEST': (
        StructuredDataError,
        'the request is not one the core accepts',
    ),
    'UNSUPPORTED_REQUEST': (
        UnsupportedRequestError,
        'the request is valid, and not one the core supports yet',
    ),
    'INVALID_RESPONSE': (
        StructuredDataError,
        'the model endpoint answered with something the core cannot read',
    ),
    'INVALID_CONFIGURATION': (
        StructuredDataError,
        'the configuration of this call is not one the core accepts',
    ),
    'TRANSPORT_ERROR': (
        StructuredDataError,
        'the transport to the model endpoint failed: the endpoint could not '
        'be reached, or its answer could not be read',
    ),
    'INTERNAL_ERROR': (
        StructuredDataError,
        'the core failed; this is a defect, not a problem with the request',
    ),
    'UNKNOWN_MODEL': (UnknownModelError, 'the model named does not exist'),
    'MISSING_EXTRA': (
        MissingExtraError,
        'an optional dependency this call needs is not installed',
    ),
    'INVALID_SCHEMA': (
        StructuredDataError,
        'the schema is not one the core can work with',
    ),
    'SESSIONS_UNSUPPORTED': (
        StructuredDataError,
        'the model endpoint does not support sessions',
    ),
    'UNSUPPORTED_TYPE': (
        StructuredDataError,
        'a column has a type the core does not take',
    ),
    'VALUE_OUT_OF_RANGE': (
        StructuredDataError,
        'a value is outside what its column type can hold in the core',
    ),
    'PROVIDER_SCHEMA_MISMATCH': (
        StructuredDataError,
        'the data source answered with columns other than the ones asked for',
    ),
    'PROVIDER_ERROR': (StructuredDataError, 'the data source failed'),
    'UNKNOWN_REPLY': (
        StructuredDataError,
        'the host answered a request the core did not make',
    ),
    'REPLY_MISSING': (
        StructuredDataError,
        'the host went on before answering every request',
    ),
    'WRONG_REPLY': (
        StructuredDataError,
        'the host answered a request with the wrong kind of reply',
    ),
    'RUN_FINISHED': (StructuredDataError, 'the run had already finished'),
    'RUN_CANCELLED': (StructuredDataError, 'the run was cancelled'),
    'RUN_POISONED': (
        StructuredDataError,
        'the run failed earlier and can no longer be used',
    ),
    'RUN_BUSY': (
        StructuredDataError,
        'the run was already in a call; a run takes one call at a time',
    ),
    'INVALID_MESSAGE': (
        StructuredDataError,
        'the host sent a message the core cannot read',
    ),
}


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
    'TIME_VALUES_UNREAD': UserWarning,
    'DATES_READ_DAY_FIRST': UserWarning,
    'DATE_LITERAL_DAY_FIRST': UserWarning,
}

#: The words of a link that does not refer to its table's key, around
#: what that table has (``{has}``), as a template of a template.
_NOT_THE_KEY = (
    '{{source_table}}.{{source_column}} refers to '
    '{{target_table}}.{{target_column}}, but {has}. A relationship joins to a '
    'primary key, so none was made from it. Pass '
    'primary_keys={{{{{{target_table!r}}: {{target_column!r}}}}}} if '
    '{{target_column!r}} identifies a row of {{target_table!r}}.'
)

#: How this client words a warning the core gives as a code and fields
#: (decision E59, the design's presentation rule): the reference's words,
#: its Python remedies included, from a template per code, or per code and
#: the ``variant`` field, which says which of the reference's sentences
#: the core meant. A template is ``str.format``'s, over the warning's
#: fields decoded, with ``!l`` for a list of names as the reference's
#: ``_and_list`` writes them, ``!t`` for a date or timestamp as pandas
#: writes one (text a time column holds as the reference names it), and
#: ``!T`` for a list of times of day, or of text, as Python writes it.
#: A code this table does not hold is shown with the core's own sentence.
SAID: dict[str, dict[str | None, str]] = {
    'TIME_COLUMNS_GUESSED': {
        'taken_earliest': (
            '{table}.{column} was taken as the time column because its values '
            'start earliest of {among!l}. If another says when a row became '
            'true, name it: time_columns={{{table!r}: {other!r}}}. If none '
            'does, clear it: time_columns={{{table!r}: None}}.'
        ),
        'taken_edit_stamp': (
            '{table}.{column} was taken as the time column although it looks '
            'like when a row was last changed, because the table has no other '
            'timestamp and no table links to it, so it reads as a log of '
            'changes. If it does not say when a row became true, clear it: '
            'time_columns={{{table!r}: None}}.'
        ),
        **{
            f'refused_{reason}': (
                '{table}.{column} was not taken as the time column because '
                + why
                + '. If it does say when a row became true, name it: '
                'time_columns={{{table!r}: {column!r}}}.'
            )
            for reason, why in (
                ('empty', 'no row holds a value'),
                ('sparse', '{missing:,} of its {rows:,} rows hold no value'),
                (
                    'constant_where_held',
                    'every row that holds a value holds {value!t}',
                ),
                ('constant', 'every row holds {value!t}'),
                (
                    'load_stamp',
                    'it looks like when a row was loaded, not when it began',
                ),
                (
                    'edit_stamp',
                    'it looks like when a row was last changed, not when it '
                    'began',
                ),
                (
                    'birth_date',
                    'it looks like a date of birth, not when a row began',
                ),
            )
        },
    },
    'COLUMNS_LEFT_OUT': {None: '{column} ({dtype})'},
    'TIME_VALUES_UNREAD': {
        None: (
            'Could not parse {unread:,} of {rows:,} value(s) of time column '
            "'{column}' (for example {examples!T}); they were dropped as 'NaT'"
        )
    },
    'DATES_READ_DAY_FIRST': {
        None: (
            'Parsing dates in {format} format when dayfirst=False (the '
            'default) was specified. Pass `dayfirst=True` or specify a format '
            'to silence this warning.'
        )
    },
    # A query's date literal read day first (date rule D4), in pandas' words,
    # as the reference's `pd.to_datetime` gives them.
    'DATE_LITERAL_DAY_FIRST': {
        None: (
            'Parsing dates in {format} format when dayfirst=False (the '
            'default) was specified. Pass `dayfirst=True` or specify a format '
            'to silence this warning.'
        )
    },
    'LINK_CANNOT_JOIN': {
        'not_the_key': _NOT_THE_KEY.format(
            has='the primary key of {target_table!r} is {key!l}'
        ),
        'no_key': _NOT_THE_KEY.format(
            has='{target_table!r} has no primary key'
        ),
        'types': (
            '{source_table}.{source_column} names the key of {target_table!r} '
            'and holds {source_dtype!r} where that key holds {target_dtype!r}, '
            'which cannot match, so no relationship was made from it. Store '
            'the two as one type if they are the same identifier.'
        ),
    },
    'TABLES_NAME_EACH_OTHER': {
        None: (
            '{source_table!r} and {target_table!r} name each other, and a '
            'relationship is crossed in both directions, so this is two ways '
            'over one pair and an aggregation between them is refused for not '
            'having a single one. Declare whichever of the two a question is '
            'about and leave the other out.'
        )
    },
    'MISSING_INDICES': {
        f'{index}_{table}': (
            'Missing {indices} '
            + index
            + ' in {tables} '
            + table
            + ' for optimal database querying; indices on primary and foreign '
            'keys speed sampling up'
        )
        for index in ('index', 'indices')
        for table in ('table', 'tables')
    },
    'CONTEXT_BEFORE_DATA': {
        None: (
            "the context's anchor time is too early, or the aggregation's "
            'window too long, for the data to reach back as far as the '
            'examples need'
        )
    },
    'ANCHOR_AFTER_DATA': {
        None: 'the anchor time is after the latest time in the data'
    },
    'CONTEXT_COVERAGE': {
        None: (
            'Failed to collect {examples:,} context examples within '
            '{candidates:,} candidates. To improve coverage, consider '
            'increasing the number of PQ iterations using the '
            "'max_pq_iterations' option. This warning will not be shown "
            'again in this run.'
        )
    },
    'TWO_WAYS_TO_A_TABLE': {
        None: (
            '{source_table!r} reaches {target_table!r} through {columns!l}, '
            'and an aggregation across the two tables has to be resolved to '
            'one of them, so a query that crosses this pair is refused until '
            'it is. They are all kept because they are all real, and which '
            'one a question means is not something this can decide. To keep '
            '{kept!r}, drop the others: schema.relationships = [link for link '
            'in schema.relationships if not (link.source_table == '
            '{source_table!r} and link.target_table == {target_table!r} and '
            'link.source_column in {others!r})].'
        )
    },
}

#: Codes whose warnings, given one after another with the same values of
#: the fields named, are shown as one, as the reference says them: the
#: fields named, the opening (a template over the first warning's
#: fields), what goes between two of them, and the closing.
GROUPED: dict[str, tuple[tuple[str, ...], str, str, str]] = {
    'TIME_COLUMNS_GUESSED': (
        (),
        'Some time columns were inferred, or refused, on a guess about what '
        'a column holds:\n- ',
        '\n- ',
        '',
    ),
    'COLUMNS_LEFT_OUT': (
        ('table',),
        '{table} leaves out ',
        ', ',
        ': a request carries one scalar value per cell, and these hold none',
    ),
}


class _Words(string.Formatter):
    r"""``str.format`` with the conversions :data:`SAID` uses."""

    def convert_field(self, value: Any, conversion: str | None) -> Any:
        if conversion == 'l':
            quoted = [repr(name) for name in value]
            if len(quoted) < 3:
                return ' and '.join(quoted)
            return f'{", ".join(quoted[:-1])} and {quoted[-1]}'
        if conversion == 't':
            if isinstance(value, str):
                return str(_as_timestamp(value))
            return str(_timestamp(value))
        if conversion == 'T':
            return repr(
                [
                    item if isinstance(item, str) else _time_of_day(item)
                    for item in value
                ]
            )
        return super().convert_field(value, conversion)


def _timestamp(value: codec.Temporal) -> Any:
    r"""A date or timestamp scalar as pandas holds it, as the reference's
    rows read it."""
    import pandas as pd

    if value.tag == 'date32':
        return pd.Timestamp(datetime.date(1970, 1, 1)) + pd.Timedelta(
            days=value.value
        )
    unit = {'us': 'us', 'ns': 'ns'}[str(value.unit)]
    if value.zone is None:
        return pd.Timestamp(value.value, unit=unit)
    return pd.Timestamp(value.value, unit=unit, tz='UTC').tz_convert(value.zone)


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


def said(warning: dict[str, Any]) -> str:
    r"""The words for one warning: :data:`SAID`'s template for its code and
    variant over its fields, or the core's sentence for a code or variant
    the table does not hold, or fields its template cannot be filled
    from."""
    templates = SAID.get(warning['code'])
    fields = {
        name: _detail(value)
        for name, value in warning.get('fields', {}).items()
    }
    template = (templates or {}).get(fields.get('variant'))
    if template is None:
        return str(warning['message'])
    try:
        return _Words().format(template, **fields)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        # A warning without the fields its words name is the core's
        # defect; it is still shown, in the core's own sentence.
        return str(warning['message'])


def _instance(
    cls: type[StructuredDataError],
    message: str,
    code: str,
    details: dict[str, Any],
) -> StructuredDataError:
    r"""An instance of ``cls`` with ``message``, ``code`` and ``details``.

    Some classes take their own arguments to build their message; the
    table's sentence replaces that message, so the base initializer is the
    one that runs.
    """
    error = cls.__new__(cls)
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
    r"""The exception an ``error`` control message raises as.

    ``driver_class`` names the class of the driver exception behind a
    ``PROVIDER_ERROR``, which the run loop saw and the core never does.
    """
    message = codec.read(control, attachments)
    error = message['error']
    code = error['code']
    cls, sentence = ERRORS[code]
    details: dict[str, Any] = {
        name: _detail(value) for name, value in error.get('fields', {}).items()
    }
    details['kind'] = error['kind']
    details['retryable'] = bool(error.get('retryable', False))
    if code == 'PROVIDER_ERROR' and driver_class is not None:
        details['driver_class'] = driver_class
    return _instance(cls, sentence, code, details)


def _detail(value: Any) -> Any:
    if isinstance(value, list):
        return [codec.decode_scalar(item) for item in value]
    return codec.decode_scalar(value)


def raise_from_core(
    control: bytes,
    attachments: dict[int, Any],
    *,
    driver_class: str | None = None,
) -> NoReturn:
    r"""Raises the exception an ``error`` control message is, ``from None``."""
    raise error_from_core(
        control, attachments, driver_class=driver_class
    ) from None


def warn_from_core(
    control: bytes,
    attachments: dict[int, Any],
    quiet: frozenset[str] = frozenset(),
    given: set[str] | None = None,
    held: list[dict[str, Any]] | None = None,
) -> None:
    r"""Shows each warning of a ``warnings`` control message, in order, but
    those whose code ``quiet`` names: one the caller has shown already, or
    may show itself, kept in ``held`` when it is given (:func:`warn_held`).
    Each code given is added to ``given``, shown or not."""
    message = codec.read(control, attachments)
    # Warnings of a code GROUPED holds, one after another with the same
    # values of its fields, are shown as one.
    shown: list[tuple[str, Any, list[dict[str, Any]]]] = []
    for warning in message['warnings']:
        code = warning['code']
        if given is not None:
            given.add(code)
        if code in quiet:
            if held is not None:
                held.append(warning)
            continue
        group = GROUPED.get(code)
        key = (
            None
            if group is None
            else tuple(repr(warning.get('fields', {}).get(f)) for f in group[0])
        )
        if group is not None and shown and shown[-1][:2] == (code, key):
            shown[-1][2].append(warning)
        else:
            shown.append((code, key, [warning]))
    for code, _, together in shown:
        if code in GROUPED:
            _, opening, between, closing = GROUPED[code]
            first = {
                name: _detail(value)
                for name, value in together[0].get('fields', {}).items()
            }
            text = (
                _Words().format(opening, **first)
                + between.join(said(warning) for warning in together)
                + closing
            )
        else:
            [warning] = together
            text = said(warning)
        warnings.warn(
            f'[{code}] {text}',
            category=WARNINGS.get(code, SDMWarning),
            stacklevel=caller_stacklevel(),
        )


def warn_held(held: list[dict[str, Any]]) -> None:
    r"""Shows warnings :func:`warn_from_core` held back, in order, each as it
    would have been shown (none of them a code :data:`GROUPED` holds)."""
    for warning in held:
        code = warning['code']
        assert code not in GROUPED, code
        warnings.warn(
            f'[{code}] {said(warning)}',
            category=WARNINGS.get(code, SDMWarning),
            stacklevel=caller_stacklevel(),
        )
