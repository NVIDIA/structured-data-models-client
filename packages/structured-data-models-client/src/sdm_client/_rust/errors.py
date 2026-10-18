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
import contextlib
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


#: Refusals this client words beyond their code's sentence, by code and
#: the ``field`` the error names, and, where one field says several
#: things, by its ``rule`` or ``remedy`` too: a request or call past the
#: endpoint's size says the size and the limit, as the old route's refusal
#: does (T1.28b, carried from T1.28a's review), and from T2.8c what to
#: change, in the reference's words (``main``'s 1fc5a686,
#: ``_said_in_this_client_s_terms``); the ceilings on what one
#: ``materialize`` gathers, and a forecast over more entities than one call
#: scores, say the counts, as the reference does. A template is
#: ``str.format``'s over the error's fields; one its fields cannot fill
#: gives the code's sentence.
_TOO_LARGE = (
    'the request is {bytes:,} bytes and the endpoint takes at most {limit:,} '
    'bytes; reduce the number of tables in the graph, their columns, the '
    'neighbourhood sampled or the context gathered (num_context_samples)'
)
_SIZE = (
    'Context size exceeds the 30MB limit: this call is {bytes:,} bytes and '
    'the endpoint takes at most {limit:,}. '
)
_ASK_IN_SMALLER_CALLS = _SIZE + (
    'What comes down is one of three. For an ordinary question it is the '
    'call: materialize(rows_per_call=N) below the {rows_per_call:,} this one '
    'holds asks for the same entities in more and smaller calls. Where the '
    'context is itself over the limit, no call is small enough, and it is '
    'num_context_samples, num_neighbors or the columns the graph carries. '
    'For a ranking or a forecast, which is one call per candidate or step '
    'and is never batched, it is the entities.'
)
_NO_CALL_IS_SMALLER = _SIZE + (
    'A call already scores one row, so there is no smaller one to ask for: '
    'what comes down is num_context_samples, num_neighbors or the columns '
    'the graph carries.'
)
_ONE_MATERIALIZE_HOLDS = (
    'is past the {limit:,} one materialize() will hold at once, since every '
    'batch is built, with its own context, before any is sent; ask about '
    'fewer entities at a time'
)
WORDED: dict[tuple[str, ...], str] = {
    ('INVALID_REQUEST', 'payload'): _TOO_LARGE,
    ('INVALID_REQUEST', 'payload', 'smaller_calls'): _ASK_IN_SMALLER_CALLS,
    ('INVALID_REQUEST', 'payload', 'no_smaller_call'): _NO_CALL_IS_SMALLER,
    ('INVALID_REQUEST', 'call'): _TOO_LARGE,
    ('INVALID_REQUEST', 'rows_per_call'): (
        'rows_per_call is not a whole number of at least one, and a call '
        'scores a whole number of rows, at least one'
    ),
    ('UNSUPPORTED_REQUEST', 'indices', 'most_entities'): (
        '{entities:,} entities ' + _ONE_MATERIALIZE_HOLDS
    ),
    ('UNSUPPORTED_REQUEST', 'indices', 'most_entities_parts'): (
        '{entities:,} entities gathered {gathered_times:,} times over, '
        '{total:,} in all, ' + _ONE_MATERIALIZE_HOLDS
    ),
    ('UNSUPPORTED_REQUEST', 'indices', 'most_calls'): (
        '{entities:,} entities are gathered as {calls:,} calls of '
        '{rows_per_call:,}, and every one is built, with its own context, '
        'before any is sent. That is past the {limit:,} calls this will hold '
        'at once; ask about fewer entities at a time'
    ),
    ('UNSUPPORTED_REQUEST', 'indices', 'batched_part'): (
        '{entities:,} entities are more than the {rows_per_call:,} one call '
        'scores, and this question is asked as {parts:,} of them, one per '
        'step. A part is not batched, and the calls would be the batches '
        'times the steps rather than the steps that were counted. Ask about '
        'fewer entities at a time'
    ),
}


def _worded(code: str, details: dict[str, Any]) -> str | None:
    r"""The template :data:`WORDED` holds for an error of ``code`` with
    ``details``: by its field and rule or remedy where one is held, else by
    its field."""
    field = str(details.get('field'))
    said = details.get('rule', details.get('remedy'))
    if said == 'most_entities' and details.get('gathered_times', 1) != 1:
        said = 'most_entities_parts'
    return WORDED.get((code, field, str(said)), WORDED.get((code, field)))


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
#: ``_and_list`` writes them, ``!j`` for one joined by commas, each as its
#: ``repr``, and ``!c`` each as it is, ``!t`` for a date or timestamp as pandas
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
        **{
            f'taken_unchecked_{reason}': (
                '{table}.{column} was taken as the time column although '
                + why
                + ', because the query over the whole table failed and the '
                'rows past them were not checked. If the whole table is like '
                'that, clear it: time_columns={{{table!r}: None}}.'
            )
            for reason, why in (
                ('empty', 'none of the {rows:,} rows read holds a value'),
                (
                    'sparse',
                    '{missing:,} of the {rows:,} rows read hold no value',
                ),
                (
                    'constant_where_held',
                    'every one of the {rows:,} rows read that holds a value '
                    'holds {value!t}',
                ),
                (
                    'constant',
                    'every one of the {rows:,} rows read holds {value!t}',
                ),
            )
        },
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
    # The key rules (T1.32e): the reference's `_warn_a_key_has_no_value`,
    # `_warn_a_key_repeats` and `_warn_a_key_was_declined`.
    # Rows a walk reached left out for a null text or decimal key beside
    # keys that are not null (T1.32e, register P1B-40), where the reference
    # fails; this client's own words.
    'NULL_KEYS_LEFT_OUT': {
        None: (
            '{table}.{column} has no value in {rows:,} of the rows sampled, '
            'beside rows that do, so those rows were left out of the request: '
            'a row with no key cannot be told apart. Fill or drop them, or '
            'name a different primary key.'
        )
    },
    'KEY_NO_VALUE': {
        None: (
            '{table}.{column} identifies a row, but {missing:,} row(s) have '
            'no value for it. Those rows cannot be scored and the model will '
            'not see them; fill or drop them, or name a different primary key.'
        )
    },
    'KEY_REPEATS': {
        None: (
            '{table}.{column} identifies a row {repeats:,} time(s) it has '
            'already been used, e.g. {example!r}. Those rows are '
            'indistinguishable to the model. Name a different primary key if '
            'this column is a foreign key rather than an identifier.'
        )
    },
    'KEY_DECLINED': {
        None: (
            '{table}.{column} is named like a key and another table names it '
            'too, but {repeats:,} of {rows:,} rows repeat a value already '
            'used, e.g. {example!r}, so it does not identify a row and '
            '{table!r} gets no primary key. Nothing can link to a table '
            'without one: every relationship pointing at {table!r} is '
            'dropped, so nothing is gathered through it, and a question about '
            'another table reaches it only where {table!r} names that table '
            'itself. Keep one row per {column!r} in the source, or name a '
            'column that does identify a row with '
            'primary_keys={{{table!r}: ...}}.'
        )
    },
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
    # The client's notices of what a request leaves out, which the core
    # gives from T1.28a, in the client's words (SDMWarning, as the client
    # shows them).
    'CONTEXT_CAPPED': {
        None: (
            '{cap:,} context examples were gathered, which is all this route '
            'takes by default; pass num_context_samples to ask for up to '
            '{most:,}.'
        )
    },
    'TABLES_UNLINKED': {
        many: (
            '{tables!j} '
            + ('are' if many == 'several' else 'is')
            + ' in the schema and not in this request: no relationship '
            'connects '
            + them
            + ' to {entity_table!r}, so the model answers without '
            + them
            + '. Say how '
            + ('they join' if many == 'several' else 'it joins')
            + ': schema.relationships.append(Relationship({tables[0]!r}, '
            '<foreign key>, <parent table>, <its primary key>)), or leave '
            + them
            + ' out of the schema to say '
            + ('they are' if many == 'several' else 'it is')
            + ' not wanted.'
        )
        for many, them in (('one', 'it'), ('several', 'them'))
    },
    'TABLES_UNREACHED': {
        many: (
            '{tables!j} '
            + ('are' if many == 'several' else 'is')
            + ' linked into the schema and not in this request: the sampler '
            'reached no row of '
            + them
            + ' from the entities being scored, so the model answers without '
            + them
            + '. Raise num_hops if '
            + ('they lie' if many == 'several' else 'it lies')
            + ' further out than the hops sampled, and check the foreign key '
            'holds values the parent key actually has if '
            + ('they do' if many == 'several' else 'it does')
            + ' not.'
        )
        for many, them in (('one', 'it'), ('several', 'them'))
    },
    'COLUMNS_UNREAD': {
        many: (
            '{columns!c} '
            + ('are' if many == 'several' else 'is')
            + ' typed text, which this model does not read, so the request '
            'is answered without '
            + them
            + ' and '
            + ('they contribute' if many == 'several' else 'it contributes')
            + ' nothing to the prediction. Type a column categorical if its '
            'values are labels rather than prose, or leave it out to say so.'
        )
        for many, them in (('one', 'it'), ('several', 'them'))
    },
    'CONTEXT_ANSWERS': {
        variant: (
            '{examples} of the rows sampled as context are the rows being '
            'scored, '
            + when
            + ', carrying the value being asked for. Rows that carry the '
            'answer to the question being asked are not sent as context, so '
            'these do not go into it.'
        )
        for variant, when in (
            ('windowed', 'at the same moment'),
            ('static', 'at one moment or another'),
        )
    },
    # The client's notices around the model call, which the core gives from
    # T1.28b: a regression's constant context (the adapter's
    # `_check_target_varies`, the target named as `_how_to_name_the_target`
    # names it) and an output the caller named that the answer left out
    # (`_report_absent`).
    'CONTEXT_CONSTANT': {
        variant: (
            'every context row has the same '
            + said
            + ', so nothing in the context distinguishes one answer from '
            'another. The endpoint will answer, but the answer cannot say '
            'much. Gather a wider context with num_context_samples, or ask '
            'about entities whose answers differ.'
        )
        for variant, said in (
            ('column', '{target}'),
            ('predicted', '{predicted}'),
            ('question', 'answer to {question!r}'),
            ('answer', 'answer'),
        )
    },
    'OUTPUTS_ABSENT': {
        None: (
            '{model} answered without {absent!r}, which this request asked '
            'for, so no column of to_pandas() will carry it. The answer to '
            'the rest of the request is unaffected.'
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
        if conversion == 'j':
            return ', '.join(repr(name) for name in value)
        if conversion == 'c':
            return ', '.join(str(name) for name in value)
        if conversion == 't':
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

        unit = {'us': 'us', 'ns': 'ns'}[str(value.unit)]
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
    micros = count // 1_000 if unit == 'ns' else count
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
    template = _worded(code, details)
    if template is not None:
        # Fields its template cannot fill give the code's sentence.
        with contextlib.suppress(KeyError, IndexError, TypeError, ValueError):
            sentence = template.format(**details)
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
