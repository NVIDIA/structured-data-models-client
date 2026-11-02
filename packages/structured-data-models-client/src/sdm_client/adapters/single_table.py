# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Rules every single-table request obeys, whichever model answers it.

A request with one table and no relationships is read the same way for every
such model: the caller's key goes back on each answer and is checked against
the batch it claims to belong to, and a column's declared meaning is held to
something its dtype can actually carry. Both were written for `kumo-tabular`
and neither is about it, so they live here rather than being reached for
across adapters or written a second time.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from sdm_client.errors import StructuredDataError
from sdm_client.schema import Column, supports_dtype
from sdm_client.wire import as_declared

# What the contract allows a `PredictionItem.id` to be.
MAX_ID_LENGTH = 1024


def key_named_by(value: str) -> str | None:
    r"""The key an id names, or ``None`` if it names no key this can read.

    The endpoint answers under the list of primary-key values, and how it
    renders that list is its own business: a key that travelled as a JSON
    string comes back quoted, one that travelled as a number does not, and one
    carrying a newline or a backslash comes back with that escaped. So the
    rendering is read back into the value it stands for, and the value is what
    gets compared. Taking the text apart by hand instead would refuse a correct
    answer for any key an escape appears in.

    Read as JSON first, because JSON is what the endpoint answers with, and
    only as a Python literal if that fails. The two agree on everything except
    a character above U+FFFF: JSON writes it as a surrogate pair and
    `ast.literal_eval` reads the halves as two separate characters that never
    recombine, so a key containing one could never be matched back and the
    request failed at the client's own id check.

    Anything not written that way is left alone rather than refused. The
    contract calls this field a *service-generated* row identifier and names
    ``row_index`` as what to align on, so an endpoint is entitled to answer
    under a token of its own and refusing one would break a compliant
    deployment. It bounds an id at 1024 characters, so anything longer is not
    one either, and is declined before being parsed.

    A value that is not JSON is read as the Python rendering it must then be.
    """
    import ast

    if len(value) > MAX_ID_LENGTH:
        return None
    named: Any = None
    try:
        named = json.loads(value)
    except (ValueError, TypeError, RecursionError):
        try:
            named = ast.literal_eval(value)
        except (
            ValueError,
            SyntaxError,
            TypeError,
            MemoryError,
            RecursionError,
        ):
            return None
    if not isinstance(named, list) or len(named) != 1:
        return None
    return str(named[0])


def named_entities(
    items: Sequence[Any], payload: dict[str, Any] | None
) -> list[Any]:
    r"""Put the caller's key back on each row, and check it belongs there.

    The endpoint answers under the stringified list of primary-key values, so
    a row the caller asked about as ``1`` comes back as ``'[1]'``. The
    relational path already returns the caller's own key, and
    :mod:`sdm_client.result` tells callers to join on it, so a key that only
    joins on one of the two models is a difference they have no reason to
    expect.

    The key is read from the payload that produced this response, so a run
    split into several calls maps each batch against its own rows.

    **A mismatch is an error, not a silent pass.** Rewriting an id is exactly
    the operation that can misattribute an answer, so a row is checked against
    the batch it claims to belong to before its id is replaced. The endpoint
    answers under the list of key values and the relational endpoint answers
    under the key itself, so a key is recognised in either shape. An id
    recognisable as a key has to be the key of the row it was paired with. An
    id in any other shape is not second-guessed: it may be a correlation token
    of the endpoint's own.

    Those checks are about the batch rather than the key, so they also hold for
    a request that has no key to put back: a composite key, or rows the caller
    supplied without the key column, still have to be answered once each and
    no more. Such a request has no value to restore. A null key is the same
    case. ``str`` would turn it into ``'None'``, which
    reads as a key that was asked about. Rows in a shape this cannot read at
    all, a format other than ``arrays``, are left alone entirely.
    """
    import dataclasses

    if payload is None:
        return list(items)
    table = payload.get('predict', {}).get('instance_table', {})
    rows = table.get('rows')
    if not isinstance(rows, list):
        return list(items)
    columns = table.get('columns') or []

    key = (
        payload.get('schema', {}).get('instance_table', {}).get('primary_key')
        or []
    )
    named_key = key[0] if len(key) == 1 else None
    key_dtype = (
        payload.get('schema', {})
        .get('instance_table', {})
        .get('columns', {})
        .get(named_key, {})
        .get('dtype')
        if named_key
        else None
    )
    keys = (
        [as_declared(row[columns.index(named_key)], key_dtype) for row in rows]
        if named_key and named_key in columns
        else None
    )
    keys_in_batch = {str(value) for value in keys or ()}

    named: list[Any] = []
    seen: set[int] = set()
    for item in items:
        index = item.row_index
        if index is None:
            raise StructuredDataError(
                'the response is missing a row_index, so its answer cannot be '
                'attributed to a row',
                code='INVALID_RESPONSE',
            )
        if not 0 <= index < len(rows):
            raise StructuredDataError(
                f'the response names row_index {index}, and this batch asked '
                f'about {len(rows)} rows',
                code='INVALID_RESPONSE',
            )
        if index in seen:
            raise StructuredDataError(
                f'the response answers row {index} more than once',
                code='INVALID_RESPONSE',
            )
        if keys is not None and item.id is not None:
            answered = str(item.id)
            named_key = key_named_by(answered)
            if named_key is None and answered in keys_in_batch:
                named_key = answered
            if named_key is not None and named_key != str(keys[index]):
                raise StructuredDataError(
                    f'the response pairs row_index {index} with id '
                    f'{item.id!r}, and this batch asked about '
                    f'{keys[index]!r} there',
                    code='INVALID_RESPONSE',
                )
        seen.add(index)
        value = None if keys is None else keys[index]
        named.append(
            item if value is None else dataclasses.replace(item, id=value)
        )

    if seen != set(range(len(rows))):
        missing = sorted(set(range(len(rows))) - seen)
        raise StructuredDataError(
            f'the response left row(s) {missing} unanswered',
            code='INVALID_RESPONSE',
        )
    return named


#: What a dtype means when nothing else says. Only reached for a column whose
#: meaning was never claimed, so it is the dtype's own reading rather than a


#: guess about the data.
MEANING_OF = {
    'int64': 'numerical',
    'int32': 'numerical',
    'float64': 'numerical',
    'float32': 'numerical',
    'bool': 'categorical',
}


def coherent_stype(stype: str, dtype: str, column: Column) -> str:
    r"""A meaning the dtype can actually carry.

    An all-null context column is typed `categorical` because a schema has to
    say something. The scored rows then decide its dtype, and if they hold
    timestamps the column goes out as `timestamp[us]` meaning `categorical`,
    which is not a thing a column can be, and is the same self-contradiction on
    the wire that a scored frame re-typing a real column used to produce.

    Only the placeholder is overridden. A meaning the caller assigned is a
    claim and is refused rather than quietly rewritten, before it reaches here.
    """
    if supports_dtype(stype, dtype) or column.stype_declared:
        return stype
    # Every spelling of a timestamp, not one of them. The contract writes
    # `timestamp[us]`, but a hand-built schema may say `[ns]`, `[ms]` or `[s]`,
    # and reading those as categorical puts a dtype and a meaning on the wire
    # that contradict each other. That is what this exists to prevent.
    if dtype.startswith('timestamp'):
        return 'timestamp'
    return MEANING_OF.get(dtype, 'categorical')
