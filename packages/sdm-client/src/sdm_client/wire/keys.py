# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Which rows an encoded request is about.

The keys are in the envelope rather than in anything the caller still holds,
and the two models write them differently: a relational envelope carries an
opaque entity reference, a tabular one the caller's own key column. Reading
that is a question about the wire, so it is answered here rather than by the
client, which should have to ask only whether two sets of rows overlap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from numbers import Real
from typing import TYPE_CHECKING, Any

from sdm_client.schema import _as_key_tuple

if TYPE_CHECKING:
    from sdm_client.request import Request

#: What a relational envelope calls the moment, when nothing of the caller's
#: is called that already.
_ANCHOR = '__kumo_anchor_time'

#: How a relational envelope names the entity a row is about.
_ENTITY_REF = '__kumo_entity_ref'

#: What ties a related row to the instance it hangs off.
INSTANCE_ID = 'instance_id'


def _anchor_column(payload: dict[str, Any] | None) -> str:
    r"""What this envelope calls the moment.

    Read from the task rather than assumed. The engine suffixes the name when
    a column of the caller's own is called that, so looking only for the bare
    name goes blind on exactly the graphs that have one.
    """
    task = (payload or {}).get('task')
    named = (
        task.get('anchor_time_column') if isinstance(task, Mapping) else None
    )
    return str(named) if named else _ANCHOR


def rows_in(
    payload: dict[str, Any] | None, half: str, *, with_anchor: bool = False
) -> list[Any]:
    r"""The key of every row one half of a wire envelope holds, one per row.

    The row-wise reading. :func:`keys_in` is this without the duplicates, and
    is what a question about overlap wants; a sentence that counts rows wants
    this, because two rows about one entity are two rows.
    """
    if not isinstance(payload, dict):
        return []
    table = (payload.get(half) or {}).get('instance_table') or {}
    columns = table.get('columns')
    rows = table.get('rows')
    if not isinstance(columns, list) or not isinstance(rows, list):
        return []
    named: Any = next(
        (name for name in columns if str(name).startswith(_ENTITY_REF)),
        None,
    )
    if named is None:
        named = ((payload.get('schema') or {}).get('instance_table') or {}).get(
            'primary_key'
        )
    if named is None:
        return []
    wanted = named if isinstance(named, list) else [named]
    if any(name not in columns for name in wanted):
        return []
    at = [columns.index(name) for name in wanted]
    if with_anchor:
        anchor = _anchor_column(payload)
        if anchor in columns:
            at = [*at, columns.index(anchor)]
    return [
        _named_by(row[position] for position in at)
        for row in rows
        if isinstance(row, list) and len(row) > max(at)
    ]


def keys_in(
    payload: dict[str, Any] | None, half: str, *, with_anchor: bool = False
) -> frozenset[Any]:
    r"""The entity keys one half of a wire envelope holds.

    Kept when a session is opened, and only the keys: the context itself is
    deliberately not kept, since holding a second encoded copy of a whole
    dataset for the life of the handle is what `pinned` exists to avoid. A set
    of keys is small enough to keep and is the only thing needed to notice that
    a later call scores rows the context already answers.

    The key is named in the schema rather than in the rows, and the two models
    name it differently: a relational envelope carries an opaque entity
    reference, a tabular one carries the caller's own key column.

    `with_anchor` puts the moment into the key. An entity asked about over a
    window is a different question at a different moment, so the context has
    only answered the one it was asked at; an entity whose answer is a
    standing value has been given away at any moment at all.
    """
    return frozenset(rows_in(payload, half, with_anchor=with_anchor))


def _named_by(values: Iterable[Any]) -> str:
    r"""One way of writing a key, whichever side of the wire it came from.

    A pinned key is read from the encoded payload and a scored key from the
    frame, and the two disagree about the same value: an integer past the
    JS-safe range travels as a base-10 string by contract, so
    `9007199254740993` on one side and `'9007199254740993'` on the other would
    never meet, and the warning would go quiet on precisely the warehouse keys
    it matters most for.

    A whole number written as a float is the same entity as the integer: a
    context typed `int64` pins entity `1` and a scored frame of floats sends
    `1.0`, and the endpoint answers the second with the first's id. Comparing
    the spellings called that two entities, and a session scored rows its
    context answers without saying so.

    A composite key becomes a tuple of its parts rather than nothing at all,
    which is what reading only single-column keys amounted to.
    """
    return repr(tuple(_one_spelling(value) for value in values))


def _one_spelling(value: Any) -> str:
    r"""A value as the name of an entity, not as the type that carried it.

    Every real number is read as the number it is, rather than a list of the
    types that can hold one: ``1``, ``1.0``, a numpy float of any width and
    ``Decimal("1.00")`` all name entity 1, and ``Decimal("1.50")`` names the
    same entity as ``1.5``. Written by type instead, a holdout given as a
    ``float32`` removed nothing and a trailing zero made two names for one row.

    Anything that is not a number is left as it is written. A key column of
    text has ``"1.50"`` and ``"1.5"`` as two different entities, and reading
    them as numbers would merge two rows the schema says are distinct.
    """
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        return str(value)
    try:
        exact = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return str(value)
    if exact == exact.to_integral_value():
        return str(int(exact))
    return str(exact.normalize())


def key_dtypes(
    table: Any, key: Sequence[str], widened_by: Any = None
) -> dict[str, str] | None:
    r"""One wire dtype per key column, settled the way the adapter settles it.

    One per column, not one per half. Settled separately for each frame, the
    same entity came out ``1.0`` on the side read from the context and ``1``
    on the side read from the rows being scored, and the comparison between
    them reported overlaps the encoded request does not have.

    ``None`` where no dtype can be settled. Every caller is a warning, and a
    warning is never worth a failure.
    """
    from sdm_client.adapters.tabular import _raw_dtype_of
    from sdm_client.wire import UNTYPED, widen_tfm_dtype

    try:
        dtypes = {}
        for name in key:
            column = table.columns[name]
            dtype = UNTYPED if column.untyped else column.dtype
            if widened_by is not None and name in widened_by.columns:
                dtype = widen_tfm_dtype(
                    (dtype, _raw_dtype_of(widened_by[name], dtype))
                )
            dtypes[name] = dtype or 'string'
        return dtypes
    except Exception:  # noqa: BLE001 - a warning is never worth a failure
        return None


def keys_from_a_frame(
    frame: Any, key: Sequence[str], dtypes: Mapping[str, str] | None
) -> list[Any]:
    r"""``frame``'s keys, written the way the wire will write them.

    The other side of every comparison here is read from an encoded envelope,
    so a frame read raw does not meet it: an entity held as ``1.0`` in a frame
    is ``1`` on the wire, and the two would never match -- which is how a
    session came to score rows its pinned context answers without saying so.

    Through the encoder that writes the request, given the dtypes
    :func:`key_dtypes` settled for the request as a whole. Empty where there
    are none or the encoding fails: every caller is a warning, and a warning
    is never worth a failure.
    """
    from sdm_client.wire.tables import encode_table

    if dtypes is None:
        return []
    try:
        encoded = encode_table(
            frame[list(key)], {name: dtypes[name] for name in key}
        )
    except Exception:  # noqa: BLE001 - a warning is never worth a failure
        return []
    return [_named_by(row) for row in encoded['rows']]


def scored_rows(
    request: Request,
    *,
    with_anchor: bool = False,
    dtypes: Mapping[str, str] | None = None,
) -> list[Any]:
    r"""The key of every row a request asks about, one per row.

    A sentence that counts rows has to be given rows: two rows asking about
    one entity are two rows, and a set of keys had already lost one of them.

    ``dtypes`` are the ones the other side of the comparison was written with
    -- a session's pinned schema. Settled here instead, the same entity has
    one name on each side and neither meets the other.
    """
    if request.payloads:
        return [
            named
            for payload in request.payloads
            for named in rows_in(payload, 'predict', with_anchor=with_anchor)
        ]
    frame = getattr(request.predict, 'instance_table', None)
    table = request.schema[request.entity_table] if request.schema else None
    names = _as_key_tuple(getattr(table, 'primary_key', None))
    if frame is None or not names or any(n not in frame for n in names):
        return []
    if dtypes is None:
        dtypes = key_dtypes(table, names, frame)
    return keys_from_a_frame(frame, names, dtypes)


def scored_keys(
    request: Request, *, with_anchor: bool = False
) -> frozenset[Any]:
    r"""The keys a request is asking about.

    Read from the envelopes where there are any, and from the frame where
    there are not. Only the first can carry a moment: a request holding
    frames is a single-table one, and a single-table model is not asked for a
    window over time -- the query is refused before it reaches here, so there
    is no moment in the frame to put in the key and none to look for.
    """
    return frozenset(scored_rows(request, with_anchor=with_anchor))


def without_the_keys(payload: dict[str, Any], keys: frozenset[Any]) -> int:
    r"""Take out of ``payload``'s context every row naming one of ``keys``.

    Returns how many instance rows went. Related rows that hang off them go
    too, by the ``instance_id`` they were tied with, because a related row
    whose instance has gone describes nothing.

    The key is found the way `keys_in` finds it, so what is removed here is
    exactly what would have been reported as an overlap there.
    """
    context = payload.get('context')
    if not isinstance(context, Mapping) or not keys:
        return 0
    table = context.get('instance_table')
    columns = (table or {}).get('columns')
    rows = (table or {}).get('rows')
    if not isinstance(columns, list) or not isinstance(rows, list):
        return 0

    named = next(
        (name for name in columns if str(name).startswith(_ENTITY_REF)), None
    )
    if named is None:
        named = ((payload.get('schema') or {}).get('instance_table') or {}).get(
            'primary_key'
        )
    wanted = named if isinstance(named, list) else [named]
    if named is None or any(name not in columns for name in wanted):
        return 0

    at = [columns.index(name) for name in wanted]
    instance = columns.index(INSTANCE_ID) if INSTANCE_ID in columns else None
    going, leaving = [], set()
    for row in rows:
        if not isinstance(row, list) or len(row) <= max(at):
            going.append(row)
            continue
        if _named_by(row[position] for position in at) in keys:
            if instance is not None and len(row) > instance:
                leaving.add(row[instance])
            continue
        going.append(row)

    removed = len(rows) - len(going)
    if not removed:
        return 0
    table['rows'] = going
    for related in (context.get('related_tables') or {}).values():
        _drop_instances(related, leaving)
    return removed


def _drop_instances(table: Any, instances: set[Any]) -> None:
    r"""Take out of a related table the rows tied to instances that have gone."""
    columns = (table or {}).get('columns')
    rows = (table or {}).get('rows')
    if not isinstance(columns, list) or not isinstance(rows, list):
        return
    if INSTANCE_ID not in columns or not instances:
        return
    at = columns.index(INSTANCE_ID)
    table['rows'] = [
        row
        for row in rows
        if not (
            isinstance(row, list) and len(row) > at and row[at] in instances
        )
    ]
