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

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from sdm_client.schema import _as_key_tuple

if TYPE_CHECKING:
    from sdm_client.request import Request

#: What a relational envelope calls the moment, when nothing of the caller's
#: is called that already.
_ANCHOR = '__kumo_anchor_time'


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
    if not isinstance(payload, dict):
        return frozenset()
    table = (payload.get(half) or {}).get('instance_table') or {}
    columns = table.get('columns')
    rows = table.get('rows')
    if not isinstance(columns, list) or not isinstance(rows, list):
        return frozenset()
    named: Any = next(
        (name for name in columns if str(name).startswith('__kumo_entity_ref')),
        None,
    )
    if named is None:
        named = ((payload.get('schema') or {}).get('instance_table') or {}).get(
            'primary_key'
        )
    if named is None:
        return frozenset()
    wanted = named if isinstance(named, list) else [named]
    if any(name not in columns for name in wanted):
        return frozenset()
    at = [columns.index(name) for name in wanted]
    if with_anchor:
        anchor = _anchor_column(payload)
        if anchor in columns:
            at = [*at, columns.index(anchor)]
    return frozenset(
        _named_by(row[position] for position in at)
        for row in rows
        if isinstance(row, list) and len(row) > max(at)
    )


def _named_by(values: Iterable[Any]) -> str:
    r"""One way of writing a key, whichever side of the wire it came from.

    A pinned key is read from the encoded payload and a scored key from the
    frame, and the two disagree about the same value: an integer past the
    JS-safe range travels as a base-10 string by contract, so
    `9007199254740993` on one side and `'9007199254740993'` on the other would
    never meet, and the warning would go quiet on precisely the warehouse keys
    it matters most for.

    A composite key becomes a tuple of its parts rather than nothing at all,
    which is what reading only single-column keys amounted to.
    """
    return repr(tuple(str(value) for value in values))


def scored_keys(
    request: Request, *, with_anchor: bool = False
) -> frozenset[Any]:
    if request.payloads:
        return frozenset().union(
            *(
                keys_in(payload, 'predict', with_anchor=with_anchor)
                for payload in request.payloads
            )
        )
    frame = getattr(request.predict, 'instance_table', None)
    table = request.schema[request.entity_table] if request.schema else None
    names = _as_key_tuple(getattr(table, 'primary_key', None))
    if frame is None or not names or any(n not in frame for n in names):
        return frozenset()
    return frozenset(
        _named_by(values)
        for values in zip(*(frame[n] for n in names), strict=True)
    )
