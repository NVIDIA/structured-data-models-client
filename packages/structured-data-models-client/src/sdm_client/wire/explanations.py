# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Reading the explanation each prediction record carries.

Both models answer an explanation in one format, ``sdm_gradient_v1``, one per
record, about that record's own row. Its blocks name rows by their position in
the tables of the call that produced it, and a request sent in several calls
starts those positions at 0 in each. So each record is read here, beside the
payload of its own call, into an entry that needs no payload to be read: the
row it is about becomes its key, and a related row its key and its values.

The format is the endpoint's: ``gradients`` is ``[output][row][column]``, and
``columns`` are the model's preprocessed numerical columns, passed through.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sdm_client.result import Explanations

#: Reads one record's ``related_tables`` into the entry's ``related``, and
#: says how many links each of those rows is from the entity.
RelatedReader = Callable[
    [Mapping[str, Any]],
    tuple[dict[str, Any], dict[str, list[int | None]]],
]


def explanations_of(
    items: Sequence[Any], related: RelatedReader | None = None
) -> Explanations:
    r"""Each answered row's explanation, under the key naming that row.

    ``items`` have their ``id`` restored already, so the key is the one the
    frame shows: the ``id``, or the ``row_index`` when the rows carry none.
    ``related`` is given for a model that reads related tables. Then an entry
    has ``related`` and no ``instance``: that model's ``instance_table`` is the
    question row its engine sends, not the caller's, and the entity's own table
    is among the related ones. Without it an entry has ``instance`` and no
    ``related`` at all, rather than an empty one.
    """
    explained = [item for item in items if item.explanation]
    if not explained:
        return Explanations()
    by_id = any(item.id is not None for item in explained)
    entries: dict[Any, dict[str, Any]] = {}
    hops: dict[Any, dict[str, list[int | None]]] = {}
    for item in explained:
        key = item.id if by_id else item.row_index
        details = item.explanation.get('details') or {}
        entry: dict[str, Any] = {'outputs': list(details.get('outputs') or [])}
        if related is None:
            instance = details.get('instance_table') or {}
            entry['instance'] = {
                'columns': list(instance.get('columns') or []),
                # One row, the record's own, so the row axis says nothing.
                'gradients': [
                    by_row[0] for by_row in instance.get('gradients') or []
                ],
            }
        else:
            entry['related'], hops[key] = related(
                details.get('related_tables') or {}
            )
        entries[key] = entry
    return Explanations(
        entries,
        keyed_by=('id',) if by_id else ('row_index',),
        hops=hops,
    )
