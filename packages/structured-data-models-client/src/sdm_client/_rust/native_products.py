# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call-local ownership of closed native preparation products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sdm_client._rust import codec


@dataclass(frozen=True)
class Product:
    """A decoded useful value and exactly its explicitly named attachments."""

    kind: str
    data: dict[str, Any]
    owners: dict[int, Any]


class Bindings:
    """One message's attachments, with no persistent numbering or registry."""

    def __init__(self) -> None:
        self.attachments: dict[int, Any] = {}

    def add(self, owner: Any) -> int:
        key = len(self.attachments) + 2
        self.attachments[key] = owner
        return key

    def json(self, value: Any, *, options: bool = False) -> int:
        return self.add(
            codec.attachment(value, named=options, option_values=options)
        )


def label_input_ids(value: dict[str, Any]) -> list[int]:
    """Explicit typed column views and completed value resources only."""
    out = [value['owner']]
    history = value.get('eligibility')
    if history is not None:
        for column in history['columns']:
            out.extend(label_input_ids(column))
        out.extend(column['owner'] for column in history['values'])
    return out


def bound_label_input(
    value: dict[str, Any], bound: dict[int, int]
) -> dict[str, Any]:
    out = {**value, 'owner': bound[value['owner']]}
    history = value.get('eligibility')
    if history is not None:
        out['eligibility'] = {
            'columns': [
                bound_label_input(column, bound)
                for column in history['columns']
            ],
            'values': [
                {**column, 'owner': bound[column['owner']]}
                for column in history['values']
            ],
        }
    return out


def group_ids(value: dict[str, Any]) -> list[int]:
    out = [value['owner']]
    restored = value.get('restored')
    if restored is not None:
        out.append(restored['rows'])
    history = value.get('what_if')
    if history is not None:
        for column in history['columns']:
            out.extend(label_input_ids(column))
        out.extend(column['owner'] for column in history['values'])
    return out


def bound_groups(
    value: dict[str, Any], bound: dict[int, int]
) -> dict[str, Any]:
    out = {**value, 'owner': bound[value['owner']]}
    restored = value.get('restored')
    if restored is not None:
        out['restored'] = {**restored, 'rows': bound[restored['rows']]}
    history = value.get('what_if')
    if history is not None:
        out['what_if'] = {
            'columns': [
                bound_label_input(column, bound)
                for column in history['columns']
            ],
            'values': [
                {**column, 'owner': bound[column['owner']]}
                for column in history['values']
            ],
        }
    return out
