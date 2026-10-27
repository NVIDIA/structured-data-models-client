# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual native grouping observations between fixed ranking score reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sdm_client._rust import errors
from sdm_client._rust.relational_response import CompletedResponse, _call


@dataclass(frozen=True)
class _Displayed:
    text: str

    def __repr__(self) -> str:
        return self.text


def population(
    source: dict[str, Any], attached: dict[int, bytes]
) -> dict[str, Any]:
    """Ask Rust for numbered child rows and normalized probability data."""
    return _call(
        {'stage': 'prepare_relational_rank_population', 'source': source},
        attached,
    )


def groups(
    population: dict[str, Any],
    owners: Mapping[str, CompletedResponse],
    candidates: Mapping[int, Any],
) -> dict[str, Any]:
    """Observe the original two dictionaries in their actual source order.

    The values are local source ordinals. Only actual returned dictionary
    membership crosses the boundary; no process identity, surrogate equality
    or host-selected confidence/rank is reported.
    """
    scores: dict[Any, list[int]] = {}
    asked_about: dict[Any, tuple[int, Any, int]] = {}
    membership = []
    displayed: dict[int, str] = {}

    def selected_candidate(fields: dict[str, Any]) -> str:
        position = fields['candidate']
        shown = repr(candidates[position])
        displayed[position] = shown
        return shown

    def candidate(fields: dict[str, Any]) -> Any:
        return _Displayed(displayed[fields['candidate']])

    with errors.holding(
        relational_rank_candidate=selected_candidate, candidate=candidate
    ):
        for item in population['items']:
            lineage = item['lineage']
            entity = (
                item['unkeyed_id']
                if lineage['source_row'] is None
                else owners[lineage['call_request_id']].entity(lineage)
            )
            where = (item['row_index'], entity)
            # Native hashing/equality occurs here before confidence can fail.
            scored = scores.setdefault(where, [])
            _call({'stage': 'rank_relational_score', 'item': item})
            scored.append(item['ordinal'])
            # This is a second real native dict operation, after the append.
            asked = asked_about.setdefault(
                where, (item['row_index'], entity, item['ordinal'])
            )
            membership.append(
                {
                    'ordinal': item['ordinal'],
                    'score_first': scored[0],
                    'asked_first': asked[2],
                }
            )
        grouped = []
        for where, scored in scores.items():
            # Retain the final real lookup, including mutable hash/equality
            # behavior and its native failure, before pure core sorting.
            asked = asked_about[where]
            grouped.append(
                {'members': list(scored), 'representative': asked[2]}
            )
    return {'membership': membership, 'groups': grouped}
