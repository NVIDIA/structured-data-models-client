# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What every provider of the client's core routes shares (design fix F):
the :class:`Provider` protocol the run loop calls, and the rules of a SQL
plan each driver's provider applies before it runs anything, written once.

A provider answers one fetch step: it runs the statements the core's plan
carries on the user's own driver and hands back the last one's result as
Arrow. What differs per driver, how a statement is executed and how a key
set reaches it, stays in that driver's provider
(``sdm_client._rust.providers``); what a plan must be for a SQL provider to
run it, how a key set the core sent is read, and the refusal of a plan
that is not, are here, so no two providers hold their own copy.

It imports nothing of the client, which depends on this package, never
the reverse, and nothing of pandas; it reads a key set with pyarrow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class PlanNotRun(Exception):
    r"""A plan a provider does not run. Its text names no value."""


class Provider(Protocol):
    r"""What the run loop calls for each fetch step: ``step`` the step the
    core asked (its ``plan`` among it), ``keys`` the key sets it sent with
    it, Arrow streams by attachment id. It answers with an Arrow table or
    stream, or raises; a driver's exception is handed to the core as a
    provider error.

    A provider may also say what a driver's failure tells the core beyond
    the attributes every driver sets, with an ``error_fields(error)``
    method answering the contract's provider-error fields (``sqlstate``,
    ``error_class``) or ``None`` for a failure it does not read;
    the run loop asks it where the provider has one.
    """

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any: ...


def statements(step: dict[str, Any], dialect: str) -> list[dict[str, Any]]:
    r"""The statements of ``step``'s plan, a SQL plan of ``dialect``.

    Raises:
        PlanNotRun: The plan is a host provider's, has no statement, or has
            one of another dialect.
    """
    plan = step['plan']
    if plan['plan'] != 'sql':
        raise PlanNotRun("a host plan is not a SQL provider's to run")
    found = plan['statements']
    if not found or any(statement['dialect'] != dialect for statement in found):
        raise PlanNotRun('a plan with no statement or another dialect')
    return found


def key_set(
    param: dict[str, Any], keys: Mapping[int, Any] | None
) -> tuple[int, Any]:
    r"""The key set ``param`` binds, one of ``keys``: its attachment id and
    the Arrow stream handed over under it.

    Raises:
        PlanNotRun: The parameter is not a key set (a scalar, which a
            provider binding only key sets does not take), or names one
            not handed over.
    """
    if param['param'] != 'key_set':
        raise PlanNotRun('a plan with a scalar parameter')
    attachment = param['keys']['attachment']
    if keys is None or attachment not in keys:
        raise PlanNotRun('a plan naming a key set not handed over')
    return attachment, keys[attachment]


def bound_relations(
    statements: list[dict[str, Any]], keys: Mapping[int, Any] | None
) -> dict[str, int]:
    r"""The key set each relation name of ``statements`` binds, by name, for
    a provider that binds each key set as a relation of the name its
    parameter gives (DuckDB registers it, SQLite ingests it): every
    parameter a key set handed over (:func:`key_set`), one name binding one
    key set however often it is named.

    Raises:
        PlanNotRun: A parameter is not a key set handed over, or one name
            binds two key sets.
    """
    relations: dict[str, int] = {}
    for statement in statements:
        for param in statement['params']:
            attachment, _ = key_set(param, keys)
            bound = relations.setdefault(param['relation'], attachment)
            if bound != attachment:
                raise PlanNotRun('a plan binding two key sets to one name')
    return relations


def arrow_table(stream: Any) -> Any:
    r"""A key set the core sent, an object exporting an Arrow stream, read
    whole as a pyarrow table, through the PyCapsule interface pyarrow 14.0.1
    reads (decision E16)."""
    import pyarrow as pa

    # The private importer, because pyarrow 14.0.1, the floor, has no public
    # reader of a stream capsule (RecordBatchReader.from_stream is 15's).
    return pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()


__all__ = [
    'PlanNotRun',
    'Provider',
    'arrow_table',
    'bound_relations',
    'key_set',
    'statements',
]
