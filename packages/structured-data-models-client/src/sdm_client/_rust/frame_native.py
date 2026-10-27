# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local native column/error owners for fixed frame domain values.

This module is not connected to dispatch. Core-selected source bindings name
native dtype owners retained at fresh conversion; caller Schema metadata is
never reread here. Graph policy, sampling, normalization and error policy
remain in Rust. No exception is displayed while capturing native facts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Any

import pandas as pd

from sdm_client._rust.codec import relation_name
from sdm_client.errors import StructuredDataError


@dataclass(frozen=True)
class NativeClass:
    """Raw native ancestry metadata, without importing an engine runtime.

    Qualified names describe classes; they are not proof of canonical class
    identity and must not authorize arbitrary constructors or coercion.
    """

    module: str
    name: str


@dataclass(frozen=True)
class NativeFailureFacts:
    ancestry: tuple[str, ...]
    classes: tuple[NativeClass, ...]
    deepest_module: str | None


@dataclass(frozen=True, repr=False, eq=False)
class NativeFrameFailure:
    """The exact local exception, independent of an encoded failure value.

    Retaining it does not call str/repr, read args, choose an SDK error code,
    inspect message phrases or append sanitization advice. The eventual core
    guard must select any native text observation at its actual boundary.
    """

    error: Exception
    facts: NativeFailureFacts
    traceback: TracebackType | None
    cause: BaseException | None
    context: BaseException | None
    suppress_context: bool

    @classmethod
    def capture(cls, error: Exception) -> NativeFrameFailure:
        traceback = error.__traceback__
        deepest = traceback
        while deepest is not None and deepest.tb_next is not None:
            deepest = deepest.tb_next
        ancestry = tuple(
            name
            for kind, name in (
                (SyntaxError, 'syntax_error'),
                (ValueError, 'value_error'),
                (TypeError, 'type_error'),
                (MemoryError, 'memory_error'),
                (RecursionError, 'recursion_error'),
                (KeyError, 'key_error'),
                (StructuredDataError, 'structured_data_error'),
            )
            if isinstance(error, kind)
        )
        classes = tuple(
            NativeClass(
                type.__getattribute__(kind, '__module__'),
                type.__getattribute__(kind, '__qualname__'),
            )
            for kind in type.__getattribute__(type(error), '__mro__')
        )
        return cls(
            error,
            NativeFailureFacts(
                ancestry,
                classes,
                None
                if deepest is None
                else deepest.tb_frame.f_globals.get('__name__'),
            ),
            traceback,
            error.__cause__,
            error.__context__,
            error.__suppress_context__,
        )

    def read_message(self) -> str:
        """One reached native observation, deliberately neither cached nor caught."""
        return str(self.error)

    def read_type_name(self) -> str:
        """Observe the native class name only when selected for presentation."""
        return type(self.error).__name__


@dataclass(frozen=True, repr=False, eq=False)
class NativeColumn:
    """An actual successful whole-column native result, not an Arrow remake."""

    values: pd.Series


class NativeFrameColumns:
    """Native dtype lifetime companion to one materializer's core graph owner.

    This owns no caller Schema, frame copy, mutable neighbor stream or task
    continuation. The exact frozen core graph class remains binding-owned.
    Construct this companion from the fresh converter's ordered native types
    and retain it with that graph even if a later warning/conversion fails.
    """

    def __init__(self, columns: Mapping[str, Mapping[str, Any]]) -> None:
        self._columns = MappingProxyType(
            {
                source: MappingProxyType(dict(types))
                for source, types in columns.items()
            }
        )

    def dtype(self, binding: Mapping[str, Any]) -> Any:
        """Read exactly the source Relation/column resolved by the core graph."""
        source = relation_name(binding['source'])
        return self._columns[source][binding['column']]

    def prediction(
        self, binding: Mapping[str, Any], indices: list[Any]
    ) -> NativeColumn | NativeFrameFailure:
        """Execute the reached whole constructor with the retained dtype owner.

        Failed construction does not start diagnostics or display anything.
        Those positions require their core-selected constructor outcome and
        are distinct from complete-task and per-batch reference conversion.
        """
        dtype = self.dtype(binding)
        try:
            return NativeColumn(pd.Series(indices, dtype=dtype))
        except Exception as error:
            return NativeFrameFailure.capture(error)


def convert_text(selected: pd.Series) -> NativeColumn | NativeFrameFailure:
    """Convert one actual selected native column before Rust normalization.

    Selection and original dtype reconstruction belong to the fixed sampled
    context boundary. There is no row loop, cap, lower, split, casing profile,
    fabricated result after failure or scan of unselected source values.
    """
    try:
        return NativeColumn(selected.fillna('').astype(str))
    except Exception as error:
        return NativeFrameFailure.capture(error)
