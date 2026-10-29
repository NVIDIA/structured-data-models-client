# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual raw source assembly before helper validation or native sanitation.

Arrow ports retain their fetched tables through cursor closure.
Scalar ports construct object frames inside each cursor scope.
No source policy, seed arithmetic, business-column conversion or Arrow
boxing of driver scalars takes place here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import pandas as pd
import pyarrow as pa

from sdm_client._rust.codec import CodecError
from sdm_client._rust.providers.postgres import ScalarRows
from sdm_client._rust.run import FetchRefused


class SourceRowsProvider(Protocol):
    def source_rows(
        self, step: dict[str, Any], keys: Mapping[int, object]
    ) -> AbstractContextManager[object]: ...


@dataclass(frozen=True)
class SourceRowsBatch:
    """Complete raw data with helpers retained and actual part row counts."""

    rows: pa.Table | pd.DataFrame
    row_counts: tuple[int, ...]

    @property
    def storage(self) -> Literal['typed_columns', 'cell_rows']:
        return (
            'typed_columns' if isinstance(self.rows, pa.Table) else 'cell_rows'
        )


@dataclass(frozen=True)
class SourceReadFailure:
    """An actual failed I/O scope, without reading its diagnostic fields."""

    step: dict[str, Any]
    error: Exception


def _column_name(column: Any) -> str:
    # The old scalar reader checks and then reads the actual description
    # name, falling back to its DB-API positional description.
    name = column.name if hasattr(column, 'name') else column[0]
    return {
        '__kumo_batch__': '__KUMO_BATCH__',
        '__kumo_row__': '__KUMO_ROW__',
        '__kumo_matches__': '__KUMO_MATCHES__',
    }.get(name, name)


def read_source_rows(
    provider: SourceRowsProvider,
    steps: Sequence[dict[str, Any]],
    keys: Mapping[int, Mapping[int, object]],
) -> SourceRowsBatch | SourceReadFailure | FetchRefused:
    """Read one already validated complete issued partition, in order.

    Body failures are native only when the same exception leaves the scope.
    A cursor entry/read/exit failure keeps its actual selected I/O owner.
    Process-control exceptions propagate directly through both boundaries.
    """
    if not steps:
        raise CodecError('a native source read needs its issued partition')
    # Missing scoped support is a boundary error, never an ordinary-provider
    # fallback that would prematurely box the scalar rows into Arrow.
    read = provider.source_rows
    tables: list[pa.Table] = []
    frames: list[pd.DataFrame] = []
    row_counts: list[int] = []
    for step in steps:
        bound = keys[step['id']]
        native_error: Exception | None = None
        try:
            with read(step, bound) as source:
                try:
                    if isinstance(source, ScalarRows):
                        part: pa.Table | pd.DataFrame = pd.DataFrame(
                            source.rows,
                            columns=[
                                _column_name(column)
                                for column in source.description
                            ],
                            dtype=object,
                        )
                    elif isinstance(source, pa.Table):
                        part = source
                    else:
                        raise TypeError(
                            'a scoped source returned no raw row representation'
                        )
                except Exception as error:
                    native_error = error
                    raise
        except Exception as error:
            if error is native_error:
                return FetchRefused(error)
            return SourceReadFailure(step, error)
        if native_error is not None:
            # A scoped port suppressing a body exception still supplied no
            # completed native data. Never publish a partially built part.
            return FetchRefused(native_error)
        try:
            if isinstance(part, pa.Table):
                tables.append(part)
                row_counts.append(part.num_rows)
            else:
                frames.append(part)
                row_counts.append(len(part))
        except Exception as error:
            return FetchRefused(error)
    try:
        if tables and frames:
            raise CodecError(
                'a source partition changed its raw row representation'
            )
        if tables:
            rows: pa.Table | pd.DataFrame = (
                tables[0] if len(tables) == 1 else pa.concat_tables(tables)
            )
        elif (
            len(frames) == 1
            and steps[0]['request']['request'] == 'sample_entities'
        ):
            rows = frames[0]
        else:
            # Keyed scalar reads concatenate even one frame, at their actual
            # reached post-I/O position, before any helper is removed.
            rows = pd.concat(frames, ignore_index=True)
    except Exception as error:
        return FetchRefused(error)
    return SourceRowsBatch(rows, tuple(row_counts))
