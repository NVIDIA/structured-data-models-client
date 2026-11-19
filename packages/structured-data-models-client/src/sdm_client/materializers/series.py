# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Build a request about one series from a query that names it.

A time-series model reads a history rather than rows to score, so gathering
for one is choosing the series and putting its steps in order:

>>> query = PredictiveQuery(
...     'PREDICT readings.value FORECAST 72 TIMEFRAMES '
...     'FOR EACH readings.sensor_id',
...     {'readings': readings},
... )
>>> request = query.materialize(indices=['s-1'], model=KumoForecast())

The entity is what says which series. A column holding one value per row
names the row, and then the whole table is one series. A column that repeats
names the series each row belongs to, and ``indices`` picks one of them.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from predictive_query import ANOMALY, TIMESERIES, PQLParser
from predictive_query.profiles import Mode
from predictive_query.schema.typing import ProblemType
from predictive_query.validator.utils import col_name

from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import StructuredDataError
from sdm_client.request import Request
from sdm_client.schema import (
    Schema,
    SourceTable,
    _as_key_tuple,
    to_schema_definition,
)

if TYPE_CHECKING:
    from predictive_query import Profile, ValidatedPredictiveQuery

    from sdm_client.base import ModelSpec

#: What each profile is asked with, and what the clause is called in a refusal.
_CLAUSES = {
    TIMESERIES.capabilities.model: (
        ProblemType.FORECAST,
        'PREDICT <table>.<column> FORECAST <n> TIMEFRAMES '
        'FOR EACH <table>.<key>',
    ),
    ANOMALY.capabilities.model: (
        ProblemType.ANOMALY,
        'PREDICT <table>.* DETECT ANOMALY FOR EACH <table>.<key>',
    ),
}


def _a_name_not_among(wanted: str, taken: Any) -> str:
    r"""``wanted``, lengthened until no column already has it."""
    names = set(map(str, taken))
    while wanted in names:
        wanted += '_'
    return wanted


def reads_a_series(spec: ModelSpec | None) -> bool:
    r"""Whether ``spec`` is a model that is handed a history, not rows."""
    return spec is not None and spec.profile in (TIMESERIES, ANOMALY)


class SeriesMaterializer:
    r"""Materialize one series from a single in-memory table."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def materialize(
        self,
        *,
        text: str,
        spec: ModelSpec,
        resolve: Callable[[Profile, Any], ValidatedPredictiveQuery],
        indices: Sequence[Any] | None,
        inference: Mapping[str, Any],
    ) -> Request:
        r"""The request that asks ``spec`` about the series ``text`` names.

        Args:
            text: The query as written.
            spec: The time-series model the request is gathered for.
            resolve: Validates the query against a schema definition, raising
                the caller's own refusal when it does not resolve.
            indices: The series to read, when the entity is a series key.
            inference: Options for the model, passed through to the adapter.

        The language holds that an entity is a primary key. A column with one
        value per row is the key of the row, and one that repeats is the key
        of the series, so the query is resolved against the table read at the
        grain the entity names. Where the entity is the row's key, the
        table's own key is a second name for the same row and is left out
        rather than read as a reading.

        The rows are picked and put in time order, so where an anomaly score
        falls in the answer is not where its row was. Each score names its
        row by the table's key. A table with no key, or a composite one that
        an answer cannot carry back, names it by the row's place in the
        caller's table.
        """
        name, table = self._the_one_table(spec)
        key = self._entity_column(text, name)
        definition = to_schema_definition(self.schema)
        whole_table = (
            key is None
            or key == table.primary_key
            or bool(table.data[key].is_unique)
        )
        if key is not None and key != table.primary_key:
            definition.tables[name].pkey = key

        asked = resolve(spec.profile, definition)
        wanted, form = _CLAUSES[spec.profile.capabilities.model]
        if asked.problem_type != wanted:
            raise StructuredDataError(
                f'{text!r} does not ask {spec.model_id} anything: it is asked '
                f'as {form}',
                code='INVALID_REQUEST',
                details={'model': spec.model_id},
            )

        if asked.get_rfm_entity_id_list():
            column = key or '<key>'
            raise StructuredDataError(
                f'{text!r} names its series with FOR, and a series is named '
                f'by indices. Write FOR EACH {name}.{column} and pass '
                f'indices=[<{column}>]',
                code='INVALID_REQUEST',
                details={'model': spec.model_id},
            )

        position = _a_name_not_among('row', table.data.columns)
        frame = table.data.assign(**{position: np.arange(len(table.data))})
        if whole_table:
            if indices is not None:
                raise StructuredDataError(
                    f'{name}.{key} names a row, so the whole of {name!r} is '
                    f'the series and there is no series to choose. Name the '
                    f'column that says which series a row belongs to in FOR '
                    f'EACH to pick one of several',
                    code='INVALID_REQUEST',
                )
        else:
            assert key is not None
            frame = self._the_series(frame, name, key, indices)

        frame = self._in_time_order(frame, name, table.time_column)
        dropped = [
            column
            for column in (table.time_column, None if whole_table else key)
            if column is not None
        ]
        frame = frame.drop(columns=dropped).reset_index(drop=True)
        row_key = next(
            (
                named
                for named in (key if whole_table else None, table.primary_key)
                if isinstance(named, str) and named in frame.columns
            ),
            None,
        )
        displaced = [
            column
            for column in _as_key_tuple(table.primary_key)
            if column != row_key and column in frame.columns
        ]
        frame = frame.drop(columns=displaced)
        if wanted == ProblemType.FORECAST:
            frame = frame.drop(columns=[position])
            target = col_name(asked.target_column)
            request = Request.forecast(
                frame,
                target=target,
                horizon=asked.num_forecasts,
                schema=self._schema_of(name, frame, row_key, target),
                entity_table=name,
                inference=inference,
            )
        else:
            if row_key is not None:
                frame = frame.drop(columns=[position])
            else:
                row_key = position
            request = Request.anomaly(
                frame,
                schema=self._schema_of(name, frame, row_key),
                entity_table=name,
                inference=inference,
            )
        request.query = text
        return request

    def _schema_of(
        self,
        name: str,
        frame: pd.DataFrame,
        row_key: Any,
        target: str | None = None,
    ) -> Schema:
        r"""The schema of the series, meaning what the caller's schema means.

        Inferred afresh from the rows of one series, a column could change
        what it is: a reading that never repeats within the series looks like
        a key. So the key and each column's meaning are carried over from the
        schema the query was written against, and only the target is settled
        here, as the number being forecast. A column empty within the series
        has no type of its own, so it takes the one the table gave it.
        """
        schema = Schema.from_data({name: frame}, time_columns={name: None})
        built, given = schema[name], self.schema[name]
        built.primary_key = (
            row_key
            if isinstance(row_key, str)
            and row_key in frame.columns
            and row_key != target
            else None
        )
        for column in frame.columns:
            if built[column].untyped and column in given.columns:
                built[column].dtype = given[column].dtype
            if column == target:
                built[column].stype = 'numerical'
            elif column != built.primary_key and column in given.columns:
                built[column].stype = given[column].stype
        return schema

    def _the_one_table(self, spec: ModelSpec) -> tuple[str, Any]:
        if len(self.schema) != 1 or self.schema.relationships:
            raise StructuredDataError(
                f'{spec.model_id} reads one series from one table, and this '
                f'schema has {len(self.schema)} tables and '
                f'{len(self.schema.relationships)} relationships. Build the '
                f'schema from the one table that holds the series',
                code='INVALID_REQUEST',
            )
        name = next(iter(self.schema.tables))
        table = self.schema[name]
        if isinstance(table, SourceTable):
            raise StructuredDataError(
                f'{spec.model_id} reads the series from rows in memory, and '
                f'{name!r} is a table in its source. Build the schema with '
                f'Schema.from_data from the rows of the series',
                code='INVALID_REQUEST',
            )
        return name, table

    def _entity_column(self, text: str, name: str) -> str | None:
        r"""The column ``FOR EACH`` names, when it is one of this table's.

        Read from the parse tree alone. A query that does not parse, or names
        another table, is left for resolution to refuse in its own words.
        """
        try:
            parsed = PQLParser(mode=Mode.INFERENCE).to_parsed_predictive_query(
                text
            )
        except Exception:  # noqa: BLE001 - resolution reports it properly.
            return None
        if parsed.entity_table != name:
            return None
        column = col_name(parsed.entity_column)
        return column if column in self.schema[name].columns else None

    def _the_series(
        self,
        frame: pd.DataFrame,
        name: str,
        key: str,
        indices: Sequence[Any] | None,
    ) -> pd.DataFrame:
        r"""The rows of the one series ``indices`` names."""
        present = frame[key].dropna().unique()
        if len(present) == 0:
            raise StructuredDataError(
                f'no row of {name!r} has a {key}, so there is no series to '
                f'read; name a column that says which series a row belongs to',
                code='INVALID_REQUEST',
            )
        if indices is None:
            if len(present) == 1:
                return frame[frame[key] == present[0]]
            raise StructuredDataError(
                f'{name!r} holds {len(present):,} series by {key}, and one '
                f'request reads one. Pass indices=[<{key}>] naming it, for '
                f'example indices=[{present[0]!r}]',
                code='INVALID_REQUEST',
            )
        if isinstance(indices, (str, bytes)) or not isinstance(
            indices, Sequence
        ):
            raise StructuredDataError(
                f'indices must be a sequence naming one {key}, got '
                f'{type(indices).__name__}',
                code='INVALID_REQUEST',
            )
        if len(indices) != 1:
            raise StructuredDataError(
                f'indices names {len(indices)} series, and one request reads '
                f'one. Materialize once for each {key}',
                code='INVALID_REQUEST',
            )
        if not pd.api.types.is_scalar(indices[0]):
            raise StructuredDataError(
                f'indices names a series by one {key}, and its one entry is '
                f'a {type(indices[0]).__name__}',
                code='INVALID_REQUEST',
            )
        rows = frame[frame[key] == indices[0]]
        if rows.empty:
            raise StructuredDataError(
                f'no row of {name!r} has {key} {indices[0]!r}; it holds '
                f'{len(present):,} series, for example {present[0]!r}',
                code='INVALID_REQUEST',
            )
        return rows

    def _in_time_order(
        self, frame: pd.DataFrame, name: str, time_column: str | None
    ) -> pd.DataFrame:
        r"""The steps in the order they happened, one row per step."""
        if time_column is None:
            return frame
        times = frame[time_column]
        missing = int(times.isna().sum())
        if missing:
            raise StructuredDataError(
                f'{missing:,} rows of {name!r} have no {time_column}, so '
                f'where they fall in the series is unknown. Drop them or fill '
                f'the time in',
                code='INVALID_REQUEST',
            )
        times = self._as_times(times, name, time_column)
        repeated = int(times.duplicated().sum())
        if repeated:
            raise StructuredDataError(
                f'{repeated:,} rows of {name!r} repeat a {time_column} '
                f'another row already has, and a series has one row per '
                f'step. Name the column that tells the series apart in FOR '
                f'EACH, or aggregate the rows to one per step',
                code='INVALID_REQUEST',
            )
        if times.is_monotonic_increasing:
            return frame
        warnings.warn(
            f'the rows of {name!r} were not in {time_column} order; the '
            f'series is read sorted by {time_column}',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
        return frame.iloc[np.argsort(times.to_numpy(), kind='stable')]

    def _as_times(
        self, times: pd.Series, name: str, time_column: str
    ) -> pd.Series:
        r"""The time of each row as something that orders the way time does.

        A time written as text orders by its spelling, which puts
        ``'2026-1-10'`` before ``'2026-1-2'``, so text is read as the times it
        spells. Only ISO 8601 is read: ``'1/2/2026'`` is a day in January to
        one writer and a day in February to another, and a guess between
        them would put the rows in a confident wrong order.
        """
        if pd.api.types.is_datetime64_any_dtype(
            times
        ) or pd.api.types.is_numeric_dtype(times):
            return times
        read = pd.to_datetime(
            times, errors='coerce', format='ISO8601', utc=True
        )
        unread = int(read.isna().sum())
        if unread:
            raise StructuredDataError(
                f'{unread:,} rows of {name!r} have a {time_column} that is '
                f'not an ISO 8601 time, for example '
                f'{times[read.isna()].iloc[0]!r}, so the rows cannot be put '
                f'in order. Give the column as times, as numbers, or as text '
                f'like 2026-01-31 or 2026-01-31T09:00:00+00:00',
                code='INVALID_REQUEST',
            )
        return read
