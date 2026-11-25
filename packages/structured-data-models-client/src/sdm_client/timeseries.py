# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare array histories for the ordinary relational forecasting workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow as pa

from sdm_client.errors import StructuredDataError
from sdm_client.query import PredictiveQuery
from sdm_client.schema import Relationship, Schema


@dataclass(frozen=True)
class ForecastInput:
    """Prepared independent schema, query, and optional UTC forecast cutoff."""

    schema: Schema
    query: PredictiveQuery
    anchor_time: pd.Timestamp | None


def _bad(reason: str) -> StructuredDataError:
    return StructuredDataError(reason, code='INVALID_REQUEST')


def build_timeseries(
    frame: pd.DataFrame | pa.Table,
    timeseries_col: str,
    *,
    timestamps_col: str | None = None,
    time_delta: pd.Timedelta | None = None,
    anchor_time: pd.Timestamp | None = None,
    entity_col: str | None = None,
    num_timeframes: int = 1,
) -> ForecastInput:
    """Expand numeric histories into entity/target tables and a MAX forecast query.

    Naive times mean UTC, never the host timezone. Explicit timestamps retain
    their instants; synthetic history ends one interval before anchor_time.
    Observations become Float64 model values; large integers/decimals can round.
    Entity keys retain their types. Input buffers are not modified.
    """
    from sdm_client._rust import codec, results
    from sdm_client._rust.run import run

    if not isinstance(timeseries_col, str) or not timeseries_col:
        raise _bad('timeseries_col must be a nonempty string')
    if any(
        v is not None and (not isinstance(v, str) or not v)
        for v in (timestamps_col, entity_col)
    ):
        raise _bad('column names must be nonempty strings')
    if type(num_timeframes) is not int or not 0 < num_timeframes <= 2**32 - 1:
        raise _bad('num_timeframes must be a positive integer')
    try:
        step = None if time_delta is None else pd.Timedelta(time_delta).value
        anchor = None if anchor_time is None else pd.Timestamp(anchor_time)
        if anchor is not None:
            if pd.isna(anchor):
                raise _bad('anchor_time cannot be missing')
            anchor = (
                anchor.tz_localize('UTC')
                if anchor.tzinfo is None
                else anchor.tz_convert('UTC')
            )
        anchor_ns = None if anchor is None else anchor.value
        if isinstance(frame, pd.DataFrame):
            if not frame.columns.is_unique or any(
                not isinstance(n, str) or not n for n in frame.columns
            ):
                raise _bad('input columns must be unique nonempty strings')
            arrays = []
            for name in frame.columns:
                values: Any = frame[name]
                if name == timestamps_col:
                    converted: list[list[Any] | None] = []
                    aware = set()
                    for sequence in values:
                        if sequence is None:
                            converted.append(None)
                            continue
                        row = []
                        for value in sequence:
                            stamp = pd.Timestamp(value)
                            if not pd.isna(stamp):
                                aware.add(stamp.tzinfo is not None)
                            row.append(stamp)
                        converted.append(row)
                    if len(aware) > 1:
                        raise _bad(
                            'timestamp histories cannot mix naive and timezone-aware values'
                        )
                    values = [
                        None
                        if row is None
                        else [
                            None
                            if pd.isna(v)
                            else (
                                v.tz_localize('UTC')
                                if v.tzinfo is None
                                else v.tz_convert('UTC')
                            )
                            for v in row
                        ]
                        for row in converted
                    ]
                    arrays.append(
                        pa.array(
                            values, type=pa.list_(pa.timestamp('ns', tz='UTC'))
                        )
                    )
                else:
                    arrays.append(pa.array(values))
            table = pa.Table.from_arrays(arrays, names=list(frame.columns))
        elif isinstance(frame, pa.Table):
            table = frame
        else:
            raise _bad('frame must be a pandas DataFrame or Arrow Table')
    except (TypeError, ValueError, OverflowError, pa.ArrowException) as error:
        raise _bad(f'invalid time-series input: {error}') from error
    finished = run(
        {'op': 'prepare_time_series', 'input': 1, 'table': 2},
        {
            1: codec.attachment(
                dict(
                    history=timeseries_col,
                    timestamps=timestamps_col,
                    entity_key=entity_col,
                    step_ns=step,
                    anchor_ns=anchor_ns,
                    forecasts=num_timeframes,
                )
            ),
            2: table,
        },
    )
    answer = results.expect(finished.result, results.DirectValue)
    metadata = json.loads(finished.take(answer.value))
    frames = {}
    for name in ('entity', 'target'):
        stream = finished.take(metadata[name])
        with pa.RecordBatchReader._import_from_c_capsule(
            stream.__arrow_c_stream__()
        ) as reader:
            frames[name] = reader.read_all().to_pandas()
    key = metadata['key']
    schema = Schema.from_data(
        frames,
        primary_keys={'entity': key},
        time_columns={'entity': None, 'target': 'timestamp'},
        relationships=[],
    )
    schema['target'].primary_key = None
    schema['target']['value'].stype = 'numerical'
    schema.relationships = [Relationship('target', key, 'entity', key)]
    minutes = metadata['step_ns'] // 60_000_000_000
    amount, unit = (
        (minutes // 1440, 'days')
        if minutes % 1440 == 0
        else (
            (minutes // 60, 'hours')
            if minutes % 60 == 0
            else (minutes, 'minutes')
        )
    )
    quoted = '`' + key.replace('`', '``') + '`'
    text = f'PREDICT MAX(target.value, 0, {amount}, {unit}) FORECAST {num_timeframes} TIMEFRAMES FOR EACH entity.{quoted}'
    return ForecastInput(schema, PredictiveQuery(text, schema), anchor)
