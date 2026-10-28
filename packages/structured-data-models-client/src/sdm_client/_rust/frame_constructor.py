# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual native samples for one core-selected physical column meaning."""

from __future__ import annotations

import json
import warnings
from typing import Any

import pandas as pd
from dateutil.parser import UnknownTimezoneWarning

from sdm_client._rust import codec, errors


def column(
    table: str,
    name: str,
    frame: pd.DataFrame,
    facts: dict[str, Any],
    declared: dict[str, Any] | None,
) -> dict[str, Any]:
    """Complete each native sample before observing the following column.

    The core owns eligibility, short circuiting, separator selection and
    semantic thresholds. This host owns the actual vector operations and
    their native exceptions, warnings, scalar hooks and index reset.
    """
    from sdm_client import _native

    try:
        meaning = _native.FrameNativeConstructor(
            table, name, codec.attachment(facts), codec.attachment(declared)
        )
    except _native.CoreError as error:
        errors.raise_from_core(
            error.control, error.attachments, internal_context=error
        )
    dtype = json.loads(meaning.dtype())
    try:
        if meaning.timestamp_sample():
            sample = frame.reset_index(drop=True)[name].iloc[:100]
            sample = sample.dropna()
            sample = sample[sample != '']
            if len(sample) == 0:
                meaning.timestamp_counts(0, 0)
            else:
                sample = sample.astype(str)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UnknownTimezoneWarning)
                    warnings.filterwarnings(
                        'ignore', message='Could not infer format'
                    )
                    mask = pd.to_datetime(sample, errors='coerce').notna()
                    meaning.timestamp_counts(int(mask.sum()), len(mask))
        if meaning.separated_sample():
            sample = frame.reset_index(drop=True)[name].iloc[:500].dropna()
            sample = sample.astype(str)
            text = '\n'.join(sample)
            separator = meaning.separator([ord(char) for char in text])
            if separator is not None:
                distinct = int(sample.nunique())
                sample = sample.str.split(separator)
                parts = int(sample.astype('object').explode().nunique())
                meaning.separated_counts(distinct, parts)
        if meaning.element_sample():
            sample = frame.reset_index(drop=True)[name].iloc[:500].dropna()
            meaning.element_count(
                int(sample.astype('object').explode().nunique())
            )
        if meaning.distinct_sample():
            sample = frame.reset_index(drop=True)[name].iloc[:1000].dropna()
            if meaning.distinct_count(int(sample.nunique())):
                meaning.population_count(len(sample))
        return json.loads(meaning.meaning())
    except _native.CoreError as error:
        errors.raise_from_core(
            error.control, error.attachments, internal_context=error
        )
    except Exception as error:
        # This fixed native inference wrapper is reached before later
        # table/key metadata, just as the original physical constructor.
        raise RuntimeError(
            f"Could not determine semantic type for column '{name}' "
            f"with data type '{dtype}' in table '{table}'. Please either "
            "change the column's data type or remove the column from this "
            'table.'
        ) from error
