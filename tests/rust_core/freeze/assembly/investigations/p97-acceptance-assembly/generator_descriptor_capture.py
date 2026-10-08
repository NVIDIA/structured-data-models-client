# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure existing host facts from authenticated shipping helper code."""

from __future__ import annotations

import datetime
import inspect
import sys
import time
from pathlib import Path


def collect_descriptor(root, name):
    from generator_oracles import source_codes
    from sdm_client import _impl
    from sdm_core import _native, describe
    from support import require

    require(_impl.OLD is True, 'descriptor.original_selector')
    require(
        name
        in {'labels.question_batches', 'schema.date_columns', 'rfm.sequences'},
        'descriptor.family',
    )
    require(sys.getprofile() is None, 'descriptor.profile_busy')
    source = root / 'rust/bindings/python/python/sdm_core/describe.py'
    functions = [describe.timestamp_series_units]
    if name == 'rfm.sequences':
        functions.append(describe.numpy_text_prefix)
    raw = source.read_bytes()
    for function in functions:
        require(
            Path(inspect.getsourcefile(function)).read_bytes() == raw
            and function.__code__
            in source_codes(raw, function.__code__.co_filename),
            'descriptor.source_code',
        )
    native_functions = tuple(
        function
        for function in vars(_native).values()
        if inspect.isbuiltin(function)
    )
    calls = {function.__code__: 0 for function in functions}
    native_calls = 0

    def observe(frame, event, function):
        nonlocal native_calls
        if event == 'call' and frame.f_code in calls:
            calls[frame.f_code] += 1
        elif event == 'c_call' and any(
            function is candidate for candidate in native_functions
        ):
            native_calls += 1

    sys.setprofile(observe)
    try:
        now = time.localtime()
        measured = {
            'year': now.tm_year,
            'today': [now.tm_year, now.tm_mon, now.tm_mday],
            'pads_year': datetime.datetime(1, 1, 1).strftime('%Y') == '0001',
            'tznames': list(time.tzname),
            'timestamp_series_units': describe.timestamp_series_units()
            or ['s', 'ms', 'us', 'ns'],
        }
        if name == 'rfm.sequences':
            measured['numpy_text_prefix'] = describe.numpy_text_prefix()
    finally:
        sys.setprofile(None)
    require(
        all(count == 1 for count in calls.values()) and native_calls == 0,
        'descriptor.measured_calls',
    )
    require(
        source.read_bytes() == raw
        and all(
            Path(inspect.getsourcefile(function)).read_bytes() == raw
            for function in functions
        ),
        'descriptor.source_changed',
    )
    return {'host': measured, 'primitives': []}
