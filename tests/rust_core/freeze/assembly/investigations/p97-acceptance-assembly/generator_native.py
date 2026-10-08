# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retained native consistency consumers, with owned temporary Arrow inputs."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

from generator_primitives import module
from support import require

NATIVE_STREAMS = {
    'sampling.debug_jobs',
    'sampling.hashseed_python_jobs',
    'sampling.hashseed_array_jobs',
}


def consumer_kind(name):
    if name in NATIVE_STREAMS:
        return 'native-consistency'
    if name == 'sampling.mt19937':
        return 'original-cpp'
    return 'original-python'


def restore_arrow(value, scratch):
    if isinstance(value, dict):
        if set(value) == {'arrow_ipc_base64'}:
            raw = base64.b64decode(value['arrow_ipc_base64'], validate=True)
            with tempfile.NamedTemporaryFile(
                dir=scratch, suffix='.arrow', delete=False
            ) as output:
                output.write(raw)
                return output.name
        return {
            key: restore_arrow(item, scratch) for key, item in value.items()
        }
    if isinstance(value, list):
        return [restore_arrow(item, scratch) for item in value]
    return value


def consume(root, name, value, observer, variant):
    import sdm_core._native as native

    require(name in NATIVE_STREAMS, 'generator.native_family')
    if name != 'sampling.debug_jobs':
        original = module(root, 'tests/rust_core/sampling/accept_hashseed.py')
        require(
            variant in original.SEEDS
            and os.environ.get('PYTHONHASHSEED') == variant,
            'generator.actual_hashseed',
        )
        consumer = module(root, 'tests/rust_core/sampling/_jobs.py')
        with tempfile.TemporaryDirectory(
            prefix='p97-original-job-'
        ) as temporary:
            jobs = [restore_arrow(value, Path(temporary))]
            result = observer.invoke(consumer.answer_all, jobs)
        require(len(result) == 1, 'generator.native_answer_count')
        return {
            'answers': result,
            'safe_python': [],
            'purpose': 'native-hashseed-consistency',
        }
    require(variant in {'debug', 'release'}, 'generator.native_variant')
    consumer = module(root, 'tests/rust_core/sampling/driver/_debug_run.py')
    original = module(
        root, 'tests/rust_core/sampling/driver/accept_driver_debug.py'
    )
    answers, safe_python = [], []
    for ordinal, job in enumerate(value['calls']):
        answer = observer.invoke(consumer.answer, native, job)
        require(
            not observer.invoke(original._panicked, answer),
            'generator.native_panicked',
        )
        outcome = observer.invoke(original._outcome, answer)
        answers.append(outcome)
        if (
            variant == 'release'
            and 'case' in job
            and not job.get('crashes_python')
        ):
            expected = observer.invoke(original.python_fixture, job['case'])
            require(
                json.dumps(outcome) == json.dumps(expected),
                'generator.debug_safe_python',
            )
            safe_python.append({'call_ordinal': ordinal, 'outcome': expected})
    observer.safe_python = safe_python
    return {
        'answers': answers,
        'purpose': 'native-debug-release-consistency',
    }
