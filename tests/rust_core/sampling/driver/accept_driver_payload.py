# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P75.payload: today's Python encoder over the core's sampling outputs
gives every transcript's payloads.

Every transcript replays through the public API with each call of
``sample_target`` and ``sample_subgraph`` answered by the core: the
interceptor starts the core's run from what ``describe`` gives, answers each
of its queries from the statements today's method ran (which the replay
holds to the transcript), and hands the caller ``to_python`` of the run's
``done`` value, with the run's warnings, instead of today's result. The
replay must then reproduce the transcript: every step, the payloads parsed
equal, the decoded answer and the warnings.
"""

from __future__ import annotations

import json

from _driver import (
    METHODS,
    Interceptor,
    capture_module,
    native,
    replaying,
    transcripts,
)


def test_payloads_from_the_core_outputs() -> None:
    core = native(*METHODS, 'Outcome')
    capture = capture_module()
    interceptor = Interceptor(capture.describe, core, capture.to_python)
    problems = []
    swapped = scenarios = 0
    with replaying(interceptor) as replay:
        for path in transcripts():
            transcript = json.loads(path.read_text('utf-8'))
            name = transcript['name']
            reason, calls = replay(path)
            if reason is not None:
                problems.append(f'{name}: {reason}')
                continue
            if transcript['backend'] != 'tabular' and not calls:
                problems.append(f'{name}: no call went to the core')
                continue
            swapped += len(calls)
            scenarios += 1
    assert not problems, (
        f"{len(problems)} scenarios do not reproduce over the core's "
        f'sampling outputs:\n' + '\n'.join(problems[:25])
    )
    assert scenarios >= 71 and swapped >= 230, (
        f'only {scenarios} scenarios and {swapped} calls'
    )
