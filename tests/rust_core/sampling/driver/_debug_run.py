# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Answer the task 6b jobs with whichever ``sdm_core`` this process imports,
for ``accept_driver_debug.py``.

    python _debug_run.py JOBS OUT

Each job is ``{"call": NAME, "args": [...]}``: ``sampling_op`` with an
operation and its input as JSON text, ``date_offset_apply`` with its three
arguments, or ``date_offset_to_seconds`` with its two. Each answer is
``{"text": T}`` (a ``str`` result, as it came), ``{"value": V}`` (any other
result, as JSON), or ``{"raised": {"code", "message"}}``. The first line of
OUT names the module file answered from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def answer(module: Any, job: dict[str, Any]) -> dict[str, Any]:
    try:
        got = getattr(module, job['call'])(*job['args'])
    except BaseException as error:  # noqa: BLE001 - a panic is an answer too
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return {'raised': {'code': type(error).__name__, 'message': str(error)}}
    if isinstance(got, str):
        return {'text': got}
    return {'value': json.loads(json.dumps(got))}


def main() -> None:
    import sdm_core._native as module

    jobs = json.loads(Path(sys.argv[1]).read_text('utf-8'))
    out = [answer(module, job) for job in jobs]
    Path(sys.argv[2]).write_text(
        json.dumps({'module': module.__file__, 'answers': out}), 'utf-8'
    )


if __name__ == '__main__':
    main()
