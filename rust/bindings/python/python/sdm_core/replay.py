# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable snapshot transport to the shared Rust replay dispatcher."""

import json
from functools import lru_cache

from . import _native


@lru_cache(maxsize=1)
def _prepare(files):
    return _native.prepare_snapshot(
        json.dumps({name: value.hex() for name, value in files})
    )


def replay_snapshot(contents, *, case_id):
    files = tuple(contents.items())
    if any(
        type(name) is not str or type(value) is not bytes
        for name, value in files
    ):
        raise TypeError('snapshot files must map text paths to immutable bytes')
    snapshot = _prepare(tuple(sorted(files)))
    return json.loads(_native.replay_snapshot(snapshot, case_id))
