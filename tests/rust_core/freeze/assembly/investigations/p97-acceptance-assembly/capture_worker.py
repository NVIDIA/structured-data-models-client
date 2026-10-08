# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small per-case resource lifecycle atop the reviewed bounded batch supervisor."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from providers import load_provider
from public_capture import materialize
from support import json_bytes, require, unique_json


def worker(
    repo: Path,
    cases: list,
    templates: dict,
    reference_scopes: dict,
    *,
    provider=None,
) -> list:
    records = []
    for case in cases:
        previous = tempfile.tempdir
        previous_env = os.environ.get('TMPDIR')
        with tempfile.TemporaryDirectory(prefix='p97-case-') as temporary:
            try:
                tempfile.tempdir = temporary
                os.environ['TMPDIR'] = temporary
                record = materialize(
                    repo,
                    case['recipe'],
                    templates,
                    reference_scopes,
                    postgres_source_factory=None
                    if provider is None
                    else provider.postgres_source_factory,
                    postgres_connection_factory=None
                    if provider is None
                    else provider.postgres_connection_factory,
                )
                records.append(
                    {
                        'id': case['id'],
                        'recipe': case['recipe'],
                        'capture': record,
                    }
                )
            finally:
                tempfile.tempdir = previous
                if previous_env is None:
                    os.environ.pop('TMPDIR', None)
                else:
                    os.environ['TMPDIR'] = previous_env
        require(
            not Path(temporary).exists()
            and sys.getprofile() is None
            and sys.gettrace() is None,
            'capture.case_cleanup',
        )
    return records


if __name__ == '__main__':
    request = unique_json(sys.stdin.buffer.read())
    provider = load_provider(
        Path(sys.argv[2]) if len(sys.argv) == 4 else None,
        sys.argv[3] if len(sys.argv) == 4 else None,
    )
    sys.stdout.buffer.write(
        json_bytes(
            worker(
                Path(sys.argv[1]),
                request['cases'],
                request['templates'],
                request['reference_scopes'],
                provider=provider,
            )
        )
    )
