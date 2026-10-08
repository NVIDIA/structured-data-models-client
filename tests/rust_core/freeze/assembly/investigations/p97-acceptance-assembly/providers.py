# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit runtime-only PostgreSQL capabilities, outside recorded recipes."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from support import Rejected, load, require, sha


def load_provider(path: Path | None, expected_sha256: str | None):
    if path is None:
        require(expected_sha256 is None, 'provider.path_required')
        return None
    require(expected_sha256 is not None, 'provider.source_hash_required')
    try:
        require(sha(path.read_bytes()) == expected_sha256, 'provider.source')
        provider = load('_p97_explicit_provider', path)
        require(sha(path.read_bytes()) == expected_sha256, 'provider.changed')
        require(
            callable(provider.postgres_source_factory)
            and callable(provider.postgres_connection_factory),
            'provider.callables',
        )
    except Rejected:
        raise
    except Exception:
        raise Rejected('provider.unavailable') from None
    return provider


@contextlib.contextmanager
def reference_connection(runner, factory):
    require(callable(factory), 'reference.postgres_provider_required')
    original = runner._postgres
    failures = []

    @contextlib.contextmanager
    def connection(frames, spec):
        try:
            manager = factory(frames, spec)
            resource = manager.__enter__()
        except Exception:
            failures.append('reference.postgres_setup_failed')
            raise Rejected(failures[-1]) from None
        try:
            yield resource
        finally:
            exception = sys.exc_info()
            try:
                manager.__exit__(*exception)
            except Exception:
                failures.append('reference.postgres_cleanup_failed')
                raise Rejected(failures[-1]) from None

    runner._postgres = connection
    try:
        yield
    finally:
        runner._postgres = original
        if failures:
            raise Rejected(failures[-1]) from None
