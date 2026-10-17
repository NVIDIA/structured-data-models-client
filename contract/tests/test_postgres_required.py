# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The Postgres service a Postgres job runs against is there (H26; T2.1a).

A job that runs the contract on Postgres sets ``SDM_REQUIRE_POSTGRES`` and
names its server in ``SDM_POSTGRES_TEST_DSN`` (the pinned service of the
Postgres ``contract-python`` jobs and ``rust-test``). There, a missing DSN,
a missing driver or an unreachable server fails this test by name: it never
stands down, so a Postgres job can't pass without its database. Where
``SDM_REQUIRE_POSTGRES`` is not set, nothing is required of the
environment, and the test has nothing to hold.

``sdm-testkit``'s ``tests/postgres_required.rs`` holds the same for the Rust
side; ``tests/test_t21a_postgres_required.py`` runs this file against an
unreachable server and holds it to failing.
"""

from __future__ import annotations

import os

#: The major version of the pinned image (``POSTGRES_IMAGE`` in
#: ``.github/workflows/ci.yml``, postgres:17).
MAJOR = 17


def test_the_postgres_service_is_reachable_where_required() -> None:
    if not os.environ.get('SDM_REQUIRE_POSTGRES'):
        return
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    assert dsn, 'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    # Not installed is a failure here, never a reason to stand down.
    import psycopg

    try:
        connection = psycopg.connect(dsn, connect_timeout=10)
    except psycopg.OperationalError as error:
        raise AssertionError(
            'SDM_REQUIRE_POSTGRES is set and the Postgres service at '
            f'SDM_POSTGRES_TEST_DSN is unreachable: {type(error).__name__}'
        ) from None
    with connection:
        assert connection.execute('SELECT 1').fetchone() == (1,)
        assert connection.info.server_version // 10000 == MAJOR
