# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A core route over frames runs where the warehouse package
(``sdm_connectors``, an optional dependency of the client) is not installed
(T3.4 fix 3, round 3's Codex P1-1): the run loop imports nothing of it."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_an_in_memory_core_query_runs_without_the_connectors_package() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import os
        import sys

        class Absent(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] == 'sdm_connectors':
                    raise ModuleNotFoundError(
                        "No module named 'sdm_connectors'", name=fullname
                    )

        for name in [m for m in sys.modules if m.split('.')[0] == 'sdm_connectors']:
            del sys.modules[name]
        sys.meta_path.insert(0, Absent())

        import pandas as pd
        from sdm_client import KumoTabular, PredictiveQuery, Schema, _routing

        os.environ[_routing.ENVIRONMENT] = ','.join(_routing.S0)
        schema = Schema.from_data(
            {'users': pd.DataFrame({'id': [1, 2, 3, 4], 'age': [10.5, 20.5, 30.5, 40.5]})}
        )
        request = PredictiveQuery(
            'PREDICT users.age FOR EACH users.id', schema
        ).materialize(model=KumoTabular(), indices=[1])
        assert 'sdm_connectors' not in sys.modules
        print(type(request).__name__)
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'Request'
