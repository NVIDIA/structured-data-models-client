# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run selected public journeys with legacy computation tripwires and call counts.

Usage: installed-python phase4_route_audit.py --out evidence.json test.py ...
Only function identities/counts are recorded, never arguments or user data.
Legacy PQL validation/schema editing is observed, not hidden by the tripwires.
"""

import argparse
import collections
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class Audit:
    def __init__(self):
        self.cases = []

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        # Run default routes even when a reused test fixture selected core routes.
        os.environ.pop('SDM_CORE_ROUTES', None)
        counts = collections.Counter()
        traps = []

        def trace(frame, event, arg):
            if event == 'call':
                path = frame.f_code.co_filename
                for root in (
                    'sdm_client/',
                    'predictive_query/',
                    'sdm_relational_engine/',
                ):
                    if root in path:
                        counts[
                            (
                                root + path.split(root, 1)[1],
                                frame.f_code.co_name,
                            )
                        ] += 1
                        break

        def forbidden(label):
            def fail(*args, **kwargs):
                traps.append(label)
                raise AssertionError(f'Legacy computation invoked: {label}')

            return fail

        targets = {
            'sdm_client.materializers.relational.RelationalMaterializer': (
                '_gather',
                '_get_model',
                '_build_graph',
            ),
            'sdm_client.materializers.tabular.TabularMaterializer': (
                'indices',
                'table',
            ),
            'sdm_client.adapters.relational.RelationalAdapter': (
                'prediction_plan',
                'session_plan',
                'session_prediction_plan',
                'normalize',
            ),
            'sdm_client.adapters.tabular.TabularAdapter': (
                'prediction_plan',
                'session_plan',
                'session_prediction_plan',
                'normalize',
            ),
        }
        previous = sys.getprofile()
        with contextlib.ExitStack() as stack:
            for name, methods in targets.items():
                module, cls = name.rsplit('.', 1)
                owner = getattr(importlib.import_module(module), cls)
                for method in methods:
                    stack.enter_context(
                        patch.object(
                            owner, method, forbidden(name + '.' + method)
                        )
                    )
            client = importlib.import_module('sdm_client.client')
            for name in ('_held_back', '_ranked', '_by_step'):
                stack.enter_context(
                    patch.object(client, name, forbidden('client.' + name))
                )
            sys.setprofile(trace)
            try:
                yield
            finally:
                sys.setprofile(previous)
                self.cases.append(
                    {
                        'case': item.nodeid,
                        'legacy_traps': traps,
                        'route_override_after_call': os.environ.get(
                            'SDM_CORE_ROUTES'
                        ),
                        'calls': [
                            {'file': p, 'function': f, 'count': n}
                            for (p, f), n in sorted(counts.items())
                        ],
                    }
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args, tests = parser.parse_known_args()
    import sdm_client
    from sdm_client import _native

    plugin = Audit()
    status = pytest.main(tests, plugins=[plugin])
    args.out.write_text(
        json.dumps(
            {
                'client': sdm_client.__file__,
                'native': _native.__file__,
                'contract': _native.CONTRACT_VERSION,
                'pytest_exit': status,
                'cases': plugin.cases,
            },
            indent=2,
        )
        + '\n'
    )
    return status


if __name__ == '__main__':
    raise SystemExit(main())
