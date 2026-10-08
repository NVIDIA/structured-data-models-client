# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed oracle-free binding replay with actual per-case native observation."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import MappingProxyType

from support import INVESTIGATIONS, json_bytes, load, require, sha, unique_json


def replay(request: dict, repo: Path) -> dict:
    probe = load(
        '_p97_oracle_probe',
        INVESTIGATIONS / 'p97-acceptance-staging/oracle_free_probe.py',
    )
    absence = probe.probe()
    native = importlib.import_module('sdm_core._native')
    require(
        sha(Path(native.__file__).read_bytes()) == request['native_sha256'],
        'replay.native_binding',
    )
    contents = {
        name: bytes.fromhex(value) for name, value in request['files'].items()
    }
    before = {name: sha(value) for name, value in contents.items()}
    immutable_contents = MappingProxyType(contents)
    require(
        bool(contents)
        and all(type(value) is bytes for value in contents.values()),
        'replay.snapshot',
    )
    cases = request['cases']
    require(
        type(cases) is list and len(cases) == len(set(cases)) and bool(cases),
        'replay.cases',
    )
    observed, counts, host_events = {}, {}, {}
    product = None
    for identity in cases:
        route = request['routes'][identity]
        if route['kind'] == 'public_host_refusal':
            from expectations import public_expected
            from public_capture import materialize
            from routes import validate_capture_route

            _expected, inputs = public_expected(contents)
            require(identity in inputs, 'replay.host_input_missing')
            record = materialize(repo, inputs[identity]['recipe'])
            validate_capture_route(route, record, record['host_event'])
            observed[identity] = {
                'family': 'public-observation-v1',
                'observation': record['observation'],
            }
            counts[identity] = record['native_calls']
            host_events[identity] = record['host_event']
            continue
        require(route['kind'] == 'native_semantic', 'replay.route_kind')
        if product is None:
            product = importlib.import_module('sdm_core.replay')
            require(
                Path(product.__file__).read_bytes()
                == (
                    repo / 'rust/bindings/python/python/sdm_core/replay.py'
                ).read_bytes(),
                'replay.installed_product_source',
            )
        names = route['native_functions']
        require(
            type(names) is list
            and bool(names)
            and len(names) == len(set(names)),
            'replay.native_functions',
        )
        functions = tuple(getattr(native, name) for name in names)
        require(
            all(inspect.isbuiltin(function) for function in functions),
            'replay.native_exports',
        )
        called = 0

        def profile(_frame, event, argument, functions=functions):
            nonlocal called
            if event == 'c_call' and any(
                argument is function for function in functions
            ):
                called += 1

        require(sys.getprofile() is None, 'replay.profile_active')
        sys.setprofile(profile)
        try:
            result = product.replay_snapshot(
                immutable_contents, case_id=identity
            )
        finally:
            sys.setprofile(None)
        require(
            type(result) is dict and set(result) == {identity},
            'replay.case_result',
        )
        require(called > 0, 'replay.no_native_computation')
        observed[identity] = result[identity]
        counts[identity] = called
    require(probe.probe() == absence, 'replay.oracle_changed')
    require(
        {name: sha(value) for name, value in contents.items()} == before,
        'replay.snapshot_mutated',
    )
    return {
        'observations': observed,
        'native_calls': sum(counts.values()),
        'case_native_calls': counts,
        'host_events': host_events,
        'files_verified': sorted(contents),
        'oracle': absence,
    }


if __name__ == '__main__':
    request = unique_json(sys.stdin.buffer.read())
    sys.stdout.buffer.write(json_bytes(replay(request, Path(sys.argv[1]))))
