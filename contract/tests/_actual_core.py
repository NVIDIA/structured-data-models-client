# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The existing checked reader supplies consumer roles, never fake starts."""

import functools
import importlib.util
from pathlib import Path
from typing import Any


@functools.cache
def _reader(name='prediction_observation'):
    assert name in ('prediction_observation', 'fetch_bindings')
    path = Path(__file__).parents[1] / 'runners' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(
        'contract_actual_core_reader'
        if name == 'prediction_observation'
        else 'contract_actual_core_bindings',
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_view(observation: dict[str, Any]) -> dict[str, Any]:
    return _reader().view(
        observation['core'], observation['identity']['operations']
    )


def checked_operations(observation: dict[str, Any]) -> list[str]:
    return checked_view(observation)['operations']


def checked_fetch_bindings(
    observation: dict[str, Any], fetched: list[Any]
) -> None:
    """Check actual association and complete facts before historical projection."""
    try:
        logical = checked_view(observation)
        assert _reader()._same_jsonl_snapshot(logical['fetches'], fetched), (
            'fetches differ from the complete actual observation'
        )
        raw = observation['core']['fetches']
        operations = observation['identity']['operations']
        for fetch in raw:
            if 'bindings' in fetch:
                facts = fetch['bindings']
                assert type(facts) is dict, (
                    'actual binding facts are not an object'
                )
                run = facts.get('run')
                assert type(run) is int and 0 <= run < len(operations), (
                    'actual binding Run is outside its observation'
                )
        # Raw facts are checked too: a malformed null binding must not disappear
        # merely because the older logical projection has no binding object.
        reader = _reader('fetch_bindings')
        assert reader.agree(raw, raw), (
            'actual raw fetch bindings are inconsistent'
        )
        assert reader.agree(fetched, fetched), (
            'actual logical fetch bindings are inconsistent'
        )
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise AssertionError('actual fetch binding proof is invalid') from error
