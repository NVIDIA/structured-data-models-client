# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entity-anchor validation in the maintained prediction case format."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
CASES = ROOT / 'contract/cases'


def _module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python')


def test_case_format_admits_only_the_entity_anchor_literal() -> None:
    validate = run_python.validator('case')
    case = next(
        case
        for case, _ in run_python.load_cases([CASES])
        if 'anchor_time' in case['input'].get('materialize', {})
    )
    assert run_python._invalid(validate, case) is None
    for key in ('anchor_time', 'context_anchor_time'):
        named = copy.deepcopy(case)
        named['input']['materialize'][key] = 'entity'
        assert run_python._invalid(validate, named) is None, key
        named['input']['materialize'][key] = 'entities'
        assert run_python._invalid(validate, named) is not None, key
