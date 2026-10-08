# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse reviewed acceptance primitives without changing their source."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[1]
INVESTIGATIONS = HERE.parent
sys.path.insert(0, str(INVESTIGATIONS / 'p97-acceptance-staging'))
import accept_contract as contract  # noqa: E402

Rejected = contract.Rejected
require = contract.require


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'assembly.import')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def same_json(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def unique_json(raw: bytes | str):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'assembly.duplicate_json_key')
            result[key] = value
        return result

    def constant(_value):
        raise Rejected('assembly.nonfinite_json')

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def checked_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(
        isinstance(relative, str)
        and bool(relative)
        and not path.is_absolute()
        and '..' not in path.parts
        and '\\' not in relative,
        'assembly.relative_path',
    )
    target = root / path
    require(
        target.resolve().is_relative_to(root.resolve())
        and all(
            not parent.is_symlink()
            for parent in (target, *target.parents)
            if parent.is_relative_to(root)
        ),
        'assembly.path_escape',
    )
    return target
