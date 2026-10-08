# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite C29 proof of execution of the original public method's old branch."""

from __future__ import annotations

import ast
import contextlib
import sys
import types
from pathlib import Path

from support import contract, require, same_json, sha

SOURCE = 'packages/structured-data-models-client/src/sdm_client/result.py'
QUALNAME = 'PredictionResult.to_pandas'


def binding(root: Path) -> tuple[dict, types.CodeType]:
    path = (root / SOURCE).resolve()
    raw = path.read_bytes()
    tree = ast.parse(raw, filename=str(path))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'PredictionResult'
    ]
    require(len(classes) == 1, 'inline_old.class')
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == 'to_pandas'
    ]
    require(len(methods) == 1, 'inline_old.method')
    method = methods[0]
    guard, branch = method.body[1:3]
    require(
        isinstance(method.body[0], ast.Expr)
        and isinstance(guard, ast.If)
        and ast.unparse(guard.test) == 'not _impl.OLD'
        and not guard.orelse
        and isinstance(guard.body[-1], ast.Return)
        and ast.unparse(guard.body[-1].value) == '_request_core.to_pandas(self)'
        and isinstance(branch, ast.AnnAssign)
        and ast.unparse(branch.target) == 'rows'
        and isinstance(branch.value, ast.List)
        and not branch.value.elts,
        'inline_old.branch_shape',
    )
    compiled = compile(raw, str(path), 'exec', dont_inherit=True)
    class_codes = [
        value
        for value in compiled.co_consts
        if isinstance(value, types.CodeType)
        and value.co_name == 'PredictionResult'
    ]
    require(len(class_codes) == 1, 'inline_old.class_code')
    codes = [
        value
        for value in class_codes[0].co_consts
        if isinstance(value, types.CodeType) and value.co_name == 'to_pandas'
    ]
    require(len(codes) == 1, 'inline_old.method_code')
    return {
        'path': SOURCE,
        'source_sha256': sha(raw),
        'function': QUALNAME,
        'line': branch.lineno,
    }, codes[0]


@contextlib.contextmanager
def observe(root: Path, function, input_sha256: str):
    descriptor, original = binding(root)
    code = function.__code__
    require(
        code == original
        and code.co_filename == original.co_filename
        and code.co_firstlineno == original.co_firstlineno,
        'inline_old.code_identity',
    )
    require(sys.gettrace() is None, 'inline_old.existing_trace')
    events = []

    def trace(frame, event, _argument):
        if frame.f_code is not code:
            return None
        if event == 'line' and frame.f_lineno == descriptor['line']:
            require(
                getattr(frame.f_globals.get('_impl'), 'OLD', None) is True,
                'inline_old.selector',
            )
            events.append({**descriptor, 'input_sha256': input_sha256})
        return trace

    sys.settrace(trace)
    try:
        yield events
    finally:
        sys.settrace(None)
        require(binding(root)[0] == descriptor, 'inline_old.source_changed')


def validate(root: Path, envelope: dict, captured: dict):
    events = captured['legacy_branch_events']
    require(type(events) is list, 'inline_old.events_type')
    if not events:
        return
    from recipes import open_recipe

    recipe = open_recipe(envelope)
    descriptor, _code = binding(root)
    require(
        recipe['operation'] == 'result.to_pandas'
        and recipe['entrypoint'] == 'PredictionResult.to_pandas'
        and captured['selector'] == 'python'
        and captured['native_calls'] == 0
        and captured['public_calls'] > 0
        and len(events) == captured['public_calls'] == 1
        and same_json(
            events[0], {**descriptor, 'input_sha256': envelope['input_sha256']}
        ),
        'inline_old.event_binding',
    )


def validate_c29(
    root: Path, envelope: dict, old: dict, core: dict, expected: dict
):
    validate(root, envelope, old)
    validate(root, envelope, core)
    require(
        old['selector'] == 'python'
        and core['selector'] == 'core'
        and old['pandas_major'] == core['pandas_major']
        and len(old['legacy_branch_events']) == 1
        and not core['legacy_branch_events']
        and old['native_calls'] == 0
        and core['native_calls'] > 0
        and core['legacy_calls'] == 0
        and core['public_calls'] > 0,
        'carry.c29_execution',
    )
    contract.observation(old['observation'])
    contract.observation(core['observation'])
    require(
        old['observation']['status']
        == core['observation']['status']
        == 'success',
        'carry.c29_success',
    )
    require(same_json(core['observation'], expected), 'carry.frozen_disagree')
