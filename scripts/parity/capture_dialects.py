#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Record every call into the SQL dialects, as the Rust port is checked
against.

    capture_dialects.py [--out DIR]

Every SQL transcript of the parity corpus, the live ones too, is replayed
with each call into a backend's ``dialect.py`` recorded: its arguments, and
the ``(sql, params)`` or other value it returned. So are the calls into the
SQL fragments of ``base/utils.py`` a backend's ``dialect.py`` or ``table.py``
imports, into ``sdm_connectors.sql.quote_ident`` where a ``dialect.py``
imports it, into ``view_expr``'s checks, and into the helpers of the tables
and samplers that compute SQL or its parameters (the moved helpers,
:data:`HELPERS`). Then the branches no scenario reaches are called directly
(:mod:`_dialect_synthetic`).

It writes ``<backend>.json`` for each backend into ``DIR``, by default
``tests/parity/dialects/``, as ``tests/rust_core/dialects/CONTRACT.md``
describes. It needs no network, database or credentials, and writes the same
bytes on every run.

:func:`replay_helper` calls a moved helper on an object built from the
arguments the capture recorded for it.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import inspect
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'tests' / 'rust_core'))
sys.path.insert(0, str(REPO / 'tests' / 'rust_core' / 'dialects'))

from _codec import encode  # noqa: E402
from _dialect_helpers import HELPERS, replay_helper  # noqa: E402
from _old_python import fixed_scenarios, use_old_python  # noqa: E402
from _parity.values import encode as encode_params  # noqa: E402

__all__ = ['main', 'replay_helper']

DEFAULT_OUT = REPO / 'tests' / 'parity' / 'dialects'
TRANSCRIPTS = REPO / 'tests' / 'parity' / 'transcripts'
ENGINE = 'sdm_relational_engine.rfm'
FORMAT = 1

BACKENDS = ('databricks', 'duckdb', 'postgres', 'snow', 'sqlite')

#: The backend whose dialect runs a transcript's SQL.
TRANSCRIPT_DIALECT = {
    'databricks': 'databricks',
    'duckdb': 'duckdb',
    'lakebase': 'postgres',
    'postgres': 'postgres',
    'snowflake': 'snow',
    'sqlite': 'sqlite',
}

#: The SQL fragments of ``base/utils.py`` the dialects take over.
UTILS_FRAGMENTS = (
    'ordered_sample_sql',
    'key_count_select',
    'count_distinct_values',
    'repeated_key_select',
    'sample_order_terms',
    'percent_literal',
    'prefilter_fraction',
)

#: The backend a ``view_expr`` check is made for, by its ``dialect``.
VIEW_DIALECT = {'databricks': 'databricks', 'snowflake': 'snow'}


def _is_statement(function: Callable[..., Any]) -> bool:
    annotation = inspect.signature(function).return_annotation
    return annotation in ('Statement', 'Statement | None')


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


class Capture:
    r"""The calls recorded so far, by backend, each once, with every source
    it was made from."""

    def __init__(self) -> None:
        self.calls: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
            backend: {} for backend in BACKENDS
        }
        self.source: str | None = None

    @contextlib.contextmanager
    def under(self, source: str) -> Iterator[None]:
        r"""Record the calls made in the block as made from ``source``."""
        previous, self.source = self.source, source
        try:
            yield
        finally:
            self.source = previous

    def add(
        self,
        backend: str,
        function: str,
        args: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        if self.source is None:
            return
        key = (function, _canonical(args))
        entry = self.calls[backend].get(key)
        if entry is None:
            entry = {'function': function, 'args': args, 'sources': []}
            entry.update(output)
            self.calls[backend][key] = entry
        elif _canonical({k: entry[k] for k in output}) != _canonical(output):
            raise AssertionError(
                f'{backend} {function}: one call gave two outputs, '
                f'{entry} and {output}'
            )
        if self.source not in entry['sources']:
            entry['sources'].append(self.source)

    def files(self) -> dict[str, str]:
        r"""The text of each backend's file."""
        out = {}
        for backend in BACKENDS:
            calls = list(self.calls[backend].values())
            lines = [
                '{',
                f' "format": {FORMAT},',
                f' "backend": {json.dumps(backend)},',
                ' "calls": [',
            ]
            for i, call in enumerate(calls):
                comma = ',' if i < len(calls) - 1 else ''
                text = json.dumps(call, ensure_ascii=False, allow_nan=False)
                lines.append(f'  {text}{comma}')
            lines += [' ]', '}', '']
            out[backend] = '\n'.join(lines)
        return out


# --- Recording -----------------------------------------------------------------


def _output(function: Callable[..., Any], value: Any) -> dict[str, Any]:
    if _is_statement(function):
        sql, params = value
        return {
            'sql': sql,
            'params': None if params is None else encode_params(list(params)),
        }
    return {'result': encode(value)}


def _recording(
    capture: Capture,
    backend: str | Callable[[inspect.BoundArguments], str | None],
    name: str,
    function: Callable[..., Any],
) -> Callable[..., Any]:
    r"""``function``, recording each call that returns under ``name``, in
    ``backend``'s capture, or the one ``backend`` picks from the call."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def recorded(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        encoded = {key: encode(value) for key, value in bound.arguments.items()}
        value = function(*args, **kwargs)
        where = backend(bound) if callable(backend) else backend
        if where is not None:
            capture.add(where, name, encoded, _output(function, value))
        return value

    return recorded


def _patch(module: Any, name: str, value: Any, undo: list) -> None:
    undo.append((module, name, vars(module)[name]))
    setattr(module, name, value)


@contextlib.contextmanager
def installed(capture: Capture) -> Iterator[None]:
    r"""Record every call the capture takes, while the block runs."""
    from sdm_connectors import sql as connector_sql

    utils = importlib.import_module(f'{ENGINE}.base.utils')
    undo: list[tuple[Any, str, Any]] = []
    try:
        for backend in BACKENDS:
            dialect = importlib.import_module(
                f'{ENGINE}.backend.{backend}.dialect'
            )
            for name, value in list(vars(dialect).items()):
                if (
                    inspect.isfunction(value)
                    and value.__module__ == dialect.__name__
                ):
                    wrapped = _recording(capture, backend, name, value)
                    _patch(dialect, name, wrapped, undo)
            if vars(dialect).get('quote_ident') is connector_sql.quote_ident:
                wrapped = _recording(
                    capture,
                    backend,
                    'sql.quote_ident',
                    connector_sql.quote_ident,
                )
                _patch(dialect, 'quote_ident', wrapped, undo)
            for module_name in ('dialect', 'table'):
                module = importlib.import_module(
                    f'{ENGINE}.backend.{backend}.{module_name}'
                )
                for name in UTILS_FRAGMENTS:
                    original = getattr(utils, name)
                    if vars(module).get(name) is original:
                        wrapped = _recording(
                            capture, backend, f'utils.{name}', original
                        )
                        _patch(module, name, wrapped, undo)
        _install_view_expr(capture, undo)
        for backend, helpers in HELPERS.items():
            for helper in helpers:
                undo.append(helper.install(capture, backend))
        yield
    finally:
        for target, name, original in reversed(undo):
            setattr(target, name, original)


def _install_view_expr(capture: Capture, undo: list) -> None:
    r"""Record the view expression checks, in the capture of the backend
    whose dialect each is made for."""
    view_expr = importlib.import_module(f'{ENGINE}.view_expr')

    def by_dialect(bound: inspect.BoundArguments) -> str | None:
        return VIEW_DIALECT.get(bound.arguments.get('dialect') or '')

    for module_name in ('graph', 'backend.databricks.metric_view'):
        module = importlib.import_module(f'{ENGINE}.{module_name}')
        for name in ('_unsafe_expr_reason', '_untrusted_call_reason'):
            original = getattr(view_expr, name)
            if vars(module).get(name) is original:
                wrapped = _recording(
                    capture, by_dialect, f'view_expr.{name}', original
                )
                _patch(module, name, wrapped, undo)


# --- The corpus ----------------------------------------------------------------


def replay_corpus(capture: Capture) -> int:
    r"""Replay every SQL transcript under ``capture``; the number replayed.

    A transcript a fix in the core re-recorded must satisfy its exact old
    refusal and complete governed native replay before its earlier calls
    are kept."""
    from _fixed_contracts import fixed_replay
    from _parity.corpus import check

    fixed = fixed_scenarios()
    count = 0
    for path in sorted(TRANSCRIPTS.rglob('transcript.json')):
        transcript = json.loads(path.read_text())
        if transcript['backend'] not in TRANSCRIPT_DIALECT:
            continue
        with capture.under(f'scenario:{transcript["name"]}'):
            reason = check(path)
        if transcript['name'] in fixed:
            fixed_replay(path, reason)
        elif reason is not None:
            raise SystemExit(
                f'{transcript["name"]} does not replay, so it cannot be '
                f'captured: {reason}'
            )
        count += 1
    return count


def capture_all() -> Capture:
    from _dialect_synthetic import synthetic

    capture = Capture()
    with installed(capture):
        replay_corpus(capture)
        synthetic(capture)
    return capture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    use_old_python()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for backend, text in capture_all().files().items():
        (out / f'{backend}.json').write_text(text, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
