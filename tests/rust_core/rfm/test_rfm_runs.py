# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core's runs above and beside the sampler's two boundary calls,
against today's Python over the parity corpus.

Every transcript replays with today's method running as it does, the call's
inputs described as ``scripts/parity/capture_sampling.py`` describes them,
and the statements the method ran and the warnings it gave kept; the core's
run of the same name is then driven from those inputs, each query answered
from the kept statements, and must ask exactly them and give what today's
method gave, or raise what it raised, with the warnings it gave first:

- ``Sampler.sample_lagged_target``: ``sdm_core._native.sample_lagged_target``,
  its frame of lagged labels and the warnings;
- ``KumoRelational._get_task_table``: ``task_table``, the task's frames
  (``ENTITY``, ``TARGET``, ``ANCHOR_TIMESTAMP`` and lagged targets), its
  type, entity tables, time column and the warnings;
- ``KumoRelational._get_context``: ``context``, the sampled subgraph and the
  task's feature table.

Frames compare by Arrow value and type, as the boundary checks compare them.
No recording is committed: each run compares with today's Python directly.
Without the core or the engine installed, as in CI's test job, it skips.
"""

from __future__ import annotations

import contextlib
import inspect
import re
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('sdm_core')
pytest.importorskip('sdm_relational_engine')

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DRIVER = REPO / 'tests' / 'rust_core' / 'sampling' / 'driver'
for _path in (DRIVER, REPO / 'scripts' / 'parity'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Today's Python, which these hold the core to, runs public calls here.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _old_path import python_path  # noqa: E402,F401 - the fixture

pytestmark = pytest.mark.usefixtures('python_path')

from _driver import (  # noqa: E402
    Guard,
    Mismatch,
    _source_classes,
    canon_frame,
    canon_warnings,
    capture_module,
    difference,
    drive,
    encode_params,
    error_of,
    native,
    python_outputs,
    rust_outputs,
    transcripts,
    warning_of,
)

#: Describes a call: ``(this, arguments) -> (inputs_json, tables)``.
Describer = Callable[[Any, dict[str, Any]], tuple[str, dict[str, Any]]]


class Boundary:
    r"""Sits on one method while transcripts replay, keeping each outermost
    call's description, statements, warnings and result."""

    def __init__(
        self, owner: type, method: str, sampler_of: Callable[[Any], Any]
    ) -> None:
        self.owner = owner
        self.method = method
        self.sampler_of = sampler_of
        self.calls: list[dict[str, Any]] = []
        self.guard = Guard()
        self._steps: list[dict[str, Any]] | None = None
        self._depth = 0
        self._run_depth = 0

    @contextlib.contextmanager
    def installed(self, describe: Describer) -> Iterator[Boundary]:
        undo = []

        def patch(owner: Any, name: str, value: Any) -> None:
            undo.append((owner, name, vars(owner)[name]))
            setattr(owner, name, value)

        original = vars(self.owner)[self.method]
        signature = inspect.signature(original)
        boundary = self

        def call(this: Any, *args: Any, **kwargs: Any) -> Any:
            if boundary._depth:
                return original(this, *args, **kwargs)
            bound = signature.bind(this, *args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arguments.pop('self', None)
            with boundary.guard.on(boundary.sampler_of(this)):
                described = describe(this, arguments)
            boundary._steps = []
            boundary._depth += 1
            error = None
            result = None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    result = original(this, *args, **kwargs)
                except Exception as raised:  # noqa: BLE001 - kept, then raised
                    error = raised
                finally:
                    boundary._depth -= 1
                    steps, boundary._steps = boundary._steps, None
            given = [w for w in caught if issubclass(w.category, UserWarning)]
            boundary.calls.append(
                {
                    'described': described,
                    'steps': steps,
                    'result': result,
                    'error': error,
                    'warnings': [warning_of(w) for w in given],
                }
            )
            for w in caught:
                warnings.warn_explicit(
                    w.message, w.category, w.filename, w.lineno
                )
            if error is not None:
                raise error
            return result

        def source_run(original_run: Any) -> Any:
            def run(source: Any, sql: Any, params: Any = None) -> Any:
                if boundary._steps is None or boundary._run_depth:
                    return original_run(source, sql, params)
                step = {
                    'sql': sql,
                    'params': encode_params(params),
                    'result': None,
                    'error': None,
                }
                boundary._run_depth += 1
                try:
                    step['result'] = original_run(source, sql, params)
                    return step['result']
                except Exception as error:
                    step['error'] = {
                        'message': str(error),
                        'sqlstate': None,
                        'vendor_code': None,
                    }
                    raise
                finally:
                    boundary._run_depth -= 1
                    boundary._steps.append(step)

            return run

        try:
            patch(self.owner, self.method, call)
            for cls in _source_classes():
                patch(cls, '_run', source_run(vars(cls)['_run']))
            yield self
        finally:
            for owner, name, value in reversed(undo):
                setattr(owner, name, value)


def recorded_as_fixed(path: Any) -> bool:
    r"""Whether the transcript at ``path`` stores the answer a Phase 9 task
    7b fix gives (``tests/rust_core/shadow/fix_changes.json``), which today's
    code, run here, does not give."""
    import json

    listed = REPO / 'tests' / 'rust_core' / 'shadow' / 'fix_changes.json'
    name = json.loads(Path(path).read_text(encoding='utf-8'))['name']
    return name in json.loads(listed.read_text(encoding='utf-8'))


def _replay_all(boundary: Boundary, describe: Describer) -> list[str]:
    from _fixed_contracts import fixed_replay
    from _parity.corpus import check

    reasons = []
    for path in transcripts():
        with boundary.installed(describe):
            reason = check(path)
        if recorded_as_fixed(path):
            try:
                fixed_replay(path, reason)
            except AssertionError as error:
                reasons.append(f'{path}: {error}')
        elif reason is not None:
            reasons.append(f'{path}: {reason}')
    return reasons


#: Today's errors that a Phase 9 task 7b fix answers instead: the core's run
#: finishes where today's raised these (``CHANGELOG.md``; the shadow
#: comparison's ``fix_changes.json``).
ANSWERED_BY_A_FIX = (
    # A whole decimal entity key, read as the integers it holds.
    re.compile(
        r"Encountered unsupported data type 'decimal128\(\d+, 0\)\[pyarrow\]' "
        r"for column 'ENTITY' in task table"
    ),
)


def _answered_by_a_fix(error: BaseException) -> bool:
    return type(error) is RuntimeError and any(
        fix.fullmatch(str(error).split('. Please either ')[0])
        for fix in ANSWERED_BY_A_FIX
    )


def _decimal_task_want(call: dict[str, Any]) -> Any:
    import pyarrow as pa
    from sdm_relational_engine.rfm.inference.task import TaskTable

    error = call['error']
    assert _answered_by_a_fix(error)
    traceback = error.__traceback__
    constructors = []
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code is TaskTable.__init__.__code__:
            constructors.append(frame.f_locals)
        traceback = traceback.tb_next
    assert len(constructors) == 1, 'missing exact old task constructor'
    arguments = constructors[0]
    expected_error = (
        f"Encountered unsupported data type '{arguments['context_df']['ENTITY'].dtype}' "
        "for column 'ENTITY' in task table. Please either manually override "
        "the columns's data type or remove the column from this table."
    )
    assert str(error) == expected_error
    assert arguments['entity_column'] == 'ENTITY'
    assert arguments['entity_table_name'] == ('products',)

    def frame(name):
        table = pa.Table.from_pandas(
            arguments[name].reset_index(drop=True), preserve_index=False
        )
        position = table.schema.get_field_index('ENTITY')
        assert position >= 0
        column = table.column(position)
        assert pa.types.is_decimal(column.type) and column.type.scale == 0
        if any(
            value is not None and not -(2**63) <= value < 2**63
            for value in column.to_pylist()
        ):
            return canon_frame(table)
        return canon_frame(
            table.set_column(
                position, 'ENTITY', column.cast(pa.int64(), safe=True)
            )
        )

    return {
        'task_type': arguments['task_type'].value,
        'entity_table_names': ['products'],
        'context': frame('context_df'),
        'prediction': frame('pred_df'),
        'time_column': arguments['time_column'],
        'num_forecasts': arguments['num_forecasts'],
        'step_size': arguments['step_size'],
        'warnings': call['warnings'],
    }


def _compare(
    method: str,
    boundary: Boundary,
    want_of: Callable[[dict[str, Any]], Any],
    got_of: Callable[[Any], Any],
) -> tuple[int, list[str]]:
    core = native(method, 'Outcome')
    problems = []
    for i, call in enumerate(boundary.calls):
        inputs_json, tables = call['described']
        steps = call['steps']
        try:
            run = getattr(core, method)(inputs_json, tables)
            value, _ = drive(
                core, run, steps, lambda k, steps=steps: steps[k]['result']
            )
        except Mismatch as mismatch:
            problems.append(f'call {i}: {mismatch}')
            continue
        except Exception as raised:  # noqa: BLE001 - compared
            if method == 'task_table' and _answered_by_a_fix(call['error']):
                problems.append(
                    f'call {i}: fixed decimal task raised {error_of(raised)}'
                )
                continue
            if call['error'] is None or error_of(raised) != error_of(
                call['error']
            ):
                problems.append(f'call {i}: raised {error_of(raised)}')
                continue
            found = difference(
                call['warnings'],
                canon_warnings(getattr(raised, 'warnings', None), 'raised'),
            )
            if found:
                problems.append(f'call {i}: warnings before the error: {found}')
            continue
        if call['error'] is not None:
            if method != 'task_table' or not _answered_by_a_fix(call['error']):
                problems.append(f'call {i}: finished where today raised')
                continue
            found = difference(_decimal_task_want(call), got_of(value))
            if found:
                problems.append(f'call {i}: fixed decimal task: {found}')
            continue
        found = difference(want_of(call), got_of(value))
        if found:
            problems.append(f'call {i}: {found}')
    return len(boundary.calls), problems


def _sampler_describer(method: str) -> Describer:
    describe = capture_module().describe

    def run(this: Any, arguments: dict[str, Any]) -> Any:
        return describe(this, _graph_of_sampler(this), method, arguments)

    return run


#: Each sampler's graph, kept as the samplers are made.
_GRAPHS: dict[int, Any] = {}


def _graph_of_sampler(sampler: Any) -> Any:
    return _GRAPHS[id(sampler)]


@contextlib.contextmanager
def _keeping_graphs() -> Iterator[None]:
    from sdm_relational_engine.rfm.base.sampler import Sampler

    original = vars(Sampler)['__init__']

    def init(sampler: Any, graph: Any, *args: Any, **kwargs: Any) -> None:
        _GRAPHS[id(sampler)] = graph
        original(sampler, graph, *args, **kwargs)

    setattr(Sampler, '__init__', init)  # noqa: B010 - typed as a method
    try:
        yield
    finally:
        setattr(Sampler, '__init__', original)  # noqa: B010 - typed as a method


def _frame(frame: Any) -> Any:
    import pyarrow as pa

    return canon_frame(
        pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    )


def task_table_want(call: dict[str, Any]) -> Any:
    r"""Today's task table, in canonical form."""
    task = call['result']
    return {
        'task_type': str(task.task_type.value),
        'entity_table_names': list(task.entity_table_names),
        'context': _frame(task._context_df),
        'prediction': _frame(task._pred_df),
        'time_column': task._time_column,
        'num_forecasts': task.num_forecasts,
        'step_size': task.step_size,
        'warnings': call['warnings'],
    }


def task_table_got(value: Any) -> Any:
    r"""The core's ``task_table`` done value, in the same form."""
    return {
        'task_type': value['task_type'],
        'entity_table_names': list(value['entity_table_names']),
        'context': canon_frame(value['context']),
        'prediction': canon_frame(value['prediction']),
        'time_column': value['time_column'],
        'num_forecasts': value['num_forecasts'],
        'step_size': value['step_size'],
        'warnings': canon_warnings(value['warnings'], 'done'),
    }


def context_want(call: dict[str, Any]) -> Any:
    r"""Today's context: its sampled subgraph and task table."""
    context = call['result']
    table = context.task_table
    return {
        'subgraph': python_outputs(
            'sample_subgraph', context.sampled_subgraph, None, call['warnings']
        ),
        'task_table': None if table is None else _frame(table.df),
    }


def context_got(value: Any) -> Any:
    r"""The core's ``context`` done value, in the same form."""
    table = value['task_table']
    return {
        'subgraph': rust_outputs('sample_subgraph', value['subgraph']),
        'task_table': None if table is None else canon_frame(table),
    }


def duckdb_missing_want(call: dict[str, Any]) -> Any:
    return {
        table: sorted([column] for column in columns)
        for table, columns in call['result'].items()
        if columns
    }


def duckdb_missing_got(value: Any) -> Any:
    return {
        table: sorted(list(index) for index in indices)
        for table, indices in value['missing'].items()
    }


def sqlite_missing_want(call: dict[str, Any]) -> Any:
    return call['warnings']


def sqlite_missing_got(value: Any) -> Any:
    return canon_warnings(value['warnings'], 'done')


def test_lagged_targets() -> None:
    from sdm_relational_engine.rfm.base.sampler import Sampler

    boundary = Boundary(Sampler, 'sample_lagged_target', lambda s: s)
    with _keeping_graphs():
        reasons = _replay_all(
            boundary, _sampler_describer('sample_lagged_target')
        )
    assert not reasons, reasons[:3]
    count, problems = _compare(
        'sample_lagged_target',
        boundary,
        lambda call: {
            'frame': _frame(call['result']),
            'warnings': call['warnings'],
        },
        lambda value: {
            'frame': canon_frame(value['frame']),
            'warnings': canon_warnings(value['warnings'], 'done'),
        },
    )
    assert count >= 4, f'only {count} calls'
    assert not problems, '\n'.join(problems[:20])


def _model_describer(method: str) -> Describer:
    describe = capture_module().describe

    def run(model: Any, arguments: dict[str, Any]) -> Any:
        return describe(model._sampler, model._graph, method, arguments)

    return run


def test_task_tables() -> None:
    from sdm_relational_engine.rfm.rfm import KumoRelational

    boundary = Boundary(
        KumoRelational, '_get_task_table', lambda model: model._sampler
    )
    reasons = _replay_all(boundary, _model_describer('_get_task_table'))
    assert not reasons, reasons[:3]
    count, problems = _compare(
        'task_table', boundary, task_table_want, task_table_got
    )
    assert count >= 70, f'only {count} calls'
    assert not problems, '\n'.join(problems[:20])


def test_contexts() -> None:
    from sdm_relational_engine.rfm.rfm import KumoRelational

    boundary = Boundary(
        KumoRelational, '_get_context', lambda model: model._sampler
    )
    reasons = _replay_all(boundary, _model_describer('_get_context'))
    assert not reasons, reasons[:3]
    count, problems = _compare('context', boundary, context_want, context_got)
    assert count >= 70, f'only {count} calls'
    assert not problems, '\n'.join(problems[:20])


def _missing_indices(method: str, owner: type, want: Any, got: Any) -> None:
    boundary = Boundary(owner, method, lambda s: s)

    def describe(sampler: Any, arguments: dict[str, Any]) -> Any:
        return capture_module().describe(
            sampler, arguments['graph'], method, arguments
        )

    reasons = _replay_all(boundary, describe)
    assert not reasons, reasons[:3]
    count, problems = _compare('missing_indices', boundary, want, got)
    assert count >= 6, f'only {count} calls'
    assert not problems, '\n'.join(problems[:20])


def test_duckdb_missing_indices() -> None:
    from sdm_relational_engine.rfm.backend.duckdb import DuckDBSampler

    _missing_indices(
        '_get_missing_index_dict',
        DuckDBSampler,
        duckdb_missing_want,
        duckdb_missing_got,
    )


def test_sqlite_missing_indices() -> None:
    from sdm_relational_engine.rfm.backend.sqlite import SQLiteSampler

    _missing_indices(
        '_warn_missing_indices',
        SQLiteSampler,
        sqlite_missing_want,
        sqlite_missing_got,
    )
