# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Phase 7.6 checks share (``CONTRACT.md`` beside this file).

Finding the link-prediction transcripts, reading them, and a boundary that
sits on one of ``KumoRelational``'s methods while a transcript replays,
keeping each outermost call's description, statements, warnings and result,
so the core's run of the same name can be driven from them.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import re
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DRIVER = REPO / 'tests' / 'rust_core' / 'sampling' / 'driver'
SAMPLING = REPO / 'tests' / 'rust_core' / 'sampling'
RUST_CORE = REPO / 'tests' / 'rust_core'
PARITY_SCRIPTS = REPO / 'scripts' / 'parity'
for _path in (PARITY_SCRIPTS, RUST_CORE, SAMPLING, DRIVER):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

TRANSCRIPTS = REPO / 'tests' / 'parity' / 'transcripts'
RECORDINGS = REPO / 'tests' / 'parity' / 'sampling'
GAPS = PARITY_SCRIPTS / 'sampling_coverage_gaps.json'
COVERAGE_JSON = REPO / 'tests' / 'parity' / 'coverage.json'
#: The request modules' gap list and unreachable-diagnostics list (Phase 8).
REQUEST_GAP_DOCS = (
    REPO / 'tests' / 'parity' / 'COVERAGE_GAPS.md',
    REPO / 'tests' / 'parity' / 'UNREACHABLE_DIAGNOSTICS.md',
)
RFM = 'packages/sdm-relational-engine/src/sdm_relational_engine/rfm/rfm.py'

OFFLINE = ('memory', 'duckdb', 'sqlite', 'postgres')

#: What ``KumoRelational._get_context`` raises when the sampled subgraph
#: holds no target entity.
NO_TARGET_ENTITIES = (
    'Cannot perform link prediction on subgraphs without any historical '
    'target entities. Please increase the number of hops and try again.'
)

#: What the retired gap reasons said: that no task reaching the engine is
#: link prediction.
RETIRED_REASONS = (
    'no task is link prediction',
    'no question reaching the engine has a list_distinct target',
)


def says_link_prediction_is_never_gathered(text: str) -> bool:
    r"""Whether ``text`` says what the retired request-gap reasons said:
    that the client never gathers a link prediction."""
    flat = ' '.join(text.lower().split())
    return bool(
        re.search(r'link[ _-]?predict', flat)
        and re.search(r'never (be )?gather', flat)
    )


#: A citation of a file of the migration's harness, which is not in the
#: repository: a phase's report, review, brief, status or audit, its open
#: questions, or the harness folder itself.
_HARNESS_CITATION = re.compile(
    r"phase [\d.]+(?:'s)? (?:report|review|brief|status|audit)\b"
    r'|\b(?:report|review-r\d+|brief|status|audit)\.md\b'
    r'|\bopen question'
    r'|sdm-dbx|rust-core/(?:phases|investigations|agents)\b',
    re.IGNORECASE,
)


def cites_the_harness(text: str) -> str | None:
    r"""What in ``text`` cites a harness file, if anything does."""
    found = _HARNESS_CITATION.search(' '.join(text.split()))
    return None if found is None else found[0]


#: The rfm.py branches the link-prediction scenarios take, as coverage.py
#: numbers them, each with the text its two lines start with (checked, so a
#: moved line fails loudly rather than silently checking another branch).
LINK_BRANCHES: dict[tuple[int, int], tuple[str, str]] = {
    (2020, 2021): (
        'if target.aggr == AggregationType.LIST_DISTINCT:',
        'table_name, col_name = target.get_target_column_name().split(',
    ),
    (2029, 2037): (
        'if len(target_edge_types) != 1:',
        'return TaskType.TEMPORAL_LINK_PREDICTION',
    ),
    (2385, 2386): (
        'if task_type.is_link_pred:',
        'final_aggr = query.get_final_target_aggregation()',
    ),
    (2389, 2390): (
        'for edge_type in self._sampler.edge_types:',
        "if edge_fkey == f'{edge_type[0]}.{edge_type[1]}':",
    ),
    (2389, 2398): (
        'for edge_type in self._sampler.edge_types:',
        "context_df = pd.DataFrame({'ENTITY': train_pkey, 'TARGET': train_y})",
    ),
    (2390, 2389): (
        "if edge_fkey == f'{edge_type[0]}.{edge_type[1]}':",
        'for edge_type in self._sampler.edge_types:',
    ),
    (2390, 2391): (
        "if edge_fkey == f'{edge_type[0]}.{edge_type[1]}':",
        'entity_table_names = (',
    ),
    (2529, 2533): ('if (', 'raise ValueError('),
}

#: The multilabel ``NotImplementedError`` of ``_get_task_type``, which a
#: ``LIST_DISTINCT`` over a foreign key linked to two tables reaches: a
#: scenario takes it.
MULTILABEL_BRANCH = (2029, 2030)
MULTILABEL_LINES = (
    'if len(target_edge_types) != 1:',
    'raise NotImplementedError(',
)


def multilabel_text(column: str) -> str:
    r"""What ``_get_task_type`` raises for a ``LIST_DISTINCT`` over
    ``column`` that has more than one edge."""
    return (
        "Multilabel-classification queries based on 'LIST_DISTINCT' are not "
        'supported yet. If you planned to write a link prediction query '
        f"instead, make sure to register '{column}' as a foreign key."
    )


#: What every multilabel text starts with.
MULTILABEL_START = multilabel_text('').split(' If you')[0]

_LIST_DISTINCT = re.compile(r'\bLIST_DISTINCT\s*\(', re.IGNORECASE)
_RANK = re.compile(r'\bRANK\b', re.IGNORECASE)
_ENTITY = re.compile(r'\bFOR\s+EACH\s+([A-Za-z_][\w]*)\.', re.IGNORECASE)


def query_of(transcript: dict[str, Any]) -> str | None:
    for call in transcript.get('calls', []):
        if call.get('call') == 'PredictiveQuery':
            query = call.get('args', {}).get('query')
            if isinstance(query, str):
                return query
    return None


def is_link_query(query: str | None) -> bool:
    return bool(
        query and _LIST_DISTINCT.search(query) and not _RANK.search(query)
    )


_TARGET = re.compile(
    r'\bLIST_DISTINCT\s*\(\s*([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)',
    re.IGNORECASE,
)


def entity_table(query: str) -> str:
    found = _ENTITY.search(query)
    assert found, f'no FOR EACH table in {query!r}'
    return found[1]


def target_column(query: str) -> tuple[str, str]:
    r"""The ``(table, column)`` a ``LIST_DISTINCT`` query lists."""
    found = _TARGET.search(query)
    assert found, f'no LIST_DISTINCT(table.column) in {query!r}'
    return found[1], found[2]


def is_multilabel(transcript: dict[str, Any]) -> bool:
    r"""Whether the run ended in the engine's multilabel refusal."""
    error = transcript.get('error')
    return isinstance(error, dict) and MULTILABEL_START in str(
        error.get('message', '')
    )


def _list_distinct_transcripts() -> dict[str, tuple[Path, dict[str, Any]]]:
    out = {}
    for path in sorted(TRANSCRIPTS.rglob('transcript.json')):
        transcript = json.loads(path.read_text('utf-8'))
        if is_link_query(query_of(transcript)):
            out[transcript['name']] = (path, transcript)
    return out


def link_transcripts() -> dict[str, Path]:
    r"""Every link-prediction transcript, by scenario name: a plain
    ``LIST_DISTINCT`` query, except a multilabel one."""
    return {
        name: path
        for name, (path, transcript) in _list_distinct_transcripts().items()
        if not is_multilabel(transcript)
    }


def multilabel_transcripts() -> dict[str, Path]:
    r"""Every multilabel transcript, by scenario name: a plain
    ``LIST_DISTINCT`` query that ended in the engine's multilabel refusal."""
    return {
        name: path
        for name, (path, transcript) in _list_distinct_transcripts().items()
        if is_multilabel(transcript)
    }


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text('utf-8'))


def is_refused(transcript: dict[str, Any]) -> bool:
    return transcript.get('error') is not None


def recording_dir_of(transcript_path: Path) -> Path:
    return RECORDINGS / transcript_path.parent.relative_to(TRANSCRIPTS)


def source_lines(relative: str) -> list[str]:
    return (REPO / _current_path(relative)).read_text('utf-8').splitlines()


def check_rfm_lines() -> None:
    r"""The branches named above are where this file says they are."""
    lines = source_lines(RFM)
    expected = {**LINK_BRANCHES, MULTILABEL_BRANCH: MULTILABEL_LINES}
    for (start, end), (first, second) in expected.items():
        got = (lines[start - 1].strip(), lines[end - 1].strip())
        assert got[0].startswith(first) and got[1].startswith(second), (
            f'rfm.py {start}->{end} is now {got}, not {(first, second)}: '
            'the engine changed under the checks'
        )


# --- A boundary on one of KumoRelational's methods ---------------------------


#: Describes a call: ``(this, arguments) -> (inputs_json, tables)``.
Describer = Callable[[Any, dict[str, Any]], tuple[str, dict[str, Any]]]


class Boundary:
    r"""Sits on one method while transcripts replay, keeping each outermost
    call's description, statements, warnings and result or error."""

    def __init__(
        self, owner: type, method: str, sampler_of: Callable[[Any], Any]
    ):
        from _driver import Guard

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
        from _driver import _source_classes, encode_params, warning_of

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
                    'arguments': arguments,
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


def model_describer(method: str) -> Describer:
    from _driver import capture_module

    describe = capture_module().describe

    def run(model: Any, arguments: dict[str, Any]) -> Any:
        return describe(model._sampler, model._graph, method, arguments)

    return run


def replay_with(
    boundary: Boundary, method: str, paths: list[Path]
) -> list[str]:
    r"""Replay ``paths`` with ``boundary`` installed; why any failed."""
    from _parity.corpus import check

    reasons = []
    with boundary.installed(model_describer(method)):
        for path in paths:
            before = len(boundary.calls)
            reason = check(path)
            if reason is not None:
                reasons.append(f'{path}: {reason}')
            for call in boundary.calls[before:]:
                call['transcript'] = path
    return reasons


def frame(df: Any) -> Any:
    import pyarrow as pa
    from _driver import canon_frame

    return canon_frame(
        pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False)
    )


def compare_runs(
    method: str,
    boundary: Boundary,
    want_of: Callable[[dict[str, Any]], Any],
    got_of: Callable[[Any], Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    r"""Drive the core's ``method`` run of every kept call; each call gets
    ``core`` (``'done'`` or ``'raised'``) and ``core_error``; and the
    problems found."""
    from _driver import (
        Mismatch,
        canon_warnings,
        difference,
        drive,
        error_of,
        native,
    )

    core = native(method, 'Outcome')
    problems = []
    for i, call in enumerate(boundary.calls):
        where = f'{call.get("transcript")} call {i} ({method})'
        inputs_json, tables = call['described']
        steps = call['steps']
        call['core'] = None
        call['core_error'] = None
        try:
            run = getattr(core, method)(inputs_json, tables)
            value, asked = drive(
                core, run, steps, lambda k, steps=steps: steps[k]['result']
            )
        except Mismatch as mismatch:
            problems.append(f'{where}: {mismatch}')
            continue
        except Exception as raised:  # noqa: BLE001 - compared
            call['core'] = 'raised'
            call['core_error'] = error_of(raised)
            if call['error'] is None or error_of(raised) != error_of(
                call['error']
            ):
                problems.append(f'{where}: raised {error_of(raised)}')
                continue
            found = difference(
                call['warnings'],
                canon_warnings(getattr(raised, 'warnings', None), 'raised'),
            )
            if found:
                problems.append(f'{where}: warnings before the error: {found}')
            continue
        call['core'] = 'done'
        if call['error'] is not None:
            problems.append(
                f'{where}: finished where today raised {error_of(call["error"])}'
            )
            continue
        if len(asked) != len(steps):
            problems.append(
                f'{where}: {len(asked)} queries for {len(steps)} statements'
            )
            continue
        found = difference(want_of(call), got_of(value))
        if found:
            problems.append(f'{where}: {found}')
    return boundary.calls, problems


def task_table_want(call: dict[str, Any]) -> Any:
    task = call['result']
    return {
        'task_type': str(task.task_type.value),
        'entity_table_names': list(task.entity_table_names),
        'context': frame(task._context_df),
        'prediction': frame(task._pred_df),
        'time_column': task._time_column,
        'num_forecasts': task.num_forecasts,
        'step_size': task.step_size,
        'warnings': call['warnings'],
    }


def task_table_got(value: Any) -> Any:
    from _driver import canon_frame, canon_warnings

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
    from _driver import python_outputs

    context = call['result']
    table = context.task_table
    return {
        'subgraph': python_outputs(
            'sample_subgraph', context.sampled_subgraph, None, call['warnings']
        ),
        'task_table': None if table is None else frame(table.df),
    }


def context_got(value: Any) -> Any:
    from _driver import canon_frame, rust_outputs

    table = value['task_table']
    return {
        'subgraph': rust_outputs('sample_subgraph', value['subgraph']),
        'task_table': None if table is None else canon_frame(table),
    }


# --- The core driven from recordings alone -----------------------------------


def drive_recordings(
    found: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    r"""The boundary recordings of ``found`` (scenario names to transcript
    paths), and the core driven over them by Phase 7.5's ``_driver_run.py``
    in a process without today's Python."""
    import os
    import shutil
    import subprocess
    import tempfile

    from _driver import METHODS, native, read_recording
    from _sampling import INSTALLED

    native(*METHODS, 'Outcome')
    recordings = {}
    for name, path in found.items():
        directory = recording_dir_of(path)
        assert (directory / 'boundary.json').is_file(), (
            f'{name}: no boundary recording at {directory}'
        )
        recording = read_recording(directory)
        assert recording['scenario'] == name, (
            f'{directory} records {recording["scenario"]!r}, not {name!r}'
        )
        recordings[name] = recording
    env = dict(os.environ)
    env[INSTALLED] = '1'
    with tempfile.TemporaryDirectory(prefix='sdm-p76-core-') as tmp:
        root = Path(tmp) / 'recordings'
        for name in recordings:
            source = recording_dir_of(found[name])
            shutil.copytree(source, root / source.relative_to(RECORDINGS))
        out = Path(tmp) / 'out.json'
        proc = subprocess.run(
            [
                sys.executable,
                str(DRIVER / '_driver_run.py'),
                str(root),
                str(out),
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert proc.returncode == 0, (
            f'_driver_run.py failed ({proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}'
        )
        return recordings, json.loads(out.read_text('utf-8'))


def recording_problems(
    recordings: dict[str, Any], got: dict[str, list[dict[str, Any]]]
) -> tuple[list[str], int]:
    r"""Where the core's driven calls differ from the recorded ones, and how
    many calls were compared and agreed."""
    from _driver import difference

    problems = []
    compared = 0
    for name, recording in recordings.items():
        entries = got.get(name) or []
        if len(entries) != len(recording['calls']):
            problems.append(
                f'{name}: {len(entries)} of {len(recording["calls"])} calls '
                'driven'
            )
            continue
        for i, (call, entry) in enumerate(
            zip(recording['calls'], entries, strict=True)
        ):
            where = f'{name} call {i} ({call["method"]})'
            outputs = entry['outputs']
            if entry['sql'] is not None:
                problems.append(f'{where}: {entry["sql"]}')
            elif outputs is None or 'unreadable' in outputs:
                problems.append(f'{where}: {outputs}')
            elif 'error' not in call['outputs'] and entry['queries'] != len(
                call['steps']
            ):
                problems.append(
                    f'{where}: {entry["queries"]} queries for '
                    f'{len(call["steps"])} statements'
                )
            else:
                found_ = difference(call['outputs'], outputs)
                if found_:
                    problems.append(f'{where}: {found_}')
                else:
                    compared += 1
    return problems, compared


def _current_path(path: str) -> str:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'switch'))
    from _switch import current_package_path

    return current_package_path(path)
