# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Reading and writing transcripts and the Arrow files beside them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa

#: The file a scenario's transcript is written to, in its own directory.
TRANSCRIPT_FILE = 'transcript.json'
#: The decoded result's Arrow file.
RESULT_FILE = 'result.arrow'

#: The order keys are written in, so a transcript reads top to bottom.
_KEY_ORDER = (
    'format_version',
    'name',
    'backend',
    'task_kind',
    'status',
    'data',
    'stand_in',
    'calls',
    'steps',
    'payloads',
    'responses',
    'error',
    'result',
    'answer',
    'warnings',
)


def step_file(index: int) -> str:
    return f'step-{index:04d}.arrow'


def scenario_dir(corpus: Path, name: str) -> Path:
    r"""Where a scenario's files go: its name as a path, ``#`` as ``--``."""
    return corpus.joinpath(*name.replace('#', '--').split('/'))


def read_arrow(path: Path) -> pa.Table:
    data = Path(path).read_bytes()
    try:
        return pa.ipc.open_file(pa.BufferReader(data)).read_all()
    except pa.ArrowInvalid:
        return pa.ipc.open_stream(pa.BufferReader(data)).read_all()


def write_arrow(path: Path, table: pa.Table) -> None:
    r"""An Arrow IPC file, uncompressed, so any Arrow reader opens it."""
    with pa.OSFile(str(path), 'wb') as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dumps(transcript: dict[str, Any]) -> str:
    r"""A transcript as text: one key or row per line, scalars inline."""
    ordered = {key: transcript[key] for key in _KEY_ORDER if key in transcript}
    ordered.update(
        (key, value) for key, value in transcript.items() if key not in ordered
    )
    return _pretty(ordered, 0) + '\n'


def _flat(value: Any) -> bool:
    return not isinstance(value, (list, dict)) or (
        not value and isinstance(value, (list, dict))
    )


def _pretty(value: Any, level: int) -> str:
    r"""JSON with an array of scalars, such as a row, kept on one line."""
    if isinstance(value, list) and all(_flat(item) for item in value):
        return json.dumps(value, ensure_ascii=False)
    pad = ' ' * (level + 1)
    if isinstance(value, list):
        items = [pad + _pretty(item, level + 1) for item in value]
        return '[\n' + ',\n'.join(items) + '\n' + ' ' * level + ']'
    if isinstance(value, dict) and value:
        items = [
            f'{pad}{json.dumps(key, ensure_ascii=False)}: '
            f'{_pretty(item, level + 1)}'
            for key, item in value.items()
        ]
        return '{\n' + ',\n'.join(items) + '\n' + ' ' * level + '}'
    return json.dumps(value, ensure_ascii=False)


def save(
    directory: Path,
    transcript: dict[str, Any],
    step_results: dict[int, pa.Table],
    result: pa.Table | None,
) -> Path:
    r"""Write a scenario's directory afresh: the transcript and its tables."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for index, table in sorted(step_results.items()):
        transcript['steps'][index]['result'] = step_file(index)
        write_arrow(directory / step_file(index), table)
    if result is not None:
        transcript['result'] = RESULT_FILE
        write_arrow(directory / RESULT_FILE, result)
    path = directory / TRANSCRIPT_FILE
    path.write_text(dumps(transcript), encoding='utf-8')
    return path


def frame_to_table(frame: Any) -> pa.Table:
    r"""A ``PredictionResult.to_pandas()`` frame as an Arrow table.

    The index is kept as a column only when it is not the default one, and
    no pandas metadata is stored, so the file does not change with pandas.
    """
    import pandas as pd

    if not (
        isinstance(frame.index, pd.RangeIndex)
        and frame.index.start == 0
        and frame.index.step == 1
        and frame.index.name is None
    ):
        frame = frame.reset_index()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    return table.replace_schema_metadata(None)
