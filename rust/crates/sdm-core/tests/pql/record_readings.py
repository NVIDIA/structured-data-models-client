# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records the reference's reading of every date literal in the PQL corpus.

The corpus's trees (``corpus.json`` beside this file) hold a date literal as
the query wrote it. The reference reads one as ``Constant.typed_value``
does, ``pd.to_datetime`` of that text; this script asks it, for each date
constant of each accepted case's tree, and writes the answer into the node
as ``reading``: the date and time, ``YYYY-MM-DDTHH:MM:SS``, the UTC offset
in seconds when the literal gives one, and the nanoseconds past the second,
which the validated-query format cannot hold and so must be 0.

Run it with the reference's interpreter, isolated, from the repository root:

    ~/code/sdmc-reference/.venv/bin/python -I \\
        rust/crates/sdm-core/tests/pql/record_readings.py ~/code/sdmc-reference
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CORPUS = HERE / 'corpus.json'


def _guard(reference: Path) -> Any:
    path = reference / 'scripts/reference_guard.py'
    spec = importlib.util.spec_from_file_location('reference_guard', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _readings(node: Any, read: Any) -> int:
    """Writes the reading into every date constant under ``node``; how
    many it wrote."""
    count = 0
    if isinstance(node, dict):
        value = node.get('value')
        if node.get('dtype_maybe') in ('date', 'time') and isinstance(
            value, str
        ):
            node['reading'] = read(value)
            count += 1
        for child in node.values():
            count += _readings(child, read)
    elif isinstance(node, list):
        for child in node:
            count += _readings(child, read)
    return count


def main(reference: Path) -> int:
    guard = _guard(reference)
    guard.require_isolation()

    from predictive_query.ast.constant import Constant
    from predictive_query.schema.typing import Dtype

    def read(text: str) -> dict[str, Any]:
        stamp = Constant(value=text, dtype_maybe=Dtype.time).typed_value()
        offset = stamp.utcoffset()
        return {
            'value': stamp.strftime('%Y-%m-%dT%H:%M:%S'),
            'offset': None if offset is None else int(offset.total_seconds()),
            'nanos': stamp.microsecond * 1000 + stamp.nanosecond,
        }

    corpus = json.loads(CORPUS.read_text(encoding='utf-8'))
    count = sum(
        _readings(case['tree'], read)
        for case in corpus['cases']
        if 'tree' in case
    )
    guard.require_isolation()

    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    corpus['format'] = 'sdm-pql-corpus/2'
    corpus['readings'] = (
        f'rust-core-reference at {commit}: each date constant of an accepted '
        f"case's tree carries the reference's reading of it, recorded by "
        f'record_readings.py beside this file'
    )
    CORPUS.write_text(
        json.dumps(corpus, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'{count} date literals read at {commit[:12]}')
    return 0


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1]).expanduser()))
