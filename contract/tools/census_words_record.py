#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the old client's words for each refusal census probe on DuckDB
into ``contract/census-words/duckdb.jsonl`` (T3.17 review r2), so
``test_t317_wording.py`` compares the core route's words with them wherever
the suite runs. Run from this repository's root with this checkout's
interpreter, naming the reference checkout and its interpreter::

    .venv/bin/python contract/tools/census_words_record.py \
        --reference ~/code/sdmc-reference \
        --reference-python ~/code/sdmc-reference/.venv/bin/python

Each probe's case is built as the census builds it
(``refusal_census.py``'s ``run``) and run by the census's ``route`` on the
reference's interpreter, with ``SDM_CENSUS_MESSAGES`` set, so each outcome
carries its refusal's text. One line a probe, in the probe list's order:
its ``site``, the ``result`` and the ``message`` (``str()`` of the refusal,
none for a success), and the reference commit the words are from.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/census-words/duckdb.jsonl'


def _census() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        'refusal_census', ROOT / 'contract/tools/refusal_census.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['refusal_census'] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-python', required=True)
    arguments = parser.parse_args(argv)
    census = _census()
    routes = census._routes()
    probes = [
        probe
        for probe in census.probes(ROOT)
        if 'duckdb' in probe.get('databases', census.DATABASES)
    ]
    with tempfile.TemporaryDirectory(prefix='sdm-census-words-') as scratch:
        cases = Path(scratch)
        for number, probe in enumerate(probes):
            (cases / f'{number}.json').write_text(
                json.dumps(
                    {
                        'case': census._case(probe, number, routes),
                        'options': probe.get('options', {}),
                        'model': probe.get('model', 'KumoRelational'),
                    }
                ),
                encoding='utf-8',
            )
        out = cases / 'out.json'
        subprocess.run(
            [
                arguments.reference_python,
                '-I',
                str(ROOT / 'contract/tools/refusal_census.py'),
                'route',
                '--reference',
                str(arguments.reference),
                '--database',
                'duckdb',
                '--cases',
                str(cases),
                '--out',
                str(out),
            ],
            stdin=subprocess.DEVNULL,
            env={
                **{k: v for k, v in os.environ.items() if k != 'PYTHONPATH'},
                census.MESSAGES: '1',
            },
            timeout=3600,
            check=True,
        )
        outcomes = json.loads(out.read_text(encoding='utf-8'))
    commit = census._runner()._git(arguments.reference)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        ''.join(
            json.dumps(
                {
                    'site': probe['site'],
                    'result': outcome['result'],
                    'message': outcome.get('message'),
                    'reference': commit,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + '\n'
            for probe, outcome in zip(probes, outcomes, strict=True)
        ),
        encoding='utf-8',
    )
    print(f'census words: {len(probes)} probes from {commit}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
