# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Write every schema and answer ``accept_hashseed.py`` compares, as JSON,
to the file named on the command line. Run once per ``PYTHONHASHSEED``.

The Rust core's outcomes: every corpus schema, every ``schema.from_data``
fixture, every describer shape beside a key, and the first generated
semantic views, metric views and date columns. Today's Python's for the
corpus and the fixtures, which the comparisons stand on.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _dates  # noqa: E402
import _schema  # noqa: E402
import _views  # noqa: E402
import _warehouse  # noqa: E402

SEMANTIC = 300
METRIC = 2000
DATES = 2000


def main(out: Path) -> None:
    warnings.simplefilter('ignore')
    written: dict[str, object] = {}
    for path in _schema.transcript_paths():
        transcript, calls = _schema.scenario_calls(path)
        for i, call in enumerate(calls):
            name = f'corpus:{transcript["name"]}#{i}'
            if call.kind == 'from_data':
                rust = _schema.rust_scenario_from_data(call)
            else:
                rust, _ = _schema.rust_scenario_from_source(
                    transcript, path, call
                )
            written[name] = {'python': call.python, 'rust': rust}
    for fixture in ('time-and-keys', 'relationships'):
        for case in _schema.fixture_cases(fixture):
            written[f'fixture:{case["id"]}'] = {
                'python': _schema.python_fixture_from_data(case['input']),
                'rust': _schema.rust_fixture_from_data(case['input']),
            }
    import accept_describer

    for name, values in accept_describer.ALL_SHAPES.items():
        written[f'describer:{name}'] = _schema.rust_from_data(
            accept_describer._frames(values, True)
        )
    for i, (text, tables, trust) in enumerate(_views.semantic_views(SEMANTIC)):
        warehouse = _warehouse.Warehouse({'TEST.PARITY_SHOP.SV': text}, tables)
        model = _warehouse.source(warehouse).semantic_model(
            'SV', trust_expressions=trust
        )
        written[f'semantic:{i}'] = _schema.rust_from_source(model, {})
    module = _schema.native('metric_view_parse', 'dates_op')
    for i, (definition, trust) in enumerate(_views.metric_views(METRIC)):
        written[f'metric:{i}'] = module.metric_view_parse(definition, trust)
    for i, generated in enumerate(_dates.columns(DATES)):
        for operation, given in _dates.inputs(generated):
            written[f'dates:{i}:{operation}'] = _schema.rust_dates_op(
                operation, given
            )
    out.write_text(json.dumps(written, default=str))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
