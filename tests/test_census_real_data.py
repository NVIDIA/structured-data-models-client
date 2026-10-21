# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The guard under the real-data counts of the two rules kept until Phase 4
(KO2-22; T2.16a, check 2's ``docs/design/rust-core/phase-2/evidence/census/
real-data.md``).

Both counts are zero by construction, so they hold only while the code
says so:

- ``text_time_rules`` fires only when a provider declares
  ``TextTimes::ClientRules`` (``sdm-core/src/materialize.rs``); no provider
  in the tree declares it, every SQL provider declaring ``ValueByValue``;
- ``model`` fires only for a ``KumoTabular`` request
  (``sdm-request/src/predict.rs``); the real-data benchmark's six tasks ask
  every question of ``KumoRelational``.

A provider declaring the client rules, or a benchmark asking the tabular
model, fails here, and the counts are to be measured again.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: The places ``ClientRules`` may be named: its definition and the core's
#: two refusals of it.
ALLOWED = {
    'rust/crates/sdm-contract/src/data.rs': 1,
    'rust/crates/sdm-core/src/materialize.rs': 2,
}
SQL_PROVIDERS = ('duckdb', 'sqlite', 'postgres')


def _sources() -> list[Path]:
    found = [
        *ROOT.glob('rust/crates/*/src/**/*.rs'),
        *ROOT.glob('rust/bindings/*/src/**/*.rs'),
        *ROOT.glob('packages/*/src/**/*.py'),
    ]
    assert found, 'no source read'
    return found


def test_no_provider_reads_text_times_by_the_client_rules():
    named = {}
    for path in _sources():
        text = path.read_text(encoding='utf-8')
        count = len(re.findall(r'\bClientRules\b|["\']client_rules["\']', text))
        if count:
            named[path.relative_to(ROOT).as_posix()] = count
    assert named == ALLOWED
    for name in SQL_PROVIDERS:
        source = (ROOT / f'rust/crates/sdm-data/src/{name}.rs').read_text()
        assert 'text_times: Some(TextTimes::ValueByValue)' in source, name


def test_the_real_data_benchmark_asks_no_tabular_model():
    source = (ROOT / 'scripts/live/real_data_benchmark.py').read_text()
    assert 'KumoTabular' not in source
    assert 'model=KumoRelational()' in source
    predict = (ROOT / 'rust/crates/sdm-request/src/predict.rs').read_text()
    assert 'return Err(not_yet("model"));' in predict
