# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.imports: after every scenario, ``sys.modules`` holds neither
``predictive_query`` nor ``sdm_relational_engine``.

The scenarios are every transcript and every public fixture case, run on the
core in fresh processes, the selector unset; a case whose input builds an
engine object itself (``$inference_config``) runs last and is not judged.
The replay harness's own engine adjustment is made only where the engine is
already loaded, so nothing but the packages can load it. Where a package was
imported, the report shows from where.
"""

from __future__ import annotations

from _switch import CORE, capture


def _loaded(record):
    return sorted(name for name, held in record['modules'].items() if held)


def test_no_transcript_leaves_a_ported_package_loaded():
    data = capture(CORE, 'transcripts')
    assert len(data['records']) >= 137
    bad = [
        f'{r["name"]}: {", ".join(_loaded(r))}'
        for r in data['records']
        if _loaded(r)
    ]
    where = '\n'.join(
        f'{name} first imported from:\n    ' + '\n    '.join(stack)
        for name, stack in sorted(data['first_imports'].items())
    )
    assert not bad, (
        f'{len(bad)} transcripts left a ported package loaded, the first '
        f'{bad[0]}\n{where}'
    )


def test_no_public_fixture_case_leaves_a_ported_package_loaded():
    data = capture(CORE, 'fixtures')
    judged = [r for r in data['records'] if not r['engine_input']]
    assert len(judged) >= 400, len(judged)
    bad = [f'{r["id"]}: {", ".join(_loaded(r))}' for r in judged if _loaded(r)]
    where = '\n'.join(
        f'{name} first imported from:\n    ' + '\n    '.join(stack)
        for name, stack in sorted(data['first_imports'].items())
    )
    assert not bad, (
        f'{len(bad)} fixture cases left a ported package loaded, the first '
        f'{bad[0]}\n{where}'
    )
