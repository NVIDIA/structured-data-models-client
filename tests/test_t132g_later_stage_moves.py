# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32g part 2d, fix: which approved entries are a move past the
reference's stop, and the large buffers' setting as DuckDB applies it.

Written by the package's independent test author, after the subagent's
review of part 2d (P2-1, P3-1, P3-3), on ``tests/
test_t132g_later_stage.py``'s real fixtures, through the real
``ratchet.check --effective-out`` and ``compare.compare``.

The rule for ``--expect``'s ``@moved`` (the orchestrator's decision), as
these tests read it: a runner is listed for a case only where the
CHECKPOINT's reference line for the case is a refusal at a stage S and an
approved entry naming that case, runner and cell moves it, ``new`` an
outcome past S, from where the reference left it: ``old`` ``none`` (a
first admission, part 2a) or the reference's own outcome. An entry whose
``old`` was never the reference's records no move away from it (the
reviewer's probe). Where the case's reference line changes in this
change, only the entries this change adds or changes count. A move held
since in a baseline (the admission landed, the baseline in place of the
``none`` move) keeps the relaxation, under the same rule.

``large_buffers``: the last word on ``arrow_large_buffer_size`` before the
case's statements wins, as DuckDB applies it: a ``RESET`` or ``SET ... =
false`` after a ``SET ... = true`` turns it off, and a setting made in the
case's base (``bases.yaml``'s statements run before the case's setup)
counts.

Gating (``contract/tests/t132g_gate.py``): a test named in
``t132g-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written. It still runs and must still fail; once it passes it fails,
naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132g_gate', ROOT / 'contract/tests/t132g_gate.py')
later = _module(
    't132g_later_stage_tests', ROOT / 'tests/test_t132g_later_stage.py'
)
helpers = later.helpers
compare, ratchet, run_python = later.compare, later.ratchet, later.run_python

origin, repo = helpers.origin, helpers.repo
commit, land = helpers.commit, helpers.land
append_entry, outcome = helpers.append_entry, helpers.outcome
HUGEINT, LARGE, CORES = later.HUGEINT, later.LARGE, later.CORES
LARGE_SETUP = later.LARGE_SETUP
lines, case_files, of, has = later.lines, later.case_files, later.of, later.has
item, entry = later.item, later.entry
CELL = helpers.CELL

Edit = Callable[[Path], None]
WHERE = 'python-core against python-ref'
#: The reference's outcome for HUGEINT, as an entry states it.
REFERENCE_TEXT = 'refusal INTERNAL_ERROR at materialize (StructuredDataError)'


def _edits(*edits: Edit | None) -> Edit:
    def edit(r: Path) -> None:
        for each in edits:
            if each is not None:
                each(r)

    return edit


def _reference_file(r: Path) -> Path:
    found: Path = next((r / 'contract/reference').glob('*.jsonl'))
    return found


def set_reference(case: str, line: dict[str, Any]) -> Edit:
    r"""``case``'s reference line replaced by ``line``."""

    def edit(r: Path) -> None:
        path = _reference_file(r)
        kept = [
            text
            for text in path.read_text().splitlines()
            if json.loads(text)['case'] != case
        ]
        path.write_text('\n'.join([*kept, json.dumps(line)]) + '\n')

    return edit


def admitted(case: str) -> Edit:
    def edit(r: Path) -> None:
        [path] = (r / 'contract/cases').glob(f'*/{case}.json')
        spec = json.loads(path.read_text())
        spec['status'] = 'acceptance'
        path.write_text(json.dumps(spec, indent=2))

    return edit


def held(case: str, runner: str, line: dict[str, Any]) -> Edit:
    r"""``runner``'s baseline of ``case``: ``line``'s projection."""

    def edit(r: Path) -> None:
        policy = yaml.safe_load(
            (r / 'contract/compare-policy.yaml').read_text()
        )
        projected = ratchet.projection(
            line, [str(f) for f in policy['baseline']]
        )
        path = r / f'contract/baselines/{CELL}/{runner}.jsonl'
        by_case = {
            json.loads(text)['case']: json.loads(text)
            for text in path.read_text().splitlines()
        }
        by_case[case] = {
            'case': case,
            'runner': runner,
            'cell': CELL,
            'projection': projected,
            'digest': ratchet.digest(projected),
        }
        path.write_text(
            ''.join(
                json.dumps(by_case[c], sort_keys=True) + '\n'
                for c in sorted(by_case)
            )
        )

    return edit


def holding_the_core(case: str) -> Edit:
    r"""python-core's and rust-host's baselines of ``case`` held at their
    real lines' outcomes (success)."""
    found = lines()
    return _edits(*(held(case, who, found[who][case]) for who in CORES))


def with_entry(seed: dict[str, Any]) -> Edit:
    return lambda r: append_entry(r, seed)


def compared(
    r: Path,
    tmp: Path,
    at_checkpoint: Edit | None = None,
    in_change: Edit | None = None,
    reference: dict[str, Any] | None = None,
    change: Callable[[Any, dict[str, Any]], None] | None = None,
    bases: dict[str, list[str]] | None = None,
) -> Any:
    r"""The fixtures compared under the expectations ``ratchet.check``
    writes: the fixture cases and their reference lines landed, with
    ``at_checkpoint``, as the checkpoint; ``in_change`` the candidate.
    ``reference`` replaces HUGEINT's reference line compared (the
    candidate's, where ``in_change`` re-records it); ``bases`` adds bases
    to the bases file compared."""
    landed = land(r, 'T1.32f', _edits(later.with_fixtures, at_checkpoint))
    commit(r, 'T1.32g: the change', in_change)
    expect = tmp / 'expect.json'
    ratchet.check(r, landed, [], '', effective_out=expect)
    found, cases = lines(), case_files()
    if reference is not None:
        found['python-ref'][HUGEINT] = copy.deepcopy(reference)
    if change is not None:
        change(found, cases)
    bases_file = run_python.BASES
    if bases is not None:
        loaded = yaml.safe_load(run_python.BASES.read_text())
        loaded['bases'].update(bases)
        bases_file = tmp / 'bases.yaml'
        bases_file.write_text(yaml.safe_dump(loaded, sort_keys=False))
    directory = tmp / 'cases'
    directory.mkdir(exist_ok=True)
    digests = {}
    for name, spec in cases.items():
        path = directory / f'{name}.json'
        path.write_text(json.dumps(spec, indent=2))
        digests[name] = run_python.digest(path.read_bytes())
    bases_digest = run_python.digest(bases_file.read_bytes())
    files = {}
    for who, by_case in found.items():
        for name, line in by_case.items():
            line['identity']['inputs']['case'] = digests[name]
            line['identity']['inputs']['bases'] = bases_digest
        path = tmp / f'{who}.jsonl'
        path.write_text(''.join(json.dumps(x) + '\n' for x in by_case.values()))
        files[who] = path
    return compare.compare(
        [directory],
        files['python-ref'],
        [files[who] for who in CORES],
        bases_file=bases_file,
        expect_file=expect,
    )


def unrelaxed(found: list[tuple[str | None, str]]) -> bool:
    r"""python-core still held to the reference past its stop."""
    return (
        has(found, 'sql.count', WHERE, 'materialize')
        and has(found, 'payload', WHERE)
        and has(found, 'result', WHERE)
    )


#: The genuine first admission: from none, past the reference's stop.
GENUINE = entry(*(item(HUGEINT, who) for who in CORES))
#: The reviewer's probe: an entry whose old was never the reference's.
UNRELATED = entry(
    item(
        HUGEINT,
        'python-core',
    )
    | {'old': 'refusal INVALID_REQUEST at predict (ValueError)'},
    item(HUGEINT, 'rust-host') | {'old': 'refusal INVALID_REQUEST'},
)
#: The large-buffer case's own first admission past its reference's stop.
LARGE_MOVE = entry(*(item(LARGE, who) for who in CORES))
#: A move from the reference's own outcome, as an entry states it.
FROM_THE_REFERENCE = entry(
    item(HUGEINT, 'python-core') | {'old': REFERENCE_TEXT},
    item(HUGEINT, 'rust-host') | {'old': 'refusal INTERNAL_ERROR'},
)


# --- a genuine move, made now or held since ----------------------------------------


def test_a_genuine_first_admission_compares_clean(
    repo: Path, tmp_path: Path
) -> None:
    found = of(compared(repo, tmp_path, in_change=with_entry(GENUINE)), HUGEINT)
    assert found == [], found


def test_a_move_from_the_reference_s_own_outcome_compares_clean(
    repo: Path, tmp_path: Path
) -> None:
    r"""Held at the checkpoint, moved from the reference's outcome."""
    found = of(
        compared(
            repo,
            tmp_path,
            at_checkpoint=_edits(admitted(HUGEINT), holding_the_core(HUGEINT)),
            in_change=with_entry(FROM_THE_REFERENCE),
        ),
        HUGEINT,
    )
    assert found == [], found


def test_a_genuine_move_held_since_in_a_baseline_still_compares_clean(
    repo: Path, tmp_path: Path
) -> None:
    r"""P3-1: the admission landed; the baselines hold the move and the
    entry is the checkpoint's; the next change changes nothing of it."""
    found = of(
        compared(
            repo,
            tmp_path,
            at_checkpoint=_edits(
                admitted(HUGEINT),
                holding_the_core(HUGEINT),
                with_entry(GENUINE),
            ),
        ),
        HUGEINT,
    )
    assert found == [], found


# --- no move: still held to the reference -----------------------------------------


def test_the_reviewer_s_probe_relaxes_nothing(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    r"""P2-1: success baselines held and an approved entry whose ``old``
    was never the reference's: no move away from the reference."""
    found = of(
        compared(
            repo,
            tmp_path,
            at_checkpoint=_edits(admitted(HUGEINT), holding_the_core(HUGEINT)),
            in_change=with_entry(UNRELATED),
        ),
        HUGEINT,
    )

    def fails() -> None:
        assert unrelaxed(found), found

    G.held('exits', f'py:{request.node.name}', fails)


def test_the_probe_s_entry_held_since_relaxes_nothing(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    r"""P3-1's negative: the held-since route under the same rule."""
    found = of(
        compared(
            repo,
            tmp_path,
            at_checkpoint=_edits(
                admitted(HUGEINT),
                holding_the_core(HUGEINT),
                with_entry(UNRELATED),
            ),
        ),
        HUGEINT,
    )

    def fails() -> None:
        assert unrelaxed(found), found

    G.held('exits', f'py:{request.node.name}', fails)


def test_success_baselines_with_no_entry_relax_nothing(
    repo: Path, tmp_path: Path
) -> None:
    found = of(
        compared(
            repo,
            tmp_path,
            at_checkpoint=_edits(admitted(HUGEINT), holding_the_core(HUGEINT)),
        ),
        HUGEINT,
    )
    assert unrelaxed(found), found


def _success_reference() -> dict[str, Any]:
    r"""HUGEINT's reference line as if the reference had succeeded through
    predict: for the checkpoint's reference only (the ratchet reads its
    outcome)."""
    line = copy.deepcopy(lines()['python-ref'][HUGEINT])
    line['outcome'] = {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    }
    return line


def _refusal_reference(stage: str = 'materialize') -> dict[str, Any]:
    r"""HUGEINT's reference line recorded again as a named refusal (no
    entry may state INTERNAL_ERROR, so a re-recording entry can name this
    one)."""
    line = copy.deepcopy(lines()['python-ref'][HUGEINT])
    line['outcome'] = {
        'result': 'refusal',
        'stage': stage,
        'error': 'ValueError',
        'code': 'INVALID_REQUEST',
    }
    return line


def _expecting(line: dict[str, Any]) -> Callable[[Any, dict[str, Any]], None]:
    r"""HUGEINT's case file expecting what its re-recorded reference line
    says, as the change re-recording it would have it (the case's own
    expectation is held to the reference, apart from this rule)."""

    def change(found: Any, cases: dict[str, Any]) -> None:
        expect = cases[HUGEINT]['expect']
        expect.update(
            outcome=line['outcome']['result'],
            stage=line['outcome']['stage'],
            error=line['outcome']['error'],
            code=line['outcome']['code'],
        )

    return change


def _re_recorded(old: str, new: str) -> dict[str, Any]:
    return {
        'id': 'SEED-7',
        'kind': 'change',
        'cases': [HUGEINT],
        'approved_by': 'owner',
        'case_outcomes': [outcome(HUGEINT, 'python-ref', old, new)],
    }


#: Entries that only record agreement with a reference that succeeded
#: (as P1B-26 and P1B-72 do): none to the reference's own success.
AGREEMENT = entry(*(item(HUGEINT, who) for who in CORES), id='SEED-8')


def test_a_reference_re_recorded_as_a_refusal_under_agreement_entries_relaxes_nothing(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    r"""P2-1's scenario: at the checkpoint the reference succeeded, its
    agreement entries approved and the core's success baselines held; this
    change records the reference again as a refusal at materialize (an
    owner's entry naming it on python-ref). The core's success is the
    baselines', and nothing moved it past a refusal."""
    refusal = _refusal_reference()
    found = of(
        compared(
            repo,
            tmp_path,
            at_checkpoint=_edits(
                set_reference(HUGEINT, _success_reference()),
                admitted(HUGEINT),
                holding_the_core(HUGEINT),
                with_entry(AGREEMENT),
            ),
            in_change=_edits(
                set_reference(HUGEINT, refusal),
                with_entry(
                    _re_recorded(
                        'success at predict',
                        'refusal INVALID_REQUEST at materialize (ValueError)',
                    )
                ),
            ),
            reference=refusal,
            change=_expecting(refusal),
        ),
        HUGEINT,
    )

    def fails() -> None:
        assert unrelaxed(found), found

    G.held('exits', f'py:{request.node.name}', fails)


def test_a_reference_re_recorded_counts_only_this_change_s_entries(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    r"""The checkpoint's reference refused at materialize and a genuine
    move held since; this change records it again refusing at
    materialize with another code. The checkpoint's entries no longer
    count for the case: only this change's would."""
    refusal = _refusal_reference()
    found = of(
        compared(
            repo,
            tmp_path,
            at_checkpoint=_edits(
                admitted(HUGEINT),
                holding_the_core(HUGEINT),
                with_entry(GENUINE),
            ),
            in_change=_edits(
                set_reference(HUGEINT, refusal),
                with_entry(
                    _re_recorded(
                        REFERENCE_TEXT,
                        'refusal INVALID_REQUEST at materialize (ValueError)',
                    )
                ),
            ),
            reference=refusal,
            change=_expecting(refusal),
        ),
        HUGEINT,
    )

    def fails() -> None:
        assert unrelaxed(found), found

    G.held('exits', f'py:{request.node.name}', fails)


# --- the large buffers' setting as DuckDB applies it -------------------------------


def _catalog(found: list[tuple[str | None, str]]) -> list[str]:
    return later._catalog_mismatches(found)


def _setup(*steps: str) -> Callable[[Any, dict[str, Any]], None]:
    def change(found: Any, cases: dict[str, Any]) -> None:
        cases[LARGE]['input']['setup'] = [
            *cases[LARGE]['input']['setup'],
            *steps,
        ]

    return change


@pytest.mark.parametrize(
    'step',
    ['RESET arrow_large_buffer_size', 'SET arrow_large_buffer_size = false'],
    ids=['reset', 'set-false'],
)
def test_a_later_reset_or_false_turns_the_large_buffers_off(
    repo: Path, tmp_path: Path, step: str, request: pytest.FixtureRequest
) -> None:
    r"""The real lines read large types; the setting turned off after it
    was set means DuckDB writes small ones, so those reads must not be
    taken for the rows' types."""
    found = of(
        compared(
            repo,
            tmp_path,
            in_change=with_entry(LARGE_MOVE),
            change=_setup(step),
        ),
        LARGE,
    )

    def fails() -> None:
        assert _catalog(found), found

    G.held('exits', f'py:{request.node.name}', fails)


def test_a_setting_made_in_the_case_s_base_counts(
    repo: Path, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    r"""The setting moved from the case's setup into its base, which runs
    before it: the catalog reads still agree as large types."""
    shop = yaml.safe_load(run_python.BASES.read_text())['bases']['shop']

    def change(found: Any, cases: dict[str, Any]) -> None:
        spec = cases[LARGE]['input']
        spec['setup'] = [s for s in spec['setup'] if s != LARGE_SETUP]
        spec['base'] = 'shop_large_buffers'

    found = of(
        compared(
            repo,
            tmp_path,
            in_change=with_entry(LARGE_MOVE),
            change=change,
            bases={'shop_large_buffers': [LARGE_SETUP, *shop]},
        ),
        LARGE,
    )

    def agree() -> None:
        assert _catalog(found) == [], found

    G.held('exits', f'py:{request.node.name}', agree)
