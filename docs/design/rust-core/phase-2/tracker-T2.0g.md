# Phase 2 tracker: T2.0g

The design branch owns `tracker.md`; this file holds T2.0g's section until it folds in.
The previous package, T2.0a, landed at `5e972867` (`Land T2.0a: the Phase 2 design, decisions and locked set`), the base of this lane.

## T2.0g

T2.0g makes three narrow gate fixes T2.0c's merge of `main` needs, each judged by the checkpoint's copy:

1. The bridge freeze accepts a covered function that `main` deleted.
2. `compare.py` leaves out the fields where the core is more exact than the reference, as an owner-approved register entry lists them (the owner's "A ok", 2026-10-17; a KO2 row for the next owner-row docs landing).
3. `ratchet.py`'s `moves()` counts a case new in a change as moved past the reference's refusal when an entry of that change moves it from `none`.

### Judging-path files touched

`scripts/rust_core/bridge_freeze.py`, `scripts/rust_core/ratchet.py`, `scripts/rust_core/compare.py`, `scripts/rust_core/gate.sh` (passes the candidate's register to `compare.py`) and `docs/design/rust-core/register.schema.json` (the `more_exact` field).
The package does not rely on its own change for its gate: it declares no `main` reason, deletes no covered function and lists no `more_exact` field.

### What changed: the bridge freeze

- `exactness` no longer reports a name `COVERED` holds that the file does not define when that key is declared `main` in `bridge_changes.json`.
  Any other missing covered name still fails by name.
- The ratchet judges that declaration as for any gone entry (`main_judge`).
  Without a merge base holding the baseline, or with the entry not gone against it and the declaration not spent, the check still fails.
- `main_judge` now also requires, for an entry that is gone, that the named commit deleted it, as the module docstring already said ("an entry gone, gone there too").

### What changed: more exact than the reference

- A register entry may list `more_exact` items: a case, the core runners (`python-core`, `rust-host`) and the fields, each a path as `compare.py` prints a difference, `*` for any one key or index, under `public.schema`, `public.result`, `public.payloads` or `core.predicted.body`, at least one level down.
- `compare.py` reads them only from entries `status: approved`, `approved_by: owner` whose `decision_ref` names a row of the harness's own (the checkpoint's) decisions approved by the owner, as a policy relaxation is read (fix 1); P2B-1 cites KO2-35, at the checkpoint.
- For that case and runner the `schema`, `result` and `payload` rules replace the value at each path by one marker on both sides, only where both sides hold one value there (a string, a number or a boolean); a dict, a list or a null there, a field on one side only, another length, or any other field still differs (fix 1).
- It reads the candidate's register (`--register`, which `gate.sh` gives); HOST-5 is still read from the harness's own register.
- `register.py validate` holds each `more_exact` item's case to one of its entry's `cases` (fix 1).
- For T2.0c: P2B-1 lists `decimal.feature-inexact-38-18`, `decimal.sum-10-2` and `key.decimal-38-10` with the paths its gate run prints.

### What changed: a new case moved past the reference's stop

`moves()` skipped every case with no line at the checkpoint's reference, so a new case whose core answer an approved entry moves from `none` past the new reference line's refusal was never moved, and `compare.py` held its payload, result and statement counts to that refusal.
It now takes a new case's new line as the starting point, as for a line this change re-records: only entries this change adds or changes count.
A case with no line now still moves nothing.

### Tests and probes

- `tests/test_t20g_bridge_main_deleted.py`: on a temporary repository with T2.0c's history shape, a covered function `main` deleted fails undeclared, by name, and passes declared `main` naming the deleting commit.
  Without the fix the passing half fails (`COVERED names it`); with it the test passes, and `tests/test_t20e_bridge_main.py` still passes (21 passed together).
- T2.0c's candidate (`t2.0c-linear`, `32e24d96`), in a scratch worktree, against `origin/rust-core-phase-2` at `5e972867`: the checkpoint's copy fails only on `_check_entity_count` ("COVERED names it"); this copy prints `freeze: ok`.
  Judged against itself (the post-landing state), both this copy and the candidate's own print `freeze: ok`.
- A one-off probe of the deleting-commit rule: `main` changes the function in one commit and deletes it in the next; the declaration naming the changing commit is refused ("does not delete the entry"), the one naming the deleting commit passes.
- `contract/tests/test_t20g_more_exact.py`: `base.regression`'s reference observed as `python-core` but for a payload target and the result digest; proposed, both fail; approved by the owner and naming both, neither fails; a third, unnamed field (`metadata.source_query`) still fails `payload`.
  Fix 1 adds: the same entry with no owner-approved decision row excuses nothing, and `public.payloads/*` (whole payloads) excuses nothing; on the round-1 code both pass wrongly.
  Without the change the test fails (`compare()` takes no register of improvements).
- `tests/test_t20g_new_case_moves.py`: a new case refused at materialize by its new reference line, moved to success at predict by an entry of this change, is moved; by an entry of an earlier change, it is not.
  Without the fix the first half fails.
- On T2.0c's candidate (`f36a8e9f`, checkpoint `5e972867`), `write_effective`'s `@moved` names `python-core` and `rust-host` for `decimal.sum-past-38` and `decimal.class-target-collide`; without the fix it names neither.

### Review rounds

- Round 1 (Codex approve; Claude two P2s, two P3s): fix 1 ties a `more_exact` entry to an owner-approved row of the checkpoint's decisions and excuses one value only, never a dict, a list or a null; P3-1 taken (the item's case one of the entry's cases); P3-2, the brief's scope, is the orchestrator's (the owner's answers of 2026-10-17 re-admit the rule and add the `moves()` fix).
