# Phase 2 tracker: T2.0g

The design branch owns `tracker.md`; this file holds T2.0g's section until it folds in.
The previous package, T2.0a, landed at `5e972867` (`Land T2.0a: the Phase 2 design, decisions and locked set`), the base of this lane.

## T2.0g

T2.0g lets the bridge freeze accept a covered function that `main` deleted, so T2.0c's merge of `main` passes the checkpoint's copy of `bridge_freeze.py`.
Item 2 of the brief (exact decimals) was dropped by the owner on 2026-10-17; T2.0c fixes it instead.

### Judging-path files touched

`scripts/rust_core/bridge_freeze.py` only.
The package does not rely on its own change for its gate: it declares no `main` reason and deletes no covered function.

### What changed

- `exactness` no longer reports a name `COVERED` holds that the file does not define when that key is declared `main` in `bridge_changes.json`.
  Any other missing covered name still fails by name.
- The ratchet judges that declaration as for any gone entry (`main_judge`).
  Without a merge base holding the baseline, or with the entry not gone against it and the declaration not spent, the check still fails.
- `main_judge` now also requires, for an entry that is gone, that the named commit deleted it, as the module docstring already said ("an entry gone, gone there too").

### Tests and probes

- `tests/test_t20g_bridge_main_deleted.py`: on a temporary repository with T2.0c's history shape, a covered function `main` deleted fails undeclared, by name, and passes declared `main` naming the deleting commit.
  Without the fix the passing half fails (`COVERED names it`); with it the test passes, and `tests/test_t20e_bridge_main.py` still passes (21 passed together).
- T2.0c's candidate (`t2.0c-linear`, `32e24d96`), in a scratch worktree, against `origin/rust-core-phase-2` at `5e972867`: the checkpoint's copy fails only on `_check_entity_count` ("COVERED names it"); this copy prints `freeze: ok`.
  Judged against itself (the post-landing state), both this copy and the candidate's own print `freeze: ok`.
- A one-off probe of the deleting-commit rule: `main` changes the function in one commit and deletes it in the next; the declaration naming the changing commit is refused ("does not delete the entry"), the one naming the deleting commit passes.
