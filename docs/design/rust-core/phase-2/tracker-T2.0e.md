# Phase 2 tracker: T2.0e

The design branch owns `tracker.md`; this file holds T2.0e's section until it folds in.
The previous package, T2.0, landed at `292f8d8a` (`Land T2.0: phase-generic tooling and rust-core CI`), the base of this lane.

## T2.0e

T2.0e lets the gate accept a new reference the owner approved, gives the bridge freeze a reason for behaviour merged from `main`, and lets a merge of `main` land.
It is the gate side of T2.0c (`main` merged into the phase branch, the reference re-recorded at `6298913c`), split out of the design's T2.0b so it lands first.
It changes judging paths (`ratchet.py`, `bridge_freeze.py`), so it lands alone, with two reviewers.

### Judging-path files touched

`scripts/rust_core/ratchet.py`, `scripts/rust_core/bridge_freeze.py` and `scripts/rust_core/bridge_changes.json` (its `about` text only).
Also changed, outside the judging path: `land.py`, `lane.py`, `prepush.sh` (a comment), `land_selftest.py` (its scripted git answers `rev-list --merges`), `selftest.py` (three freeze cases).
`gate.sh` and `ci.yml` are unchanged.
The package does not rely on its own judging change for its gate: its candidate adds no reference file and declares no `main` reason, so the checkpoint's harness judges it as any package.

### What changed, by exit

1. **A new reference, only on the owner's recorded approval.**
   `ratchet.check_reference` hands a candidate that adds a file under `contract/reference/` to `check_new_reference`.
   The file `contract/reference/<NEW>.jsonl` passes only when a row of the checkpoint's decisions approves `NEW` (`approved_references`).
   Such a row's status is the owner's approval as `ratchet.owner_approved` reads it, and its Decision cell (the second) holds the word "reference" and exactly one full 40-hex SHA in backticks.
   Without such a row, the refusal stays by name: "a new reference commit replaces the recorded one: the owner's", now naming the file.
   With the row, each of these fails by name:
   - `NEW` is a recorded reference's name, or more than one file is added;
   - a reference file other than the new one is left (the old one must be removed);
   - `contract/reference-env.txt`'s one `reference:` line does not name `NEW`;
   - a file outside `docs/` and `contract/reference/` still cites the old SHA in full, or cited it at the checkpoint and does not cite `NEW` now (read with `git grep -F` on both commits, so every vector file the SHA consistency tests check is covered without a list to keep);
   - a line of the new file records another reference commit than `NEW` (`identity.reference.commit.sha`);
   - a case of the old recording is missing from the new one, unless its removal is approved.

   The per-case comparison does not apply across the change.
   Instead every case whose semantic projection (`ratchet.semantic`, the same one the per-case rule uses) differs from its recorded line is listed in the report's notes by case, for the reviewers, under a note naming the move and the approving row.
   A case recorded alike (identity only) is not listed, so the list is exactly the changed answers.
   After the landing the new file is the one the per-case rules hold, and the row admits no further file: another SHA needs its own row.
2. **Agents can't forge the approval.**
   Only the checkpoint's rows count (the orchestrator's decision): the rule reads `checkpoint_decisions`, every `phase-N/decisions.md` at the checkpoint through `capture_check.phase_decisions` (T2.0), as the relaxation rules do.
   A row the candidate adds never authorizes, so the owner's row (KO2-1, on the design branch) lands before T2.0c.
   The status is read by `owner_approved`, the reader `coverage.py` shares, so "proposed", "orchestrator, owner to confirm", a reviewer's approval, "not approved", a withdrawn approval, and the owner's words in the text but not the status approve nothing.
   An abbreviated SHA, another SHA, a row without the word, two SHAs in one row, and a row outside a `phase-*/decisions.md` approve nothing.
3. **The bridge freeze takes `main` as a reason.**
   `bridge_freeze.py` adds `{"main": "<the full SHA of a commit of main> <its title>"}`, documented in its docstring beside the other reasons and in `CHANGES_ABOUT` (and the `about` of `bridge_changes.json`).
   `main_judge` holds it, from git: 40 lower-case hex digits, a space and the commit's own subject; the commit exists, is an ancestor of `refs/remotes/origin/main` (never a local branch) and of HEAD, is not an ancestor of the merge base, and changes the entry's file against its first parent.
   A `main` reason may also declare an entry gone.
   A spent `main` declaration behaves like any spent one: held as it stands on the phase branch after the landing, and refused as the reason for a later change.
4. **A merge of `main` can land, from the lane.**
   `land.main_merge`, used by `land.commits_not_owned`, so by `lane.py check`, its pre-push form, `prepush.sh` and `land.py` alike:
   - with no merge commit in `base..head`, nothing changes;
   - with one, owned by the package (subject), with two parents, and its second parent an ancestor of `main` as the remote holds it at check time (`git ls-remote`, then fetched), the commits that merge brings in (from the second parent, not the first parent or the phase head) are exempt from the subject rule;
   - two or more merge commits, a merge of anything that is not the remote's `main` (another branch, a local `main`, a rewritten `main` commit, a remote-tracking ref moved by hand), or an octopus merge exempt nothing, and each such merge is named beside the commits without the ID.

   The route is the package's lane, the lightest: the lane already carries every ownership check, and `lane.py`'s and `land.py`'s docstrings say so.
   No topic branch, flag or new command was added.
5. **Post-landing state.**
   The reference rule passes the landed state judged against itself and judges the next package with no problem; the per-case rules hold the new file (the T2.0e tests' exit 5 cases).
   The freeze holds a landed `main` declaration as spent.
   A landing of a merge of `main` verifies, the lane is back at the phase head and passes `lane.py check`, and a Phase 1 landing still verifies.

### Probes

- Seeded faults on the reference rule's evidence, each refused by name: `reference-env.txt` deleted ("names no reference"), retyped (`reference: 12`), the new file empty (every case named as removed), a row with an empty Decision cell, a row naming two SHAs, a line recording another reference commit.
- KO2-1 as written on the design branch (`cc74a4c8`) is read by `approved_references` as the approval of `6298913c3b66da9459bb605d0127f8493adaa1fc`, and no other row there approves anything.
- `origin/main` today holds 20 commits past the phase branch, linear, none of them a merge or a `Land ` subject, so `ratchet.py --phase-push` (which counts `Land ` subjects in the pushed range) and CI's landing skip are unaffected by them.
- The route and outside-JSON probes don't apply: no product route changes and no new reader of outside JSON.

### Tests

- `tests/test_t20e_reference_change.py`: 29 passed (the 29th, the test author's later `6648a1e8`, cherry-picked unchanged as `750cce57`: a row the candidate writes itself admits nothing).
- `tests/test_t20e_main_merge.py`: 27 passed.
- `tests/test_t20e_bridge_main.py`: 14 passed, 1 still `PENDING` (see Deviations).
- `land_selftest.py` 194 of 194, `lane_selftest.py` 69 of 69, `selftest.py --no-clippy` with the three new freeze cases.

### Deviations

- **The row format.** The decisions header on the design branch states that a reference approval's Decision cell *begins* `Reference: ` and the SHA in backticks.
  The T2.0e tests' rows read "The reference moves to `<SHA>`: ...".
  To keep both, the rule reads the word "reference" anywhere in the Decision cell and exactly one full SHA in backticks; the header's form is a case of it, and KO2-1 passes.
  Requiring exactly one SHA keeps a row that names two (for example "not `<other>`") from approving either.
- **One exit 5 test stays `PENDING`.** `test_after_the_landing_the_declarations_are_held_and_spent` edits the frozen file with `.replace('    return value\n', '    return 1\n')`, which also matches the existing `_copied` in `_rust/predict.py` (`        return value`), so a second entry changes and the test's final declaration doesn't cover it.
  With the replacement anchored to the added function (`'\n    return value\n'`) the test passes against this implementation.
  It is left as a strict xfail for the test author to fix; its `PENDING` definition stays for that one use.
