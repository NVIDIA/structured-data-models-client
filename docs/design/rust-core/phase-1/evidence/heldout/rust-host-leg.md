# rust-host leg, completed under owner item 49

Owner item 49 (owner-decisions-2026-10-14.md, 2026-10-15): "Complete the rust-host leg of the one-shot held-out run with the opt-in --pre-given-specs conversion (branch t136-ck), on the same sealed cases. Only the leg that never ran is rerun; case answers are neither changed nor inspected to make it pass."

- In the one-shot run (started 2026-10-15T14:02:54Z), rust-host stopped with a runner failure on h11.key.composite-primary-key: `core.describe_source.source is not a SourceSpec`. The bundle was written in contract 22's SourceSpec shape (before T1.32h), which the 0a8a7412 test host refuses by design. It recorded no observations.
- Binary: `sdm-contract-run` built from branch t136-ck at `88ccc26537e4a865bcbe39e5bcdc2d23ca78fbd1` (2 commits on 0a8a7412: dbe33196 tests, 88ccc265 fix 19). Built with `cargo build --locked -p sdm-testkit --bin sdm-contract-run` (debug profile, as the run's `cargo run`). Binary SHA-256 `2f130646b6641227d31f31f3bcf415e80525053253e3cb12a8ad96d258565607`.
- Only sdm-testkit differs: every other `rust/crates/*` tree, `rust/Cargo.toml` and `rust/Cargo.lock` are git-tree-identical to 0a8a7412's. sdm-testkit tree 9efe747a at 0a8a7412, 7eca871b at 88ccc265. The diff touches only `rust/crates/sdm-testkit/{src/bin/sdm-contract-run.rs, src/lib.rs, src/pre_given.rs, tests/t136_pre_given_specs.rs}` and `contract/tests/t136ck-pending.json`.
- Command (cwd `<worktree>/rust`, so each observation's identity.commit is 88ccc265 and clean), started 2026-10-15T15:15:19Z:
  `sdm-contract-run --pre-given-specs --out heldout-run-0a8a741/rust-host.jsonl heldout-run-0a8a741/cases`. It recorded 42 cases, exit 0.
- Not rerun: python-ref, python-core and python-old (their files are the one-shot run's, unchanged). No case file was modified: `cases/` is the regenerated set from the one-shot run, and INPUTS_SHA256 is unchanged.
- The conversion (pre_given.rs) reads each case's `core.describe_source.source` as contract 22's SourceSpec and wraps each primary-key list and time-column value as `{"value": ...}`; nothing else changes.
- compare.py is unchanged (0a8a7412's). Its `inputs` rule compares each core runner's recorded calls against the case's own (contract-22) spec. inputs-format-only.txt checks each flagged pair call by call:
  - 3 pairs are format-only: they match only after the same rewrite.
  - 22 differ only in calls the run never reached, because it refused earlier. The outcome rule judges those.
  - 3 python-core pairs have a materialize call that differs. That is python-core's own, already in the one-shot compare (compare-partial.txt), and has nothing to do with item 49.
- compare-partial.txt and report-partial.md are kept from the partial step; its differences file is not in the pack.
