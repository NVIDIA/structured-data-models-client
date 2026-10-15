# CI, seeded faults, reviews and what was not done (item 9)

Hand-written summary; the reviewers' reports are the orchestrator's records,
summarised here so the pack does not depend on them.

## CI and the locked gates

- CI run 38739174702 built the native wheel whose digest identity.md records
  (on 84ba45b3, the same tree as 0a8a7412).
- CI run 38746039117 (0a8a7412) green; then `final_gate.sh --execute` on
  0a8a7412: locked 1a (checks 1, 2, 3 and 6, wheel-smoke on CI's wheels) and
  1b (checks 4, 5 and 6) both passed ([gate/](gate/)).
- The final code, 9779fcbc (landed as 852e1e26, an identical tree): CI run
  38791629299 green; locked 1a (checks 1, 2, 3 and 6, the seeded suite and
  wheel-smoke on that run's wheels) and 1b (checks 4, 5 and 6) both passed,
  judged by bbb6c67f's harness ([gate/final-9779fcb/](gate/final-9779fcb/),
  NOTES.md item 9); decisions.md's go record, E154.

## Seeded faults

The seeded suite runs inside locked 1a (`seeded` step, check 2): ok at
0a8a7412 ([gate/final-gate-1a-1b-summary.txt](gate/final-gate-1a-1b-summary.txt))
and at 9779fcbc ([gate/final-9779fcb/gate-1a-summary.txt](gate/final-9779fcb/gate-1a-summary.txt)).

## The final whole-phase reviews

| round | SHA | Codex | independent subagent |
|---|---|---|---|
| 1 | c49ee4f8 | changes required: one P2, HOST-2's approval too broad | changes required: three P2s (refusals of inputs the reference accepts; a float entity's text; evidence.py's cell rule) |
| 2 | 84d68cf5 | changes required: one P2, P1B-81(b) ignored the zone | approve |
| re-check | 4a2aee5b, then b6ac5b76 | approve, with APPROVE-P1B-81 | approve, with APPROVE-P1B-81 |

Each finding, its fix and test commits and its status are in as-built.md's
T1.36 section. Every later package (T1.36's second landing, the Snowflake
connector 5.0 fix, the held-out leg's tool, the owner's answers, forecasts in
the core) was approved by both reviewers before it landed. The held-out
classifications are in [heldout/h11-classification.md](heldout/h11-classification.md).

## Process exceptions, accepted

- Two landings missed protected paths in their `Protected-paths` trailers
  (da4ab7f0, 8af842b2); four mechanical implementer edits to test-author files
  (tracker.md, T1.36 notes).
- The stop rule (H20): T1.33 part 3's big-integer seed join failed review the
  same way three rounds running; reported, and approved at round 5
  (tracker.md's stop-rule counters).
- No phase landing has a local `gate.sh` file for its own SHA; CI gated every
  one (the owner's item 36(b)).

## Not done

- aarch64 wheels (CI builds x86_64 only).
- DBR 15.4 LTS and serverless 1 and 2; only 14.3 LTS was run (12 of 13
  checks; `import mlflow` does not apply to the standard runtime).
- The archive column (3887762 has no contract runner): not measured (C9).
- `top_k` with a forecast on rust-host: untested, a case cannot carry it.
- Size trends past checkpoint 4, and the wheel's size: not measured here.
- Phase 2's named rows: migration-plan.md section 8.
