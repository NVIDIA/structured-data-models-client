# Phase 2 tracker: T2.0f

The design branch owns `tracker.md`; this file holds T2.0f's section until it folds in.

## T2.0f

T2.0f teaches the gate Phase 2's locked set, adds the `main` step that holds locked check 7, and teaches the harness the `<case>@sqlite` and `<case>@postgres` keys.
It changes judging paths (`gate.sh`, `compare.py`, `ratchet.py`, the observation schema), so it lands alone, with two reviewers, after T2.0e and before T2.0a.
Base: T2.0's landing 292f8d8a on `rust-core-phase-2`.
Design: the T2.0f row, architecture choice 2 and "Locked acceptance checks" on `origin/rust-core-phase-2-design` at dce97ec9.

### What changed, by exit

1. **The reader accepts Phase 2's steps and keys.**
   `gate.sh`'s locked-set reader (the Python inside `gate.sh`) adds the steps `census`, `coverage-s1`, `l3`, `l7`, `ci` and `main` to a half's own steps, and the keys `limits` and `ci` to a half's keys.
   Each new key goes with the step that reads it, as `inventory` already does: `limits` with `l7`, `ci` with `ci`.
   A half with only one of the pair is a usage error.
   `limits` must be a mapping holding exactly KO2-29's eight limits:
   - `statements` must be `equal`.
   - The four ratios and `boundary-overhead-fraction` must be positive numbers (booleans, NaN and infinity are refused).
   - `boundary-overhead-floor-ms` must be a non-negative number.
   - `native-size-baseline` must be a quoted commit SHA of 7 to 40 hex digits.
   `ci` must map each check that holds the `ci` step to a non-empty list of distinct job-name patterns.
   Each pattern must be a printable, non-empty string with no leading or trailing space.
   A check that holds the `ci` step must have patterns, and only such checks may have them.
   Every refusal is a usage error (exit 2), raised before any step runs, and it names the faulty step, key, limit or check.
   Lists of unknown names are capped at ten, and a value that is not a scalar is shown by its type, never by its repr.
   `census`, `coverage-s1`, `l3`, `l7` and `ci` each run as a step that fails by name and says which package's tool it is waiting for (`not_landed` in `gate.sh`).
   The package that lands each tool wires it in at that line.
   Phase 1's halves 1a and 1b read and run exactly as before, since none of the new steps or keys appears in them (`contract/tests/test_t20_locked_set.py`, `contract/tests/test_gate_modes.py`).
2. **The `main` step (locked check 7).**
   The step runs only for a half whose `steps` list it.
   It requires the fetched `refs/remotes/origin/main` (never a local `main`) to be an ancestor of the candidate.
   It also requires it to be an ancestor of the reference commit that the candidate's `contract/reference-env.txt` names on its `reference: <sha>` line.
   Each of these fails the step by name:
   - no fetched `origin/main`;
   - a `reference-env.txt` with no single reference line;
   - a reference commit missing from the clone;
   - `main` not merged into the candidate;
   - `main` merged but the reference not recorded again at its head.
   When both ancestry conditions fail, both are reported.
3. **The `<case>@<database>` keys.**
   - The observation schema's `case` field now uses a new `$defs/key`: `^[a-z0-9][a-z0-9._-]*(@(sqlite|postgres))?$`.
     A case file's own ID stays `case.schema.json`'s plain ID.
   - In `compare.py`'s known-case check, the part of a key before `@` must be a case.
     A malformed key is already refused by the schema check, which names the file and line.
     A database line is accepted with a note: it is not compared until T2.1a brings the comparator for SQLite and Postgres.
   - `ratchet.py`: `case_of(key)` gives a key's case.
     The removed-case rule (`check_reference`) and the baseline rules (`check_baselines`, which skips a gone case's held lines) read the case part, so an approved removal of a case also covers its database lines.
     A new tree rule, `check_keys`, requires every reference line and baseline in the candidate to have a key of the schema's form (read from this harness's own `observation.schema.json`) whose case is in the candidate's corpus.
     So a removal that leaves a line or baseline on any database fails, naming its key.
     The acceptance-case baseline rule now runs over the keys observed: each runner observed on a database (that is, holding a `<case>@<database>` observation) needs a proposed baseline under that key.
     A reference line on a database where the core has no observation needs no baseline.
     `check_availability` looks a database line's case up by its case part.
     `ratchet.py accept` writes a baseline under each key observed of a corpus case, so its output passes `check` (fix 1).
   - A register entry approves a database line by naming its key: the register schema's `case_outcome.case` takes the observation schema's key form, and `register.py`'s corpus check reads the case part (fix 1).
     A case outcome names one line, so approving a Postgres move leaves the DuckDB and SQLite expectations as they were.
     The entry's `cases` list still names bare cases.

### Tests and probes

- Acceptance tests (the test author's): `tests/test_t20f_case_keys.py` and `contract/tests/test_t20f_gate.py`.
  The gate tests run a real `gate.sh`, which needs the built native core, so they run in the core's `contract-python` job.
  All 38 pass, with the `PENDING` marker and its uses removed in the commit that implements them.
- The implementer's seeded faults:
  - `contract/tests/test_t20f_main_seeds.py` (beside the gate tests, for the same reason): a `reference-env.txt` with no reference line, and a reference commit missing from the clone, each fail the `main` step and check 7 by name.
  - `tests/test_t20f_key_seeds.py`: a baseline keyed `@oracle`, a baseline of no case, and a reference line whose key is an integer each fail `check_keys` by name, while the twin passes.
- Outside-input probe on the reader: in Phase 2's file, a limit and a CI pattern were each nested 5,000 and 100,000 levels deep.
  Each is refused by name ("not a locked set it reads") with no traceback.
  A `limits` mapping with 200,000 unknown keys is refused in one line naming the first ten.
  A check with 200,000 CI patterns is read.
- Route differential: not applicable, since no route changes.

### Fix rounds

- Fix 1 (round 1, both reviewers):
  - The `main` step now requires exactly one `reference:` field naming a 40-hex SHA. Before, a malformed second field was ignored.
  - The reader's shadowed `listed` loop variable is renamed, so a check that holds `ci` with no patterns is refused by name instead of with a traceback.
  - `accept` writes database baselines.
  - A register entry can name a database line's key.
  - `check_reference` guards a key that is not a string.
  - The test author's round 1 seeds (4d8a4a42, 09508907, cherry-picked unchanged) cover each of these. They are: an ambiguous reference field; a `ci` check with no patterns; the `accept`-then-`check` round trip; the ratchet's suggested entry approving an `@postgres` baseline move; and an entry in the same form approving a changed `@postgres` reference line.
  - The test author's re-recording test stays gated until the rebase onto T2.0e's landing.
    There, T2.0e's `check_new_reference` also reads the case part of a removed key.

### Deviations and notes

- `tests/test_contract_reference.py` still requires exactly one reference line per case.
  T2.1a, which records the first database lines, has to teach it the keys.
  That test is outside this package's row.
- Not done here (round 1 P3s):
  - `compare.py` comparing database lines stays T2.1a's job. The note it prints names T2.1a.
  - Fetching `origin/main` right before the locked run belongs to T2.16's runbook. The step prints the tip it judged.
  - A removal's approval states the bare line's outcome. Its database lines are removed with it under the same entry, as the acceptance tests set out.
- Judging-path files touched: `scripts/rust_core/gate.sh`, `scripts/rust_core/compare.py`, `scripts/rust_core/ratchet.py`, `scripts/rust_core/register.py`, `contract/observation.schema.json` and `docs/design/rust-core/register.schema.json`.
