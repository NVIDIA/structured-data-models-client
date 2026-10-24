# T3.9: parity and CI cleanup

## Production corrections

Postgres zoned text uses canonical UTC on Python and Java under the explicit owner decision T39-OWNER-2026-10-24-UTC-TEXT.
The public materialize path previously emitted different range feature strings for the same instants on UTC and Los Angeles sessions.
The conversion preserves microseconds, nulls, empty and open ranges, and readable wall dates whose UTC year is outside Python's datetime range.
Scalar timestamp columns and timezone-free values retain their existing conversion, and caller sessions are unchanged.

Postgres floating seeds are bound as exact numeric values, repairing six float/integer key-pair materialize refusals.
Whole floats use their actual integer digits rather than a shortest decimal that can name a nearby integer.
Fractional seeds are not truncated, and integer columns are not rounded to double precision for matching.
This changes the binding representation of already admitted values; it does not relax entity-key admission.
The dialect matrix is regenerated canonically, changing only four PostgreSQL answers and preserving all 2,972 inputs and every other dialect.

Metadata refresh retains the accepted tier of a held big-integer column.
Fresh nullability, uniqueness and catalog facts remain current, while a later fetch wider than the held tier produces the established VALUE_OUT_OF_RANGE error.
Ordinary numeric-width compatibility remains unchanged.
The existing public DuckDB regression reproduced the lost refusal before the fix.

## Dispatched comparison gates

The saved nightly's SQLite rust-host, SQLite Python-core and floor DuckDB jobs reported 50, 71 and 174 difference lines across 18, 20 and 25 cases respectively.
Every reported case and runner already has an exact approved outcome row in the register.
The common failure was that `ratchet.py where` returned no checkpoint for `workflow_dispatch`, so those jobs compared against the original reference without its approved deviations.
The floor job explicitly reported that it was running without the ratchet.
The original `where` CLI reproduced that empty result under the nightly event and phase-ref environment.

A dispatched phase head now names `--base HEAD`.
The existing checkpoint resolver reads that landing's recorded base and rejects an unlanded head as a self-comparison.
Dispatched topics retain the phase-remote base, and dispatched epic runs retain the merged phase's reviewed checkpoint.
Push behavior and unrelated events are unchanged.

| Saved difference | Existing exact approval |
|---|---|
| The twelve backtick, dot and line-break name cases, plus `rank.odd-column` | P2E-1 and KO2-36 |
| SQLite unsigned keys beyond int64 | P2B-12, explicitly reviewer-approved |
| SQLite time-of-day text feature | P2B-13 and KO2-61 |
| SQLite empty fact table and bad text time | P2B-10 and P2B-11 |
| DuckDB large text keys and fixed arrays | P1B-33 and P1B-9 |
| DuckDB empty, map, struct, union and interval errors | P1B-11, P1B-10, P1B-7 and P1B-8 |
| Seven-hop error subclass on both databases | P1B-20 |
| LIST_DISTINCT of a foreign key refused before the endpoint | P2E-19 and KO2-75 |
| DuckDB bad text dates reported as provider failures | DT-12 |

No new outcome approval, reference observation, case expectation or comparison exclusion was added for these failures.
The actual saved rust-host SQLite artifact was downloaded for inspection; the nightly did not publish the failed Python observation files.
This classification does not claim a successful rerun of those jobs.

## View decision reconciliation

P3J-15's stale proposed status is corrected to the two reviewers' existing explicit approval of finite safe diagnostics on a678ddf2c90507e234a637b7e4231e1eee6df2ec.
Both reviewers reaffirmed that exact scope in their final round-3 reports.
P3J-12 and P3J-14 remain proposed pending explicit review of equivalent expression placement and the precise view refresh behavior.
The owner's Plan B and output-safety policies do not supply blanket approval of new register entries.
Java P3S entries remain owned by the Java package until it lands; its existing P3S-2 reviewer approval and P3S-7 retirement must be preserved, with the other unresolved entries assessed on their actual evidence.

## Remaining Databricks measurements

The genuine T3.20f SQL captures and the validated reused Spark captures were inspected for the T3.20d review's four unregistered candidates.
Both `databricks.name.three-part` cases, including `-quoted`, refuse INVALID_REQUEST at from_source on both reference legs.
`databricks.key.bigint-string` refuses INVALID_REQUEST at materialize on both, and `databricks.key.bigint-string-linked` refuses the same code at from_source on both.
All four agree with their generic case expectations; their recorded reference has no core result.
No core improvement is claimed or approved from these reference-only observations.
E-1 must measure each candidate core leg against its own actual reference and register any real improvement before accepting it.

The old-client leg differences also persist in those genuine captures.
`key.declared-int32` refuses INVALID_REQUEST at materialize on SQL and succeeds at predict on Spark.
Both `sample.anchor-entity.null-time` cases, including `.prediction-time`, succeed at predict on SQL and refuse INVALID_REQUEST at materialize on Spark.
The SQL references record actual ArrowInvalid catalog-read failures that the Spark references do not; these observations alone do not establish the cause of each outcome difference or attribute it to sampling randomness.
Under the owner's answer 9, each leg's recorded outcome remains its own baseline, without copying or normalizing these outcomes into the case files.
E-1 must check the core on all three, including the approved missing-anchor policy, before proposing any measured difference.

## Validation state

Regression tests precede each production change.
Only narrowly scoped suspected-bug reproductions and the canonical matrix writer ran during implementation.
Both independent approvals of the final integrated SHA are required before lint and package checks.
Full CI remains the phase-end check.
