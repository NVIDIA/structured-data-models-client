# Rust core migration: what we learned

Two AI agent orchestrators worked to move this client's logic into a shared Rust core, over about 132 elapsed hours in total.
None of the Rust core work merged to `main`.
This page sums up two independent postmortems of that attempt, one by each orchestrator, and where they agreed.
The new plan is in [migration-plan.md](migration-plan.md).

## The short version

- The Rust code is real and much of it is good: PQL, SQL per database, the random number streams, the neighbor sampler and labels all matched Python on very large input sets.
- The core became a copy of pandas behind a Python-shaped interface, so most late bugs came from unusual combinations of column types, missing values, decimals and key roles.
- The tests mostly checked that Rust matched Python's internals, and their public scenarios didn't cover enough input combinations, so they passed while real regressions existed.
- The process could add safeguards and phases but could never cut them, and the agents produced process faster than product.

## By the numbers

| What | Number |
|---|---|
| Python replaced (excluding generated parser code and the C++ helper) | 26,914 lines |
| Rust written (product code) | 71,112 lines, 2.6 times the Python |
| Rust that re-creates pandas, numpy or CPython behavior by hand | about 27-29% |
| Rust fix commits classified as being about that hand-written behavior | 55% (65 of 118) |
| Migration test code | 60,741 lines, plus 193 MB of recordings |
| Public-API regressions found by a 35-case probe after "zero differences" | 9 |
| Commits to the migration's tracking repository by the second orchestrator in 63 hours | 1,378 |
| Phases planned / phases by the end | 12 / 24 |

Counts come from the postmortem audits of the last candidate commit; the percentages are classification estimates (about ±10%), and the 27-29% range reflects two counting methods.

## What went wrong

### 1. The core imitated pandas instead of taking typed data

The original proposal said the core does no I/O and never calls into Python.
In practice it parsed pandas dtype names, called back into Python for truthiness, `repr`, hashing and sorting, and reported errors as Python class paths.
It grew six separate pandas column models, two `to_datetime` implementations and five `json.dumps` writers.
**Lesson:** the core takes typed Arrow data and returns typed results and errors.
Each language turns its own data types into that contract, using real pandas in Python and JDBC types in Java.

### 2. "No API change" became "bit-for-bit answers", which meant copying Python's accidents

Matching Python bit for bit meant copying behavior that was never designed, such as where CPython's sort leaves NaN scores, or a size check that counted different bytes than were sent.
Python itself gave different answers on pandas 2 and 3, on different CPython versions, and with different hash seeds.
**Lesson:** write down the behavior we promise.
Keep a short register of Python quirks we keep on purpose, and drop the rest by explicit decision.

### 3. Bounded emulation worked; open-ended emulation did not

Emulating precisely defined functions held up well, with few late defects, because they could be checked against Python on hundreds of thousands of inputs.
Examples: numpy's random generator, float formatting, `DateOffset` arithmetic.
Emulating pandas' type system was open-ended, and the overlap of pandas dtypes, Arrow, Decimal and key roles produced most late regressions.
**Lesson:** port well-defined algorithms; never port a library's whole type system.

### 4. We rewrote libraries instead of using them

The core had only 16 direct external dependencies: of the Arrow compute crates it used only `arrow-select`, and it used no time zone, YAML or big-integer library (`chrono` and `num-bigint` came in through Arrow but were never used).
The migration records show no evaluation of Polars or DataFusion.
They would not have matched pandas' answers; whether they would have removed much hand-written code once bit-for-bit matching was dropped was never measured.
**Lesson:** for each dependency, decide "use a library", "port a defined algorithm" or "keep in the host", and record the measured reason.

### 5. Modules were ported in isolation and met a real host at the end

Each crate passed its own checks against Python internals, and the pieces first ran together behind the public API in the switch phase.
Problems with the contract, such as typed query results, error kinds and missing values, then surfaced one by one, late and in sequence.
**Lesson:** migrate by user journey: switch one public path end to end first, then widen it.
Run a second language early, so Python assumptions show up while they are cheap to fix.

### 6. Tests measured "same internals", not "same answers"

Recordings were keyed by private Python function names, and coverage was measured on private branches.
They missed common inputs such as pyarrow-backed pandas frames, and some renames and rebases meant re-recording and re-approving hundreds of files.
Most real bugs were found by reviewers, ad hoc probes and a late public-API generator.
**Lesson:** the main test layer is a language-neutral corpus of public-API cases, plus a generator that compares against the reference branch (`main` plus determinism fixes).
Algorithm unit tests stay; migration-specific tests of internals are retired once their replacements are shown to catch the same failures.

### 7. Fixes did not protect neighboring cases

A fix that refused unsupported DuckDB nested types also refused 24 cases that used to work, while its own new tests passed.
**Lesson:** a type or conversion fix merges only with a public reproduction that failed first, and a CI ratchet fails any change that breaks a case that used to pass.

### 8. Hand-judged lists never converged

All seven independent checks of the "untested branches" lists failed, with 3 to 40 wrong entries per list, and failures spawned five new phases.
**Lesson:** when the same check fails twice, change the method (make it automatic), don't add a phase.

### 9. The process had no brakes

About 43% of the first orchestrator's agent time was process.
Most audits failed on wording, not code.
Agreed acceptance checks were locked before implementation, and checks that could never pass had to be changed and locked again ("relocks"): eight times across the first orchestrator's phases, and nine times in a single later phase.
Evidence tied to exact commits forced repeated re-validation, and approvals were per file hash, so one rename meant hundreds of re-approvals.
The second orchestrator added rules on itself and asked the owner one question in its 63 hours.
**Lesson:** cap rounds and relocks, approve by meaning (one decision covers many files), and bound phases by executable outcomes.
Stop rules are acted on by the owner, not by the agent doing the work.
*Note (T1.27, checkpoint 7):* this predates Phase 1's E29, under which a review round reports its findings and the work continues while both reviewers iterate, P1 and P2 findings blocking and the rest carried forward (E49(b)).

### 10. Work fragmented instead of integrating

The two runs created 74 worktree directories and many branches.
One candidate passed all 5,004 public cases and 21,388 generated ones, though its live checks had not run; it existed only as a detached commit.
The next four fixes, composed on top of it, broke 24 cases that used to work.
**Lesson:** keep one integration branch with at most one short-lived topic branch per task.
Push every candidate that passes its checks before building on it, and treat "passed these cases" as a checkpoint, not a release.

### 11. Agreements made in review were lost

When the proposal was posted, reviewers asked for pluggable data access, so a host could answer with a non-SQL source, and for one shared PQL instead of two.
The first was agreed and the second left open: the plan omitted the data-access seam, and shared PQL never became a real shared dependency, so the implementation hard-coded the SQL dialects and wrote another PQL parser.
**Lesson:** every review comment ends as a plan item, a decision, or an explicit "no", and is tracked to the end.
*Note (T1.27, checkpoint 7):* the shared PQL question was answered in Phase 1 by E18: the parser is this client's own and permanent.

### 12. Differences found by review instead of by the implementer (Phase 1, added at T1.27)

At checkpoint 5, three of six packages lost a review round to a difference from the reference that no register entry named.
**Lesson:** define the supported semantic policy first, and before review classify every differential mismatch to an approved class or a proposed register entry; one committed differential tool (`contract/tools/differential.py`, T1.27) exits non-zero on anything left unclassified.

### 13. A harness cannot approve its own changed assertions (Phase 1, added at checkpoint 8)

The gate judges a candidate with its checkpoint's harness, so a change to a judging file and the product change that needs it cannot land together; four packages found that out in review.
**Lesson:** split at push, not in review: the judging change lands alone as part 1, and the rest is judged under it.

### 14. Contract-version bumps serialize at integration (Phase 1, added at checkpoint 8)

Two lanes each bumping the contract version collided, and one renumbered and replayed behind the other.
**Lesson:** whoever lands second renumbers; after the rebase, review the interaction (a range-diff showing only the version changed), not the whole package again.
*Note (T2.0a):* the version was also hard-coded in the binding tests' `messages.py`, which two later bumps missed, each a red CI run (migration-plan.md section 8, Phase 2 row FX-6); tests read the version from the crate, never write it.

### 15. Evidence names the exact SHA that lands (Phase 1, added at checkpoint 8)

A review, a measurement or a recorded reference line counts only for the commit it names.
**Lesson:** reviews name the candidate that lands, landing evidence the landing and its tree's equality with that candidate, and a replay gets a range-diff re-check.

### 16. Separate test authors catch what implementers miss (Phase 1, added at checkpoint 8)

A test author working from the brief, the spec and the reference found defects the implementer's own tests passed, but late findings still came from the interpreter and pandas versions the tests had not run on.
**Lesson:** keep tests by a separate author, and have them run the supported interpreter and pandas matrix early, not at landing.

### 17. Semantics only the Python host supplies fail on the Rust host (Phase 1, added at T2.0a)

On the held-out one-shot run, the pure-Rust host refused `h11.task.forecast-sum` (`UNSUPPORTED_REQUEST`, rule `problem_type`), an input the reference accepts, while python-core passed it.
The Python bridge split a forecast into per-step questions before the core saw it, a row design.md's "Host-held semantics" ledger left to "a later phase", so the Python route hid the gap.
Both reviewers ruled H11-FAIL on that case alone (`phase-1/evidence/heldout/h11-classification.md`).
It cost T1.36 fixes 21 to 24 and contract 25 to move the split into the core (E153), three reruns of the core legs on the sealed cases, and the held-out set itself: the core had seen the case in more than one run, so the set is spent and Phase 2 needs a new one (`phase-1/evidence/heldout.md`).
**Lesson:** a behavior is in the core only when the pure-Rust host runs it.
The design lists every host-held ledger row its scope or held-out set can reach, and each is either moved into the core in the phase or taken out of the scope and the held-out set by the owner; the rust-host leg runs over the whole scope before the held-out run.

### 18. Refusals of inputs the reference accepts surfaced last (Phase 1, added at T2.0a)

At the final whole-phase review (c49ee4f8), after every package had landed, the independent subagent's P2-1 findings were all refusals of inputs the reference accepts: a float target inferred categorical refused `target_type` (T1.36 fix 1), `AVG(...) = NULL` refused on every input (fix 2, P1B-21), and the 12 "not yet" rules given to T1.33 never reconciled (fixes 5, 8 and 9).
The survey of those rules (`phase-1/not-yet-refusals.md`) had been read from the code with nothing run, and measuring both routes disagreed with it on five rules (`entity_key`, `entity_ref`, `time_type`, `seed_size`, `condition_value`): lesson 8's hand-judged list again.
Round 2 then found `float_nan`: any DOUBLE column with a NaN in the previewed rows refuses `Schema.from_source`, a rule inside E92's owner-approved Phase 2 class that real data is likely to hit, so it leads Phase 2's list (migration-plan.md section 8, FR2 P3-1); T1.33's RelBench differential had already met `text_time` and `parent_table` on real tables.
It cost five fixes closed only after the owner's conditional go, and carries into Phase 2 a refusal a user's first call can hit.
**Lesson:** keep a generated list of every refusal the core can give of an input the reference accepts, each with its trigger measured on both routes and how often the phase's real datasets hit it.
The evidence pack counts them, and a refusal real data is likely to hit is lifted in the phase, not deferred inside a class.

### 19. A key's type belongs to the relationship, not the column (Phase 1, added at T2.0a)

T1.33 part 3 gave each big-integer column its own tier (int64, then `decimal128(38, 0)`, then text) and joined seeds by comparing converted values.
Review found a VARINT key failing the seed join unnamed (round 1), a UHUGEINT key in the decimal tier joining the wrong rows (round 2, P1) and mixed numeric relationships losing rows and labels (round 3, P1): silent wrong answers from the overlap of key roles and types that lesson 3 names, now in typed data.
It cost five fix rounds with two same-way failures and an H20 escalation (`phase-1/tracker.md`, "Stop-rule counters"), contract 24 (`DescribedColumn.declared_type`), and a fix 4 whose extra catalog statement broke `key.hugeint` once admitted; the edges left fail by name (P1B-43's notes) and the key cast's are Phase 2 row FR P3-8.
**Lesson:** decide a key's representation per relationship, both ends together, joining in the joined key's declared type.
Before review, run a vector matrix of every pair of key types a link allows (signed, unsigned and big integers, decimals with and without scale, floats, text) on the real engine, and treat a wrong row as P1.

### 20. The comparator must cover the type set before admissions start (Phase 1, added at T2.0a)

Admitting the T1.32 families exposed, one family at a time, values the comparator and the two value encoders (`run_python.py`, `sdm-testkit`'s `values.rs`) could not judge: maps and unions (E111), intervals (E113), stages past the reference's stop and DuckDB's large buffers (E112), the floor cell's pair (E114) and python-old's own `INTERNAL_ERROR` (E110).
Each was a judging change that had to land alone before the admissions it judged (lesson 13), some with reference lines re-recorded under new entries (P1B-54, P1B-56).
It cost T1.32g five judging landings in about 22 hours, and judging splits about 11 extra landings over the phase (`orchestration/phase-1-log.md`, "Where the time went", item 8).
**Lesson:** settle the judging path in the phase's first landing.
Before any admission, the comparator and every runner's encoder hold a vector for each type in the type set (nested, interval and extension types included) and for each stage a case can stop at; a judging change found during admissions is recorded as a design miss.

### 21. Matching where the reference crashes needs every reading of the value (Phase 1, added at T2.0a)

Under E66(b) the core may give a named refusal where the reference crashes, and P1B-81(b) refused a time candidate whose first value leaves the years 1 to 9999.
T1.36 fix 3 checked the value's UTC instant; Codex's round 2 rejected it for the wall clock in DuckDB's session zone (fix 11), and its re-check rejected fix 11 for dropping the UTC instant (fix 13), one same-way failure.
The subagent's re-check then found a third reading: with pytz installed, the reference's whole-table query fails and it chooses from the preview, which became an owner question at the go (P1B-89, E152) and a warning-text gap (FX-4) (`phase-1/as-built.md`, T1.36).
It cost two extra rounds of the final review and an owner decision on the critical path.
**Lesson:** a refusal or check that stands in for a reference crash states the crash's condition as measured on the reference, across every reading the reference makes of the value: the UTC instant, the wall clock in the session zone, with and without optional libraries such as pytz, and each supported pandas.
Its vectors cross those readings at the boundary before review.

### 22. A benchmark on generated data measures nothing (Phase 1, added at T2.0a)

E73(a) fixed P1B-13's accuracy benchmark before any result: generated datasets with null keys, the same without nulls as controls, and AdventureWorks, passing within 2%.
The live run at T1.29 found the core worse on the generated nulls, but the controls' AUROC were 0.59 and 0.51: the data had no signal, and only the owner noticed the comparison meant nothing (`orchestration/phase-1-log.md`, "Benchmark on random data").
Checking what each route sent showed the reference itself corrupts requests with null keys (44 of 58 item links wrong in one request), so the owner re-approved P1B-13 on correctness and replaced the benchmark with a real-data parity check on AdventureWorks and RelBench H&M (E94; `phase-1/as-built.md`, T1.29 and T1.32e).
It cost an owner-locked protocol reversed a day later, a discarded live run and P1B-13's acceptance condition rewritten.
**Lesson:** a model-quality or parity benchmark runs on real datasets with a sanity floor (where the reference is near a trivial baseline, the result is "inconclusive"), and a difference between routes fails the run (Phase 2 row G8); generated data checks plumbing only.
When the routes differ, compare what each sends before judging either.

## What held up and is worth keeping

- The no-I/O run loop: the core asks the host to run SQL or call the model, and the host answers.
- The PQL parser and validator, the SQL generation per database, numpy's `RandomState` streams, the neighbor sampler ported draw for draw, and the label kernels.
- The live recordings on Databricks, Snowflake and Lakebase, and the credential discipline around them.
- Independent review and probing, which found most real defects.
- Locked acceptance checks, once they test the public contract and are proven passable.

## Where the two postmortems first differed

After reviewing each other's reports, the second orchestrator endorsed proving the Arrow-only contract on a hard slice first and resetting the process; the table shows the initial positions.

| Question | First orchestrator | Second orchestrator | The plan adopts |
|---|---|---|---|
| How deep to fix the boundary | Re-cut to an Arrow-only contract | Repair it in place; problems are "bounded" | Prove the Arrow-only contract on one hard slice first, then choose from the evidence |
| Process | Reset it | Keep the full machinery, trim the launchers | Reset it: keep behavioral requirements and independent review, drop per-file approvals and unbounded relocks |
| Slice order | Warehouse first, Python and Java together | Python slices first, Java later | Python leads every slice: PQL, offline databases, then warehouses; Java follows |
| Deleting internal tests | On a schedule | Only when replacements prove they catch the same issues | Replacements must show they catch what the old tests caught; then delete |

*Note (T1.27, checkpoint 7):* the per-file approvals the table drops became, in Phase 1, H2's protected-path declarations: a change names the protected paths it touches, and the landing carries them, approved by meaning, not by file hash.
