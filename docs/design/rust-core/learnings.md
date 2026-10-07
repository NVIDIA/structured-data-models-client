# Rust core migration: what we learned

In early October 2026, two AI agent orchestrators worked to move this client's logic into a shared Rust core, over about 132 elapsed hours in total.
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
**Lesson:** the main test layer is a language-neutral corpus of public-API cases, plus a generator that compares against the old release.
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
The second orchestrator added rules on itself and asked the owner one question in 62 hours.
**Lesson:** cap rounds and relocks, approve by meaning (one decision covers many files), and size phases in days.
Stop rules are acted on by the owner, not by the agent doing the work.

### 10. Work fragmented instead of integrating

The two runs created 74 worktree directories and many branches.
One candidate passed all 5,004 public cases and 21,388 generated ones, though its live checks had not run; it existed only as a detached commit.
The next four fixes, composed on top of it, broke 24 cases that used to work.
**Lesson:** keep one integration branch with at most one short-lived topic branch.
Push every candidate that passes its checks before building on it, and treat "passed these cases" as a checkpoint, not a release.

### 11. Agreements made in review were lost

When the proposal was posted, reviewers asked for pluggable data access, so a host could answer with a non-SQL source, and for one shared PQL instead of two.
The first was agreed and the second left open: the plan omitted the data-access seam, and shared PQL never became a real shared dependency, so the implementation hard-coded the SQL dialects and wrote another PQL parser.
**Lesson:** every review comment ends as a plan item, a decision, or an explicit "no", and is tracked to the end.

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
| Slice order | Warehouse first, Python and Java together | Python slices first, Java later | Warehouse first, both languages together, if the Java target can load native code |
| Deleting internal tests | On a schedule | Only when replacements prove they catch the same issues | Replacements must show they catch what the old tests caught; then delete |
