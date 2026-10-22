# Rust core migration: what we learned

Two AI agent orchestrators first worked to move this client's logic into a shared Rust core, over about 132 elapsed hours in total.
None of that first attempt merged to `main`.
The new plan ([migration-plan.md](migration-plan.md)) restarted the work in phases; Phase 1 merged into `rust-core` on 2026-10-15 and Phase 2 on 2026-10-21.
This page holds what the first attempt, Phase 1 and Phase 2 taught, one lesson per failure mode, each in the same form: what happened, what it cost, the rule that prevents it, and what enforces the rule.
It holds the process lessons as well as the product ones (the owner, 2026-10-21, RCA decision 9: "Yes"); the handbook ([orchestration/README.md](orchestration/README.md)) keeps the operating detail.
Every design and design checkpoint is reviewed against it.

**How to read "Enforced by".**
A rule is only as strong as what stops a breach.
The owner, 2026-10-21: "The problem is all these "lessons" aren't being enforced, you just seem to be updating some random md file and not officially enforcing them."
So each lesson names what enforces it, in one of four states:

- *exists*: a script, hook or CI job in the repository before Phase 3;
- *T3.0*: built by T3.0, Phase 3's first package, which landed before this page's update (the handbook's "The Phase 3 process mechanisms (T3.0)" says how to use each);
- *planned (T3.n)*: a mechanism [phase-3/design.md](phase-3/design.md) schedules in package T3.n, with the one focused test that shows it can fail;
- *wording only*: nothing stops a breach; reviewers and the owner must catch it.

A lesson whose main rule is wording only says so in its heading; there are two (lessons 18 and 20), and where part of another lesson is wording only, its line says which part.
T3.1's test `tests/test_learnings_enforced.py` (planned, T3.1) reads this page: every lesson has one "Enforced by" line, every file it names exists once the package that builds it has landed, and the lessons marked wording only are counted.

## The short version

- The Rust code is real and much of it is good, but the first attempt's core became a copy of pandas behind a Python-shaped interface, its tests checked Python's internals rather than answers, and its process could add safeguards but never cut them.
- The phased plan fixed the architecture: typed data across the boundary, a public-case corpus, a held-out set, two reviewers.
  Phase 1's held-out run exposed one host gap (lesson 4); Phase 2's passed.
- The schedule did not hold: Phase 1 took about a week against a one-day goal, and Phase 2 took 131 hours against a 50-hour plan.
- Phase 2's overrun was almost all overhead that Phase 1's lessons had already named: judging-tool changes landed one at a time, a serial landing queue with slow CI, problems found by reviewers instead of before review.
  The lessons failed because they were prose without mechanisms, contradicted each other, and lived outside the briefs agents actually read.
  T3.0 turned the owner-approved fixes into tools before Phase 3's design; this page names them.

## By the numbers

The first attempt:

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

Phase 2 (`$SDM_ORCH_STATE/phase-3/rca/`, outside the repository: `RCA.md` and `time-accounting.md`):

| What | Number |
|---|---|
| Planned critical path / actual wall time | 50 to 51 hours / 131 hours |
| Hours the phase waited on writing product code | about 5 |
| Landings / of them small judging, CI or owner-row packages found mid-phase | 52 / 20 |
| Review rounds / blocking (P1 and P2) findings | 187 / 251 |
| Blocking findings a cheaper, earlier stage could have caught / caused by an earlier fix | 249 / 105 |
| CI runs / green / repeating a commit that already had a run | 537 / 236 / 126 |
| Median green CI run, day 1 to day 6 | 35 to 93 minutes; 8 runs cut by the time limit |
| Hard-rule breaks logged by the watcher | about 25, at least 6 by the orchestrator |

## What went wrong: the product and its design

### 1. The core imitated pandas instead of taking typed data (first attempt; Phase 2 evidence added)

The original proposal said the core does no I/O and never calls into Python.
In practice the first attempt's core parsed pandas dtype names, called back into Python for truthiness, `repr`, hashing and sorting, and reported errors as Python class paths, with six pandas column models, two `to_datetime` implementations and five `json.dumps` writers.
Phase 2 met the same trap in a smaller form: T2.14b's brief moved "deciding repeated candidates" into the core, as the reference's `dict.fromkeys` does, but Python's equality for dates and times depends on `tzinfo` identity, the daylight-saving fold and PEP 495, and no plain-data key can reproduce it.
It cost eight fix rounds over about 14 hours until the host decided equality and sent the result, and a Java host still has a gap (migration-plan.md section 8, P2-T2.14b-4).
**Lesson:** the core takes typed Arrow data and returns typed results and errors; each language turns its own types into that contract.
Before a rule moves into the core, answer one question: can the core compute it from plain data without the host language's runtime?
If not, the host computes it and the contract carries the result.
**Enforced by:** the L0 guards and the emulation allowlist (*exists*: `scripts/rust_core/l0_scan.py`, `scripts/rust_core/l0_deps.py`, `scripts/rust_core/emulation-allowlist.txt`, in CI's `rust-lint`); the "Host or core" heading every package brief must fill, which `lane.py start --brief` refuses to leave out (*planned, T3.1*: `docs/design/rust-core/orchestration/briefs/package.md`).

### 2. Write down what "match" means, or matching copies the reference's accidents (first attempt and Phase 2)

In the first attempt, matching Python bit for bit meant copying behavior nobody designed, such as where CPython's sort leaves NaN scores, and Python itself answered differently on pandas 2 and 3 and across hash seeds.
In Phase 2, the owner's "match the old client by default" was read literally: packages ported NumPy's quicksort for tie order, pydantic-core's integer parser, kept NaN sign bits apart, and added a comparator rule for how many identical warnings repeat.
It cost rounds on T2.11, T2.12 and T2.13 and one judging rule.
**Lesson:** "match" means the same answer, the same refusal kind and the same user-visible message.
Statement counts, warning repeat counts, tie order among equal scores and hash order are matched only when a user can observe them; otherwise they fall in one standing register class the owner approves once (the owner, 2026-10-21, RCA decision 6: "Yes"; Phase 3 decision KO3-7).
Keep a short register of quirks kept on purpose and drop the rest by explicit decision.
**Enforced by:** `register.py validate` and the ratchet, which fail any changed outcome without an approved entry (*exists*); the reviewer report's `differences:` list, in which any difference without an approved entry blocks (*T3.0*: `docs/design/rust-core/orchestration/briefs/reviewer.md`).

### 3. Port defined algorithms and use libraries; never port a library's type system (first attempt)

Emulating precisely defined functions held up, because they could be checked against Python on hundreds of thousands of inputs: numpy's random generator, float formatting, `DateOffset` arithmetic.
Emulating pandas' type system was open-ended, and the overlap of dtypes, Arrow, Decimal and key roles produced most late regressions.
Meanwhile the first core had only 16 direct dependencies, used only `arrow-select` of the Arrow compute crates and no time zone, YAML or big-integer library, and never measured Polars or DataFusion.
The first attempt's archive still holds a 2,150-line hand-written YAML reader; Phase 3 reads view YAML and must not port it.
**Lesson:** for each piece, decide "use a library", "port a defined algorithm" or "keep in the host", and record the measured reason; the brief names the reference code a port follows, with file and line.
**Enforced by:** `l0_deps.py`, which refuses a crate the design's dependency graph does not hold (*exists*); the brief's "Reference code to port" heading, refused when missing by `lane.py start --brief` (*planned, T3.1*).

### 4. A behavior is in the core only when the pure-Rust host runs it, end to end (first attempt and Phase 1)

The first attempt ported modules in isolation, and problems with the contract (typed results, error kinds, missing values) surfaced one by one when the pieces first met a real host at the end.
In Phase 1, the held-out run's pure-Rust host refused a forecast the reference accepts, because the Python bridge had split it into per-step questions before the core saw it; it cost four fixes, a contract bump, three reruns and the held-out set itself.
Phase 2 held this lesson: its held-out run passed, with the only differences four named refusal codes.
**Lesson:** migrate by user journey, one public path end to end first, with a second host running early.
The design lists every host-held rule its scope can reach, and each is moved into the core or taken out of scope by the owner; the rust-host leg runs over the whole scope before the held-out run.
**Enforced by:** the `cross-host (<database>)` CI jobs and `gate.sh`'s rust-host leg on every case, the bridge freeze (`scripts/rust_core/bridge_freeze.py`) and the held-out runner (`scripts/rust_core/heldout_run.sh`) (*exists*); the host-held list is a section of each design, checked by the design reviewers.

### 5. Tests check public behavior, are written by someone else, and are seen to fail first (first attempt to Phase 2)

The first attempt's recordings were keyed by private Python function names, so they passed while real regressions existed.
Phase 1 showed that a separate test author working from the brief and the reference finds what implementers miss.
Phase 2 showed the remaining gap: tests that pin a mechanism instead of the behavior.
T2.14b's test asserted "the answer bytes reach the core unchanged", so when the core's reader accepted only UTF-8 where the reference accepts BOM, UTF-16 and UTF-32, every test passed for seven rounds; the whole-phase review found it.
Other tests pinned our own old behavior (an old refusal, an old locked set), one new case took a path that never reached the fixed code, and Postgres-only tests skipped silently without a database, hiding a key-spelling bug.
**Lesson:** the main test layer is public-API cases recorded from the reference by a separate test author.
A test states the behavior a user sees ("every answer the reference accepts still predicts"), is shown failing on the broken build before it counts, and a database-specific test fails rather than skips when its database is missing in CI.
Migration-specific tests of internals are retired once replacements are shown to catch the same failures.
**Enforced by:** `handoff.py`'s fix-round step, which reruns every new or changed test on the previous round's SHA and stamps nothing unless one fails there, and the review launch refusing a SHA without a stamp (*T3.0*: `scripts/rust_core/handoff.py`, `scripts/rust_core/orchestration/codex_pane.py`, `scripts/rust_core/orchestration/guard_cmd.py`); `SDM_REQUIRE_POSTGRES=1` on every Postgres step in CI (*exists* on most steps; the rest, carried row P2-T2.14c-1, *planned, T3.2*); the landing review's `git log` check that no implementer commit touched a test author's file (*exists*, review).

### 6. Reproduce against the reference before review, over every input the package touches (Phase 1 and Phase 2)

Phase 1 lost review rounds to differences no register entry named, and its final review found a crash stand-in that needed three readings of one value (the UTC instant, the session-zone wall clock, and with pytz installed).
Phase 2 repeated it at scale: of 81 blocking findings on product packages, 51 could have been caught by the implementer or the test author running the package's inputs against the reference.
Reviewers did the search instead, one example at a time: NaN, then signed NaN, then float32 NaN (T2.12); `'0_3'`, then `'0__3'` (T2.13); unsigned integers past 2^63 (T2.10); floor-cell failures on PyArrow 14 and NumPy 1.23 found only in CI.
It cost about 12.5 hours of extra review and fix rounds on the critical path and one extra package (T2.21).
**Lesson:** every brief lists the package's input axes: value classes (null, NaN and its bit patterns, signed zero, each integer width's extremes, past 2^53 and 2^63, past 38 and 76 digits, time zones and folds, text encodings, spellings of the same value, odd identifiers), every reading the reference makes of a value, the databases, and the CI cells (current, floor, Databricks).
The test author turns them into cases recorded from the reference; before the first push the package's cases run against those recorded answers, and every fix re-runs them.
The reviewers read the result rather than probing for it (the owner, 2026-10-21: "The reviewers shouldn't really need to live test, that's the point of CI or the implementer and the tests we write").
**Enforced by:** `handoff.py`'s reference step, which runs the package's cases on python-core (and rust-host when `rust/` changed) against the recorded old-client answers on each database the case names, with no stamp on a difference, and the review launch refusing a SHA without a stamp (*T3.0*); `contract/tools/differential.py` gaining the SQLite and Postgres axis (*planned, T3.2*, carried row P2-T2.16-4); the "Input axes" brief heading (*planned, T3.1*).

### 7. A key's type belongs to the relationship, not the column (Phase 1; Phase 2 evidence added)

T1.33 gave each big-integer column its own tier and joined seeds by comparing converted values; review found keys failing the join, joining the wrong rows, and mixed relationships losing rows, over five fix rounds and a contract bump.
Phase 2 met the same family: "real keys never join" on Postgres, numeric keys read as float64 past 2^53, keys named by the caller decided before the caller's own keys (which became package T2.21), and Postgres numeric keys spelled `1.0` and `1.00` breaking the graph walk, found only by the whole-phase review.
**Lesson:** decide a key's representation per relationship, both ends together, joining in the joined key's declared type.
Before review, run the matrix of every pair of key types a link allows, and every spelling of the same key value, on the real engine; a wrong row is P1.
**Enforced by:** the key-pair cases each dialect package's test author writes (Phase 3: `DECIMAL(38,x)`, `BIGINT` and `STRING` keys on Databricks and Snowflake, in the replayed legs), run by `handoff.py` before review (*T3.0*) and held by the ratchet afterwards (*exists*).

### 8. A fix ports the reference's rule, with a test over the whole family; the second hit stops for a short note (first attempt to Phase 2)

In the first attempt, a fix refusing unsupported DuckDB nested types also refused 24 cases that used to work, and seven hand-judged lists all failed their checks.
Phase 2: 105 of 251 blocking findings were caused by an earlier fix, restack or companion test.
Fixes made the reported example pass instead of matching the reference's rule, so the next member of the family came back next round; 68 fix commits changed code and no test file.
The stop rule ("a check that fails twice the same way changes method") fired three times and each time ended the chain, but it was read narrowly: T2.14b's eight date-and-time rounds never triggered it because each example differed.
**Lesson:** a fix names the reference rule it now matches, carries a test over the class that fails on the previous commit, and re-runs the package's cases against the reference.
When the same family blocks two rounds in a row, or a fix's own change causes a new blocking finding twice in a row, stop patching: write a half-page note with the reference's rule or a simple invariant, decide host or core, write the class test, then make one fix reviewed as a whole range.
The note is short, not a redesign meeting (the owner, 2026-10-21: "sometimes you might've just missed some dumb case so we shouldn't need a large redesign sesion for that").
A CI ratchet fails any change that breaks a case that used to pass.
**Enforced by:** the commit-msg hook and `land.py` refusing a product fix commit without a `Rule:` line naming the old client's rule (*T3.0*: `scripts/rust_core/commit_msg.py`); `handoff.py`'s failing-first step (*T3.0*); the ratchet (*exists*); the stop counter per finding family in each package's tracker is *wording only*.

### 9. Refusals of inputs the reference accepts surface last unless they are generated and counted (Phase 1 and Phase 2)

At Phase 1's final review every finding was a refusal of input the reference accepts, and a survey of "not yet" rules read from the code disagreed with measurement on five rules.
Phase 2 repeated it: the response-encoding refusal and the numeric-spelling failure surfaced in the whole-phase review, and T2.11's refusal census still probed ten lifted rules, red in CI on all four candidates and missed by eight reviews.
**Lesson:** keep a generated list of every refusal the core can give of an input the reference accepts, each with its trigger measured on both routes and how often real data hits it.
Lifting a refusal updates the census, its bound and the differential's class list in the same commit.
A refusal real data is likely to hit is lifted in the phase, not deferred inside a class.
**Enforced by:** `tests/test_t21a_refusal_census.py` (*exists*), run by `handoff.py` whenever a change touches a file `scripts/rust_core/affected.py` maps to it (*T3.0*).

### 10. A benchmark on generated data measures nothing (Phase 1)

Phase 1's accuracy benchmark on generated data had control AUROCs of 0.59 and 0.51: the data had no signal, and only the owner noticed.
Comparing what each route sent then showed the reference itself corrupts requests with null keys.
It cost an owner-locked protocol reversed a day later and a discarded live run.
**Lesson:** a model-quality or parity benchmark runs on real datasets with a sanity floor (near a trivial baseline, the result is "inconclusive"), and a difference between routes fails the run; generated data checks plumbing only.
When the routes differ, compare what each sends before judging either.
**Enforced by:** `scripts/live/real_data_benchmark.py` failing the run when the two routes send different requests (*planned, T3.6*, carried row G8); its sanity floor (*exists*).

### 11. Contract-version bumps collide across lanes (Phase 1 and Phase 2)

Two Phase 1 lanes each bumped the contract version and collided; the version was also hard-coded in the binding tests, which two bumps missed.
In Phase 2, T2.12 changed the contract under v26 after a mid-phase merge froze it and moved to v27 in its ninth fix, forcing re-merges in three packages.
**Lesson:** a phase bumps the contract in one early package; a later bump is renumbered by whoever lands second and reviewed as the interaction only (a range-diff showing only the version changed).
Tests read the version from the crate, never write it.
**Enforced by:** `scripts/rust_core/contract_lock.py`, which refuses a contract change without a version bump (*exists*); the binding tests reading the version from the crate (*planned, T3.3*, carried row FX-6); "one bump per phase" itself is *wording only*, held by the design scheduling it in one package.

## What went wrong: the process

### 12. A plan states its forecast, and the slip is reported, not hidden (first attempt to Phase 2)

The first attempt's process had no brakes: about 43% of agent time was process, acceptance checks were relocked nine times in one phase, and the agent acted on its own stop rules.
Phase 1 wrote "cap rounds", "fewer, bigger landings" and "one day per phase", with no numbers.
So Phase 2's design planned 47 landings and a 50-hour critical path, double the one-day goal, and was approved after 26 review rounds without anyone calling it a breach.
The overrun was visible on day 2, when the owner asked why the phase was still on its setup packages after a night; nothing had reported it.
The owner then rejected the obvious fix, hard budgets (2026-10-21: "The problem with setting "budgets" like the 10 landings and 16-hour critical path is it can lead to shoddy work. ... Imagine 16 hours go by and it just throws its hands in the air").
**Lesson:** the design states a forecast, package by package, in a file a script reads; the slip against it is reported in plain language as it happens, and never stops work.
Round 4 of any package, the design included, goes to the owner with options; a P3 never starts a round.
The mid-phase design check runs at half the planned landings, not when remembered.
Stop rules are acted on by the owner, not by the agent doing the work; approve by meaning, not by file hash.
**Enforced by:** the slip report, `scripts/rust_core/orchestration/phase_stats.py --forecast`, which the heartbeat runs against the design's forecast file (*T3.0*; Phase 3's is `docs/design/rust-core/phase-3/forecast.yaml`); round 4 and the mid-phase check are *wording only*, read from the tracker's counters at each check-in.

### 13. Judging changes are found from the planned cases and land together up front (Phase 1 and Phase 2)

The gate judges a candidate with the last landed copy of its judging tools (comparator, runners, case format, CI config), so a judging change must land before what it judges.
Phase 1 found its judging gaps one family at a time: five judging landings in 22 hours for one package and about 11 extra landings over the phase.
Its lesson said "settle the judging path in the phase's first landing", but it was settled from the design, not from the cases.
Phase 2 repeated it at twice the scale: 20 of 52 landings were judging, CI or owner-row packages found mid-flight; five case-format widenings were each a test author's first commit, visible on day 0.
A "gate freeze" declared at 22:06 on 10-17 was broken at 22:29; "no new judging splits" on 10-20 lasted three hours.
It cost about 30 critical-path hours, the largest single cost of Phase 2.
**Lesson:** before the design locks, one stub case of every planned case kind runs through the landed judging tools, and every gap they report goes into one up-front judging package.
A judging change found later rides one tooling train at the front of the next stack, never a package of its own.
When the product cannot match exactly, the order is: fix the product, change the case, then the train; never a new package.
Owner decision rows and CI config ride with the package that needs them, and package IDs are plain integers (T3.1, T3.2), with no sub-letters.
A reference move is the phase's last landing before the gate.
**Enforced by:** `scripts/rust_core/gate_dryrun.sh` over the design's case kinds (*T3.0*; Phase 3's are `docs/design/rust-core/phase-3/dryrun-plan.yaml`); `scripts/rust_core/judging-paths.txt` and `land.py` refusing a judging change outside the phase's first package or a train (*T3.0*); `lane.py start` refusing a package ID with a letter from Phase 3 on (*planned, T3.1*).

### 14. Landing throughput sets the phase's length: stack the queue, keep CI short, one environment per lane (first attempt to Phase 2)

The first attempt fragmented into 74 worktrees and detached candidates that later fixes broke.
Phase 1 measured its day-killers (rebases across other lanes, red CI, owner waits, stalls) and ranked the fixes: a merge-queue flow, a CI split, a fast local loop.
Phase 2 dropped the queue and the CI split as "machinery" on day 2.
Every landing then forced every approved package to move and wait for a fresh CI run (15 hours); the median CI run grew from 35 to 93 minutes because about 170 judging-tool tests each re-ran the gate over the whole 41 MB corpus; eight runs hit the time limit, which was raised three times instead of fixed (11 hours).
Pushing both a lane and a topic branch made 126 duplicate runs, a stale shared environment hid a real failure for 12 hours, the disk filled or nearly filled five times, and five lanes ran on 8 CPUs at a load of 26 to 47.
**Lesson:** the landing pipeline is product work, built first and never cut as "machinery" (the owner, 2026-10-21, RCA decision 3: "Yes please.").
Up to four approved packages stack and land on one CI run on the top (the owner: "every N packages (e.g. 4 packages) before CI is mandatory"); the stack is ordered by what each landing unblocks.
The checking tools' own tests run once, only when the tools change; a job near its time limit is fixed at its cause.
One environment per lane, reused until what it installs changes, and heavy jobs start only when the machine can take them.
**Enforced by:** `land.py --stack`, `stack-land` and `restack` (*T3.0*); the `tooling` marker, held by `scripts/rust_core/tooling_tests.py check` in `tests/test_ci_workflow.py`, and the `tooling (contract)` and `tooling (root)` jobs (*T3.0*); the timing guard `scripts/rust_core/orchestration/ci_timing.py`, which names a job over 80% of its limit on two of its last three runs and writes a fix item (*T3.0*); CI cancelling a superseded run on the same lane (*T3.0*); `scripts/rust_core/cells.sh` lane environments, the runners refusing the shared `.venv`, and `scripts/rust_core/orchestration/heavy` admitting by CPU, memory and disk (*T3.0*).

### 15. Evidence names the exact commit that lands, and CI judges against the base it was pushed on (Phase 1 and Phase 2)

A review, a measurement or a recorded reference line counts only for the commit it names.
Phase 2 met the edges: the two reviewers sometimes judged different commits, or different reference commits, and both once approved a merge that did not compile under the landed judging tools; moves cancelled CI on approved commits; and landing T2.16 before its CI finished made four comparison jobs compare the candidate with itself, because CI resolved its base when the job ran.
**Lesson:** reviews name the candidate that lands, both reviewers judge the same commit against the same reference, and landing evidence names the landing and its tree's equality with that candidate.
CI fixes its comparison base when the commit is pushed.
**Enforced by:** `land.py` requiring two approvals naming the exact commit and a green exact-commit run (*exists*); the landing job recording its push-time base with `ratchet.py push-base` for every judging job (*T3.0*).

### 16. A brief answers the hard questions before anyone writes code (Phase 2)

104 of Phase 2's 251 blocking findings had the brief or the design as their earliest catch point.
The design specified a YAML file for a reader being written in parallel, and eight rounds each found a gap in the last fix; one decision about staging databases touched six readers and was fixed one reader per round for six rounds.
Briefs stated no threat model, so reviewers spent rounds on forgery attacks against internal tools; one brief mandated an equality the core could not compute (lesson 1); owner answers arrived after the design was approved and reopened it twice.
**Lesson:** every package brief fills, before the test author starts: the threat model (honest mistakes unless the owner says otherwise); the reference code to port, with file and line; host or core; the input axes (lesson 6); the "when you change X, also update Y" list; the owner's words verbatim for every decision row; and every reader of every new or changed file, with the command that runs it.
A design section that specifies an executable file is not reviewable until the file exists and its real reader has run on it.
Owner questions are answered before the first review round.
**Enforced by:** `docs/design/rust-core/orchestration/briefs/package.md` and `lane.py start --brief`, which refuses a brief missing one of its headings (*planned, T3.1*); the "executable file" rule is *wording only*, which Phase 3's design follows by running the locked-set reader and `gate_dryrun.sh` on its own files.

### 17. Two reviewers, one bar, reading the stamp and CI (Phase 1 and Phase 2)

Two reviewers pay for themselves: in Phase 2 each alone raised 30 to 40 percent of blocking findings, so dropping either would have let about a third through.
But they used different bars: Claude rated rare reference mismatches P3 a round before Codex blocked on them, three rounds running on T2.14b; Codex approved owner-wording and register text that Claude found wrong.
Both wrote verdicts before CI finished and, after round 1, reviewed only the fix, so a census failure red on round 1 surfaced in round 5 and an encoding regression survived seven rounds.
**Lesson:** both reviewers start from the same evidence: the handoff stamp, the previous round's finished CI, and the package's cases against the reference; they probe only when the code looks wrong.
Every third round, and whenever a fix changes a shared mechanism, the whole range is re-read.
One severity bar: any measured difference from the reference on a public call, any red gate or CI job the change caused, and any decision row wider than the owner's words is at least P2; attack-style hardening of internal tools stays P3 (Phase 2 decision KO2-52).
The reviewer who raised a finding on a docs-only fix round confirms its fix, and a round writing decision rows or register entries keeps Claude (the owner, 2026-10-21, RCA decision 4: "Sure...").
**Enforced by:** `land.py` requiring two `APPROVE` reports naming the same SHA (*exists*); the review launch refusing a SHA without a handoff stamp, and the report's `differences:` list (*T3.0*); the severity bar itself is *wording only* in the reviewer brief, and the owner's answer on it is still open (Phase 3 owner question G2).

### 18. Every review comment ends as a plan item, a decision or an explicit no (first attempt and Phase 2; wording only in part)

In the first attempt, reviewers asked for pluggable data access and one shared PQL; the plan dropped both silently.
In Phase 2, two P3s carried from T2.1g came back 36 hours later as parts of two new packages, an owner's "keep refusing" never became a decision row until the mid-phase check, and the carry list grew past 100 items with no owner and no date.
**Lesson:** every review comment is tracked to a plan item, a decision or an explicit no; a carried P3 that touches the judging path goes into the phase's up-front judging package or its train, not into a new package; the carry list is closed out at phase end into named rows for the next phase, and each design gives every carried row a package or a recorded no.
**Enforced by:** `scripts/rust_core/orchestration/packet_checks.sh`, which lists rows awaiting the owner and proposed register entries at phase end (*exists*); tracking each P3 to its end is *wording only*.

### 19. A rule without a mechanism in the repository is a wish (Phase 2)

Phase 2's rules held where a script enforced them (commit subjects, two approvals before landing, the ratchet) and decayed where only prose did: the gate freeze lasted 23 minutes, "push one branch" did not survive the next three packages, the banned-command rule (watched, not blocked) was broken about 25 times, and after the orchestrator's own `--no-verify` pushes and its note "not repeating", an agent did it again.
Process changes decided during the phase lived in private checklists and memory, so the repository's implementer brief went on telling every agent to run the full slow suite before each push, to add a seeded-fault test for every rule, and to compare with the reference only where a route changes, all three contrary to the rules in force.
Rules also contradicted each other ("match the reference", "only the owner relaxes a comparison", "judging changes land alone", "propose the lightest compliant route"), and the orchestrator resolved each conflict locally, usually with one more small package.
**Lesson:** a process rule is enforced by a script, a hook or a counter the check-in reads, or it is recorded here as wording only and expected to fail.
Banned commands are blocked, not only watched.
A rule decided during a phase reaches the repository's handbook and briefs at the next landing.
"The lightest compliant route" is judged over the phase, not the package: a new package is never the light route.
**Enforced by:** the command hook `scripts/rust_core/orchestration/guard_cmd.py`, wired in `.claude/settings.json`, which denies the banned commands for every Claude agent and names the alternative (*T3.0*; Codex panes keep the watcher, as the owner chose); `tests/test_learnings_enforced.py`, which fails when a lesson loses its "Enforced by" line or names a file that does not exist (*planned, T3.1*); "the next landing carries the rule" is *wording only*.

### 20. The owner reads it once, without context (Phase 1 and Phase 2; wording only in part)

Phase 2's kickoff questions were "sent" in a form the owner could not find and answered within 16 minutes of a readable page appearing, eight hours later.
Ten decisions made overnight under the owner's delegation were written in register language and handed over at the very end; five needed explaining and one was reversed, costing a round on the final candidate.
The owner replied "explain simply" or "I don't understand" at least ten times, and reviewers caught about eight decision rows written wider than the owner's answer.
**Lesson:** every owner message opens with two plain sentences on what it means, one concrete example and a recommendation; IDs go in a footnote.
Questions go on a readable page the moment they exist, all at once.
An overnight decision is written that way when it is made and put to the owner in the morning, before the final candidate (the owner, 2026-10-21, RCA decision 8: "Yes").
A decision row quotes the owner's words and says no more; nobody but the owner writes `approved_by: owner`.
**Enforced by:** `register.py validate` refusing an `approved_by: owner` entry whose `decision_ref` names no row the owner approved (*planned, T3.2*); the message form is *wording only*.

## What held up and is worth keeping

- The no-I/O run loop: the core asks the host to run SQL or call the model, and the host answers.
- The PQL parser and validator, the SQL generation per database, numpy's `RandomState` streams, the neighbor sampler ported draw for draw, and the label kernels.
- The live recordings on Databricks, Snowflake and Lakebase, and the credential discipline around them.
- Two independent reviewers on product code, and separate test authors.
- Locked acceptance checks, once they test the public contract and are proven passable, and a held-out set used once: Phase 2's ran once and passed.
- Scripted confirmation of clean moves (`mech_confirm.py`): 27 hand-run confirmation rounds across both phases found nothing a script would not have.
- The stop rule, when it fired: each of its three Phase 2 firings ended a chain of fixes by changing method (tree equality, an explicit table, a type-tree walk).
- Replaying recorded answers into the core (Phase 2's Postgres replay for rust-host): the same pattern lets Phase 3 develop the warehouse dialects offline.

## Where the two first-attempt postmortems first differed

After reviewing each other's reports, the second orchestrator endorsed proving the Arrow-only contract on a hard slice first and resetting the process; the table shows the initial positions.

| Question | First orchestrator | Second orchestrator | The plan adopts |
|---|---|---|---|
| How deep to fix the boundary | Re-cut to an Arrow-only contract | Repair it in place; problems are "bounded" | Prove the Arrow-only contract on one hard slice first, then choose from the evidence |
| Process | Reset it | Keep the full machinery, trim the launchers | Reset it: keep behavioral requirements and independent review, drop per-file approvals and unbounded relocks |
| Slice order | Warehouse first, Python and Java together | Python slices first, Java later | Python leads every slice: PQL, offline databases, then warehouses; Java follows |
| Deleting internal tests | On a schedule | Only when replacements prove they catch the same issues | Replacements must show they catch what the old tests caught; then delete |

In Phase 1 the per-file approvals became H2's protected-path declarations: a change names the protected paths it touches, and the landing carries them, approved by meaning, not by file hash.
The shared-PQL question was answered by Phase 1's E18: the parser is this client's own and permanent.

## Where each earlier lesson went

Lessons 1 to 22 were numbered before Phase 3; this table maps each to its lesson now.

| Old | New | Old | New | Old | New |
|---|---|---|---|---|---|
| 1 | 1 | 9 | 12 | 17 | 4 |
| 2 | 2 | 10 | 14 | 18 | 9 |
| 3 | 3 | 11 | 18 | 19 | 7 |
| 4 | 3 | 12 | 6 | 20 | 13 |
| 5 | 4 | 13 | 13 | 21 | 6 |
| 6 | 5 | 14 | 11 | 22 | 10 |
| 7 | 8 | 15 | 15 | | |
| 8 | 8 | 16 | 5 and 6 | | |
