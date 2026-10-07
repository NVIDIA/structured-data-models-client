# Rust core migration: the new plan

**Status: proposed.**
It replaces the first plan once the owner approves its scope, the decisions in section 2 and the process rules in section 9.
The postmortem behind it is [learnings.md](learnings.md).

The goal is the same: one implementation of the client's logic shared by Python and Java, so the two can't drift.
What changes is where the line between the core and each language sits, how data access and PQL plug in, how the work is sliced and tested, and how the rules are enforced.

![New architecture](images/architecture.png)

## 1. What changes from the original proposal

| Original proposal | New plan | Why |
|---|---|---|
| No API change, read as bit-for-bit answers | Behavior is defined by a public contract corpus and a register of kept Python quirks; anything else changes only by a named, approved entry | Bit-for-bit copying spread pandas' accidents through the core |
| Core receives query results as Arrow; frames described by the host | Core takes only a closed set of Arrow types; each language converts its own types through a reviewed table | The core grew six pandas column models and parsed dtype names |
| Core never calls into the host (stated) | Checked in CI: core crates can't depend on Python or Java bindings or name Python types, and a pure-Rust test host runs every contract case | The implementation called back into Python in 19 ways |
| SQL dialects inside the core | Data access is pluggable: the core emits typed data requests; SQL dialects are one built-in provider, and a host can answer with its own (for example Pyramid) | Agreed when the proposal was posted, then lost: the first attempt hard-coded the dialects |
| PQL: a hand-written port | One PQL implementation from a shared library, built from declarative grammar files and used by Python, Java and the core | Several PQL copies already drift |
| Port the client logic | Port defined algorithms; use Rust libraries for general data work; keep presentation and object behavior in the host | 27-29% of the Rust re-created libraries by hand |
| Python first, then Java | Each slice ships in Python and Java together, warehouse path first | Python assumptions surfaced only at the end |
| Module-by-module phases, merged at the end | Vertical slices through the public API, each merged to `main` at its gate | Modules passed alone and broke together; nothing merged |

Unchanged: Rust, no I/O in the core, the run loop that asks the host to fetch data or call the model, official database drivers per language, and per-language packaging.

## 2. Decide before writing code

The owner decides 2-5, 8 and 9 in R0.
Decisions 1, 6 and 7 need other teams, so they start in R0 and each names the gate it blocks: 1 before R1a's Java work, 6 before the shared crate replaces the interim parser, 7 before the Pyramid slice.

1. **The Java target.**
   Who is the first Java consumer, what do they need (warehouse path only, or more), and can their environment load a native library?
   If not: check whether a pure-Java WebAssembly runtime can run the same core within the memory it needs (32-bit WebAssembly is limited to 4 GiB).
   A separate Java implementation would give up the shared-code goal and needs explicit approval.
   The existing hand-written Java prototype is retired or kept as a reference by the same decision.
2. **Calendar budget and deadline**, as dates, so stop rules can fire on them.
3. **The reference environment**: the exact old release and its resolved Python, pandas, NumPy and PyArrow versions, and whether torch is installed (it changes an edge order), that define "today's answer", including which random generators it uses (section 5).
4. **Is refusing an input the old client accepted a regression?** Default: yes, unless an approved register entry names it.
5. **The quirk register** (section 5): which Python behaviors to keep and which to drop.
6. **PQL** (section 3): which library is the shared one (the proposed open-source PQL library and Kumo's own PQL package are separate today), one agreed PQL dialect, and who owns the crate.
   A Rust crate in the PQL repository changes the earlier position that it holds only the grammar and a Python package, so it needs the library owners' agreement.
7. **Data access and Pyramid** (section 3): is Pyramid in scope and who builds its provider; who gets its open questions answered; the minimum a custom provider must support; whether the provider interface is a public, versioned API; and the seeding promise for providers whose own sampling isn't seeded.
8. **Scope inventories**: the exact backends, scenarios, Databricks runtimes and platforms each language must support.
9. **Who owns and reviews the Rust core long term.**

## 3. Architecture

### The contract

- **Inputs:** Arrow record batches restricted to a closed type set (below).
  Each language converts its native types through a reviewed table, without losing values or the metadata that matters (precision, scale, units, missing values, order).
  Inputs the old client supported stay supported unless the register says otherwise; anything else is refused by name.
  The core never sees a pandas dtype or a Java class.
- **Operations:** a versioned list of typed requests and responses (serde types, published as JSON Schema), not functions looked up by string name.
- **Runs:** the existing `Run`/`Step` loop.
  A data step carries either SQL (for SQL providers) or a typed data request (for custom providers), with ordered typed parameters and the Arrow schema it expects back, or a marker that the schema is being discovered.
  The host returns Arrow or a structured error.
- **Errors and warnings:** `{kind, code, message, fields, cause}` with a closed list of kinds, one channel for each, and warnings kept in order.
  Each language maps kinds to its own exception and warning classes.
- **Host facts:** clock, time zone, locale and similar facts are passed in explicitly, so tests can pin them.
- **State and memory:** typed, versioned values the host holds; Arrow buffers and runs have defined owners and release rules; no handles into host objects.

### The type set

| Kind | Arrow types |
|---|---|
| Integers | int8, int16, int32, int64, uint8, uint16, uint32, uint64 (kept as they are: today's payloads distinguish them) |
| Floats | float32, float64 |
| Decimals | decimal128, decimal256 |
| Text and bytes | utf8, binary (hosts convert `large_*` without loss, or refuse by name) |
| Time | date32, timestamp(ns or us, with or without zone), duration(ns or us), time64 |
| Nested | list, struct, map, dictionary of utf8; child types come from this set (hosts convert `large_list` and `fixed_size_list` without loss, or refuse by name) |
| Other | bool, null (all-null columns) |

Every accepted type is preserved or refused with a named error, never silently turned into nulls.
R1b validates this set against the first attempt's column-type cases before it is final.

### What lives where

| Concern | Same in every language? | Where |
|---|---|---|
| PQL parsing, validation, error codes and spans | Yes | Shared PQL crate, used by the core |
| Model capability checks and profiles | Yes | Core |
| Semantic inference over typed columns (keys, time column, links, semantic types) | Yes | Core |
| What data to fetch, as typed data requests; SQL for each built-in dialect | Yes | Core |
| Sampling plan, neighbor sampler, random streams | Yes | Core |
| Labels and aggregations | Yes | Core |
| Materialization, decomposition, payload limits, session state | Yes | Core |
| Payload encoding, response decoding, result layout | Yes | Core |
| Native types to Arrow (pandas dtypes, JDBC types) | No | Host, as a reviewed table |
| Running SQL on the user's connection, or answering data requests another way | No | Host data providers |
| Object behavior, exceptions, presentation (`to_pandas`, Java records) | No | Host |
| HTTP and authentication | No | Host |

### Data access

The core asks for data through a small, fixed vocabulary of typed data requests.
The proposed minimum, which decision 7 confirms, is two operations: describe the catalog, and scan a table with a column list and key and time-range filters.
Optional operations make common steps faster, and each provider declares which it supports: rows by key, rows in a time window, counts and minimum or maximum times, the latest neighbors per key, and a seeded sample of entities.
Before a run, the core picks pushdown or an in-memory fallback from the provider's declared capabilities.
Core-side sampling uses the registered random streams over a defined ordering of the same data, so it gives the same rows on every provider; a provider's own seeded sampling is repeatable only for that provider, and changing a backend's sampling needs a register entry.
A declared capability that fails is reported as an error, never silently replaced.

- **SQL providers:** the core compiles requests to SQL for each built-in dialect (Databricks, Snowflake, Postgres, SQLite, DuckDB), and the host runs it with the vendor's driver.
- **Custom providers:** the host receives the typed request and answers it its own way, for example by compiling it to a Pyramid query.
- **The provider contract** is versioned and defines null and duplicate keys, time units and zones, interval bounds, ordering of ties, typed errors (with retryability and driver codes), and complete results: scans page through every matching row, nothing is silently truncated, and the in-memory fallback has declared memory and transfer limits that fail loudly when exceeded.
- **Testing:** a provider conformance kit (a fixture dataset, multi-page results, limits, and expected answers including hand-checked ones) that every provider passes, the built-in SQL ones included, ending with a full prediction compared with the reference provider.

Pyramid stays provisional until its row-level reads, result types, paging and filters (including long key lists) are confirmed with its team; the provider itself is an optional slice after S2.

### PQL

![One PQL implementation](images/pql.png)

Several separate PQL implementations exist today, and only two share tests.
The grammar is small (about 216 lines); the drift risk is in the code around it, the tree building, error handling and validators, so "defined once" has to mean one implementation, not one grammar file feeding several generators.

- **One crate in the shared PQL repository:** a token table for `logos` and an LR(1) grammar for `lalrpop`, both declarative files, plus the validator, one table of error codes, and the validated query as versioned JSON with a JSON Schema.
  ANTLR has no official Rust target; a prototype of this approach accepted and rejected the same 41,780 inputs as today's ANTLR parser.
  Adoption still has to prove trees, validation, date literals, error codes, spans over non-ASCII text, and deep nesting, against the agreed dialect.
- **Everyone uses that crate:** the core directly (so this client's Python and Java get it through the core), and Kumo's Python and Java through thin wrappers once Kumo adopts it, a commitment its owners make separately.
- **One conformance suite** of valid and invalid queries with expected codes and spans, seeded with this client's 417 cases, run by every consumer.
- **During migration:** this client's Python and Java both use the first attempt's Rust parser as an interim parser, held to the shared suite; today's ANTLR grammar is a CI reference only.
  Before R5 this client switches to a pinned release of the shared crate and deletes the interim parser; deferring that is an explicit scope decision.
  The ANTLR reference is retired once its remaining users have moved.
- **This client keeps** only the model capability checks on top of the validated query.
- **If a future PQL feature doesn't fit LR(1)**, fall back to one hand-written parser in the shared crate, with the ANTLR grammar kept as the grammar of record and checked against it in CI.

### Crates

The names and boundaries below are the target ownership, not a reason to rename or rewrite everything before the first slice works.

| Crate | Holds | From today's tree |
|---|---|---|
| `sdm-contract` | The type set, operations, data requests and provider capabilities, errors and warnings, run steps | New, folding in `protocol` |
| `pql` (shared repository) | Grammar, parser, validator, error codes, validated query | New, built from the grammar files; replaces both parsers |
| `sdm-capabilities` | Model capability checks and profiles over the validated query | From today's `pql` crate |
| `sdm-data` | Data planning, the in-memory fallback, and SQL compilation per dialect | Keep the dialect backends behind a provider interface; drop the capture and render scaffolding |
| `sdm-schema` | Inference over Arrow, metric and semantic views | Keep the engine; drop the pandas column models and dtype parsing; YAML through a library |
| `sdm-time` | One calendar, the date parsing rules recorded in R0, `DateOffset`, time zones | New, merging today's duplicates; built on a time library; not a general pandas date-inference clone |
| `sdm-random` | The random streams the register keeps | Keep |
| `sdm-sampling` | Sampling plan, neighbor sampler, graph store | Keep the algorithms; one value model |
| `sdm-labels` | Label kernels and windows | Keep; use Arrow compute where answers allow |
| `sdm-request` | Materialization, decomposition, limits, session state, payload, response, result | Keep the logic; drop the CPython sort and `json.dumps` size emulation |
| `sdm-core` | The public operations, built from the crates above | Rewrite the host API; typed operations only |
| `sdm-testkit` | Contract runner, provider conformance kit, fixtures | Dev-only; nothing in the shipped crates |
| `sdm-py`, `sdm-jni` | Thin bindings: convert types, map errors, drive runs | Rewrite `sdm-py` slimmer; new `sdm-jni` |

Today's `wire` crate is already generated from the model API; `sdm-request` either uses it or it is deleted.

### Dependencies

| Need | Choice |
|---|---|
| Columnar data and kernels | `arrow` crates, including `arrow-cast`, `arrow-ord`, `arrow-arith`, `arrow-select`, `arrow-row` and `arrow-ipc` |
| PQL parsing | `logos` and `lalrpop`, in the shared PQL crate |
| Dates, times, zones | One time library, chosen in R2 against the date corpus, including zone rules after 2037: `chrono` (already in the build through Arrow) with `chrono-tz` (to be added), or `jiff` |
| YAML (metric and semantic views) | A maintained YAML 1.2 crate such as `saphyr` or `yaml-rust2` (not the archived `serde_yaml`); behavior differences go through the register |
| JSON | `serde`, `serde_json` |
| Bindings | `pyo3` (abi3 wheels), `jni` |
| Dataframe engine | Decided in R2: compare plain Arrow kernels with Polars or DataFusion on one label aggregation and one sampling join; correct answers are required, then weigh hand-written code, maintenance, speed, binary size and JNI loading; record the results |

On the Python side, pandas and pyarrow stay as host dependencies, and the old engine goes away at the end.

## 4. Delivery: slices, not modules

![Delivery](images/delivery.png)

| Slice | Public path | Languages |
|---|---|---|
| S0 | PQL validation | Python and Java |
| S1 | `Schema.from_source` → `materialize` → `predict` on DuckDB, SQLite and Postgres, offline; a toy custom data provider passes the conformance kit | Python and Java |
| S2 | The same live on Databricks (with a metric view), Snowflake (with a semantic view) and Lakebase; the Databricks job matrix across runtimes; Python and Java agree on the same data | Python on all three; Java on Databricks and Snowflake, Lakebase if in scope |
| S3 | In-memory frames, ranking, forecasting, link prediction, sessions, tabular requests | Python; Java scope set by the owner |
| Pyramid (optional, after S2) | A Pyramid data provider | Java, if in scope |

A path is one public operation with one input mode on one backend (for example `predict` of a regression query on Snowflake), and one routing boundary chooses the old implementation or the core for each path.
Python switches a path only after its slice passes all its gates, live ones included: S1 and S2 paths after S2, S3 paths (ranking, forecasting, sessions, on any backend) after S3.
R0 maps every Databricks job in the matrix to its paths, so S2 includes what its jobs need.
Old paths are deleted in R5, and the oracle (section 6) stays available after that.
Each slice gate merges to `main`.
If decision 1 or the Java binding blocks a slice for more than two days, the slice ships in Python only; Java keeps running the contract corpus and catches up at the next gate.

## 5. The quirk register

The register is a machine-readable file in the repository.
Every entry names the old and new behavior, the affected inputs and contract case IDs, and the owner's approval by name; silence is not approval.

| Keep (part of the promise) | Drop (by recorded decision) |
|---|---|
| The random streams of the reference release (numpy `RandomState` and `default_rng` where it uses them, `std::mt19937` for neighbor draws), pinned in R0; keeping `default_rng` means porting its generator too | CPython's sort order around NaN scores (NaN ranks last) |
| Label arithmetic as today: float32 Kahan sums in row order | The size limit counted as `json.dumps` bytes (count the bytes sent) |
| `DateOffset` month-end rules | Python object protocols inside the core |
| PQL codes, spans and grammar quirks, as agreed with the shared library | Errors as Python class paths |
| PQL date-literal rules, written down | Silent nulls or internal errors for types the old code mishandled |
| Today's public exception and warning classes and codes | Time zone rules ignored after 2037 |
| Text-column timestamp detection, under the reference pandas major | |
| Integer widths in payloads | |
| Payload equality after parsing; SQL per database as reviewed golden files | |

Each needs its own decision, not a blanket rule: answers that differ between pandas 2 and 3, SQL text changes, and YAML 1.1 values such as `yes` and `on`.

## 6. Testing

| Layer | What it checks | When |
|---|---|---|
| L0 Boundary guards | Core crates' production dependency graph has no `pyo3`, `jni` or `arrow-pyarrow`; the run steps are only data and model steps (no requests for language behavior such as truthiness, sorting or dtype inference); a pure-Rust test host passes the contract subset; the contract's JSON Schema, run steps and the PQL validated-query format don't change without an owner-approved version bump | Every PR |
| L1 Unit and property tests | Algorithms in each crate, with every past mismatch seed; the shared PQL conformance suite | Every PR |
| L2 Contract corpus | One shared list of cases, run in each language: a public call and its inputs in; ordered data requests, SQL and typed parameters, payload, result, warnings and error kind, code, details and cause out; a coverage matrix of type × role × backend is generated from the executed cases and the approved scope, and an empty cell fails (unsupported combinations have explicit refusal cases) | Every PR (subset, for feedback), before each slice gate (all) |
| L3 Differential generator | Random public-API inputs compared with the old release, installed from a pinned wheel in its pinned environment (the "oracle"), before and after the old code is deleted | Nightly |
| L4 SQL goldens and provider kit | Reviewed SQL per database, executed on DuckDB, SQLite and Postgres; the local providers pass the conformance kit | Every PR |
| L5 Conversion tables | Each language's native-to-Arrow table, both pandas majors; the same tables loaded into DuckDB, SQLite and Postgres reach the core with identical Arrow schemas, values, null masks, order and required metadata from Python's drivers and Java's JDBC, checked against expected values (large integers, decimal scale, timestamp units, nested nulls, list widths, dictionaries) | Every PR |
| L6 Live matrix | The scope inventory's scenarios on Databricks, Snowflake and Lakebase (Databricks' managed Postgres), plus the Databricks job matrix (one-time jobs across Databricks runtimes, compared with the old release's results); old versus new, and Python versus Java on the same data | Weekly and before release |
| L7 Performance | Query counts equal; push-down median over at least five interleaved runs, caches off, within 10% on Databricks and Snowflake | Nightly and before release |
| L8 Packaging and safety | Wheels and JAR (JDK 17 and 21, Linux x86_64 and aarch64) install and load; native memory, panic and leak tests; HTTP retry, timeout and redaction tests | Before release |

- **Ratchet:** CI compares every case's full observed outcome with the protected last-green baseline, so a changed answer, a new refusal or a removed case fails unless a register entry states that case's exact old and new outcome.
  This replaces choosing "neighboring cases" by hand.
- **Budgets:** about 10 minutes per PR and 60 minutes nightly, as targets to optimize, never a reason to skip cases.
- **Test the tests:** before the L2 runners count for acceptance, show they fail on a corrupted expectation, a missing case, a skipped call and a missing native library.
- **Local first:** reproduce a failure from the installed package before spending a remote run on it; a failure that only happens remotely gets one bounded diagnostic run.
- **Retiring old tests:** the first attempt's internal-equivalence tests go once the L2 and L3 failure families they caught are covered, measured by seeding those failures and checking the new layers catch them.
  Algorithm unit tests stay.

## 7. Release gates

| Gate | Passes when |
|---|---|
| Merge a change | L0, L1, L4, L5 and the L2 subset green; the ratchet holds; a type or conversion fix comes with a public reproduction that failed first |
| Slice done (merges to `main`) | The full L2 for the slice green in both languages; L3 on the agreed seed sets with no unexplained divergence; every difference has an approved register entry; for S2, L6 green for its scenarios |
| Python switch for a path | Its slice done; L6 green for it; the Databricks job matrix equal to the baseline; L7 within budget |
| Release | Every layer green on the release commit, including L6 and the job matrix re-run after the old code is deleted; the existing Python test suites green; live test resources cleaned up |

## 8. Phases

| Phase | Work | Exit | Estimate |
|---|---|---|---|
| R0 | Decisions (section 2) and calendar dates; pin the oracle environment; record the accepted text-date formats and inference rules; name the shared PQL crate's owner and adoption milestone; verify branch protection, CI and approval identities; push the first attempt's best candidate as a named branch and run CI once, as a fallback; archive the rest; one integration branch | Decisions and dates written down | 1 day |
| R1a | `sdm-contract`, `sdm-testkit`, L0 guards, L2 runners for Python, Java and the oracle, CI tiers | S0 green in both languages; the runners fail on seeded faults | 3-4 days |
| R1b | Timeboxed spike: DuckDB `from_source` → `materialize` → `predict` → result through the new contract and the data provider interface in both languages, and the same path through a minimal non-SQL test provider, against a scripted model endpoint; the text-date rules recorded in R0 checked against the oracle | Matches the oracle on inferred types, keys, time columns, data requests and SQL, payloads, results, warnings and refusals, over the first attempt's column-type cases; the owner decides go or no-go (go: R1a and R1b merge to `main`; no-go: repair the first attempt in place, or stop) | 3 days, hard stop |
| R2 | S1, moving only the code S1 needs off the Python value model and onto the contract; the provider conformance kit; the time-library and dataframe-engine comparisons | Slice gate | 8-11 days |
| R3 | S2 and the Python switch for warehouse paths | Switch gate | 5-7 days |
| R4 | S3, moving the rest of the code as its paths need it | Slice gate and switch gate | 6-8 days |
| R5 | Switch to the shared PQL crate release; delete the interim parser and the old engine; release candidate | Release gate | 2-3 days |

About 28-37 working days before contingency, plus the optional Pyramid slice.
The basis: the first attempt's time for comparable pieces, plus time for the second language in every slice, the data provider seam, and moving code off the Python value model.
R0 turns this into calendar dates with a stated hours-per-day assumption, and R1b's measurements revise it before the owner approves R2.

## 9. Process rules

| Rule | How it's enforced |
|---|---|
| One integration branch holds the only candidate; work on the next slice starts only after this slice's gate passes; independent tasks within a slice may run in parallel | Required CI and review on the integration branch and `main`; the status page lists open branches |
| A candidate that passes its checks is pushed before more work builds on it; each slice gate merges to `main`; rebase daily | CI on the integration branch; the slice gate is a pull request to `main` |
| Acceptance is what CI runs, plus one independent reviewer per change, the owner's decisions, and one independent release audit of the existing evidence | Required checks on the pull request |
| At most two review rounds per change; one independent reviewer checks code and behavior, not prose; a third round goes to the owner | Recorded on the pull request |
| A phase's acceptance checks are locked before implementation; each change states what and why; a third change in one phase goes to the owner | Locked files guarded by code owners |
| Approval by meaning: one decision approves all the files it explains | Register entries and pull request approvals; no per-file hashes |
| Stop rules (below) are lifted only by the owner, never by delegation; on a stop, implementation and live runs pause and the evidence is reported | The owner's approval on the record; the executor's credentials can't approve its own exceptions (R0 verifies this, or the owner checks it by hand) |
| One status page: candidate commit, gates passed, open failures, decisions needed, days used against the approved dates | Generated from CI and git, not written by the executor |
| Anything that changes behavior, scope or dates goes to the owner the same day | Register entries and the status page |
| A new test or harness tool over about 200 lines needs the owner's approval and a timebox | Listed on the status page |

**Stop rules:**
- a phase at 150% of its estimate stops for a re-plan with concrete cuts;
- a check failing twice the same way changes method, never adds a phase;
- after two fix rounds on the same divergence, the owner chooses between repairing it, deferring the slice with the old path unchanged, and approving a named behavior change; nothing becomes an accepted refusal automatically;
- three live failures for non-product reasons stop live work;
- if S1 isn't green in both languages by its date, the owner chooses the next step;
- if the owner hasn't answered a blocking question within one working day, work narrows to the current slice; silence never authorizes going on.

## 10. Risks

| Risk | Mitigation |
|---|---|
| The Java target can't load native code | Decision 1 before any Java work; WebAssembly check as the fallback |
| The team has little Rust experience to own a large core | Decision 9; small crates, boundary guards, and the contract corpus make changes reviewable |
| The shared PQL crate isn't ready when needed | The interim parser, held to the shared conformance suite |
| Pyramid's request and result formats are unknown | The provider is optional and comes after S2; the open questions go to its team in R0 |
| Host conversion tables drift between Python and Java | L5 checks both languages convert the same table to the same Arrow types |
| Estimates are optimistic again | Calendar dates, stop rules lifted only by the owner, and re-estimation after R1b |
| Process grows faster than product again | Rules enforced by CI and the owner, not by the executor; the status page counts days used |
