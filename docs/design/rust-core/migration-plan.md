# Rust core migration: the new plan

**Status: proposed.**
It replaces the first plan once the owner approves it; the owner's decisions are in section 2.
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
| SQL dialects inside the core | Data access is pluggable: the core emits typed data requests; SQL dialects are one built-in provider, and the layer can be swapped or a host can answer with its own provider (for example for Pyramid) | Agreed when the proposal was posted, then lost: the first attempt hard-coded the dialects |
| PQL: a hand-written port | One PQL implementation, the PQL team's C++ core, used through its published Rust crate | Several PQL copies already drift |
| Port the client logic | Port defined algorithms; use Rust libraries for general data work; keep presentation and object behavior in the host | 27-29% of the Rust re-created libraries by hand |
| Python first, then Java | Each slice ships in Python and Java together, warehouse path first | Python assumptions surfaced only at the end |
| Module-by-module phases, merged at the end | Vertical slices through the public API, each merged to `main` at its gate | Modules passed alone and broke together; nothing merged |

Unchanged: Rust, no I/O in the core, the run loop that asks the host to fetch data or call the model, official database drivers per language, and per-language packaging.

## 2. Decisions

The owner made these before any code:

1. **Java:** the first Java consumer is a team using Pyramid, which brings its own data layer and uses none of the data sources this client supports.
   The Java binding still gets basic support for the offline backends and Databricks, proven working on Databricks.
   It loads the core as a native library; WebAssembly is out, because its size limits are too restrictive.
2. **Scope:** Python supports every backend it supports today (Databricks, Snowflake, Lakebase, Postgres, SQLite, DuckDB and in-memory frames).
   Java starts with DuckDB, SQLite, Postgres and Databricks.
3. **Data layer:** Pyramid is not in scope, but the data layer must be switchable: a team given access to this repository should be able to replace it, for example to work with Pyramid, without touching the rest of the core.
4. **PQL:** the PQL team owns one C++ core and publishes it as a Rust crate and a Python package; this client's core uses the Rust crate, with the hand-written parser as a stopgap (section 3).
5. **Regressions:** refusing an input the old Python client accepted is a regression.
   Java on Snowflake is not prioritized, so it is not held to this yet.
6. **Python quirks:** the register (section 5) follows common sense, with a small bias towards today's Python behavior but without catering to its accidents.
7. **Reference environment:** the old release is installed locally in its own worktree and environment, with its Python, pandas, NumPy, PyArrow and torch versions recorded; it is the oracle the tests compare against.
8. **No calendar budget and no time estimates:** the work is paced by its gates, not by dates, and dates must not change the quality of the work.
9. **Ownership:** this team owns the core long term; that does not change how it is built.

Still to agree with the PQL team: one PQL dialect, the versioned validated-query format, the conformance suite, platform builds, and when the Rust crate ships.

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
| PQL parsing, validation, error codes and spans | Yes | The PQL team's C++ core, through its Rust crate |
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
The minimum every provider supports is two operations: describe the catalog, and scan a table with a column list and key and time-range filters.
Optional operations make common steps faster, and each provider declares which it supports: rows by key, rows in a time window, counts and minimum or maximum times, the latest neighbors per key, and a seeded sample of entities.
Before a run, the core picks pushdown or an in-memory fallback from the provider's declared capabilities.
Core-side sampling uses the registered random streams over a defined ordering of the same data, so it gives the same rows on every provider; a provider's own seeded sampling is repeatable only for that provider, and changing a backend's sampling needs a register entry.
A declared capability that fails is reported as an error, never silently replaced.

- **SQL providers:** the core compiles requests to SQL for each built-in dialect (Databricks, Snowflake, Postgres, SQLite, DuckDB), and the host runs it with the vendor's driver.
- **Custom providers:** the host receives the typed request and answers it its own way, for example by compiling it to a Pyramid query.
- **The provider contract** is versioned and defines null and duplicate keys, time units and zones, interval bounds, ordering of ties, typed errors (with retryability and driver codes), and complete results: scans page through every matching row, nothing is silently truncated, and the in-memory fallback has declared memory and transfer limits that fail loudly when exceeded.
- **Testing:** a provider conformance kit (a fixture dataset, multi-page results, limits, and expected answers including hand-checked ones) that every provider passes, the built-in SQL ones included, ending with a full prediction compared with the reference provider.

The data layer is one crate behind one documented interface, with the SQL providers as its reference implementation and the conformance kit as its acceptance test, so a team with access to this repository can swap it, or implement a provider in Java, without touching the rest of the core.
A Pyramid provider is not part of this plan.

### PQL

![One PQL implementation](images/pql.png)

Several separate PQL implementations exist today, and only two share tests.
The drift risk is less in the grammar than in the code around it (tree building, error handling, validators), so "defined once" means one implementation.

- **One implementation, owned by the PQL team:** a C++ core holding the grammar, parser, validator, error codes and spans, and the validated query as versioned JSON.
  The team publishes it as a Rust crate (wrapping the C++ core) and a Python package.
- **This client's core uses the Rust crate**, so this client's Python and Java get PQL through the core; Kumo's consumers use the Python package or the C++ core directly.
- **One conformance suite** of valid and invalid queries with expected codes and spans, seeded with this client's 417 cases, run by every consumer.
- **Until the Rust crate is published,** this client's Python and Java both use the first attempt's hand-written Rust parser as an interim parser, held to the shared suite; today's ANTLR grammar stays as a CI reference.
  Before R5 this client switches to a pinned release of the crate and deletes the interim parser; deferring that is an explicit scope decision.
- **What to ask the PQL team for:** a C interface or a Rust crate that never lets a C++ exception cross into Rust, builds for every platform this client ships (Linux x86_64 and aarch64, macOS) with private C++ symbols, so two copies can live in one process, a versioned validated-query format with a JSON Schema, and the conformance suite in their repository.
- **This client keeps** only the model capability checks on top of the validated query.

### Bindings

![Bindings](images/bindings.png)

The core is plain Rust and doesn't know which language loads it.
Each language uses its standard binding tool, over one shared Rust layer:

- **`sdm-host`**, plain Rust, written once: the run protocol, encoding messages as versioned JSON and data as Arrow, catching panics, mapping errors to kinds, and the threading rules.
- **Python:** a thin PyO3 binding built with maturin as `abi3` wheels (one wheel per platform for every supported Python version); it converts types, releases the GIL while the core runs, and passes Arrow through the standard PyCapsule interface without copying.
- **Java:** a thin `jni-rs` binding loaded from the JAR (JDK 17 and 21); it passes Arrow through Arrow Java's C Data module without copying.
- **Rules at every boundary:** no Rust panic and no C++ exception crosses into another language; memory has one owner and an explicit release; the threading model is written down.
- **Java runtime flags:** users run with `--enable-native-access` (Java warns without it from JDK 24 and will later refuse) and the flag Arrow Java needs; the Java consumer's environment must allow both.
- **PQL's C++ is linked into both native libraries** with its symbols kept private, so a Python process that also loads the PQL team's own package gets no clashes.

Considered and not chosen: one C interface (`extern "C"`) with generic glue per language.
It ships one native file per platform, but means designing and maintaining our own unsafe boundary where PyO3 and `jni-rs` already provide tested ones; it stays the route for a future language without a Rust binding tool (such as C# or Go), built on the same `sdm-host`.
Binding generators such as UniFFI and Diplomat don't carry Arrow data, and their Java support needs JDK 22 or later.
If a Java environment can't load native code at all, the fallback is running the core in a separate process.

### Crates

The names and boundaries below are the target ownership, not a reason to rename or rewrite everything before the first slice works.

| Crate | Holds | From today's tree |
|---|---|---|
| `sdm-contract` | The type set, operations, data requests and provider capabilities, errors and warnings, run steps | New, folding in `protocol` |
| `pql` (from the PQL team) | Rust crate wrapping the C++ PQL core | External dependency; replaces the interim parser |
| `sdm-capabilities` | Model capability checks and profiles over the validated query; the interim parser until the PQL crate ships | From today's `pql` crate |
| `sdm-data` | Data planning, the in-memory fallback, and SQL compilation per dialect | Keep the dialect backends behind a provider interface; drop the capture and render scaffolding |
| `sdm-schema` | Inference over Arrow, metric and semantic views | Keep the engine; drop the pandas column models and dtype parsing; YAML through a library |
| `sdm-time` | One calendar, the date parsing rules recorded in R0, `DateOffset`, time zones | New, merging today's duplicates; built on a time library; not a general pandas date-inference clone |
| `sdm-random` | The random streams the register keeps | Keep |
| `sdm-sampling` | Sampling plan, neighbor sampler, graph store | Keep the algorithms; one value model |
| `sdm-labels` | Label kernels and windows | Keep; use Arrow compute where answers allow |
| `sdm-request` | Materialization, decomposition, limits, session state, payload, response, result | Keep the logic; drop the CPython sort and `json.dumps` size emulation |
| `sdm-core` | The public operations, built from the crates above | Rewrite the host API; typed operations only |
| `sdm-testkit` | Contract runner, provider conformance kit, fixtures | Dev-only; nothing in the shipped crates |
| `sdm-host` | The run protocol for hosts, message and Arrow encoding, panic catching, error kinds, threading rules | New, written once |
| `sdm-py`, `sdm-jni` | Thin PyO3 and `jni-rs` bindings over `sdm-host`: convert types, map errors, release the GIL | Rewrite `sdm-py` much slimmer; new `sdm-jni` |

Today's `wire` crate is already generated from the model API; `sdm-request` either uses it or it is deleted.

### Dependencies

| Need | Choice |
|---|---|
| Columnar data and kernels | `arrow` crates, including `arrow-cast`, `arrow-ord`, `arrow-arith`, `arrow-select`, `arrow-row` and `arrow-ipc` |
| PQL parsing | The PQL team's Rust crate over its C++ core (so every build also compiles and links C++) |
| Dates, times, zones | One time library, chosen in R2 against the date corpus, including zone rules after 2037: `chrono` (already in the build through Arrow) with `chrono-tz` (to be added), or `jiff` |
| YAML (metric and semantic views) | A maintained YAML 1.2 crate such as `saphyr` or `yaml-rust2` (not the archived `serde_yaml`); behavior differences go through the register |
| JSON | `serde`, `serde_json` |
| Bindings | `pyo3` with `maturin` (abi3 wheels); `jni` (`jni-rs`); the Arrow C Data Interface in `arrow`, `pyarrow` and Arrow Java |
| Dataframe engine | Decided in R2: compare plain Arrow kernels with Polars or DataFusion on one label aggregation and one sampling join; correct answers are required, then weigh hand-written code, maintenance, speed, binary size and JNI loading; record the results |

On the Python side, pandas and pyarrow stay as host dependencies, and the old engine goes away at the end.

## 4. Delivery: slices, not modules

![Delivery](images/delivery.png)

| Slice | Public path | Languages |
|---|---|---|
| S0 | PQL validation | Python and Java |
| S1 | `Schema.from_source` → `materialize` → `predict` on DuckDB, SQLite and Postgres, offline; a sample custom data provider, written in Java, passes the conformance kit | Python and Java |
| S2 | The same live on Databricks (with a metric view), Snowflake (with a semantic view) and Lakebase; the Databricks job matrix across runtimes; Python and Java agree on the same Databricks data | Python on all three; Java on Databricks |
| S3 | In-memory frames, ranking, forecasting, link prediction, sessions, tabular requests | Python; Java as the owner adds scope |

A path is one public operation with one input mode on one backend (for example `predict` of a regression query on Snowflake), and one routing boundary chooses the old implementation or the core for each path.
Python switches a path only after its slice passes all its gates, live ones included: S1 and S2 paths after S2, S3 paths (ranking, forecasting, sessions, on any backend) after S3.
R0 maps every Databricks job in the matrix to its paths, so S2 includes what its jobs need.
Old paths are deleted in R5, and the oracle (section 6) stays available after that.
Each slice gate merges to `main`.
If the Java binding blocks a slice, the slice can ship in Python first; Java keeps running the contract corpus and catches up at the next gate.

## 5. The quirk register

The register is a machine-readable file in the repository.
Entries follow common sense, with a small bias towards today's Python behavior but without catering to its accidents.
Every entry names the old and new behavior, the affected inputs and contract case IDs, and who approved it: the independent reviewer approves entries, and an entry that refuses an input Python accepted needs the owner's approval, since that is a regression.

| Keep (part of the promise) | Drop (by recorded decision) |
|---|---|
| The random streams of the reference release (numpy `RandomState` and `default_rng` where it uses them, `std::mt19937` for neighbor draws), pinned in R0; keeping `default_rng` means porting its generator too | CPython's sort order around NaN scores (NaN ranks last) |
| Label arithmetic as today: float32 Kahan sums in row order | The size limit counted as `json.dumps` bytes (count the bytes sent) |
| `DateOffset` month-end rules | Python object protocols inside the core |
| PQL codes, spans and grammar quirks, as agreed with the PQL team | Errors as Python class paths |
| PQL date-literal rules, written down | Silent nulls or internal errors for types the old code mishandled |
| Today's public exception and warning classes and codes | Time zone rules ignored after 2037 |
| Text-column timestamp detection, under the reference pandas major | |
| Integer widths in payloads | |
| Payload equality after parsing; SQL per database as reviewed golden files | |

Each needs its own decision, not a blanket rule: answers that differ between pandas 2 and 3, SQL text changes, and YAML 1.1 values such as `yes` and `on`.

## 6. Testing

| Layer | What it checks | When |
|---|---|---|
| L0 Boundary guards | Core crates have no `unsafe` code and their production dependency graph has no `pyo3`, `jni` or `arrow-pyarrow` (only the binding crates may use them); the run steps are only data and model steps (no requests for language behavior such as truthiness, sorting or dtype inference); a pure-Rust test host passes the contract subset; the contract's JSON Schema, run steps and the PQL validated-query format don't change without an owner-approved version bump | Every PR |
| L1 Unit and property tests | Algorithms in each crate, with every past mismatch seed; the shared PQL conformance suite | Every PR |
| L2 Contract corpus | One shared list of cases, run in each language: a public call and its inputs in; ordered data requests, SQL and typed parameters, payload, result, warnings and error kind, code, details and cause out; a coverage matrix of type × role × backend is generated from the executed cases and the approved scope, and an empty cell fails (unsupported combinations have explicit refusal cases) | Every PR (subset, for feedback), before each slice gate (all) |
| L3 Differential generator | Random public-API inputs compared with the old release, installed from a pinned wheel in its pinned environment (the "oracle"), before and after the old code is deleted | Nightly |
| L4 SQL goldens and provider kit | Reviewed SQL per database, executed on DuckDB, SQLite and Postgres; the local providers pass the conformance kit | Every PR |
| L5 Conversion tables | Each language's native-to-Arrow table, both pandas majors; the same tables loaded into DuckDB, SQLite and Postgres reach the core with identical Arrow schemas, values, null masks, order and required metadata from Python's drivers and Java's JDBC, checked against expected values (large integers, decimal scale, timestamp units, nested nulls, list widths, dictionaries) | Every PR |
| L6 Live matrix | The scope inventory's scenarios on Databricks, Snowflake and Lakebase (Databricks' managed Postgres), plus the Databricks job matrix (one-time jobs across Databricks runtimes, compared with the old release's results); old versus new, and Python versus Java on the same data | Weekly and before release |
| L7 Performance | Query counts equal; push-down median over at least five interleaved runs, caches off, within 10% on Databricks and Snowflake | Nightly and before release |
| L8 Packaging and safety | The real wheel and JAR install and load in clean containers (JDK 17 and 21, Linux x86_64 and aarch64, macOS); exported symbols, private C++ symbols and minimum system library versions checked; a Python process loading both this client and the PQL team's package; `sdm-host` under sanitizers; panic, C++ error, crash and leak tests in each language; HTTP retry, timeout and redaction tests | Before release |

- **Ratchet:** CI compares every case's full observed outcome with the protected last-green baseline, so a changed answer, a new refusal or a removed case fails unless a register entry states that case's exact old and new outcome.
  This replaces choosing "neighboring cases" by hand.
- **Run time:** keep the PR checks fast and move slow ones to nightly; run time is something to optimize, never a reason to skip cases.
- **Test the tests:** before the L2 runners count for acceptance, show they fail on a corrupted expectation, a missing case, a skipped call and a missing native library.
- **Local first:** reproduce a failure from the installed package before spending a remote run on it; a failure that only happens remotely gets one bounded diagnostic run.
- **Retiring old tests:** the first attempt's internal-equivalence tests go once the L2 and L3 failure families they caught are covered, measured by seeding those failures and checking the new layers catch them.
  Algorithm unit tests stay.

## 7. Release gates

| Gate | Passes when |
|---|---|
| Merge a change | L0, L1, L4, L5 and the L2 subset green; the ratchet holds; a type or conversion fix comes with a public reproduction that failed first |
| Slice done (merges to `main`) | The full L2 for the slice green in both languages; L3 on the agreed seed sets with no unexplained divergence; every difference has an approved register entry; for S2, L6 green for its scenarios |
| Python switch for a path | Its slice done; L6 green for it; the Databricks job matrix equal to the baseline; L7 within its limits |
| Release | Every layer green on the release commit, including L6 and the job matrix re-run after the old code is deleted; the existing Python test suites green; live test resources cleaned up |

## 8. Phases

| Phase | Work | Exit |
|---|---|---|
| R0 | Build the reference environment (decision 7) in its own worktree; record the accepted text-date formats and inference rules; agree the PQL crate's release milestone with the PQL team; verify branch protection, CI and approval identities; push the first attempt's best candidate as a named branch and run CI once, as a fallback; archive the rest; one integration branch | Decisions and dates written down | 1 day |
| R1a | `sdm-contract`, `sdm-testkit`, L0 guards, L2 runners for Python, Java and the oracle, CI tiers | S0 green in both languages; the runners fail on seeded faults |
| R1b | Spike, fixed in scope: DuckDB `from_source` → `materialize` → `predict` → result through the new contract and the data provider interface in both languages, and the same path through a minimal non-SQL test provider, against a scripted model endpoint; the text-date rules recorded in R0 checked against the oracle | Matches the oracle on inferred types, keys, time columns, data requests and SQL, payloads, results, warnings and refusals, over the first attempt's column-type cases; the owner decides go or no-go (go: R1a and R1b merge to `main`; no-go: repair the first attempt in place, or stop) | 3 days, hard stop |
| R2 | S1, moving only the code S1 needs off the Python value model and onto the contract; the provider conformance kit; the time-library and dataframe-engine comparisons | Slice gate |
| R3 | S2 and the Python switch for warehouse paths | Switch gate |
| R4 | S3, moving the rest of the code as its paths need it | Slice gate and switch gate |
| R5 | Switch to the PQL crate release; delete the interim parser and the old engine; release candidate | Release gate |

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
| One status page: candidate commit, gates passed, open failures, decisions needed | Generated from CI and git, not written by the executor |
| Anything that changes behavior or scope goes to the owner the same day | Register entries and the status page |
| A new test or harness tool over about 200 lines needs the owner's approval | Listed on the status page |

**Stop rules:**
- a phase that keeps finding new work instead of closing its exit checks stops for a re-plan with concrete cuts;
- a check failing twice the same way changes method, never adds a phase;
- after two fix rounds on the same divergence, the owner chooses between repairing it, deferring the slice with the old path unchanged, and approving a named behavior change; nothing becomes an accepted refusal automatically;
- three live failures for non-product reasons stop live work;
- if S1 can't go green in both languages, the owner chooses the next step;
- while a blocking question waits for the owner, work narrows to the current slice; silence never authorizes going on.

## 10. Risks

| Risk | Mitigation |
|---|---|
| The Java consumer's environment can't load a native library | Check it with the consumer before Java packaging; running the core in a separate process is the fallback |
| The PQL Rust crate isn't ready when needed | The interim parser, held to the shared conformance suite |
| The C++ PQL core complicates every build and can crash across the boundary | Ask the PQL team for a C interface that never lets exceptions escape, static libraries per platform built for manylinux with private symbols, matching crate and PyPI releases, no aborts or console output, a recursion limit, and the conformance suite in their CI; load and crash tests in L8 |
| A native library builds but doesn't load on some platform | L8 installs the real artifacts in clean containers for every platform |
| The Java consumer's environment disallows the needed runtime flags | Confirm `--enable-native-access` and Arrow Java's flag with the consumer before Java packaging |
| A team swapping the data layer for Pyramid finds the interface insufficient | The interface is documented, a Java sample provider proves it in S1, and the conformance kit defines what a replacement must do |
| Host conversion tables drift between Python and Java | L5 checks both languages convert the same table to the same Arrow types |
| Process grows faster than product again | Rules enforced by CI and the owner, not by the executor; new harness tooling needs approval |
