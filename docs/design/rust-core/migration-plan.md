# Rust core migration: the new plan

**Status: proposed.**
It replaces the first plan once the owner approves its scope, the decisions in section 2 and the process rules in section 9.
The postmortem behind it is [learnings.md](learnings.md).

The goal is the same: one implementation of the client's logic shared by Python and Java, so the two can't drift.
What changes is where the line between the core and each language sits, how the work is sliced, how it is tested, and how the process is kept small.

## 1. What changes from the original proposal

| Original proposal | New plan | Why |
|---|---|---|
| No API change, read as bit-for-bit answers | Behavior is defined by a public contract corpus and a short register of kept Python quirks; anything else changes only by explicit decision | Bit-for-bit copying spread pandas' accidents through the core (learnings 1-3) |
| Core receives query results as Arrow; frames described by the host | Core takes only a closed set of Arrow types, for query results and in-memory frames alike; each language converts its own types through a reviewed table | The core grew six pandas column models and parsed dtype names |
| Core never calls into the host (stated, not enforced) | Enforced: no calls back into the host for language behavior, no Python names, no host facts read implicitly | The implementation called back into Python in 19 ways |
| Port the client logic | Port defined algorithms; use Rust libraries for general data work; keep presentation and object behavior in the host | 27-29% of the Rust re-created libraries by hand |
| Python first, then Java | Each slice ships in Python and Java together, warehouse path first | Python assumptions surfaced only at the end |
| Module-by-module phases | Vertical slices through the public API | Modules passed alone and broke together |

Unchanged: Rust, no I/O in the core, the run loop that asks the host to run SQL or call the model, official database drivers per language, and per-language packaging.

## 2. Decide before writing code

The owner answers these in one sitting, before any code.

1. **Where does the first Java consumer run, and can it load a native library?**
   If yes: JNI over the Rust core.
   If no: check whether a pure-Java WebAssembly runtime can run the same core within the memory it needs (32-bit WebAssembly is limited to 4 GiB; larger memory depends on the runtime).
   A separate Java implementation would give up the shared-code goal and needs explicit approval.
2. **Calendar budget and the Java deadline**, as dates, so stop rules can fire on them.
3. **The reference environment**: the exact old release and its resolved Python, pandas, NumPy and PyArrow versions that define "today's answer".
   The Databricks validation runtimes use pandas 2, and current `main` doesn't include the first attempt's determinism fixes.
4. **Is refusing an input the old client accepted a regression?** Default: yes, unless an approved register entry says otherwise.
5. **The quirk register** (section 5): which Python behaviors to keep and which to drop.
6. **Where PQL lives** (section 3, "PQL").
   Proposed: one PQL, owned by the shared PQL library, agreed with its owners.
   This includes which library that is: the proposed open-source PQL library and Kumo's own PQL package are separate today.
7. **Scope inventories**: the exact backends, scenarios, Databricks runtimes and platforms each language must support.
   For example: is Lakebase in Java's first release?
8. **Who owns the Rust core long term.**

## 3. Architecture

### The contract

- **Inputs:** Arrow record batches restricted to a closed type set (below).
  Each language converts its native types through a reviewed table.
  A conversion must not lose values or the metadata that matters: precision, scale, units, missing values, order.
  Inputs the old client supported stay supported unless the register says otherwise; anything else is refused by name.
  The core never sees a pandas dtype or a Java class.
- **Operations:** a versioned list of typed requests and responses (serde types, published as JSON Schema), not functions looked up by string name.
- **Runs:** the existing `Run`/`Step` loop.
  A query step carries the SQL, its ordered typed parameters, its placeholder style, and either the Arrow schema it expects back or a marker that the schema is being discovered.
  The host returns Arrow or a structured database error.
- **Errors and warnings:** `{kind, code, message, fields, cause}` with a closed list of kinds, one channel for each, and warnings kept in order.
  Each language maps kinds to its own exception and warning classes.
- **Host facts:** clock, time zone, locale and similar facts are passed in explicitly, so tests can pin them.
- **State and memory:** typed, versioned values the host holds; Arrow buffers and runs have defined owners and release rules; no handles into host objects.

### The type set

| Kind | Arrow types |
|---|---|
| Integers | int64, uint64 (hosts widen narrower integers) |
| Floats | float32, float64 |
| Decimals | decimal128, decimal256 |
| Text and bytes | utf8, binary (hosts convert `large_*` without loss, or refuse by name) |
| Time | date32, timestamp(ns or us, with or without zone), duration(ns or us), time64 |
| Nested | list, struct, map, dictionary of utf8; child types come from this set (hosts convert `large_list` and `fixed_size_list` without loss, or refuse by name) |
| Other | bool, null (all-null columns) |

Every accepted type is preserved or refused with a named error, never silently turned into nulls.
R1 validates this set against the first attempt's column-type cases before it is final.

### What lives where

| Concern | Same in every language? | Where |
|---|---|---|
| PQL parsing, validation, error codes and spans | Yes | Core |
| SQL per database, including push-down sampling | Yes | Core |
| Semantic inference over typed columns (keys, time column, links, semantic types) | Yes | Core |
| Sampling plan, neighbor sampler, random streams | Yes | Core |
| Labels and aggregations | Yes | Core |
| Materialization, model checks, decomposition, payload limits, session state | Yes | Core |
| Payload encoding, response decoding, result layout | Yes | Core |
| Native types to Arrow (pandas dtypes, JDBC types) | No | Host, as a reviewed table |
| Object behavior, exceptions, presentation (`to_pandas`, Java records) | No | Host |
| Database drivers, HTTP, authentication | No | Host |

### PQL

Several separate PQL implementations exist today, and only two of them share tests.
Keeping a private copy of the grammar in this client lets it drift from everyone else's.

| Part | Owner |
|---|---|
| Grammar, parser, generic validation, error codes and spans, written date-literal rules | The shared PQL library |
| A versioned, serialized form of a validated query (JSON plus a schema) | The shared PQL library |
| A shared test corpus of valid and invalid queries (seeded with this client's 417 cases) | The shared PQL library, run by every implementation |
| Model capability checks and profiles | This client's core |
| Turning a validated query into tasks, samples, labels, SQL and payloads | This client's core |

The core, the Python package and the Java package must all call the same parser without calling back into another language.
So the library's reference implementation should be a Rust crate, with its Python package a thin wrapper over it.
Until that crate exists, this client's Python uses the library's Python package, and the core keeps its own Rust parser, held to the shared corpus and deleted once the shared crate exists.

### Crates

The names and boundaries below are the target ownership, not a reason to rename or rewrite everything before the first slice works.

| Crate | Holds | From today's tree |
|---|---|---|
| `sdm-contract` | The type set, operations, errors and warnings, run steps | New, folding in `protocol` |
| `sdm-pql` | Model capability checks and profiles over the shared PQL library's validated query; a temporary parser until the library's Rust crate exists | Keep the capability checks; the parser goes once the shared crate exists |
| `sdm-sql` | One module per database | Keep the dialect backends; drop the capture and render scaffolding |
| `sdm-schema` | Inference over Arrow, metric and semantic views | Keep the engine; drop the pandas column models and dtype parsing; YAML through a library |
| `sdm-time` | One calendar, date parsing rules, `DateOffset`, time zones | New, merging today's duplicates; built on a time library |
| `sdm-random` | numpy `RandomState` and `std::mt19937` streams | Keep as is |
| `sdm-sampling` | Sampling plan, push-down driver, neighbor sampler, graph store | Keep the algorithms; one value model |
| `sdm-labels` | Label kernels and windows | Keep; use Arrow compute where answers allow |
| `sdm-request` | Materialization, model checks, decomposition, limits, session state, payload encoding, response decoding, result layout | Keep the logic; drop the CPython sort and `json.dumps` size emulation; split if it grows |
| `sdm-core` | The public operations, built from the crates above | Rewrite the host API; typed operations only |
| `sdm-testkit` | Corpus runner, fixtures, replay | Dev-only; nothing in the shipped crates |
| `sdm-py`, `sdm-jni` | Thin bindings: convert types, map errors, drive runs | Rewrite `sdm-py` slimmer; new `sdm-jni` |

Today's `wire` crate is already generated from the model API; `sdm-request` either uses it or it is deleted.

### Dependencies

| Need | Choice |
|---|---|
| Columnar data and kernels | `arrow` crates, including `arrow-cast`, `arrow-ord`, `arrow-arith`, `arrow-select`, `arrow-row` and `arrow-ipc` |
| Dates, times, zones | One time library, chosen in R1 against the date corpus, including zone rules after 2037: `chrono` (already in the build through Arrow) with `chrono-tz` (to be added), or `jiff` (then two time libraries in the build) |
| YAML (metric and semantic views) | A maintained YAML 1.2 crate such as `saphyr` or `yaml-rust2` (not the archived `serde_yaml`); behavior differences go through the register |
| JSON | `serde`, `serde_json` |
| Bindings | `pyo3` (abi3 wheels), `jni` |
| Dataframe engine | Decided in R1: compare plain Arrow kernels with Polars or DataFusion on one label aggregation and one sampling join; correct answers are required, then weigh hand-written code, maintenance, speed, binary size and JNI loading; record the results |

On the Python side, pandas and pyarrow stay as host dependencies, and the old engine and `predictive_query` go away at the end.

## 4. Delivery: slices, not modules

| Slice | Public path | Languages |
|---|---|---|
| S0 | PQL validation | Python and Java |
| S1 | `Schema.from_source` → `materialize` → `predict` on DuckDB, SQLite and Postgres, offline | Python and Java |
| S2 | The same live on Databricks (with a metric view), Snowflake (with a semantic view) and Lakebase; the Databricks job matrix across runtimes; Python and Java agree on the same data | Python on all three; Java on Databricks and Snowflake, Lakebase if in scope |
| S3 | In-memory frames, ranking, forecasting, link prediction, sessions, tabular requests | Python; Java scope set by the owner |

Python switches a path to the core only after its slice passes all its gates, live ones included.
Warehouse paths can switch after S2; S3 paths stay on the old implementation until S3 passes.
Old paths are deleted in R6, and the oracle (section 6) stays available after that.

## 5. The quirk register

The owner approves the first version once.
Every entry names the old and new behavior, the affected inputs and test cases, and the approval.

| Keep (part of the promise) | Drop (by recorded decision) |
|---|---|
| numpy `RandomState` and `std::mt19937` streams | CPython's sort order around NaN scores (NaN ranks last) |
| Label arithmetic as today: float32 Kahan sums in row order | The size limit counted as `json.dumps` bytes (count the bytes sent) |
| `DateOffset` month-end rules | Python object protocols inside the core |
| PQL codes, spans and grammar quirks | Errors as Python class paths |
| PQL date-literal rules | Silent nulls or internal errors for types the old code mishandled |
| Today's public exception and warning classes and codes | Time zone rules ignored after 2037 |
| Payload equality after parsing | |
| SQL per database, as reviewed golden files | |

Each needs its own decision, not a blanket rule: answers that differ between pandas 2 and 3, SQL text changes, and YAML 1.1 values such as `yes` and `on`.

## 6. Testing

| Layer | What it checks | When |
|---|---|---|
| L1 Unit and property tests | Algorithms in each crate, with every past mismatch seed; the shared PQL corpus | Every PR |
| L2 Contract corpus | One shared list of cases run by a runner in each language: a public call and its inputs in; ordered SQL and parameters, payload, result, warnings and error kind, code and details out | Every PR (subset), nightly (all) |
| L3 Differential generator | Random public-API inputs (type × role × backend × task) compared with the old release, installed from a pinned wheel in its pinned environment (the "oracle"), before and after the old code is deleted | Nightly |
| L4 SQL goldens | Reviewed SQL per database, also executed on DuckDB, SQLite and Postgres | Every PR |
| L5 Conversion tables | Each language's native-to-Arrow table, both pandas majors, no value or metadata lost | Every PR |
| L6 Live matrix | The scenarios in the scope inventory on Databricks, Snowflake and Lakebase, plus the Databricks job matrix; old versus new answers, and Python versus Java on the same data | Weekly and before release |
| L7 Performance | Query counts equal; push-down median over at least five interleaved runs, caches off, within 10% on Databricks and Snowflake | Nightly and before release |
| L8 Packaging and safety | Wheels and JAR (JDK 17 and 21, Linux x86_64 and aarch64) install and load; native memory, panic and leak tests; HTTP retry, timeout and redaction tests | Before release |

- **Budgets:** 10 minutes per PR, 60 minutes nightly.
- **Test the tests:** before the L2 runners count for acceptance, show they fail on a corrupted expectation, a missing case, a skipped call and a missing native library.
- **Local first:** reproduce a failure from the installed package before spending a remote run on it; a failure that only happens remotely gets one bounded diagnostic run.
- **Retiring old tests:** internal-equivalence tests from the first attempt go only once an L2 or L3 replacement is shown to catch their failures.
  Algorithm unit tests stay.

## 7. Release gates

| Gate | Passes when |
|---|---|
| Merge to the integration branch | L1, L4, L5 and the L2 subset green; any type or conversion fix comes with a public reproduction that failed first and tests for the neighboring supported and refused cases |
| Slice done | The full L2 for the slice green in both languages; L3 with no unexplained divergence for two nights; every difference has an approved register entry |
| Python switch for a path | Its slice done; L6 green for it; the Databricks job matrix equal to the baseline; L7 within budget |
| Release | Every layer green on the release commit, including L6 and the job matrix re-run after the old code is deleted; the existing Python test suites green; live test resources cleaned up |

## 8. Phases

| Phase | Work | Exit | Estimate |
|---|---|---|---|
| R0 | Decisions (section 2) and calendar dates; push the first attempt's best candidate as a named branch and run CI once, as a fallback; archive the rest; one integration branch | Decisions and dates written down | 1 day |
| R1 | `sdm-contract`, `sdm-testkit`, L2 runners for Python, Java and the oracle, CI tiers; the time-library and dataframe-engine comparisons; a spike running DuckDB `from_source` → `materialize` → `predict` → result through the new contract in both languages, against a scripted model endpoint | S0 green in both languages; the spike matches the oracle on inferred types, keys, time columns, SQL, payloads, results, warnings and refusals, over the first attempt's column-type cases (populated, all-null, empty, nested, decimal, time, keys); the owner decides go or no-go (no-go: repair the first attempt in place, or stop) | 4-5 days |
| R2 | S1, moving only the code S1 needs onto the contract | Slice gate | 6-8 days |
| R3 | S2 and the Python switch for warehouse paths | Switch gate | 5-7 days |
| R4 | S3, moving the rest of the code as its paths need it | Slice gate and switch gate | 6-8 days |
| R5 | Delete the old engine; release candidate | Release gate | 2-3 days |

About 24-32 working days before contingency.
R0 turns them into calendar dates with a stated hours-per-day assumption, and R1's measurements revise them before the owner approves R2.

## 9. Process rules

- **One branch.** One integration branch and at most one topic branch living under a day; no stacked or parallel phases.
  A candidate that passes its checks is pushed before more work builds on it.
- **Stop rules, acted on by the owner:**
  - a phase at 150% of its estimate stops for a re-plan with concrete cuts;
  - a check failing twice the same way changes method, never adds a phase;
  - after two fix rounds on the same divergence, the owner chooses between repairing it, deferring the slice with the old path unchanged, and approving a named behavior change; nothing becomes an accepted refusal automatically;
  - three live failures for non-product reasons stop live work;
  - if S1 isn't green in both languages by its date, the owner chooses the next step.
- **Reviews.** One implementer and one independent reviewer per change, checking code and behavior, not prose; one independent audit of scope and evidence before release.
- **Approval by meaning.** One decision approves all the files it explains; no per-file hash approvals.
- **Changes to agreed checks.** A phase's acceptance checks are locked before implementation.
  Each change states what it changes and why; a third change in one phase goes to the owner.
- **One status page.** Candidate commit, gates passed, open failures, decisions needed, days used against the approved dates; five lines a day.
- **Ask early.** Anything that changes behavior, scope or dates goes to the owner the same day.
