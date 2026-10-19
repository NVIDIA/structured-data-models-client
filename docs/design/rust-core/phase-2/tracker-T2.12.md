# Phase 2 tracker: T2.12

The design branch owns `tracker.md`; this file holds T2.12's section until it folds in.
Lane e, on the phase head after T2.18's landing (f42d92ea), then the test author's acceptance tests (`T2.12 tests:` commits, from `t212-tests`), then this package (`T2.12:` commits).

## T2.12

T2.12 lifts the five schema rules of E92's Phase 2 class it owns (`float_nan` first, `float_signed_zero`, `key_tie`, `key_name_order`, `composite_key`; `phase-1/not-yet-refusals.md` rows 2, 4, 5, 8 and 9) on DuckDB, SQLite and Postgres, and admits the nine `edit.*` S1 cases as acceptance cases.
The retirement of the interim edit gate is a later landing of its own and is not here.

### What changed

- **`float_nan`, `float_signed_zero`** (`sdm-schema/src/values.rs`): a float preview is read as the reference's pandas frame holds it.
  A NaN is missing, as a null is (pandas' `isna`, `dropna` and `nunique` drop it), and `-0.0` is one value with `0.0` (`nunique` counts them once, measured on the reference venv for float64 and float32).
  So the category and key scores count what the reference counts.
- **`key_tie`** (`infer.rs`): a tie for the guessed key leaves the table keyless and says so, as the reference's `infer_primary_key` warns and returns `None`.
  The warning is `KEY_CANDIDATES_TIED` (table, tied columns in the reference's order), listed in `assemble::WARNINGS`; the Python host words it in the engine's words as its `UserWarning`.
- **`key_name_order`** (`infer.rs`): removed as unreachable.
  The reference adds, for a column named as one form of its table's name, what the forms it meets first give, in an order that varies by run; but no form ever gives such a name anything, since the forms differ only by endings (`s`, `ies`/`y`, `ing`, `info`) and underscores, none of them a key suffix.
  `a_name_equal_to_a_form_scores_four` holds it over every name of up to three characters of an alphabet of those endings, the suffixes' letters and characters whose lowercase changes length; an offline search of 3.4 million names of up to five characters found none either.
- **`composite_key`**, as the reference behaves on every engine:
  - a key of several columns given to `Schema.from_source` is taken as the reference's setter takes one (`assemble::apply_keys_and_times`): each column held by `_check_key` (able to identify a row, not the time column) and marked an identifier, the key the columns in order;
  - it is counted over every row and held to the key rules (`keys::hold`): a row missing any part refused, repeats warned within 1% and refused past it; a key of several columns is named by a `columns` field and its repeated value is the list of its parts, which the Python host shows as the reference's `table.a.b` and tuple;
  - the catalog's keys of several columns are read as no key on all three databases, Postgres included (the reference's Postgres discovery fails unnamed there, so the refusal of P2C-11 is lifted; P2E-4);
  - at materialize, after the query resolves (the reference resolves it against a schema with no key of several columns, so a query whose entity is such a table is refused by the language first), each such table is folded as the reference's engine folds it, in table order: a float among the key's columns or a derived name past 200 characters refused, else the source's fold of the first rows read (a new optional request, `folded_key`, rendered as the reference's `composite_key.sql_expression` on each dialect); then the schema is refused `INVALID_REQUEST` naming the table and the key's columns, as the reference's SQL samplers refuse every such table (their check of reserved names refuses its folded column).
    A provider that declares no `folded_key` (one that is not SQL) is refused without the read.
    The reference compares the source's fold with its own of its preview, which only chooses between two refusals of that code; the core keeps no preview and compares nothing.
- **The key counts' labels** (`describe::count_keys`): each key counted is labelled by its table's place among the tables with a key, a key the source enforces included, as the reference labels it; `TableKey` gains an optional `label`.
  Found by `rule.composite-foreign-key-declared`: an enforced key before a counted one shifted the core's labels, a gated `sql.result` difference on DuckDB.
- **Contract v26**, regenerated (`SDM_CONTRACT_WRITE=1 cargo test -p sdm-contract --test schema_lock`): `TableKey.label` (optional) and the `folded_key` request with its optional-request name; v26 is still the phase's working version, so no bump.
- **FR P3-8**: admission reads run-end values tagged `arrow.bool8` as booleans, as the same column unencoded is, and refuses the tag on other storage (`sdm-host`'s `admission.rs`); a timedelta or a time of day given for an integer key stays refused as a type (P1B-34, which now names the two vectors); `Decimal('1E-7')` into a text key and a float of 2**63 into int64 are already the reference's.
- **Census and refusal maps**: the five probes whose sites are gone leave `contract/refusal-census.json` (`refusal_census.py check` passes); refusal-map citations follow their lines (`empty-null.json`, `nested.json`, `time.json`).
- **Pending list**: 26 of the test author's 32 pending tests pass and leave `contract/tests/t212-pending.json`'s pending; six wait (below).

### Probes

- Reference against core, outside the cases, through the public calls (`Schema.from_source`, then `materialize`) on all three databases:
  - 15 composite-key shapes (given, given without the key check, a few and many repeats, a null part, a float, timestamp, date, bool or decimal part, a key on the entity's table, a one-element list, a column named twice, a missing column, a key over a foreign key): the same outcome, class, code, schema, links and warnings on each database, but the two already registered differences (a one-element list read as the column, P1B-15; a column named twice refused, P1B-73);
  - declared composite primary keys, unique constraints and composite foreign keys: the same on DuckDB and SQLite; on Postgres the reference fails unnamed and the core answers as on the other two (P2E-4);
  - NaN in a feature, in the aggregated column and in a float32, a NaN key candidate, many distinct values with NaNs, signed zeros in a feature, with many distinct values and the census's own probe, ties of two and four candidates, a tie in the entity's table, a key named over a tie, a key named as the table's singular and the census's `key_name_order` probe: the same outcome, schema, links and warnings (class and words, the core's `[CODE]` prefix aside) on each database.
- The test author's 18 cases (9 rule cases, 9 edit cases) on python-core and rust-host against python-ref recorded on this machine, through `compare.py`, on each database (rust-host's Postgres leg replaying python-core's recorded fetches): no difference but the Postgres lines of P2E-4 and `rule.key-tie-entity`'s case file (below).
- The test author's acceptance tests: `test_t212_schema_rules.py` (cases per rule, the composite key pair matrix of 75 vectors on each database, FR P3-8's key casts, the edit cases), `sdm-testkit`'s `t212_schema_rules.rs` and `sdm-host`'s `t212_bool8_runs.rs` pass but the six pending. Two more composite shapes beside them (a decimal part, refused before any read on DuckDB and Postgres and read on SQLite, where the loader stores a decimal as text; a text part holding a backslash): the same statements and outcome as the reference on all three databases.
- L4: the new cases' statements pinned as goldens on each database (`sdm-data`'s `tests/<database>_goldens.rs`), held to python-core's run of the cases there.
- Baselines: the nine rule cases on DuckDB for python-core, python-old and rust-host (`ratchet.py accept`, merged into the committed lines); `ratchet.py check` fails only on the nine edit cases' changed inputs, which P2E-3 names once approved.
- `pytest contract/tests -n 4 --dist loadfile` (this machine's venv, not the ref-pandas3 cell): every failure is one of the environment's (the `test_t136_host2_*` recorded answers, the gate's full-corpus runs judged in another cell), `rule.key-tie-entity`'s case file, the test author's two Postgres regression vectors of P2C-11, the protected tests of the lifted refusals, or KO2-65's exclusion, which waits on the owner's row.
- `cargo test --workspace --no-fail-fast`, the census check, `register.py validate`, the refusal-map test: see the handoff report.

### Deviations and questions

- **P2E-4 (owner)**: on Postgres a catalog key of several columns is read as no key, where the reference fails unnamed; E66(b) approves a named refusal in place of such a failure, not an answer, so the owner decides between this answer (DuckDB's and SQLite's, and so the one sensible match) and keeping P2C-11's named refusal, which the orchestrator called acceptable, as a permanent E66(b) refusal under a rule outside E92's class.
  It refuses no input the reference accepts.
  The two Postgres lines of `rule.composite-key-declared` and `rule.composite-foreign-key-declared` differ from the reference until it is approved.
- **P2E-3 (reviewers)**: the nine edit cases' files changed (status and paths), which the ratchet's case rule holds to an approved entry naming each.
- **`folded_key`**: a read the core makes to match the reference's statements (sql.count and sql.result gate them); its answer chooses nothing, since the reference's comparison of it only chooses between two refusals of one code.
  The owner approved skipping reads whose results go unused on Postgres (KO2-58); if the owner prefers that here too, the request and the read go, and the `rule.composite-key-given` lines need the comparator to take one statement fewer.
- **Composite keys are never sampled**: the reference cannot materialize any schema keyed by several columns on a SQL source, so neither does the core; sampling them would be a change beyond the reference, for the owner.
- **Register refs**: P1B-34 (FR P3-8's two time vectors; for the owner, in a commit of its own, by the orchestrator's decision), P2B-1 (the matrix's five `DECIMAL(12, 4)` foreign vectors on DuckDB, the key sent exact, KO2-35) and P2E-4 (the matrix on Postgres): each a ref under the entry's approved meaning, so each changed entry needs approving again.
- **Waiting**: the three `rule.key-tie-entity` tests (its case file, below) and the three Postgres tests of P2E-4.
- **For the test author**: `rule.key-tie-entity`'s routes lack `query.validate/*` and its core block names a `materialize` the core never starts on DuckDB (compare `refuse.unknown-column`); the protected tests that hold the lifted refusals (`sdm-testkit`'s `t132h_given.rs` and `t136_pre_given_specs.rs`, the client's `test_t132h_keys_and_times.py` and `test_the_schema_route.py`, `contract/tests/test_t132h_given_vectors.py` and `contract/given-vectors.json`'s `key.composite` vectors) and the Postgres regression vectors `key.composite-primary` and `key.composite-foreign`.

### Fix rounds

- Round 1 (the Claude reviewer approved; P3-1): a tied key is said after every table is read, before any left-out column, in the reference's order (`assemble::after_reading`, its unit test and a DuckDB probe).
- After round 1 (the orchestrator's items): P2C-11 retired (rejected, its ID kept), its two Postgres regression vectors named by P2E-4 as `refusal UNNAMED -> success`; the test author's `rule.key-tie-entity` fix and updated composite-key tests taken.
- Round 1, Codex (two P2s), fix 3: a float preview is read as the reference's frame of that source holds it (`DataLayer::reads_nan_as_missing`: Postgres's rows read as NumPy-backed pandas, a NaN missing and the zeros one; DuckDB's and SQLite's as Arrow-backed pandas, every NaN one value and the zeros two); on Postgres a sampled entity whose float key is NaN is refused INVALID_REQUEST before the next read, as the reference's sampler fails to coerce it; on DuckDB and SQLite a context example of such an entity is left out once labelled, as the reference leaves it out (112 context rows against 98, found by the test author); and a float32 key's instance reference and foreign keys are float64 on the wire, as pandas types them. The test author's six regression cases pass on DuckDB and Postgres on both hosts, with their goldens and DuckDB baselines.
- Round 2, the Claude reviewer (approved; P3): floats read as Arrow-backed pandas keep their own bits, so DuckDB's `0.0/0.0` (a NaN with the sign bit set) and `'NaN'` are two values, as the reference counts them; a key tie decided by one NaN of each sign now leaves no key, as on the reference (fix 5, its unit test and a DuckDB probe).
- Round 2, Codex (two P2s): the mixed-sign NaN inference is fix 5's (Codex's DOUBLE and FLOAT probes now give the reference's tie and numerical feature); fix 6 refuses a Postgres NaN key only where an example's key seeds a windowed target's read (`_by_time`), as the reference fails there, and leaves a static query's NaN-keyed example out once labelled, as the reference does (Codex's static probes and his windowed ones, DOUBLE and FLOAT, match on Postgres and DuckDB).
