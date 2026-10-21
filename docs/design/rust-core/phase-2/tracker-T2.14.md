# Phase 2 tracker: T2.14

The design branch owns `tracker.md`; this file holds T2.14's sections until they fold in.
T2.14a, T2.14b, T2.14c and T2.14d land separately, each writing its own section.

## T2.14a

T2.14a widens the bridge freeze (`scripts/rust_core/bridge_freeze.py`, decision E50) to the rest of the core route's Python bridge (FR P3-2, the design's scope item 5), and records Phase 2's ledger of host-held rules, the "before" of the evidence pack.
It is a judging-path package and lands alone, before T2.14b relies on it.

### What changed

- `WHOLE` gains `_rust/codec.py`, `_rust/edits.py`, `_rust/errors.py` and `_rust/providers/postgres.py`: every function, method and skeleton of each is now frozen.
- `COVERED` gains `SDMClient._send_body` in `client.py`, the core's call bytes sent as the core wrote them.
- The docstring's list of frozen files names the four new files.
- `bridge_freeze.json` rewritten with `--write`: 69 entries added, none changed or gone.
- `bridge_changes.json` declares each of the 69 new entries `refactor` ("newly covered by the freeze (FR P3-2), unchanged; no rule added"), as the freeze requires of an entry new against the base.
- A strict mode (the T2.1v review's P3-1 and P3-2): a copy judging its own tree, where the tree's `bridge_freeze.py` is the running file byte for byte (`own_tree`, as `prepush.sh` and CI's `rust-lint` run it), keeps every refusal T2.1v relaxed: a baseline entry for a file or name it does not freeze, and a stale declaration of an entry it does not freeze.
  Only a copy judging another tree's lists, the gate's checkpoint copy, uses T2.1v's tolerances.
- No product code changed.

### Phase 2's ledger (before)

Each row is a rule a core route still runs in the Python host at T2.14a's base (1d85bce3), from the mid-phase review MID2, with the package that moves it into the core and deletes it from the host.
Phase 1's ledger (`phase-1/design.md`, "Host-held semantics") stays in force for its own rows; its rows marked "stays host" are not repeated here.
Paths are under `packages/structured-data-models-client/src/sdm_client/`; a definition's line is its `def`.

| MID2 row | Rule held outside the core | Where, at 1d85bce3 | Owner |
|---|---|---|---|
| P2-4 | The old validator's pick: which validator resolves a query, chosen by a Python column-name check (`schema.py`'s `names_a_query_cannot_say`) | `query.py`'s `PredictiveQuery._resolve` (`:636`, the check at `:662-666`) | T2.14b |
| P2-2 (rest) | The candidate-repeat rule: a candidate equal to one given before it is a repeat by Python's equality, as the reference's `dict.fromkeys` keeps the first; stays the host's (the host language's own equality, sent as the contract's `repeat`), by the orchestrator's decision at T2.14b fix 7 | `_rust/materialize.py`'s `_candidates` | stays host |

### Deviations

- Landed in two parts: T2.1v first (the checkpoint's copy judges only what it freezes; first landed as "T2.14a1", an ID outside the package-ID pattern, rolled back and landed again as T2.1v), since the gate's checkpoint copy of the freeze refused a baseline naming files outside its own `FROZEN`, so no candidate could widen the freeze; T2.14a then, rebased onto T2.1v's landing.
Packages landing after T2.14a (T2.11, T2.13b) declare their changes to the newly frozen files in their next fix round, as the brief says.


## T2.14b

T2.14b is the host clean-up (the design's T2.14b row), with T2.14d (the Postgres value rules, MID2 P2-3) and the decimal stype fix riding along in the same landing, by the owner's granularity decision.
Its tests are the test author's (`contract/tests/test_t214b_*.py`, `rust/crates/sdm-testkit/tests/t214b_host.rs`, `rust/crates/sdm-schema/tests/t214b_decimal_stype.rs`, `packages/structured-data-models-client/tests_test_ops/test_t214d_postgres_values.py`), gated by `contract/tests/t214b-pending.json`.

### What changed

- **Response bytes untouched.**
  The HTTP transport's `predict_body` gives back the endpoint's answer as the bytes it sent, and the Databricks transport's `predict_body` the served model's answer text; `SDMClient._send_body` hands either to the core as it is.
  `_rust/predict.py`'s `_encoded`, `_ENCODED_DEPTH` and the made-up too-deep body are deleted.
  An answer the reference's transport refuses as not a JSON object is still refused by the transport (`TRANSPORT_ERROR`, as the reference); one nested past the interpreter's reader goes to the core, which refuses it as too deep (register P1B-22).
- **Facts from the core.**
  `_rust/materialize.py`'s `asked` reads the target's windowing and entity table from the core's `Materialized.query` for every query.
  A refused call's forecast step and ranking candidate are the core's `forecast_step` and `ranking_candidate`, put on the raised refusal by `_rust/run.py`'s `_of_the_part`; the host's call counter (`_forecast_step`, `_ranking_candidate`) is gone.
  The candidate repeats are the core's (`sdm-request`'s `ranking::distinct`, Python's equality by key): the host passes each candidate as given and no longer writes `repeat`.
- **State version.**
  An unknown `RequestState` version is refused by name: the core adds the version it reads (`reads`), and the host says both.
- **Wording (KO2-32 and the carried notes).**
  The core decides each refusal's code and fields; `_rust/errors.py`'s `WORDED` and `SAID` word them in the reference's words, with no rule beyond the tables.
  G15: `describe_source`'s and `edit_schema`'s refusals carry a `reason` and what the reference names (the tables or the columns the table has, the data type on the wire, the meaning, the link's key).
  The generic refusals at materialize and predict (the sampling options, the anchor rules, a missing exclusion, the hops, the seed, a one-class binary target, a text candidate on an integer key, a non-finite target) carry a `reason`, the table, the value given as the host wrote it and what the graph has, and are worded as the reference words them.
  M4: `CONTEXT_BEFORE_DATA` and `ANCHOR_AFTER_DATA` name their times and are said, with `MISSING_INDICES`, in the reference's words.
  FX-1: the core gives `FLOAT32_OVERFLOW` where a finite double becomes infinite in a target's or an aggregate's float32 cast (not on Postgres, which the reference reads as nullable `Float64`), shown as a `RuntimeWarning` in pandas' words.
  FX-4: a whole table's smallest timestamp that is DuckDB's infinity is shown as `datetime.max` or `datetime.min`, as the reference's `fetchall` hands it over.
- **T2.14d.**
  `sdm-data`'s new `pg_values.rs` holds the Postgres value rules (a JSON value as the reference writes it, a `numeric` past 76 digits by its declaration, a key set's zone-less times and binary as the reference binds them); `sdm-core` applies them to each Postgres fetch (`DataAccess::conform`, `DataAccess::bound_keys`).
  The Postgres provider only moves the driver's rows into Arrow: `json_text`, `_interval`, `_after_read`, `_Numeric`, `key_set_json` and `_isoformat` are gone.
  No statement's text changed, so the rust-host Postgres replay is not re-recorded.
- **Decimal stype.**
  `decimal_to_f64` moved from `sdm-engine`'s `labels.rs` into `sdm-contract`'s `scalar.rs`; `sdm-schema`'s inference reads a `Decimal128` of scale above 0 as that double, so stype counting, key scoring and distinct examples count what the reference's cast counts.
  Materialize and predict keep exact decimals (P2B-1).
- **Ledger.**
  The rows moved (P2-4's facts, step numbering and ranking split; P2-3; the answer re-encoding) are deleted from the ledger above; P2-2's repeats returned to the host at fix 7 and stay in it, host-held.
  The moved rows' bridge entries are declared `moved` naming T2.14b in `scripts/rust_core/bridge_changes.json`, and `bridge_freeze.json` is rewritten with `--write`.

### Deviations

- The old validator's pick (`PredictiveQuery._resolve`) stays in the ledger: the core route still builds the old validator's query object for the materializer's call, though no fact is read from it any more.
- Round 1 (KO2-86, the owner's "Echo own args"): an error may name a value of the caller's own call, so twelve of the thirteen keys first left pending now pass: the date literal and the query (`date_literal_packed`'s `literal` and `query`), the keys a ranking's question asked that the entity table lacks (`not_in_table`'s `keys`), and a key or time column no contract type holds, given spelled (`Given::Spelled`, contract 27's working version) and named as `given`.
  `key.uint64-above-int64-max` stays pending: the value its reference message names (row 0's `9223372036854775810`) is a context example's key read from the data, not one the caller passed, which H17 still keeps out.
- P1B-75 (the sampler's two warnings in the core's words) is retired by the orchestrator's decision, the core now saying them in the reference's words.

### Fix 1 (round 1 reviews)

- Claude P2-1: a declared `numeric`'s unreadable text is refused naming the column only, never the cell (`pg_values.rs`); no other message in `sdm-data` or the three providers formats a value read from data.
- Codex P2-1: a date, a time of day or a timestamp candidate is given as `{"typed": {"text", "type"}}` (`GivenCandidate::Typed`), so the core holds two of one kind and text one candidate, as Python's equality and `dict.fromkeys` do; repeated dates make the reference's parts and calls.
- Codex P2-2: KO2-89 (the orchestrator, under the owner's overnight delegation) lets M4's warnings and refusals name the data's earliest and latest times, and a non-finite value's refusal `inf` or `-inf`; noted on P1B-75.
- Codex P2-3: a Postgres `json` document escaping a lone surrogate is kept as the reference keeps it: the provider writes the driver's value as ASCII JSON, and `sdm-data` holds JSON text as UTF-16 code units, writing each unit back as the reference's writer escapes it.
- Codex P3-1: `stype = 5` is said as "5 is not a semantic type", the spelled refusal now carrying its variant.
- Claude P3-2: FX-1 also warns where an aggregation's `WHERE` reads a double past float32's range (the entity filter and ASSUMING already did, through the labels' build).
- Claude P3-6: a `json[]` column's elements are marked as JSON and handed over as JSON text, so `sdm-data`'s list rule reads them.
- Claude P3-7: the 2,000-character cap on a spelled value is registered (P2D-2, proposed; the ID is from lane D's block).

### Fix 2 (round 2 reviews)

- Codex P2-1: a timestamp candidate carries the point in time Python's equality compares (`at`: a zoned one's UTC instant, a zone-less one's wall clock, in nanoseconds), beside its kind and text, so two zoned timestamps of one instant are one candidate, the first one's text written into the question.
- Claude: the typed candidate's kind is a closed set in the contract (`TypedKind`: `date`, `datetime`, `datetime_zoned`, `time`); the contract's reader refuses another by name.
- Claude P3-3: a Postgres `json` cell whose whole value is a string holding a lone surrogate is refused by name (`VALUE_OUT_OF_RANGE`, rule `text_not_utf8`), where the reference fails unnamed (register P2D-3, proposed); in a `json[]`, a column the core leaves out where the reference refuses the table (register P2D-5, proposed), such an element is missing.
- Claude P3-4: `key.uint64-above-int64-max`'s wording without the data's value is registered (P2D-4, proposed, under KO2-86).

### Fix 4 (round 3 reviews)

- Codex P2-1: a temporal candidate's equality is Python's own, never the core's: the host groups the candidates as `dict.fromkeys` does, by Python's `==` and `hash`, and gives each date, time or timestamp the position of the first candidate equal to it (`first`); the core keeps the first of each group and its text. A host that sends no `first` gets the core's rule by kind and text. The `at` instant is gone, and with it the bare `OverflowError` of an aware timestamp at the end of the date range (Claude r3 P3-1); the fold and aware-time cases (Claude r3 P3-2) group as Python groups them.
- Claude r3 P3-3: P2D-3 says the reference refuses a table with a `json[]` column; that older difference is registered (P2D-5, proposed).
- Claude r3 P3-4: P2D-4's note names the NAMED pin as it now stands.

### Fix 5 (the floor cell)

- CI's `floor-pandas2.2` cell pins NumPy 1.23.5, which does not warn when a cast to float32 overflows; NumPy warns from 1.24. The core keeps reporting `FLOAT32_OVERFLOW` on every runtime, and the Python host shows it as `RuntimeWarning` only where the installed NumPy is 1.24 or later (`_rust/errors.py`'s `numpy_warns_on_cast_overflow`), as the reference's NumPy shows it: the orchestrator's decision under KO2-88, listed for the owner.

### Fix 7 (round 4 review)

- Claude r4 P2-1: the context-before-data refusal carries `window_outran_the_data`, as the reference's client marks it, so a forecast's step is said with the old route's whole message, its hint sentence included.
- Claude r4 P3-2 and P3-3 (the orchestrator's decision): `first` is gone from the contract before contract 27 ships. Which candidates repeat is Python's own again: the host runs `dict.fromkeys`'s grouping over every hashable candidate, by Python's `==` and `hash`, and sends each later one as the contract's `repeat`, which the core leaves out; a typed candidate keeps its closed set of kinds, and two of one kind and one text are one candidate where a host sends no `repeat`. This also groups a `pandas.Timestamp` with an equal `numpy.datetime64`. MID2 P2-2's row is therefore not moved into the core: its rule stays the host's, by the host language's own equality, with no field carrying it. No position index remains in the contract (P3-4).
- Claude r4 P3-1: CI's reference-cell Postgres job runs `contract/tests/test_t214b_regressions.py -k json`.

### Fix 8 (round 6 review)

- Codex r6: after the host's grouping, the core still held two typed candidates of one kind and text one, merging values Python keeps distinct (one wall time in New York and in Toronto). The host now sends a temporal candidate Python holds distinct from an earlier one of the same kind and text as `spelled`, which the core merges with none; `repeat` stays for values Python holds equal.

### Carried to Phase 3 (P3)

- Decimal stype on Postgres: inference reads a scaled decimal by Arrow's `ToDouble`, where the reference's Postgres double is `float(Decimal)`, correctly rounded (0.700 as 0.7000000000000001 here, 0.7 there); distinct counts agree but near collisions (Claude r1 P3-3).
- `core/transport.py`'s `predict_body` still decodes each answer to keep the transport's not-a-JSON-object refusal and the depth check, so an answer is parsed twice; the core could own that refusal (Claude r1 P3-4).
- A custom transport with no `predict_body` is served through `json.dumps`, which recurses: a very deep answer raises `RecursionError` there rather than the core's `too_deep` (Claude r1 P3-5).
- Typed candidates and the core's equality: the core could hold typed candidates never equal to each other, every host grouping its own with `repeat`, which would retire both fix 8's collision check (a distinct value of a repeated kind and text sent `spelled`) and the core's kind-and-text rule. It changes what contract 27 promises a host that does not group its candidates, so it belongs with Phase 3's Java host work (the orchestrator, T2.14b fix 8).

## T2.14c

T2.14c lowers the bridge freeze to what the core route still enters, and carries four items that ride along (the orchestrator's rides file of 2026-10-21): the whole-phase review's Postgres numeric-key finding, `labels::compare`'s deletion, and the owner's register answers and overnight delegation.

### What changed

- The freeze lowered, measured, not guessed: python-core ran every case on SQLite and Postgres (each call forced onto the core) and every DuckDB case whose routes take each of its calls to the core (275 of 315), under coverage, on T2.14b's final candidate ab74bebc (the same list as on its earlier f16493e7).
  Each `COVERED` function whose body ran in none of them is out of `COVERED`, declared `gone`, the baseline rewritten with `--write`: 29 functions, and `base.py`, `result.py` and `decompose.py`, which then held none, leave `FROZEN` (their `<module>` declared `gone`).
  Out: `check_request_against_model`; `client.py`'s `_empty_answer`, `_names_its_rows`, `SDMClient._send`, `SDMClient._spec`; `parts_of`; `PredictiveQuery._in_parts`; `PredictionResult.merge`; and in `materializers/relational.py` `_gather`, `_check_shape`, `_refuse_a_name_the_wire_reserves`, `_run_mode_for`, `_unreserved`, `_warn_two_whole_numbers_are_being_asked_as_quantities` and the module functions `_columns_to_leave_out`, `_drop_context_that_answers_the_question`, `_entities_at_anchors`, `_instances_matching`, `_joins_nothing`, `_positions`, `_record_the_statistic`, `_remove_instances`, `_statistic_asked_for`, `_tables_sent` and the five module-level `_warn_*` functions.
  No function is deleted: the old route still runs each.
- Kept in `COVERED` though the core route no longer enters them, because `selftest.py` seeds faults in them: `SDMClient._prepare`, `RelationalMaterializer._check_limits`, `_tables_linked_to`, `RelationalAdapter._preflight` and `Schema.check_consistent` (the last already kept for that reason).
  The brief's likely list held: `_in_parts`, `parts_of` and `merge` are out; `SDMClient._answer_in_parts` is still entered on the core route and stays.
- One number spelled two ways is one graph key (rides item 6): a Postgres `numeric` declared without a precision reaches the core as its exact text, and the walk took `1.0` and `1.00` for two keys, so the second hop failed with `INTERNAL_ERROR` where the reference answers.
  The walk and the entity sample key such a column by the number its text writes (`sdm_engine::keys::read_numbers`, `Key::number_text`), exactly, with no float: `1.0` and `1.01` stay two keys past 2**53.
  A foreign key is read as the key it refers to is read.
  The number text parser moves into `sdm_contract::data` (`written_number`, `exact_number_text`), which `sdm-testkit`'s non-SQL provider uses to join such a column by value, as Postgres does.
  Register: P2G-4 (caller-keyed, P2G-2's text kept) and P2G-5 (declared keys: one key where the reference reaches each user twice), decided by the orchestrator under KO2-88.
- `labels::compare` deleted (rides item 1): only tests called it since T2.10; the reference's 480 condition vectors now run through `labels::label` (the test author's commit).
- KO2-87 (P2D-1, P2J-1 and P2J-2 approved by the owner) and KO2-88 (the owner's overnight delegation of 2026-10-21), with the three entries approved citing KO2-87.
- P2D-2 to P2D-5, which T2.14b proposed, approved by the orchestrator under KO2-88 and listed for the owner's reading.
- Part B of T2.16a (rides item 7), prepared by T2.16a's implementer: `float64/target` back on the eight cases coverage now places, their python-ref lines and T2.13 option vectors re-recorded at 43e55f4f, and P2O-1 proposed for the reviewers.
- Round 1 (Claude) P3s: `key_values`' comment says the reference splits a declared key's spellings (P2G-5); KO2-87's entries' notes no longer open with "Approver: reviewer"; T2.14b's section no longer lists P2-2 among the rows moved.

### Deviations

- `approved_by` for P2G-4 and P2G-5 is `reviewer`: the register's schema takes only `owner` or `reviewer`, the decision is the orchestrator's under KO2-88, and each entry's notes say so.
- The freeze lowering lands with the product items, not alone: T2.16a's fix 4 taught the gate's checkpoint copy of the freeze to take an entry the candidate declares `gone` and leaves out of its baseline, so the gate judges this candidate's lower freeze with the checkpoint's `COVERED`.
