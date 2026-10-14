# Named "not yet" refusals in the Rust core

Recorded at T1.32d from the orchestrator's survey of 2026-10-13, read from the code at `1efd5882` (line numbers are as they were there). The owner approved the rules this table proposes for Phase 2 as one class on 2026-10-13 (decision E92): each stays refused by name, `UNSUPPORTED_REQUEST` with its rule, until Phase 2. Not in that class, by the owner's decisions the same day: `time_whole_table`, a bug T1.32e fixes, and `multicategorical`, which T1.32e ports. The rules proposed for a T1.32 family or for T1.33 are not approved by this; each waits for its own entry. T1.32e part 1 (E83, E84) lifted `time_whole_table` and, moved out of the class by the owner the same day, `key_missing`, `key_repeats` and `keyless_neighbors`: the class holds 37 rules from then. T1.32b part 1 (E100) lifted `wire_dtype` for a decimal of any precision, which it now names as the reference does (only a negative scale, and the non-decimal types T1.33's DataType pass reads, stay refused by it), and `target_type`'s decimal average (row 47's `labels.rs:227`) and a static decimal compared in a condition; the static decimal target stays `target_type` under P1B-18, now refused where it is labelled (`sdm-engine/src/labels.rs`, which `decimal.json` cites), and a float compared with a static decimal of a scale past 22 is `condition_value` (P1B-71). Row 47's other sites (a UInt64 past int64, the multiclass floats, a time, predict's float target) are not decimal and stay unowned, for T1.33. The class, the 40 rules the survey gives Phase 2 but those two: `composite_key`, `float_nan`, `float_signed_zero`, `key_missing`, `key_repeats`, `key_name_order`, `key_tie`, `aggregation`, `anchor_entity`, `assuming`, `condition`, `context_anchor_time`, `entity_filter`, `entity_ids`, `evaluate`, `exclude_cols_dict`, `explain`, `keyless_neighbors`, `lag_timesteps`, `logical_target`, `max_pq_iterations`, `num_hops`, `num_neighbors`, `open_window`, `parent_table`, `problem_type`, `return_embeddings`, `static_target`, `target_filter`, `target_transform_null`, `inference_config_text`, `text_time_rules`, `top_k`, `unseeded`, `use_prediction_time`, `instance_id_type`, `model`, `prediction_statistic`, `prediction_type`, `response_field`.

Below is the survey as written.

- There are 61 distinct `not_yet("…")` rule names in `rust/crates/*/src` at origin/rust-core-phase-1 1efd5882. One more named refusal, `schema_edited`, is built by a separate constructor; it is listed at the end and not counted.
- Proposed owners: T1.32a 3, T1.32b 2, T1.32c 1, T1.32d 1, T1.32e 0, T1.32f 0, T1.33 12, Phase 2 42. Only the 4 rules a refusal map owns have hard evidence. id_type under T1.32a is an inference.
- 42 rules need the owner to approve them as one Phase 2 class. Two of them, `time_whole_table` and `multicategorical`, need an explicit "T1.32e or Phase 2" decision first (checkpoint 8, P2-5), because T1.32f's real-data run will almost certainly hit them.
- Unregistered divergences: 0 by rule name. Every name appears in P1B-2, P1B-3 or P1B-4, and all three are still `proposed`. By trigger there is 1: `column_type` also refuses binary and duration columns, which the reference leaves out with a warning, and P1B-2 does not describe that trigger.
- Other mismatches with the register, found by reading the code (nothing was run): `time_type` is still in the code although P1B-3 says it was lifted at T1.25; `seed_size` gives UNSUPPORTED before the INVALID_REQUEST that approved RS-7 asks for; decimal.json cites `materialize.rs:804` for the static-decimal refusal, which is actually at `:868`.

## The survey: proposed owners

Base: `origin/rust-core-phase-1` at `1efd58827a36`, read in a detached checkout. Reference: the `rust-core-reference` branch at `b413e9ad`, the commit that `contract/reference/` is named after. Every finding below comes from reading the code and docs; nothing was executed.

## How the rules were found

1. The constructors: `grep -rn 'fn (not_yet|unsupported)' rust/`. Three constructors build `UNSUPPORTED_REQUEST` with a `rule` field:
   - `sdm_engine::not_yet` (`sdm-engine/src/lib.rs:37`), for operation `materialize`;
   - `sdm_schema::not_yet` (`sdm-schema/src/lib.rs:55`), for `describe_source`;
   - `sdm_request::predict::not_yet` (`sdm-request/src/predict.rs:172`), for `predict`.

   `sdm-request`'s payload and options modules call `sdm_engine::not_yet`. `unsupported_type`, `admission::unsupported` and `predict::unsupported` carry no `rule` and are left out.
2. The rule names: `grep -rhoE 'not_yet\("[a-z_]+"\)' rust --include='*.rs' --exclude-dir=tests | sort -u` gives 61 names. The two names chosen by a conditional, at `infer.rs:563-567`, are `text_time` and `text_time_order`, which the grep already finds elsewhere.
3. Other `"rule"` fields: `grep -rn '"rule"' rust --include='*.rs'` (outside tests) adds `schema_edited` (`sdm-schema/src/edit.rs:77`, UNSUPPORTED) and `date_literal_packed` (`sdm-core/src/pql.rs:216`). `date_literal_packed` is `INVALID_REQUEST`, not a "not yet" refusal, so it is left out.
4. The Python bridge: its only named refusal is `_not_taken('key_type')` (`_rust/materialize.py:219`), and the same rule name is in the core too.
5. Map ownership: the `entries[].rule` and `owner` fields in `contract/refusal-maps/*.json`. Four `not_yet` rules have an owner:
   - `key_type`: T1.32a (keys.json);
   - `target_type`: T1.32b (decimal.json);
   - `column_type`: T1.32c (nested.json);
   - `empty_table`: T1.32d (empty-null.json).

   No map probe's first refusal is any other `not_yet` rule. Every other inventory probe reaches predict or another refusal code.
6. Register: a word-boundary search of each name in each `register.yaml` entry. Entries where the name appears only as an ordinary English word were dropped. All 61 names appear in P1B-2 (`register.yaml:1371`), P1B-3 (`:1459`) or P1B-4 (`:1621`). All three are `status: proposed`, `approved_by: null`.

## Owner rule used

Each rule was assigned by the first of these tiers that applies:

1. A refusal map entry owns it.
2. A design or tracker row ties it to a family. This is cited in the row; where it is my inference it is marked "(inferred)".
3. The reference itself refuses or crashes on the input, or the rule is a deliberate fail-closed refusal or an internal guard. These go to T1.33, which reconciles them and does not lift them (design.md:487).
4. Everything else is Phase 2. The evidence is:
   - design.md:41: ranking, forecasting, link prediction and tabular requests are out of Phase 1;
   - E64(a) (decisions.md:104) and E72(c) (decisions.md:112): "every rule no map needs stays refused by name";
   - T1.32f's exit (design.md:486): "anything outside Phase 1 is a named refusal for Phase 2".

"Ref accepts" is taken from the register entry's `old_behavior` or notes, or from a code comment that cites the reference, unless the row says otherwise. Paths are relative to `rust/crates/`.

## Table

| # | Rule | Where (file:line) | What triggers it | Ref accepts? | Register | Proposed owner | Evidence | Flag |
|---|---|---|---|---|---|---|---|---|
| 1 | `column_type` | sdm-schema/src/infer.rs:450, :453; sdm-schema/src/values.rs:99 | A struct or map column; a list whose items are not int, float or text; any other admitted type that `dtype_of` does not map, such as binary or duration | Struct, map and fixed-size list: **no**, an unnamed RuntimeError (P1B-10; infer.rs:441-444 cites P1B-9). Binary and duration: **yes**, the reference leaves the column out with a warning (the reference's `sdm_client/schema.py:1639-1660`; `duckdb/table.py:276-277` maps BLOB to binary) | P1B-2 (proposed), P1B-10 (approved) | **T1.32c** | nested.json entry `describe_source/column_type`, owner T1.32c | **Unregistered by trigger**: P1B-2 names only struct, map and odd lists. Admission admits Binary and Duration (sdm-host/src/admission.rs:111, :119), and they then reach :453. Scalar types beyond nesting are T1.33's DataType classification |
| 2 | `composite_key` | sdm-schema/src/assemble.rs:385; sdm-engine/src/graph.rs:95 | `primary_keys` gives a table a key of two or more columns | yes (P1B-3 old: "composite keys"; P1B-2) | P1B-2, P1B-3 (proposed) | Phase 2 | No `key.*` case uses a composite key (keys.json cases); E72(c) "no map needs" | |
| 3 | `empty_table` | sdm-schema/src/infer.rs:261 | A table with no rows | **no**, an unnamed RuntimeError (P1B-11) | P1B-2 (proposed), P1B-11 (approved) | **T1.32d** | empty-null.json entry `describe_source/empty_table`, owner T1.32d | |
| 4 | `float_nan` | sdm-schema/src/values.rs:107 | A float column whose preview holds a NaN | yes (P1B-2 notes: "each is a refusal of an input the reference accepts") | P1B-2 (proposed) | Phase 2 | No case reaches it; the `null.*` cases use NULL, not NaN | Real data likely to hit it (inferred) |
| 5 | `float_signed_zero` | sdm-schema/src/values.rs:118 | A float preview holding both 0.0 and -0.0 | yes (P1B-2 notes) | P1B-2 (proposed) | Phase 2 | No case reaches it | |
| 6 | `key_missing` | sdm-core/src/describe.rs:418 | A key the source does not enforce holds a null | partly: the reference warns, declines or refuses by its own rules (P1B-2 old) | P1B-2 (proposed) | Phase 2 | No case reaches it; `null.some-foreign-keys` nulls a foreign key, not a key | Real data likely (inferred) Lifted at T1.32e part 1 (E83, E84). |
| 7 | `key_repeats` | sdm-core/src/describe.rs:421 | A key the source does not enforce repeats a value | partly, as `key_missing` (P1B-2 old) | P1B-2 (proposed) | Phase 2 | No case reaches it | Real data likely (inferred) Lifted at T1.32e part 1 (E83, E84). |
| 8 | `key_name_order` | sdm-schema/src/infer.rs:657 | A key guessed by name where the column equals one form of the table's name and another form also scores | yes, but its answer varies from run to run (infer.rs:653-655) | P1B-2 (proposed) | Phase 2 | No case reaches it | Could instead become a permanent refusal on determinism grounds; owner's call |
| 9 | `key_tie` | sdm-schema/src/infer.rs:674 | Two key candidates tie on score | yes (P1B-2 old: "a tie between key candidates") | P1B-2 (proposed) | Phase 2 | No case reaches it | |
| 10 | `key_type` | sdm-schema/src/infer.rs:393; sdm-engine/src/keys.rs:68 (plus the bridge, `_rust/materialize.py:219`) | describe: a key whose dtype is not int, float or string. Engine: an entity key array that is not int, uint or Utf8 (decimal keys) | Decimal and HUGEINT keys: **no**, INTERNAL_ERROR (P1B-1; keys.json "why"). Key of another dtype at describe: yes (P1B-2) | P1B-2, P1B-3 (proposed) | **T1.32a** | keys.json entry `materialize/key_type`, owner T1.32a; design.md:273, :481 move the entities' cast into the core | |
| 11 | `multicategorical` | sdm-schema/src/infer.rs:590 | Text whose first 500 values contain `;`, `:`, `\|` or a tab | yes (P1B-2 old: "text that holds … several categories"; at T1.25b, 36 corpus columns were refused this way) | P1B-2 (proposed) | Phase 2, **owner to decide T1.32e or Phase 2** | Checkpoint 8 P2-5: no task owns it; no `feature.*` or `time.*` case reaches it | T1.32f will very likely hit it: any text holding times ("10:30") or URLs is refused (inferred) Lifted at T1.32e part 2a (E86): the reference's detection ported. |
| 12 | `text_time` | sdm-schema/src/infer.rs:563-566, :575, :930; sdm-core/src/materialize.rs:2202 | Text the core's fixed format list does not read but that could make inference's score, or a value outside the list at materialize | yes (DT-15) | DT-15 (approved, E68), DT-17 (approved), P1B-2, P1B-3 | T1.33 | An owner-approved permanent design (E66(h), E68): a fixed list of formats with no pandas emulation. Nothing to lift | |
| 13 | `text_time_order` | sdm-schema/src/infer.rs:563-567, :576, :929; sdm-core/src/materialize.rs:2203 | Day and month order left open by the values | yes, the reference reads one order (DT-16) | DT-16 (approved, E68), P1B-2 | T1.33 | Approved permanent refusal (E68) | |
| 14 | `text_time_zones` | sdm-schema/src/infer.rs:577; sdm-core/src/materialize.rs:2204 | Mixed UTC offsets in a text time column | partly: the reference refuses 6 of 14 unnamed (DT-11; P1B-2 measurement) | DT-11 (approved), P1B-2 | T1.33 | Approved under E66(b)'s class (tracker.md:138) | |
| 15 | `time_whole_table` | sdm-schema/src/infer.rs:809 | A table with ≥1,000 preview rows (`PREVIEW_ROWS`, infer.rs:29) and any timestamp column that is not the key. `choose_time_column` runs unconditionally (infer.rs:398), so this fires even when `time_columns` is given | yes: the reference reads the whole table (P1B-2 old: "a preview of a whole 1,000 rows") | P1B-2 (proposed) | Phase 2, **owner to decide T1.32e or Phase 2** | Checkpoint 8 P2-5. No case reaches it: inventory tables have at most 80 rows (inventory `bases`), and AdventureWorks is held to ≤1,000 rows in all (fixtures/adventureworks/README.md:39) | **Blocks real data**: almost every RelBench table (10^5 to 10^7 rows) would be refused at `Schema.from_source` (inferred from the code path) Lifted at T1.32e part 1 (E83, E84). |
| 16 | `aggregation` | sdm-core/src/materialize.rs:462 | An aggregation other than SUM, COUNT, AVG, MIN or MAX | yes (P1B-3) | P1B-3 (proposed) | Phase 2 | E72(c) lists "other aggregations" as no map needs | |
| 17 | `anchor_entity` | sdm-core/src/materialize.rs:1270 | `anchor_time='entity'` | yes (P1B-3 old) | P1B-3 (proposed) | Phase 2 | E61(b); no case uses it | |
| 18 | `assuming` | sdm-core/src/materialize.rs:404 | PQL `ASSUMING` | yes (P1B-3 old) | P1B-3 (proposed) | Phase 2 | No case uses it | |
| 19 | `average_null` | sdm-core/src/materialize.rs:516 | `AVG(...) = NULL` or `!= NULL` | **no**, INTERNAL_ERROR (P1B-21) | P1B-21, P1B-3 (proposed) | T1.33 | E66(b) class (a reference crash becomes a named refusal); P1B-21 waits for approval | |
| 20 | `condition` | sdm-core/src/materialize.rs:485, :488, :498, :502; sdm-engine/src/labels.rs:397, :504 | IN, LIKE and similar; a non-numeric constant; a text or boolean target compared; a condition of a condition; NULL compared with `<` or `>` | yes (P1B-3), except NULL with an ordering operator: unknown (labels.rs:496-498: "another comparison with NULL the executor does not make") | P1B-3 (proposed) | Phase 2 | E72(c) | |
| 21 | `condition_value` | sdm-core/src/materialize.rs:493, :496; sdm-engine/src/labels.rs:393 | A constant the target's type cannot read; an integer past int64 compared with an integer column | **no** for int64 overflow ("where the reference raises", P1B-3); other forms unknown | P1B-3 (proposed) | T1.33 | The reference raises, so it is E66(b)-class; it has no entry of its own | |
| 22 | `context_anchor_time` | sdm-core/src/materialize.rs:1240 | That sampling option given | yes (P1B-3 old) | P1B-3 | Phase 2 | E61(b) | |
| 23 | `entity_filter` | sdm-core/src/materialize.rs:409 | `FOR EACH` that is not a plain column (an entity filter) | yes (P1B-3 old) | P1B-3 | Phase 2 | No case uses it | |
| 24 | `entity_ids` | sdm-core/src/materialize.rs:407 | Named entities in the query | yes (P1B-3 old) | P1B-3 | Phase 2 | No case uses it | |
| 25 | `entity_key` | sdm-core/src/materialize.rs:1596 | The query's entity column is not the entity table's key in the graph | unknown. The ported PQL validator refuses a non-key entity first (sdm-pql/src/validator/predictive_query.rs:188), so this may be an internal guard | P1B-3 | T1.33 | Not shown to be reachable | |
| 26 | `evaluate` | sdm-core/src/materialize.rs:395 | PQL `EVALUATE` | yes (P1B-3 old) | P1B-3 | Phase 2 | No case uses it | |
| 27 | `exclude_cols_dict` | sdm-core/src/materialize.rs:1241 | That option given | yes | P1B-3 | Phase 2 | E61(b) | |
| 28 | `explain` | sdm-core/src/materialize.rs:398 | PQL `EXPLAIN` | yes | P1B-3 | Phase 2 | No case uses it | |
| 29 | `indices_type` | sdm-core/src/materialize.rs:1775 | The indices' Arrow type differs from the sampled entity key's type | yes for anything pandas would cast. P1B-3 says the bridge's cast makes it unreachable from Python today | P1B-3 | **T1.32a** | design.md:273, :481: the entities' cast to the key's type moves into the core under T1.32a | |
| 30 | `keyless_neighbors` | sdm-engine/src/sample.rs:174 | Neighbors read from a table with no primary key | yes (P1B-3; the T1.24b differential saw 5 such sources, tracker.md:233) | P1B-3 | Phase 2 | E64(a): "stays refused (no map needs them)" | Seen in differentials; real data likely Lifted at T1.32e part 1 (E83, E84). |
| 31 | `lag_timesteps` | sdm-core/src/materialize.rs:1242 | That option given | yes | P1B-3 | Phase 2 | E61(b) | |
| 32 | `logical_target` | sdm-core/src/materialize.rs:524 | AND, OR or NOT in the target | yes | P1B-3 | Phase 2 | E72(c) | |
| 33 | `max_pq_iterations` | sdm-core/src/materialize.rs:1243 | That option given | yes | P1B-3 | Phase 2 | E61(b) | |
| 34 | `null_order` | sdm-core/src/materialize.rs:1058 | The provider declares no null order | n/a: a provider capability; the DuckDB provider declares `last` (E63(a)) | P1B-3 | T1.33 | Fail-closed by design (E63(b), design.md:157); a dialect adds its order under a new version. Reclassify it as permanent | |
| 35 | `num_hops` | sdm-core/src/materialize.rs:1244 | That option given | yes | P1B-3 | Phase 2 | E61(b) | |
| 36 | `num_neighbors` | sdm-core/src/materialize.rs:1245 | That option given | yes | P1B-3 | Phase 2 | E61(b); seen in the T1.24 differential (P1B-3 notes) | |
| 37 | `open_window` | sdm-core/src/materialize.rs:465, :466 | A temporal aggregation with no window, or no window start | yes per P1B-3; which PQL forms reach it was not checked | P1B-3 | Phase 2 | E72(c) | |
| 38 | `parent_table` | sdm-core/src/materialize.rs:1626 | The entity table itself has foreign keys | yes (P1B-3 old: "tables the entity table refers to") | P1B-3 | Phase 2 | E64(a): "stays refused (no map needs them)" | Real data likely (inferred) |
| 39 | `problem_type` | sdm-core/src/materialize.rs:401; sdm-request/src/predict.rs:253 | A PQL problem type, `num_forecasts` ≠ 1 or `top_k` (forecast or ranking); a predict task kind other than regression, binary or multiclass | yes | P1B-3, P1B-4 | Phase 2 | design.md:41 puts ranking, forecasting and link prediction out of Phase 1; the AdventureWorks FORECAST and ranking questions (design.md:503) | |
| 40 | `return_embeddings` | sdm-core/src/materialize.rs:1246 | That option given | yes | P1B-3 | Phase 2 | E61(b) | |
| 41 | `seed_size` | sdm-core/src/materialize.rs:277 | A negative seed, or one ≥ 10^38 | On DuckDB, **no** for seeds ≥ 2^31 (INTERNAL_ERROR, RS-7 notes); negative seeds unknown | P1B-3 | T1.33 | Other backends that take any seed are Phase 2 (RS-7) | **Conflicts with approved RS-7**: `seed_scalar` (:1586) runs before the RS-7 check (sdm-core/src/compose.rs:85), so a seed ≥ 10^38 on DuckDB gets UNSUPPORTED/`seed_size` where RS-7 says INVALID_REQUEST |
| 42 | `static_target` | sdm-core/src/materialize.rs:441, :447, :526 | A static target from another table than the entity's, a `*` column, a constant, or a join without an aggregation | yes (P1B-3) | P1B-3, P1B-18 (history) | Phase 2 | E72(c) lifted the in-table case; the rest is not needed by any map | |
| 43 | `target_filter` | sdm-core/src/materialize.rs:464, :525 | A target with WHERE | yes | P1B-3 | Phase 2 | E72(c) | |
| 44 | `target_stype` | sdm-core/src/materialize.rs:418 | A static target whose stype is not ID, categorical or numerical | **no**: NotImplementedError (materialize.rs:417; P1B-3) | P1B-3 | T1.33 | E66(b) class candidate; no entry of its own | |
| 45 | `target_transform_null` | sdm-request/src/options.rs:444 | A null in `inference_config.target_transforms` | yes (options.rs:441-443; P1B-3) | P1B-3 | Phase 2 | No case uses it. T1.32f owns the `inference_config` case-format gap (base.adventureworks.json), not this rule | |
| 46 | `inference_config_text` | sdm-request/src/options.rs:372 | An inference option given as text that the engine would coerce | yes (pydantic coerces it; E76(a)) | P1B-3 | Phase 2 | As row 45 | |
| 47 | `target_type` | sdm-core/src/materialize.rs:792, :804, :868; sdm-engine/src/labels.rs:227; sdm-request/src/payload.rs:579; sdm-request/src/predict.rs:854 | :868 a static decimal, time or other target; :792 an aggregated column of another type; labels.rs:227 a decimal AVG; :804 a UInt64 value past int64; payload.rs:579 a multiclass target of floats; predict.rs:854 a classification of a float target | Mixed: static decimal **no** (INTERNAL_ERROR, P1B-18); decimal AVG yes (labels.rs:224-226); multiclass floats yes (P1B-3); UInt64 past int64 **no** (pandas' cast refuses, materialize.rs:796-797); predict float target is not written by the core's materialize (P1B-4) | P1B-18, P1B-3, P1B-4 (proposed) | **T1.32b** | decimal.json entry `materialize/target_type`, owner T1.32b (P1B-18) | The non-decimal sites (multiclass floats, UInt64, time) need splitting off from T1.32b. decimal.json's `where` says :804 (the UInt64 site); the static-decimal refusal is at :868. No entry describes the UInt64 site |
| 48 | `text_time_rules` | sdm-core/src/materialize.rs:690 | The provider declares the client's rules for text times (frames) | n/a: a provider capability | P1B-3 | Phase 2 | design.md:41: frames are out of Phase 1 | |
| 49 | `text_times` | sdm-core/src/materialize.rs:691 | The provider declares no text-time reading | n/a: DuckDB declares `ValueByValue` (sdm-data/src/duckdb.rs:134) | P1B-3, DT-7 | T1.33 | Fail-closed by design (sdm-contract/src/data.rs:487-489). Reclassify it as permanent | |
| 50 | `time_type` | sdm-core/src/materialize.rs:549 | A time value (entity time, time bounds or rows) fetched as anything other than a zone-less timestamp after the core's reading | unknown; whether it can be reached is unknown | P1B-3 says "time_type lifted" at T1.25 | T1.33 | Looks like a leftover guard | **Register out of date**: it says lifted, but the code still refuses |
| 51 | `top_k` | sdm-core/src/materialize.rs:1247 | The `top_k` option | yes | P1B-3 | Phase 2 | Ranking (design.md:41, :263) | |
| 52 | `unseeded` | sdm-core/src/materialize.rs:1267 | `random_seed=None` | yes (RS-3 and RS-4 keep the unseeded stream; P1B-3) | P1B-3 | Phase 2 | E61(b) | |
| 53 | `use_prediction_time` | sdm-core/src/materialize.rs:1248 | That option given | yes | P1B-3 | Phase 2 | E61(b) | |
| 54 | `wire_dtype` | sdm-request/src/payload.rs:205, :379 | A feature with no wire dtype, such as a DECIMAL whose precision is not 38 | yes (P1B-3 old: "any column type in the payload") | P1B-3, P1B-19 (rejected, settled) | **T1.32b** | tracker.md:67: T1.32b's row notes the decimal-precision sum difference that `wire_dtype` hides; E76(a) | Non-decimal types that reach it are unknown; T1.33's DataType pass |
| 55 | `entity_ref` | sdm-request/src/predict.rs:996 | A predict payload with no entity-reference column | unknown: only reachable from a payload the core did not write, which has no core route (P1B-4) | P1B-4 | T1.33 | Looks like an internal guard | |
| 56 | `id_type` | sdm-request/src/predict.rs:1400, :1403 | An entity key declared as something other than int8 to int64 or string, or an int past int64 | yes (P1B-4 old: "an entity key of any declared dtype") | P1B-4 | **T1.32a** (inferred) | Hidden today behind `key_type` for `key.decimal-*` and `key.hugeint`. T1.32a's exit runs the family through predict (design.md:481). Not measured | |
| 57 | `instance_id_type` | sdm-request/src/predict.rs:1413 | A model answer whose instance id is neither integer nor text | yes (P1B-4) | P1B-4 | Phase 2 | P1B-4 notes: "what remains is the forecast, ranking, tabular and extra-output work of later packages" | |
| 58 | `model` | sdm-request/src/predict.rs:390 | The KumoTabular model | yes | P1B-4 | Phase 2 | design.md:41 puts tabular requests out (S3 in migration-plan.md:260) | |
| 59 | `prediction_statistic` | sdm-request/src/predict.rs:673 | A prediction statistic that is not text | yes (P1B-4 old) | P1B-4 | Phase 2 | P1B-4 notes | |
| 60 | `prediction_type` | sdm-request/src/predict.rs:933, :936, :939 | A classification answer that is not of the target's type | yes, passed through `as_declared` (P1B-4) | P1B-4 | Phase 2 | P1B-4's catch-all covers it, but its notes do not name this rule explicitly | |
| 61 | `response_field` | sdm-request/src/predict.rs:1049 | An answer carrying scores, rankings, embeddings, quantiles, an explanation or item metadata | yes (P1B-4) | P1B-4 | Phase 2 | P1B-4 notes ("extra-output work") | |
| — | `schema_edited` (not counted) | sdm-schema/src/edit.rs:77 (`edited_past`) | A schema changed past its description (table or column missing, a column retyped, a table remapped) | yes (P1B-3 old) | P1B-17 (approved), P1B-14 (approved), P1B-3 | T1.33 | Already approved; only needs reconciling | |

## Counts

| Proposed owner | Rules |
|---|---|
| T1.32a | 3: key_type, indices_type, id_type (inferred) |
| T1.32b | 2: target_type, wire_dtype |
| T1.32c | 1: column_type |
| T1.32d | 1: empty_table |
| T1.32e | 0. `multicategorical` and `time_whole_table` are the open candidates |
| T1.32f | 0. T1.32f surfaces rules; it lifts none (design.md:486) |
| T1.33 | 12: text_time, text_time_order, text_time_zones, average_null, condition_value, target_stype, seed_size, null_order, text_times, time_type, entity_key, entity_ref |
| Phase 2 | 42 |

Of the 12 under T1.33:
- 3 are approved by the owner (DT-11, DT-15, DT-16);
- 4 are cases where the reference itself raises, waiting on E66(b)'s class;
- 2 are fail-closed by design (`null_order`, `text_times`);
- 3 are guards whose reachability is unknown (`time_type`, `entity_key`, `entity_ref`).

## Findings for the owner

1. **`time_whole_table` blocks real data.** Any table with at least 1,000 rows and a non-key timestamp column is refused at `Schema.from_source`, even when `time_columns` names the column (`infer.rs:398` → `:808`). No inventory or AdventureWorks case can show this, because the fixtures stay under 1,000 rows. T1.32f's RelBench run will hit it on almost every table. Decide whether it is T1.32e or Phase 2 before T1.32f.
2. **`multicategorical` is broad.** Any `:` in a text value is enough, so times of day and URLs trigger it (`infer.rs:581-590`). It needs the same decision as finding 1.
3. **Unregistered trigger.** A binary (or duration) column is refused by `column_type` (`infer.rs:453`, after admission admits it, `admission.rs:111`). The reference leaves that column out with a warning (the reference's `schema.py:1639-1660`). P1B-2 describes `column_type` only for struct, map and odd lists. Either add the trigger to P1B-2 or P1B-10, or leave such columns out the way list columns already are (`infer.rs:290-297`).
4. **Register and code disagree.**
   - P1B-3 says `time_type` was lifted at T1.25, but `materialize.rs:549` still refuses it.
   - `seed_size` (`materialize.rs:277`, called at `:1586`) fires before RS-7's check (`compose.rs:85`). On DuckDB a seed ≥ 10^38, or a negative seed, gets UNSUPPORTED where approved RS-7 says INVALID_REQUEST.
   - decimal.json's `where` cites `materialize.rs:804`. The static-decimal refusal is at `:868`; `:804` is the UInt64-past-int64 site.
5. **`target_type` covers six sites.** Only the static-decimal site is T1.32b's by the map. The multiclass-float site (`payload.rs:579`) refuses an input the reference accepts. It is registered (P1B-3), but no family owns it.
6. **None of the 61 is approved.** All of them are listed only under the umbrella entries P1B-2, P1B-3 and P1B-4, which are `proposed`. Approving the Phase 2 class as one decision (checkpoint 8, P2-5) would settle 42 of them; the other 19 still need their family or T1.33 entries approved.
