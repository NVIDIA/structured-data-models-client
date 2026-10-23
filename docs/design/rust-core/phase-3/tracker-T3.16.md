# T3.16: plain-data inputs and typed results

Lane E, stack 2.
Design: the Phase 3 structure design (round 3, 2026-10-23), section 3's T3.16 and section 2's fixes A and D.
Started on the phase head efb1bb45 (T3.15 landed).

## Tests first

The `T3.16 tests` commit came first, each test failing on the phase head for the reason named in its message:

- `test_t316_the_schema_as_plain_data.py` runs every public call on both routes, the old client's and the core's with every core route forced, and holds the core to the old route.
  On the head the core refused `schema_edited` for a hand-built column the source lacks, a described column retyped and a table pointed at another source table; a described schema adopted into a new `Schema` or pickled and loaded was described again at materialize; `results`, `Column._source` and `Request._core` did not exist; predict of a request the core never materialized raised `RouteError`; and the core route ran nine functions of `predictive_query.schema`.
- The design's NUMERIC-key test (a Postgres `NUMERIC` primary key through `copy.copy`, `snapshot()`, adoption and pickling, linked, cleared and re-keyed, then materialized) passed on the head: it is a guard, not a failing test.
- `test_a_column_added_by_hand_to_a_described_schema_is_edited_and_sampled` was rewritten in the implementation commit: as first written it changed the source after describing, where the reference reads its cached reading of the source (P3E-4); it now writes back a column the schema left out, which the head already followed.
- `rust/bindings/sdm-py/tests/test_results.py` decodes `messages.RESULTS`, a sample of every `OperationResult` variant.
- `sdm-schema/tests/held.rs` and `sdm-core/tests/t316_held_schema.rs` did not compile on the head; the latter's host provider was fixed to declare `preview` in the implementation commit.

## What changed

| Fix | Where | Change |
|---|---|---|
| A | `sdm-contract/src/ops.rs` | `HeldColumn.source: Option<SourceFacts>` (the relation they were read from, the data type the source gives, `received`, `nullable`, `unique`, `declared_type`, `exact_number`); a column whose facts are of another relation than its table's (a table pointed elsewhere since) holds none of it; `HeldTable.duplicate_keys`; `EditSchema` and `Materialize` take a `HeldSchema` only, and `SchemaEdited` answers one. Contract v28's working entry rewritten (H4). |
| A | `sdm-core/src/discover.rs` | The per-table reading moved out of `describe.rs`: the `columns` and `keys` catalog requests, index columns, one preview of the columns asked, whole-column big-integer tiers and exact-number settling. `describe` infers from it; `materialize` reads, by `discover::facts`, the facts of each column the host holds none of, and nothing for a schema whose columns all carry theirs. |
| A | `sdm-schema/src/held.rs`, `infer.rs` | `held::read` and `held::held`: the schema as the host holds it, read into the description the core works on and written back. `infer::source_facts` is the one reading of a column's facts, shared with `preview_keyed`. |
| A | `sdm-schema/src/edit.rs` | `read_held` and the `schema_edited` refusal are gone. A column with no facts is edited by its data type. |
| A | `sdm_client/schema.py` | `Column._source`, out of repr and equality, copied by `_copy_column` and pickled with the column; `Schema._core_description` is gone; a schema is described on the core when a column holds facts (`_described_on_the_core`). |
| A | `sdm_client/_rust/spec.py` | `held_schema(schema)` for edits and materialize; `query_schema(schema)` for S0, written with no object of the old PQL package. `_described_now` and `materialize.description` are gone. |
| A | `sdm_client/_rust/schema.py` | The `Schema` is written as the core answered it, field by field, so the old route's setters and their PQL checks do not run on the core route (rule 11). |
| D | `sdm_client/_rust/results.py` | Frozen dataclasses for every result (`Validated`, `Described`, `Edited`, `Gathered`, `Predicted`), `SourceFacts`, `QueryFacts`, `CoreRequest` (`Request._core`), `CoreQuery` (moved); `decode` refuses another shape by name. |
| D | `_rust/predict.py` | Reads `Request._core`; a request with none is refused by name (`INVALID_REQUEST`, rule `materialize_again`), never a `RouteError`. |

## Cost

A schema from `from_source` asks the source nothing more at materialize, however copied (held by the both-routes test).
A column with no facts costs, per table holding one, the `columns` and `keys` requests in one step, any `index_columns` reads, one preview of those columns, and a `column_ranges` read only where the preview may have left rows unread and a big integer needs widening.

## For the tooling train

`contract/runners/run_python.py` is a judging path; its change is its own commit, compatible with the phase head, for the orchestrator to move into the train:
`edit_schema` is recorded without `held` when the operation names none, and `presented` takes the description the run's `describe_source` answered, from the run's capture, since the schema keeps none.

## Differences registered

- P1B-3: `schema_edited` lifted.
- P1B-17: its refusal is gone; the owner is asked to retire it.
- P3E-3 (proposed): predict of a request the core never materialized refused by name.
- P3E-4 (proposed): a column the source gained after describing, added by hand, is read on the core route; the reference's cached reading leaves it out.

## Review round 1 (ec2eca2a)

Codex and Claude, three P1 findings and Claude's P1-2; tests first (edacbfcb, d492f8ad, each failing on ec2eca2a), then fix 1.

| Finding | Fix | The old client's rule |
|---|---|---|
| An edit was checked against the source's data type, not the one the caller's schema holds | The schema an edit reads holds each column as the host types it; the source's type is kept apart (`held::Read::retyped`), written back in the facts and read by materialize as the engine reads the table; materialize refuses a meaning the caller's type cannot carry, then one the source's cannot | A setter checks a meaning against the column's own `dtype`; `check_consistent` reads the caller's schema, and the engine refuses an incompatible semantic type when it reads the table in place |
| Data type names: `timestamp[s|ms|ns]`, `date`, `int` and others refused as unsupported, the name rule written twice | One table, `sdm_schema::held::DTYPE_NAMES`, exported as `_native.DTYPE_NAMES` and read by the client's `language_dtype` (its own map only in a build without the core, held equal by a test); a name no type has is let through by an edit's checks, a byte joins only a byte | `_WIRE_TO_PQL_DTYPE`, the language's names passed through, `supports_dtype` letting an unknown pair through, `_join_family` by name |
| The core route still imported the old PQL package | Profiles by name (`ModelSpec.profile`, `query.RELATIONAL`), the profile's model words in `base.PROFILE_MODELS`, the meaning check from `_native.MEANINGS`, every old-route import moved into the old route's functions; the test blocks both old packages before the client's first import | The core route needs neither old package (the design's rule 11) |

Also: `Request._core` and `Column._source` typed; `to_schema_definition` built from `schema.query_schema`, the one translation (Claude P2-2); `_described_on_the_core`'s rule documented (P2-3).
Not done: P2-1 (`build` marks the time columns' meanings declared as the old setter did) stays, held by the repr comparisons; P3-2 (the tooling's copies of the held schema) is in the judging train's files.
A name no data type has is let through by an edit as the old client lets it through; so is the literal `unsupported`, which the old client checks.
