# Phase 2 tracker: T2.13

The design branch owns `tracker.md`; this file holds T2.13's section until it folds in.
Lane c, on the phase head after the lane's previous landing (e1fac9ee), then the test author's acceptance tests (`T2.13 tests:`, from `t213-tests`), then this package (`T2.13:` commits).

## T2.13

The query and option long tail: `entity_filter`, `entity_ids`, `assuming`, `evaluate`, `explain`, `target_transform_null`, `inference_config_text`; `response_field` for explanations and item metadata; carried row U1.

### Split (the orchestrator's decision, 2026-10-19)

`entity_filter` and `assuming` move to a follow-up package, T2.13b, stacked on T2.10 once T2.10 is approved.
Both mark examples by a condition over the entity's row and its windowed aggregations (the reference's executor: `execute`'s entity mask, `ASSUMING` and-ed into it), which is the labelling T2.10 rewrites (`labels::truth`, conditions with text and `IN`, `NOT`, `AND`, `OR`, the windowed fetches merged across a query's aggregations).
Built here on e1fac9ee they would duplicate T2.10's evaluator and conflict with it.
Until T2.13b they stay refused by name, `UNSUPPORTED_REQUEST` with rule `entity_filter` or `assuming`, as before.
The test author's `filter.*` and `assuming.*` cases are T2.13b's exits: `filter.entity-column`, `filter.entity-text`, `filter.entity-aggregation`, `filter.matches-none`, `filter.ranking`, `assuming.binary`, `assuming.regression` and `assuming.no-examples` fail here by the refusal, while `filter.event-column` and `assuming.static`, which the validator refuses, pass.

### What changed

- **`EVALUATE`, `EXPLAIN` and named entities** (`sdm-core/src/materialize.rs`, `asked`): the three refusals are lifted.
  The reference's engine reads neither prefix nor the entities `FOR t.c = 1` or `FOR t.c IN (...)` names when it builds a materialize's task table (`_get_task_table`; a forecast's named entity is replaced by the first index), so the payload is the plain query's but for `metadata.source_query`, which keeps the query as written.
  Probed on the reference at 6298913c: the four queries over the shop base gather the plain query's context rows, predict rows and options, the named entity ignored for the indices given.
- **Inference options given as text** (`sdm-request/src/options.rs`): read as pydantic 2.13's lax mode reads them into the engine's config classes.
  A boolean: `0`, `off`, `f`, `false`, `n`, `no` and `1`, `on`, `t`, `true`, `y`, `yes`, in any case of letters, nothing else (white space around one refused).
  `num_estimators`: Unicode white space around it dropped (Rust's `str::trim`, which keeps 0x1C as pydantic does), an optional sign, ASCII digits, and an optional point followed by one or more zeros.
  `an_option_given_as_text_is_read_as_the_engine_reads_it` holds 25 accepted and 18 refused booleans and 32 accepted and 43 refused whole numbers, each probed through `RegressionInferenceConfig` on the reference; `probe_ic.py` confirmed the payload through the reference's public `materialize`.
- **A null target transform** (`options.rs`, `sdm-wire/src/payload.rs`): taken and written as null, as the engine's `list[TargetTransform | None]` writes it.
  The payload's `target_transforms` items become a name or null: contract v27, the phase's working entry beside the merged v26 (lock line, `sdm-contract-27.json`, `state-payload.json` regenerated; `CONTRACT_VERSION` 27; the binding's test messages; T2.1b's tests hold v26 to its own file).
  The test-only schema check reads a list of types and `null` for it (`schema_check.rs`, with tests).
  As in the reference, `predict` then refuses `target_transforms` as an option `kumo-relational` does not read.
- **U1** (`sdm-core/src/pql.rs`, `dispatch.rs`, `_rust/s0.py`): `validate_query` gives D4's warnings for the day-first literals read before the error a query ends with (a packed literal, a two-digit year the run's zone cannot place) on the run's warnings channel, then the error (`pql::validate_warned`); the Python host shows the warnings it held for the shadow check when the run ends with an error, since no shadow check follows.
- **An item's explanation and metadata** (`sdm-wire/src/response.rs`, `sdm-request/src/predict.rs`, `sdm-core/src/predict.rs`, `_rust/predict.py`):
  - read as the bindings' `_dict_or_none` reads them, Python's `dict(value)` (`response::mapping`): an object as it is; empty text none; a list of pairs, each a list of two whose first member Python can hash, text of two characters, or an object of two members; a key given twice keeps its first place and takes the last value, keys equal as Python's are, every NaN one key (the transport's `json.loads` reads every `NaN` token as one float object). A mapping keyed by text alone is a JSON object, one with another key the list of its pairs. Anything else is `INVALID_RESPONSE`, `not_a_mapping`, naming the field. `a_mapping_is_made_as_python_s_dict_makes_one` holds 19 accepted and 11 refused vectors, each recorded through the reference's `PredictionItem.from_dict`;
  - laid out in the result: `explanation` and `metadata` columns (the JSON text of each answer's mapping) where any answer has one, and the result's explanation on the result's schema metadata under `explanation`: the first answer's in the endpoint's order holding a mapping not empty (`normalize`), the first part's across a merge, and for a forecast `{"by_forecast_step": [[step, explanation], ...]}` (`_by_step`); the carried answers of a failed call keep it;
  - `explanation` is absent from an answer only where the result has none (`_report_absent`), and a forecast explained in some steps and not others gives `STEPS_UNEXPLAINED` (`_warn_of_steps_that_carry_no_explanation`), worded by the Python host as the client words it;
  - the Python host hands each to `PredictionItem` and `PredictionResult` as the reference holds them (`_mapping`, `_explanation`, `_table`; bridge changes declared).
- **Wording** (`_rust/errors.py`, `sdm-request/src/predict.rs`): the refusals these options reach are said with what to change, from the fields the core names and never a value given: an inference option the engine does not read or a value it refuses (by option and reason), an explanation asked of more than one entity or of a request gathered past the fast mode's context (the core now names `field` `outputs` and a `remedy`, worded as the client's `_preflight` words them), and an item's explanation or metadata that is no mapping.
  A `WORDED` entry is found by the error's rule, remedy or, now, reason.
  Left for T2.14b: every other refusal still reaching Python as its code's sentence, notably the E92 rules' generic `UNSUPPORTED_REQUEST` sentence (`entity_filter` and `assuming` among them until T2.13b).
- **The test host** (`sdm-testkit/src/host.rs`, after T2.1n lands): predict takes the case's `outputs` and answers each call by `endpoint::answer_told` where the case tells the endpoint to answer probabilities, quantiles, an explanation or item metadata.
  It needs T2.1n's `case.rs` and `endpoint.rs`, so it is committed after the rebase onto T2.1n's landing.

### Probes

- The reference, probed directly before the code (6298913c's venv): the prefixes, the named entities, the entity filter (its context sample, and the client's `SDMWarning` naming the filter), text options through `materialize`, `dict` of every item-field shape.
- Width and depth (outside JSON): an item's explanation is read by the answer's reader, whose depth bound (`MAX_DEPTH`, P1B-22) and wide-object reading (n log n) are unchanged; `mapping` walks one level of pairs, its key lookup linear in the pairs kept.
- The implementer's own host tests (`packages/structured-data-models-client/tests/test_the_query_long_tail.py`, 22 tests): each outcome on the core route held to the old route's in the same process.

### Exits

See the before-handoff section below for the runs.

### Deviations and questions

- `sdm-wire`'s `tests/json.rs::a_wide_object_reads_in_n_log_n` failed once under the machine's load (a timing ratio past 24) and passed on rerun; not this package's code.
- Disk: the root file system was full when the first wheel build ran; this package's lane target's debug profile was cleaned (`cargo clean --profile dev`, 64.8 GiB) before the orchestrator's rule that build output is deleted only on asking.
- T2.8's `sdm_wire::response::mappings` (rankings) and this package's `response::mapping` read the same `dict()`; whichever lands second should make `mappings` call `mapping` (T2.8's reads an object element of a pair list as no pair, where Python takes its two names).
