# Phase 2 tracker: T2.11

The design branch owns `tracker.md`; this file holds T2.11's section until it folds in.
Lane i, on the phase head e154a8cc, then the test author's acceptance tests (`T2.11 tests:` commits, from `rust-core-phase-2-topic-t211-tests`), then this package (`T2.11:` commits).

## T2.11

T2.11 lifts the sampling options of E92's Phase 2 class (`anchor_entity`, `context_anchor_time`, `exclude_cols_dict`, `num_hops`, `num_neighbors`, `max_pq_iterations`, `use_prediction_time`, `return_embeddings`, `unseeded`, `parent_table`), reads a response's `embeddings`, and closes M6.
Everything is in the core; the Python host only passes the option values and words the new warnings.

### What changed

- **The option values travel in `Materialize.values`** (contract v27's working entry, documentation only; the shape is unchanged).
  The host adds the sampling options to its `TAKEN` and `VALUED` sets and writes `context_anchor_time` as it writes `anchor_time` (a `ts` scalar), the literal `entity` and none as they are, and any other value as the text `unreadable`, which the core refuses where the engine does.
  `Materialize.options` still names `lag_timesteps` and `top_k`, which other packages lift.
- **`num_hops`, `num_neighbors`** (`sdm-request/src/options.rs`, `sdm-core/src/subgraph.rs`): the walk takes any number of hops, each with its own size.
  By default the first `num_hops` (two) of the run mode's sizes (`_DEFAULT_NUM_NEIGHBORS`: fast 32, 32, 8, 8, 4, 4; best 64, 64, 8, 8, 4, 4).
  A size of zero ends the walk (`neighbors == 0`); a size between zero and one keeps no referring rows, while the hop's key lookups still run; a size below zero is refused by name, `INVALID_REQUEST` naming the hop, since the reference's DuckDB, SQLite and Postgres samplers answer it three ways (P2I-4, KO2-82).
  Each size is read as the reference's DuckDB SQL reads the value written into it (P2I-1).
  `num_hops` named as none with no `num_neighbors` is the engine's `TypeError`, the caller's error.
  `num_hops` beside `num_neighbors`, other than two, is warned of as ignored (`HOPS_IGNORED`, in the engine's words).
- **`max_pq_iterations`**: the candidates drawn are that many times the context's examples; below one, text or none is `INVALID_REQUEST`; a float as P2I-1 says.
- **`use_prediction_time`, `return_embeddings`**: held to a boolean as the engine's request model holds them (pydantic's lax mode: a boolean, 0 or 1, or the words pydantic reads).
  `use_prediction_time` is written into the payload's `inference` section; `return_embeddings` adds `embeddings` to the payload's outputs (`_output_fields`), which the call then replaces with `predict`'s, as the reference's client does.
- **Where each is read**: as the reference's engine reads it, after the client's checks and the sampler's index check, so a refusal comes after the reads and warnings the reference's does: `max_pq_iterations` and `context_anchor_time` with the task, `exclude_cols_dict` and a `num_hops` of none as the subgraph is sampled, the two booleans as the request is built.
- **`exclude_cols_dict`**: merged after the query's own exclusions (`_columns_to_leave_out`), each table and feature checked against the graph as the engine checks them (`sample_subgraph`); a mapping of lists, as the client requires.
- **`context_anchor_time`**: the context's own anchor (`_get_task_table`).
  A context anchored after the predictions, or whose window reaches past them, is warned of as the engine warns (`CONTEXT_AFTER_ANCHOR`, `CONTEXT_LEAKS`, in its words with both anchors); `'entity'` with a time anchor is refused as the engine refuses it.
- **`anchor_time='entity'`**: a static query's examples anchored at each entity's own time: every sampled entity, shuffled, then latest first, equal times in the order the reference's pandas `sort_values(ascending=False)` leaves them (NumPy's generic introsort on `datetime64[ns]`, ported exactly at fix 1); each predicted entity anchored at the time its row holds, a null one a missing anchor (`NaT`): null in the payload, and no row of a table with a time column reached for it (fix 1).
  A temporal query, an entity table with no time column, or a context anchored at a time is refused as the engine refuses it; no time bounds are read.
- **`random_seed=None`**: drawn under a fresh seed each call, mixed from the host's clock fact (`HostFacts.now`), since the core reads no clock or entropy of its own (P2I-2).
- **`parent_table`**: an entity table referring to another is sampled; the walk already looks its foreign keys up by exact value (decision E41(d)), so distinct decimal keys never merge (KO2-35).
- **Embeddings** (`sdm-wire/src/response.rs`, `sdm-request/src/predict.rs`): an item's `embeddings` is read as the bindings' `_float_tuple` reads it, laid out as a list-of-float64 `embeddings` column (contract v27's `Predicted.result`), and counted as carried where any answer holds one; the host passes it into each `PredictionItem`.
- **M6**: a sampler memo holding time bounds or stored times with the index check not made is refused by name (`sampler_state`).

### Decisions under the overnight delegation

- **Unseeded sampling by a fresh seed** (P2I-2), where design choice 8 kept SQLite's `ORDER BY RANDOM()`.
  The reference: each engine's own unseeded draw.
  The core could match it only by a new data-contract request shape (an optional seed), each dialect's unseeded SQL, and the conformance kit and fallback learning it, a judging-path change (`sdm-testkit`) that would land alone first.
  A fresh seed per call gives the customer what they asked for (a new sample each call, of the right size, from the source's rows), passes KO2-76's property checks, and keeps one sampling path.
  Recommended to the owner as is; reviewers may approve it as a stream difference only.
- **Hop sizes and iteration counts read as DuckDB reads them on every database** (P2I-1): matching the reference wherever its DuckDB sampler answers, and answering where its SQLite or Postgres samplers fail.
  Text that is no number is refused by name, where the reference fails or runs the text as SQL; owner-pending then, recommended as is; approved under KO2-82 (P2I-1).

### `sample.parent-table.relbench-stack` and P1B-13

As the test author first wrote it, the posts asked about had a null `ParentId`. The reference's hop-1 parent lookup took each such null for the last parent key it looked up (post 3, pandas' `factorize` code -1), so each of those examples gained post 3, its referring posts and votes, and every later row was numbered out of step: 46, 30 and 89 context rows of posts, users and votes where the core sent 41, 28 and 93.
The reference with only P1B-13's null handling corrected (null dropped before its `factorize`, in `SQLSampler._sample_subgraph`) sent the core's request exactly, on every table and row: the difference was exactly P1B-13's.
A payload of another length is a difference the comparator cannot excuse, so the test author gave every post a parent (5b1a58ae's data), and the case no longer meets a null parent.

### Fix 1 (round 1)

- Codex P2 and Claude P2-2: a scored entity whose own time is null was anchored at the no-anchor sentinel (1677) and, on SQLite, reached all its referring rows. It is now the reference's `NaT` (`sample::MISSING_ANCHOR`): null in `__kumo_anchor_time`, and its rows of tables with a time column are left out, as `time <= NULL` leaves them in the reference's SQL.
- Claude P2-1: equal entity times are ordered as the reference orders them, by a port of NumPy's generic `aquicksort` and `aheapsort` under pandas' `nargsort`; held index for index to NumPy 2.5.3's and pandas 3.0.6's orders (`sdm-engine/tests/numpy_sort.rs`). P2I-3 withdrawn (rejected, ID kept).
- Claude P3-1: P2I-1's notes name Postgres' answer to fractional sizes.

### Checks run

See the package report for the head SHA and each check's result.
