# Phase 7 acceptance contract

The acceptance checks in `tests/rust_core/sampling/` and `tests/rust_core/labels/`, and the Rust tests `rust/crates/sampling/tests/accept_fixtures.rs` and `rust/crates/labels/tests/accept_fixtures.rs`, test behavior through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

The checks build the binding's wheel from this checkout with maturin and install it into the phase venv before calling it (`_rc.build_local_wheel`; a clean commit's wheel is kept under `$TMPDIR/sdm-rust-core-p7/<sha>/`, or `$SDM_RC_P7_CACHE`).
No call into the binding may end in a Rust panic or a crash: a panic fails the check that made it.
Unless this file says otherwise, the binding raises `ValueError` for input it can't read.

Today's Python is the reference throughout: numpy's legacy `RandomState`, `sdm_relational_engine.relationallib`, `LocalGraphStore`, pandas' `DateOffset` arithmetic, the sampler's helpers and `PQueryPandasExecutor`, run in the phase venv (pandas 3.0.6, numpy 2.5.3).

## Errors as data

Where an outcome can be an error, it is `{"error": {"code": C, "message": M}}`, with `C` the exception's class name (such as `OutOfBoundsDatetime` or `ValueError`) and `M` its text, as the fixtures write errors (`tests/parity/FORMAT.md`).

## Random numbers

`sdm_core._native` exposes:

- `RandomState(seed)`: numpy's legacy `RandomState`.
  - An `int` from 0 to `2**32 - 1` seeds it as `np.random.RandomState(seed)` does.
  - A non-empty list of such ints seeds it as `np.random.RandomState(np.array(seed, dtype=np.uint32))` does (`init_by_array`), any length.
  - Anything else, such as a negative int, `2**32` or an empty list, raises `ValueError`.
- `random_state(seed)`: the stream `base/utils.random_state(seed)` gives, for any int from 0 up, of any size.
- A `RandomState`'s methods continue one stream, as numpy's do; each equals, drawn from the same stream:
  - `next_u32()`: `randint(0, 2**32, dtype=np.uint32)`, as an `int`;
  - `next_u64()`: `randint(0, 2**64, dtype=np.uint64)`, as an `int`;
  - `interval(max)`: `randint(0, max + 1, dtype=np.int64)`, as an `int` (the interval draw `shuffle` makes), for `0 <= max < 2**63 - 1`;
  - `randint(low, high, size)`: `randint(low, high, size=size, dtype=np.int64).tolist()`;
  - `shuffle(values)`: a list of ints, returned in the order `shuffle(np.array(values, dtype=np.int64))` leaves them;
  - `permutation(n)`: `permutation(n).tolist()`;
  - `choice(a, size)`: `choice(a, size=size, replace=False).tolist()`, what `DataFrame.sample` draws.

  Lists may come back as lists, tuples, numpy arrays or pyarrow arrays.
- `Mt19937(seed)`: C++'s `std::mt19937` seeded with `seed`, for `0 <= seed < 2**64`, which the standard reduces modulo `2**32`.
  - `next_u32()` is `rng()`.
  - `floyd(population, count)`, for `1 <= count < population`, is the list of offsets `relationallib`'s `_sample` picks for a neighborhood of `population` when it asks for `count` (case 2: `rnd = rng() % (i + 1)`, replaced by `i` when already picked), in order.

The reference is `mt19937_reference.cpp`, beside this file, compiled with the system's C++ compiler.

## The neighbor sampler

`sdm_core._native.NeighborSampler` is a drop-in for `sdm_relational_engine.relationallib.NeighborSampler`:

- the same constructor, `NeighborSampler(node_types, edge_types, colptr, row, node_time, seed=123)`, and methods `seed(seed)` and `sample(num_neighbors, sampling_time_offsets, entity_table, seed_node, seed_time)`, by position and by keyword;
- the same 6-tuple of dicts from `sample` (row, col, node ids, batch, sampled nodes per hop, sampled edges per hop), with the same keys in the same order and `numpy.ndarray`s of `int64` of equal values;
- the same random stream across calls: a sampler's draws continue from one `sample` call to the next, and `seed` restarts them;
- where `relationallib` raises `RuntimeError` or `KeyError` (invalid inputs, a non-sorted temporal neighborhood, a missing edge type), the same class with the same text;
- where it refuses an argument's type or range (pybind11's `TypeError`), a `TypeError`;
- whatever seed `relationallib` takes, in its constructor or `seed`, the drop-in takes and samples the same, and it refuses with `TypeError` the seeds `relationallib` refuses.

## The graph store

`sdm_core._native.graph_store(tables, graph_json)` is `LocalGraphStore(graph)`:

- `tables` maps each table's name to a `pyarrow.Table` of its data, as `pa.Table.from_pandas(local_table._data, preserve_index=False)` gives it;
- `graph_json` is JSON text: `{"tables": [...], "edges": [...]}`, tables in the graph's order, each `{"name", "primary_key", "time_column", "end_time_column", "columns": [{"name", "dtype", "stype"}]}`, with `dtype` and `stype` the engine's `str(column.dtype)` and `str(column.stype)`, and each edge `{"src_table", "fkey", "dst_table"}`;
- it returns a dict with exactly these keys, each a dict in the order today's store holds it:
  - `report`: by table, `{"input_rows", "output_rows", "null_primary_key_rows", "duplicate_primary_key_rows", "null_time_rows"}`, the sanitization report;
  - `masks`: `mask_dict`, by table, a list of bools;
  - `keys` and `node_ids`: `pkey_map_dict`, by table, its index (the valid keys, as Python values) and its `arange` column;
  - `time`: `time_dict`, by table, every row's time in floored seconds (a null time as today's arithmetic gives it);
  - `time_range`: `min_max_time_dict`, by table, `[min, max]` in nanoseconds;
  - `colptr` and `row`: `colptr_dict` and `row_dict`, by edge type joined with `__`, reverse edges included and named as `SampledSubgraph.rev_edge_type` names them.
  Arrays may be lists, numpy arrays or pyarrow arrays.
- Where `LocalGraphStore` raises, it raises the same class with the same text.

## DateOffset arithmetic

An offset is JSON `{"n": N, "kwds": {UNIT: VALUE}}`, standing for `pd.DateOffset(n=N, **kwds)`, with units among `years`, `months`, `weeks`, `days`, `hours`, `minutes` and `seconds`.

- `date_offset_apply(timestamps_json, offset_json, sign)`: `timestamps_json` is a JSON list of nanosecond ints; `sign` is `1` or `-1`.
  It returns JSON text: a list with, for each timestamp `t`, `{"ns": N}`, where `N` is `(pd.Timestamp(t, unit='ns') + offset).value` (or `- offset`), or the error that raises.
- `date_offset_to_seconds(offset_json, bound)`: `base/utils.date_offset_to_seconds(offset, bound)`, an `int`; `bound` is `"earliest"` or `"latest"`.
- `date_offset_to_ns(offset_json)`: JSON text, `{"ns": N}` with `N` what `rfm._date_offset_to_ns(offset)` gives, or the error it raises.
- `date_offset_pool(offsets_json, earliest)`: a JSON list of offsets; JSON text, the offset `base/sampler._pooled_offset(offsets, earliest=earliest)` gives, as `{"n", "kwds"}`.
  The order of `kwds` is not compared: pandas keeps an offset's keywords in an order that follows `PYTHONHASHSEED`.

## Text

`sdm_core._native.normalize_text(values, max_words)` is `base/sampler._normalize_text(pd.Series(values, dtype=object), max_words).tolist()`: `values` a list of `str` or `None`, `max_words` an `int` or `None`, and the result a list of lists of `str`.

## Labels

- `sdm_core._native.labels_op(op, input_json)` answers the `labels.*` operations of `tests/parity/_fixture_ops.py` (`labels.column`, `labels.rel_op`, `labels.member_op`, `labels.bool_op`, `labels.aggregation_type`, `labels.aggregation`).
  `input_json` is a case's whole `input`; it returns JSON text, `{"output": X}` with `X` what the fixture runner gives, or `{"error": {"code", "message"}}` where it raises, as a fixture writes them.
  An unknown operation, or `input_json` that isn't a JSON object, raises `ValueError`.
- `sdm_core._native.labels_execute(query_json, feat, times, batches, anchor)` is `PQueryPandasExecutor().execute_query(query, feat_dict, time_dict, batch_dict, anchor_time)`:
  - `query_json` is JSON text, the `tree` a resolved query has in the PQL corpus (`json_serde.to_json_dict` of the `ValidatedPredictiveQuery`, as Phase 5's `pql_validate` gives it);
  - `feat` maps tables to `pyarrow.Table`s, `pa.Table.from_pandas` of the frames; `times` and `batches` map tables to `pyarrow.Array`s (timestamps, and `int64` example indices); `anchor` is a timestamp `pyarrow.Array`;
  - it returns `(values, mask)`: `values` a `pyarrow.Array` of the labels, `mask` a list, numpy array or pyarrow array of bools;
  - a label of pandas dtype `float32` is `float32`, bit for bit, NaN included (any NaN is the same NaN); `int64` is `int64`; `bool` is `bool`; a timestamp is a timestamp; text is `string` or `large_string`; a list (`LIST_DISTINCT`, a list column) is a list array;
  - where today's executor raises, it raises (any `Exception`).
- In Rust, `sdm_labels::fixture_op(operation: &str, input_json: &str) -> Result<String, E>`, `E: Display`, answers the same operations with the same JSON text; `Err` is for an unknown operation or an input it can't read.
  The Rust tests read the fixtures with `serde_json`, so `sdm-labels` and `sdm-sampling` have it as a dev-dependency.

## Sampler fixtures

Task 8's fixtures are new files under `tests/parity/fixtures/`, in the format of `tests/parity/FORMAT.md`, written by `scripts/parity/make_fixtures.py` from today's Python, with their operations in `tests/parity/_fixture_ops.py`:

- every operation of these files is named `sampling.*`, and no other file has one;
- between them they hold at least 20 cases, from at least `packages/sdm-relational-engine/tests/backend/local/test_local_sampler.py` and `packages/sdm-relational-engine/tests/backend/local/test_local_graph_store.py` (each file's `source`);
- `sdm_core._native.sampling_op(op, input_json)` answers them as `labels_op` answers `labels.*`, and raises `ValueError` for an unknown operation or `input_json` that isn't a JSON object;
- in Rust, `sdm_sampling::fixture_op(operation, input_json)` answers them as `sdm_labels::fixture_op` does.

## The porting ledger

`ledger/ledger.tsv` in the harness (`~/.local/share/sdm-dbx/rust-core`, or `$SDM_RC_HARNESS`) marks these rows `ported` or `verified`, each with a `rust_symbol` whose last path segment names an item defined in `rust/`:

- `backend/local/graph_store.py`: `LocalGraphStore.sanitize`, `.get_pkey_map_dict`, `.get_time_data`, `.get_csc`, `_argsort`;
- `base/sampler.py`: `_normalize_text`, `_pooled_offset`, `min_date_offset`, `max_date_offset`;
- `base/utils.py`: `date_offset_to_seconds`, `_elastic_days`, `random_state`;
- `rfm.py`: `_date_offset_to_ns`.

(Paths are under `packages/sdm-relational-engine/src/sdm_relational_engine/rfm/`.)
The `labels` rows are checked by P7.ledger.

## What the checks compare

- **Fixes** (`accept_fixes.py`, task 1, checked against today's engine):
  - `materialize(random_seed=S)` over in-memory tables succeeds for `S` in `2**31`, `2**32 - 1`, `2**32`, `2**63`, `2**64 + 3` and `10**30`, and gives the same payloads twice; seeds below `2**31` keep their payloads (the corpus replays);
  - on a static in-memory graph where the C++ sampler draws, each of those seeds draws the same neighbors whatever was drawn before (it reseeds), and all but at most one of them draw other neighbors than the seed one above it (the seed is not ignored);
  - with `anchor_time='entity'`, the context examples are the shuffled entities in descending time, ties kept in shuffled order (a stable sort);
  - the SQL of a question over several temporal tables is the same under six `PYTHONHASHSEED` values, and its `time_ranges` statement lists the tables in the order they were given.
- **Random numbers** (`accept_rng.py`): 10,000 seeds (seed 20261010), each with a random sequence of draws, through `random_state`, `RandomState(int)` and `RandomState(list)`; and through `Mt19937`, against the C++ reference.
- **Graphs** (`accept_graphs.py`): 10,000 random graphs (seed 20261011), each sampled one to three times, through both samplers; plus invalid inputs and seeds outside the C `int` range.
- **Graph store** (`accept_graph_store.py`): 1,000 random in-memory graphs (seed 20261012), through `graph_store` and `LocalGraphStore`, then sampled from both stores' arrays.
- **Offsets** (`accept_offsets.py`): 100,000 timestamp and offset pairs (seed 20261013), applied up to five times in a row, plus `date_offset_to_seconds`, `date_offset_to_ns` and pooling of their offsets.
- **Text** (`accept_text.py`): 20,000 generated texts (seed 20261014) under `max_words` 50, 3 and `None`.
- **Labels** (`../labels/accept_generated.py`): at least 100,000 generated groups through `labels.aggregation_type`, 3,000 windows through `labels.aggregation`, and every resolvable relational question of the PQL corpus with 1,500 generated ones, each over two generated batches, through `labels_execute`.
- **Fixtures** (`accept_sampler_fixtures.py`, `../labels/accept_executor_fixtures.py` and the two `accept_fixtures.rs`).
- **Hash seeds** (`accept_hashseed.py`): the corpus replay and the fixture runner of `tests/parity` pass under `PYTHONHASHSEED` 1 and 4242, and the core's answers to every `labels.*` and `sampling.*` fixture case and to the first generated inputs of the other checks are the same bytes under both.
- **On its own** (`accept_independent.py`): the core gives the same answers in a process where numpy, pandas, dateutil, pytz, torch, `sdm_relational_engine`, `sdm_client` and `predictive_query` can't be imported (random numbers, offsets, text, fixture operations), and in one where, of those, only numpy can (the neighbor sampler, the graph store, `labels_execute`, with pyarrow).
