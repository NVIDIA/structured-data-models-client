# Phase 7.5 acceptance contract

The acceptance checks in this directory, and the Rust test `rust/crates/sampling/tests/accept_debug.rs`, test the sampling driver through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

The checks build the binding's wheel from this checkout with maturin and install it into the phase venv before calling it (Phase 7's `_sampling.native`, in `tests/rust_core/sampling/`).
No call into the binding may end in a Rust panic: a panic fails the check that made it.
Today's Python, in the phase venv (pandas 3.0.6, numpy 2.5.3), is the reference throughout.

`_driver.py`, beside this file, holds what the checks share: the recording format, the canonical form outputs are compared in, driving a run, and the interceptor that sits on the sampler while the corpus replays.
The capture script may import it, as `capture_dialects.py` imports `tests/rust_core/dialects/_codec.py`.

## Names

- A **call** is one call of `Sampler.sample_target` or `Sampler.sample_subgraph` (`packages/sdm-relational-engine/src/sdm_relational_engine/rfm/base/sampler.py`) on any sampler, in memory or over a database, that no other such call is inside.
- The call's **method** is `sample_target` or `sample_subgraph`.
- Its **arguments** are the method's parameters by name, defaults applied, as `inspect.signature(method).bind(...)` then `apply_defaults()` give them, without `self`.
- Its **graph** is the `Graph` the sampler was made from (the `graph` argument of `Sampler.__init__`).
- Its **statements** are the calls of `Source._run(sql, params)` (`sdm_connectors.source`, any subclass, the outermost only) made while today's method runs, each with what `_run` returned or raised.

## The hooks the implementer writes

`scripts/parity/capture_sampling.py` defines:

- `describe(sampler, graph, method, arguments) -> (inputs_json, tables)`: what the core's run of the call is started from.
  - `inputs_json` is JSON text (no NaN or infinities), of the implementer's design.
  - `tables` is a `dict` of names to `pyarrow.Table`s, of the implementer's design, with one rule: for a sampler whose `backend` is `local` (in memory), it holds, for each table of the graph, under the table's name, `pa.Table.from_pandas(table._data, preserve_index=False)` (equal by canonical form): the in-memory backend takes its tables directly.
  - It is called before today's method runs, and it may only read: it may not run a statement, sample, or reach a local sampler's graph store or neighbor sampler (below, "The guard").
- `to_python(sampler, graph, method, arguments, inputs_json, value)`: the object today's method returns, made from `value`, the run's `done` value (below), under the same guard.
  For `sample_target` that is the pair of `TargetOutput`s, for `sample_subgraph` a `SampledSubgraph`, as today's encoder (`rfm.py`, `predict_request_to_json`) reads them.

It also has a command line:

- `python scripts/parity/capture_sampling.py --out DIR`, run from the repository root in the phase venv, writes the recordings (below) under `DIR`, and nothing else; without `--out` it writes `tests/parity/sampling/`.
  It needs no network, database or credentials, and writes the same bytes on every run, whatever `PYTHONHASHSEED` is.
  The committed `tests/parity/sampling/` is what it writes.
  `_driver.capture_corpus(out, describe)` does all of this.

`scripts/parity/bench_sampling.py --out PATH` writes the performance record (below, "Performance").

## The guard

While `describe` or `to_python` runs, each of these raises `_driver.Tripped`, a `BaseException`, and the call fails even if something catches it:

- `Source._run`, on any source;
- on every sampler class, the methods `sample_target`, `sample_subgraph`, `sample_lagged_target`, `get_min_time`, `get_max_time`, `validate_entity_references`, `_get_min_max_time_dict`, `_sample_subgraph`, `_sample_entity_table`, `_sample_query_data`, `_sample_target`, `_sample_target_set`, `_by_pkey`, `_by_fkey`, `_by_time`, `_time_spelling`, `_probe_time_spelling`, `_run` and `_rows`, where the class defines them;
- `PQueryPandasExecutor.execute_query`;
- any attribute of a sampler's own that holds a `LocalGraphStore` or a `relationallib.NeighborSampler`.

State the sampler keeps, such as its minimum and maximum times, the SQLite time spellings it has probed, or the module's `_coverage_warned`, may be read directly and passed in `inputs_json`.

## The Python binding

`sdm_core._native` holds, besides what earlier phases put there:

- `sample_target(inputs_json, tables)` and `sample_subgraph(inputs_json, tables)`: each returns a run (Phase 4's run loop, `tests/rust_core/CONTRACT.md`, "The run loop"), started from what `describe` gave.
  - Each query's `sql` and `params` are what the host passes to `Source._run`, as `source._run(query.sql, query.params or None)`.
  - A host runs the queries of a step in the order the step lists them, and submits `Outcome.data(table)` with what `_run` returned, or, where it raised, `Outcome.error(sqlstate, vendor_code, message)` with the error's `sqlstate` (when a `str`), its `errno` (when an `int`) and its text.
  - The in-memory backend asks no query.
- Where today's method raises, the run's `next()` raises an exception whose class has the same `__name__` and whose text is the same.

### What a run gives

The `done` step's `value` is a `dict`.
An *array* below is a `pyarrow.Array` or `pyarrow.ChunkedArray`; *integers* are a list or tuple of `int`, a numpy integer array, or a pyarrow integer array without nulls; *times* are a pyarrow timestamp array (any unit) or integers in nanoseconds since the epoch.

For `sample_target`:

| Key | Holds |
|---|---|
| `train`, `test` | Each a `dict`: `entity_pkey`, an array; `anchor_time`, times; `target`, an array. A `float32` label is a `float` (32-bit) array |
| `warnings` | The `UserWarning`s the call gives, in order, each `{"category": "module.QualifiedName", "message": text}`, such as the coverage warning as `builtins.UserWarning` |

For `sample_subgraph`:

| Key | Holds |
|---|---|
| `anchor_time` | Times |
| `tables` | A `dict`, in the order `SampledSubgraph.tables` holds them, of table names to `dict`s: `df`, a `pyarrow.Table` of the frame's columns in order; `row`, integers or `None`; `batch`, integers; `num_sampled_nodes`, integers; `stype_dict`, a `dict` of column names to the stype's value (such as `numerical`), in order; `primary_key`, a `str` or `None` |
| `links` | A `dict`, in the order `SampledSubgraph.links` holds them, of `(src, fkey, dst)` tuples to `dict`s: `layout` (`REV`, `COO` or `CSC`), `row` and `col` (integers or `None`), `num_sampled_edges` (integers) |
| `warnings` | As for `sample_target` |

## Canonical form

Outputs are compared in one canonical form, which `_driver.python_outputs` makes of what today's method gives and `_driver.rust_outputs` makes of a `done` value:

- An array is `{"type": T, "values": [...]}`.
  `T` is the Arrow type's name as `_driver.type_name` gives it: a dictionary is named by its values, every string type is `string`, every binary type `binary`, every list type `list<item>`; any other type as `str(type)`, such as `int64`, `float`, `double` or `timestamp[ns]`.
  Values are integers, booleans and strings as they are; a float as the hex of its IEEE bits at its width, with a NaN and a null both `nan`; a temporal value as its integer storage; a decimal as its text; binary as base64; a list as a list.
- Today's Series and frames are made Arrow by `pa.Table.from_pandas(..., preserve_index=False)`.
- Times are integer nanoseconds.
- `stype_dict` is a list of `[column, stype]` pairs, `tables` and `links` lists in order.
- An error is `{"error": {"code": C, "message": M}}`, `C` the class's `__name__`.

Two outputs are equal when `_driver.difference` finds nothing: equal JSON, by value and type, in order; an empty `entity_pkey` or `target` of any type equals another empty one.
So labels compare bit for bit as `float32`, anchor times exactly as nanoseconds, and frames by Arrow value and type.

## Recordings

`tests/parity/sampling/` holds one directory per transcript under `tests/parity/transcripts/`, at the same relative path (`#` written as `--`), each holding `boundary.json` and the Arrow IPC files it names (`_driver.write_recording` writes them):

```
{
 "format": 1,
 "scenario": NAME,
 "calls": [
  {"method": ..., "inputs": I, "tables": {NAME: FILE}, "steps": [S, ...], "outputs": O},
  ...
 ]
}
```

- `inputs` is `inputs_json`, parsed; `tables` names the IPC file holding each of `describe`'s tables.
- Each step is a statement: `{"sql", "params", "result", "error"}`, with `params` encoded as a transcript step's (`tests/parity/FORMAT.md`, "Encoded values"), `null` for none or an empty list; `result` the IPC file holding the table `_run` returned, or `null`; `error` `null`, or `{"type", "message", "sqlstate", "vendor_code"}`.
- `outputs` is what today's method gave, in canonical form, with the `UserWarning`s it gave under `warnings`.
- Calls are in the order the replay made them.
  A transcript whose replay makes none (the tabular ones) has a recording with no calls.

## Synthetic scenarios

Task 1's synthetic scenarios are scenarios of the parity corpus: transcripts under `tests/parity/transcripts/`, recorded by `run_corpus.py record` from scenarios in `scripts/parity/_parity/scenarios.py`, over data a generator in `scripts/parity/_parity/data.py` makes (the `shop` generator unchanged).
A DuckDB scenario's tables are named as the generator's frames.
Between them, among the `memory` scenarios and again among the `duckdb` ones:

- **Static fan-out above the hop count:** a `sample_subgraph` call over an edge `(src, fkey, dst)` whose `src` has no time column, where some `dst` key in the sampled `dst` rows has more `src` rows in the scenario's data than the call's `num_neighbors` at some hop where that is above 0.
  In memory, the same call with `random_seed` one higher samples other neighbors: `relationallib` draws random numbers.
- **Fractional amounts:** a `sample_target` label of type `float` that is finite and not a whole number.
- **Two or more temporal tables:** a call whose sampler has two or more tables with a time column.
- **`anchor_time='entity'` with ties:** a `sample_target` call with `train_anchor_time` or `test_anchor_time` `'entity'`, whose train and test anchor times hold a value twice.

## Performance

`bench_sampling.py --out PATH` drives the core's run of every recorded call from its recording, as the checks do, and writes JSON:

```
{"scenarios": {NAME: {"queries": Q, "calls": [{"method": M, "queries": N, "rust_seconds": S}, ...]}}}
```

- `Q` is the number of statements the scenario runs with the core's runs standing in for today's calls: the transcript's `execute` steps.
- `N` is the number of queries the call's run asked; `S` its wall time from start to `done`, in seconds, a positive number for every call of a `memory` scenario (the in-memory sampler and labels), a number or `null` for the others.

`scripts/parity/sampling_baseline.json` is such a record, committed: the Rust timings later phases compare with.

## Coverage

The ported code is every function, method and class body whose porting-ledger destination is `sampling`, by the rules in the harness's `ledger/rules.tsv` (`$SDM_RC_HARNESS`, by default `~/.local/share/sdm-dbx/rust-core`), and, where a module's own `<module>` symbol is `sampling`, its lines outside any of them.
A branch is the ported code's when its source line is, the innermost symbol deciding.

`scripts/parity/coverage.py`'s `measure` gives branch coverage under the corpus replay and the fixture and PQL runners.
For each module with ported code, at least 90% of its ported branches are taken, or every one not taken is listed in `scripts/parity/sampling_coverage_gaps.json`:

```
{"MODULE PATH": [{"branch": [FROM_LINE, TO_LINE], "reason": TEXT}, ...]}
```

Every entry must be a ported branch not taken, with a reason that isn't empty.
A reviewer reads the reasons (`P75.coverage-gaps`).

## Debug and release (task 6b)

`maturin build` without `--release` makes a debug build of the binding.
Over the inputs `accept_driver_debug.py` makes (seed 20261016), the debug and release builds give the same answer through `sampling_op` and `date_offset_apply`, and neither panics; where today's Python answers without crashing (`tests/parity/_fixture_ops.py`), both give its answer.
`sampling.date_offset_to_seconds` gives a result past int64 as the exact integer.
`accept_debug.rs` asks `sdm_sampling::fixture_op` the same of the two inputs the Phase 7 audit found, in a debug build.

## The ledger (task 7)

Each of the 65 rows `ledger/ledger.tsv` marked `triage:sampling` when these checks were locked (`accept_driver_ledger.py` lists them) has a destination other than `triage:*`, set by a line of `ledger/rules.tsv` naming symbols (`PATH::SYMBOLS`, `SYMBOLS` not just `*`) with a reason in its third column; no line of `rules.tsv` sends anything to `triage:sampling`.
That every `sampling` row is `verified`, with a Rust symbol and fixtures, is `P75.ledger`.
