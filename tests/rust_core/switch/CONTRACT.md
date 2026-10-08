# Phase 9 acceptance contract

The acceptance checks in this directory test the switch of `sdm_client` to the core through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

The checks build the binding's wheel from this checkout with maturin and install it into the phase venv before they run anything on the core (Phase 7's `_sampling._install`, cached per clean commit).
Today's Python is BASE's code, in the phase venv (Python 3.12, pandas 3.0.6).
`_switch.py`, beside this file, holds what the checks share.

## Names

- A **transcript** is any `tests/parity/transcripts/**/transcript.json`, the live ones included; they replay offline.
- A **public fixture case** is a case of any `tests/parity/fixtures/*.json` whose `operation` is `client.run`, `schema.from_data`, `decompose.parts` or `driver.materialize`: the operations made of public calls only.
  `driver.materialize` runs as `tests/parity/_fixture_ops.py` runs it, except that it reads `Request.payloads` and resets nothing of the engine, so each process keeps its own once-only warnings (`_switch.driver_materialize`).
- A **ported package** is `predictive_query` or `sdm_relational_engine`.
- **The packages** are `sdm_client`, `sdm_connectors`, `sdm_api` (their `src` trees) and the Python of `sdm_core`.

## The selector

The environment variable `SDM_CLIENT_IMPLEMENTATION` chooses the path for a whole process (DV31):

| Value | The process runs |
|---|---|
| unset, or `core` | the core, which is what users get |
| `python` | today's Python implementation, as at BASE, with the engine and `predictive_query` |

The checks set it only in the environment of a process they start, never while a process runs, and never mix objects of the two paths.
On the `python` path every public call behaves as at BASE, apart from the fixes listed in `fix_changes.json`, which the old path keeps as bugs.
Phase 10 deletes the selector with the old code.

## The core path

On the core path, with the selector unset:

- no ported package is imported by anything a public call does (P9.imports);
- `sdm_core._native` is loaded;
- no ported code is reached from the packages (P9.unreached);
- every warning is given with `caller_stacklevel()`, so it lands on the line of the caller's code that made the public call (brief task 1).

Every comparison check also holds the core path to the first two: a comparison of two runs of today's Python proves nothing.
Imports made while a fixture case whose input builds an engine object runs (`$inference_config`, which `tests/parity/_fixture_ops.py` decodes by importing the engine) are not counted; those cases run last, and P9.imports does not judge them either.
The replay harness's own adjustment for the engine (`scripts/parity/_parity/drivers.py`, `engine_hooks`) is made, in the checks' processes, only when the engine is already loaded.

## The shadow comparison (P9.shadow, P9.perf-counts, P9.errors, P9.hashseed)

Each path runs every transcript, and every public fixture case, in a fresh process of its own (`_switch.py capture`).
For a transcript the capture keeps:

- whether `scripts/parity/_parity/corpus.check` replays it as stored;
- every statement the replayed connection ran (SQL and encoded parameters);
- every `Source._run` call (SQL and parameters), in order;
- the payloads, request id normalized; the error (class, code, message, details, status); what the result holds beyond its frame; the warnings (category, text, order, and where each landed); the result's frame;
- the classes of the error's chain (`__cause__`, or `__context__` unless suppressed), and whether the error is a public `sdm_client` error.

For a public fixture case it keeps its output or its error (as `expected_error` writes one), its warnings, its `Source._run` calls and its error's chain.

**Changed answers.**
`tests/rust_core/shadow/fix_changes.json`, the implementer's, is a JSON object of transcript names and public fixture case ids to `{"fix": ID, "why": TEXT}`: each a scenario whose answer a fix of brief task 7b changes.
`ID` is one of `composite-keys-on-sql`, `null-foreign-key`, `sqlite-unreadable-time-bounds`, `time-of-day-column`, `multilabel-link-prediction`, `empty-table-from-source`, `table-emptied-after-read`, `complex-period-interval-columns`, `about-the-part-forecast-step-1`, `column-named-like-a-derived-key`, `reserved-name-added-after-schema`, `dropped-target-or-entity-column`, `int-past-uint64`, `decimal-entity-keys`, `uint64-entity-keys`, `arrow-large-string-feature`, `arrow-duration-feature`, `arrow-dictionary-feature`, `arrow-large-string-key`, `arrow-time32-feature` or `arrow-time64-feature`; `TEXT` has five words or more.
Under DV42 and DV44, `uint64-entity-keys` covers the brief's task 7b fix for in-memory uint64 entity keys above i64::MAX, preserving their exact values as the text their schema declares instead of refusing their integer range.
`arrow-large-string-feature` covers only the addendum's large-string Arrow feature refusal changing from the legacy `INTERNAL_ERROR` to a descriptive `UNSUPPORTED_REQUEST`.
Under DV45, `arrow-duration-feature`, `arrow-dictionary-feature` and `arrow-large-string-key` cover only the named Arrow dtype and role changing from the legacy `INTERNAL_ERROR` to a descriptive `UNSUPPORTED_REQUEST`.
`arrow-time32-feature` and `arrow-time64-feature` cover only those Arrow feature dtypes changing from the legacy date/categorical mismatch to a descriptive `UNSUPPORTED_REQUEST`.
These IDs authorize no other dtype or role difference.
Every entry must differ between the paths, replay as stored on the core (the transcript or fixture re-recorded, which the corpus lock holds for Adi's approval), and not end in the engine's unforeseen-failure wrapper; when the file lists anything, `CHANGELOG.md` must have changed since BASE.
Every scenario not listed must be the same on both paths.

## Ported code (P9.unreached)

`ported.tsv` is the porting ledger's rows sent to a core crate when these checks were locked, less what stays as the public wrappers (brief task 2): the top-level code of `sdm_client/schema.py`, which defines public classes, and the public and dunder methods of `sdm_client`'s public classes.
A function, method, class body, lambda or comprehension is ported code when the innermost of those rows enclosing it is listed; a module's own top-level code when its `<module>` row is.

`_unreached.py`'s guard watches every start and resume of ported code, in any thread, during the suite-rest command and during `run_corpus.py replay`, both on the core.
It walks up the stack to the first frame that is ported code, test or harness code (`tests/`, `packages/*/tests/`, `scripts/`, any `conftest.py`), or code of the packages; code of the ported packages that is not ported, the standard library and third parties are passed through.
Reached from the packages is a breach; reached from test code is a test of the old implementation, which is allowed; reached from ported code is judged where that started.
Tests under `tests/rust_core/` are not judged.

## Pickles (P9.pickle)

`pickles/` was written by BASE's code (`_pickles.py make`): the schemas, requests and results a few in-memory transcripts' public calls make, answered by the parity stand-in model, and a schema read from DuckDB; each pickled when it can be, and in `expected.json` what it was, did and gave after `pickle`, `copy.copy`, `copy.deepcopy` and `snapshot()`, a schema copy's relationships then cleared.
Descriptions use public attributes only.
A raised error is compared by class alone.
The reads of the DuckDB schema, and of its first question on each copy, are compared too.

## The pandas 2 environment (P9.pandas2)

The second venv is `~/.local/share/sdm-dbx/venv-rc-p9-pd2` (`SDM_P9_PANDAS2_VENV` overrides it): Python 3.10, made with uv when missing, and on every run given the packages as GitHub's test job installs them (editable), the engine, this checkout's binding and `pandas>=2,<3`.
`tests/rust_core/shadow/pandas2_differences.json`, the implementer's, is `{"tests": {TEST_ID: CAUSE}, "replay": {TRANSCRIPT: CAUSE}}`: every test of the suite-rest command that fails or errors there, and every transcript (by its path under `tests/parity/transcripts/`) that does not replay there, with its cause in six words or more.
Every one listed must still differ.

## Tests (P9.tests)

`base.json` holds BASE's facts in the phase venv: the test ids the suite-rest command collects, how many passed and were skipped, and the assertions in its test code (`assert` statements, and calls of `pytest.raises`, `pytest.warns`, `pytest.approx` and any `assert_*`).
The implementer's shadow tests, under `tests/rust_core/shadow/`, count for none of these.
`tests/rust_core/shadow/changed_tests.tsv`, the implementer's, lists every `.py` file under the suite-rest paths (outside `tests/rust_core/shadow/` and this directory) added, changed, deleted or renamed since BASE, one per line, `PATH<TAB>WHAT CHANGED`, in five words or more, saying where any moved assertion now is.

## Wheels (P9.wheels)

Linux wheels come from Phase 4's cibuildwheel build in Docker; macOS wheels from `maturin build --release --target aarch64-apple-darwin` and `--target x86_64-apple-darwin`, which needs that Rust target installed for the pinned toolchain.
The four pure-Python distributions are built with `uv build --wheel`.
A clean Python 3.12 environment for each platform (on this Mac natively and under Rosetta; `python:3.12-slim` for `linux/amd64` and `linux/arm64`) installs `sdm-client` from them, third parties from the index, and runs `tests/rust_core/smoke.py` and `client_smoke.py` outside any checkout.
`sdm-client` requires `sdm-core` at its own version and `pyarrow`, neither as an extra, and the `sdm-core` wheels carry that version.
In cibuildwheel's Linux test step each wheel passes the binding's tests in its build container.
Where the x86_64 container runs under Rosetta (`vendor_id` `VirtualApple` in `/proc/cpuinfo`), `_rosetta.py` skips only `test_hostile.py::test_random_numbers_refuse_hostile_input_without_panicking`: Rosetta ends the process on its 4 PB allocation instead of failing it.
The aarch64 container runs every test, and the full x86_64 suite runs on real x86 in Phase 9.5's the Databricks validation workspace jobs.

## Performance (P9.perf-cpu)

Seven rounds, alternating which path goes first, each replaying every transcript once per path in a fresh process; a process's `time.process_time()` from before the first transcript to after the last.
The core's median must not exceed the old path's.
Both paths leave out the transcripts a brief task 7b fix re-recorded with the core's answer, which the old path cannot replay as stored by design: exactly the transcripts `fix_changes.json` lists, which P9.shadow compares.

## Fix evidence (P9.fixes-review)

Each item enumerated by P9.fixes-review is covered by an item-to-execution manifest: each actual correction, including an inherited correction, has the same behavioral regression assertion failing on an identified unmodified immutable pre-fix product revision and passing on the final revision, with historical executions distinguished from retrospective executions; an independently confirmed retained behavior or policy has BASE evidence and a nonvacuous final proof with a seeded negative control, without claiming an earlier failure.
The enumerated items, final-SHA evidence requirement, and review of every `fix_changes.json` entry and its CHANGELOG line remain unchanged.
