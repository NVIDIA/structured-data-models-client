# Phase 8.7 acceptance contract

The checks in this directory keep the request coverage-gap list (`scripts/parity/request_coverage_gaps.json`) true by machine.
They hold it to what Phase 8.6's coverage-gap verifier found (`investigations/P86-coverage-gaps-verifier.md` in the migration harness), and they turn the verifier's method into checks over the whole list.
This file is the interface they test through.
Everything it doesn't name is the implementer's choice.

Phase 8.5's contract (`tests/rust_core/request/CONTRACT.md`) and Phase 8.6's (`tests/rust_core/request_gaps/CONTRACT.md`) still hold.
Today's Python, in the phase venv, is the reference throughout, and no file under `packages/` changes, so every line number below still points at the code the verifier read.

## Names

- The **base** is `484ff35`; `base.json` holds the gap list and the fixture exclusions as they stood there.
- A **named entry** is one of the gap-list entries `_gaplist.py` lists in `ENTRIES`, by module and branch, in one of two groups.
  - **Group A** (30): a branch a public call reaches that no held case or host test took at the base.
  - **Group B** (10): a branch still unreachable whose base reason is false or names the wrong guard.
  - Each may carry the clauses of its base reason the verifier showed false, and, for group B, patterns naming the guard the verifier found.
- An entry **says "reachable only"** when its reason holds the words `reachable only`, in any case, anywhere.
- A **held case** is a stored transcript (`tests/parity/transcripts/`) or a request fixture case (an operation starting `client.`, `decompose.`, `payload.`, `tabular.` or `wire.`) that `scripts/parity/request_fixture_exclusions.json` does not exclude.
  Phase 8.5's checks, which P87.p86-core runs again, hold the core to every held case.
- A **recorded case** is a stored transcript or any fixture case under `tests/parity/fixtures`, of any operation, excluded or not.
- A **host test** is a test pytest collects under `tests/rust_core` (its `test_*.py` files, in any subdirectory).
  A host test that holds a route runs it through today's Python and through the core and compares class, code, message, `details` and warnings.
- A reason **cites** a host test by its pytest node id, written in full: `tests/rust_core/<path>.py::<test>` or `tests/rust_core/<path>.py::<test>[<param id>]`.
  A file name with a param id in parentheses is not a citation.
- A cited host test **holds** an entry when it is collected, passes (setup, call and teardown), makes at least one call into `sdm_core._native` (a function of it, or a method of an object of its types), and takes the entry's branch in today's Python.

## Measures

Both measure branch coverage of `sdm_relational_engine`, `sdm_client`, `sdm_connectors` and `predictive_query`, as `scripts/parity/coverage.py` measures them, over every module the gap list names now or at the base.

- `_corpus.py` replays every stored transcript and runs every fixture case through the Python fixture runner (`tests/parity/_fixture_ops.py`'s `run`).
  It reports the branches the held cases take (`held`), and those every recorded case takes (`all`).
- `_host.py` is a pytest plugin.
  `_gaplist.host_measure` first builds `sdm_core._native` from this checkout and installs it, then runs every host test with the plugin and reports each test's outcomes, its calls into `sdm_core._native`, and the branches it took.

## Group A (P87.taken)

- Every group A entry was in the gap list at the base.
- At least the base's 137 transcripts and 572 held fixture cases run, and every transcript replays.
- `request_fixture_exclusions.json` excludes no case the base did not.
- A group A entry gone from the list is taken by a held case.
- A group A entry still listed says "reachable only", holds none of its false clauses (case-insensitive, runs of whitespace as one space), and cites at least one host test; every host test it cites holds it.

## The whole list (P87.reachable-only)

- Every listed entry that says "reachable only" cites at least one host test, and every host test it cites holds it.
- No recorded case takes the branch of any listed entry.
- No host test takes the branch of a listed entry that does not say "reachable only".
  An entry a host test takes says so, or the route is a gap the list is wrong about.

## Group B (P87.reasons)

- Every group B entry was in the gap list at the base.
- A group B entry still listed has a reason other than its base reason (runs of whitespace as one space), holds none of its false clauses, and matches each of its guard patterns (case-insensitive regular expressions in `_gaplist.py`).
  The patterns accept the code's own names or a `file:line` citation of the guard the verifier found; for example `client.py:164-168` or `client.py:415` for `adapters/relational.py` [737,738].
- A group B entry gone from the list is taken by a held case.

## Phase 8's list (P87.phase8-gaps)

- A row of `tests/parity/COVERAGE_GAPS.md` for a group A branch, if it is still there, claims nowhere that the branch cannot be taken (`never taken`, `never returns`, `never sends`, `never settled`, `never reached`, `never asked`, `unreachable`, and the two false claims of `wire/keys.py` [240,241] and [277,278]).
- It names the host test under `tests/rust_core` that holds the route, by path.
- Whether the corpus misses each listed branch is Phase 8's check (P8.coverage), which P87.p86-regress runs.

## Not checked here

That each reason is true, and that each cited host test compares the core with today's Python on the route it is cited for, is the independent verifier's check (P87.coverage-gaps).
