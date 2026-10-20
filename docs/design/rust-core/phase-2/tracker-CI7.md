## CI7: aarch64 wheels and runtime cells

CI7 builds the client wheel for aarch64 Linux beside x86_64, smoke-tests it on Arm, and checks the install on each promised Databricks runtime's own packages (decisions KO2-38 and KO2-40).
It changes judging files (`ci.yml` and scripts under `scripts/rust_core/`), so it lands alone.

### What changed, by exit

1. **aarch64 wheels.**
   `native-wheel (aarch64, shipped)` and `native-wheel (aarch64, test-ops)` build on `ubuntu-24.04-arm` with `native-wheel`'s own steps, in PyPA's `manylinux2014_aarch64` image pinned by digest.
   rustup-init is checked against each architecture's published SHA-256.
   `native_wheel.py` knows each architecture's manylinux2014 tag, and `--check-set` refuses a shipped set without exactly one wheel per architecture of one distribution and version.
   `rust_notices.py` reads the crates both targets link; the list is unchanged, and the notices file's header names both targets.
2. **aarch64 smoke.**
   `wheel-smoke (aarch64-floor)` and `wheel-smoke (aarch64-newest)` check the wheel set, then run `wheel_smoke.py`'s floor and newest lanes on Arm, which also require the interpreter to run on aarch64.
3. **Runtime cells.**
   Six `runtime-install (<runtime>)` jobs, named as locked check 10 names them, preinstall one runtime's Python, NumPy, pandas, pyarrow and PyYAML.
   They install the candidate's wheels without dependencies, then run the installation guide's line verbatim.
   They require the runtime's NumPy kept, pandas at or above the client's floor 2.2.2 (KO2-37), `pip check` clean and the candidate's wheels still in place, print the final versions, and run the binding's and the client's tests.
   DBR 14.3's release notes list no PyYAML, so its cell preinstalls none.
   DBR 16.4 ships pandas 1.5.3 built by Databricks for Python 3.12; PyPI has neither a wheel for it nor a source build that works there, so that cell starts with no pandas rather than a stand-in.
4. **Tests.**
   `tests/test_ci_workflow.py` covers the new jobs, locked check 10's patterns and the seeded faults: a wheel set missing an architecture, an install that moves NumPy, a resolution failure, pandas left below the floor, a failing `pip check` and a swapped wheel.
   The test author's `CI7 tests:` commit makes the client tests that the runtime cells exposed correct for each supported pandas, pyarrow and NumPy version.

5. **Register.**
   P2J-1 and P2J-2, proposed for the reviewers, record two old-route differences the runtime cells exposed: the old route's crash on a NumPy 1 `numpy.uint64` context size, which the core takes, and the old route's code for a year-0001 text date, INVALID_REQUEST under pandas 2 where DT-14 measured INTERNAL_ERROR under pandas 3.

### Open

- The aarch64 build image comes from quay.io by digest; mirroring it to ghcr.io is the owner's (on the morning list).

### For the landing

Protected paths changed, for the Protected-paths trailer:
`.github/workflows/ci.yml`, `scripts/rust_core/native_wheel.py`, `scripts/rust_core/rust_notices.py`, `scripts/rust_core/wheel_smoke.py`, `packages/structured-data-models-client/THIRD-PARTY-NOTICES-RUST.txt`, `docs/design/rust-core/register.yaml` (P2J-1 and P2J-2), and the test author's `packages/structured-data-models-client/tests/test_dates_the_column_decides.py`, `test_list_valued_columns.py`, `test_semantic_model_source.py`, `test_the_materialize_route.py` and `test_the_schema_route.py` (same directory).
