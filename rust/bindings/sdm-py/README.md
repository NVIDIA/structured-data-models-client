# sdm-py

The Python binding: builds the extension module `sdm_client._native` into the client wheel, with maturin.
Main class: `Run`, with `start`, `next`, `reply`, `cancel`, `take_attachment` and `take_warnings`; control messages cross as `bytes`, Arrow crosses through the PyCapsule interface (`__arrow_c_stream__`).
The Python side that drives it is `packages/structured-data-models-client/src/sdm_client/_rust/`, and its type stub is `sdm_client/_native.pyi`.
Depends on exactly `sdm-host` and `sdm-contract`, plus `pyo3`, `pyo3-log` and `arrow-pyarrow`.
It releases the GIL during core calls and sends the core's logs to Python's `logging`.
Must not hold: domain logic, a direct dependency on `sdm-core`, a database driver, or calls back into Python code.
Tests in `tests/` run against the built module, with or without the `test-ops` feature.
