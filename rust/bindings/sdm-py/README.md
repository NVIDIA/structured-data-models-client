# sdm-py

The Python binding: builds the extension module `sdm_client._native` into the client wheel, with maturin.
Main class: `Run`, with `start`, `next`, `reply`, `cancel`, `take_attachment` and `take_warnings`; control messages cross as `bytes`, Arrow crosses through the PyCapsule interface (`__arrow_c_stream__`).
The Python side that drives it is `packages/structured-data-models-client/src/sdm_client/_rust/`, and its type stub is `sdm_client/_native.pyi`.
It depends on exactly `sdm-host` and `sdm-contract`, plus the third-party `pyo3`, `pyo3-log`, `arrow-pyarrow`, `arrow-array` and `arrow-schema`.
It releases the GIL during core calls.
It calls into Python only to convert and to log: it reads Arrow through the producer's `__arrow_c_stream__`, and it forwards the core's logs to Python's `logging`, whose handlers can raise during a core call (an `Exception` goes to `sys.unraisablehook`, an interrupt is raised in place of the answer).
Must not hold: domain logic, a direct dependency on `sdm-core`, a database driver, or a call into Python that supplies domain answers.
Tests in `tests/` run against the built module, with or without the `test-ops` feature.

```text
used by:     Python, as sdm_client._native
     |
sdm-py
     |
depends on:  sdm-host, sdm-contract
```
