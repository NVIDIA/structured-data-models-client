# sdm-host

The plain-Rust layer every language binding shares, so a binding only converts types.
Main types: `HostRun` (owns a run, contains panics, handles a call while the run is busy), `encode` and `decode` (control messages, with the 64 KB cap in `CONTROL_CAP`), `admit` (checks Arrow the host hands in), `ZoneResolver` (time zones from the host's directories, then the bundled copy).
`default_root()` builds the composition root with `sdm-data`'s data layer.
Depends on `sdm-core` to run operations, `sdm-data` to plug in the data layer, and `sdm-contract` for messages.
Must not hold: domain rules (they belong in the core), a language runtime such as `pyo3` or `jni`, or a database driver.
