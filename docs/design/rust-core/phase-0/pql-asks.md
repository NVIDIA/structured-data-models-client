# Asks to the PQL team

Draft for the owner to send; answers are tracked here.
Agreement is not a Phase 0 exit.

## Draft message

We plan to use your PQL core through its Rust crate inside the sdm-client Rust core, which ships in our Python wheel and, later, a Java JAR.
To do that safely we'd like to agree on:

1. **No exceptions across the boundary:** a C interface, or a Rust crate whose bridged functions all return errors as values, so no C++ exception ever reaches Rust.
2. **Builds for every platform we ship:** Linux x86_64 and aarch64 (manylinux) and macOS, as static libraries with private C++ symbols, so your Python package and ours can load in one process.
3. **A versioned validated-query format:** JSON with a JSON Schema, plus stable error codes and source spans.
4. **Matching releases:** the same version on crates.io and PyPI, pinned by version and digest on our side.
5. **Safe behavior in a host process:** no aborts, signal handlers or console output from the library, and a recursion limit on nesting.
6. **One PQL dialect** agreed between your core and our current grammar, and a shared conformance suite (we can contribute 417 cases) run in your CI.
7. **When the Rust crate will be available,** so we know how long to keep our interim parser.

## Answers

| Ask | Answer | Date |
|---|---|---|
