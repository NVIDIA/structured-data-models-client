# Prerequisites

- **CPython 3.10 through 3.14** with a compatible client wheel.
- **A reachable NIM** serving `kumo-tabular` or `kumo-relational`. The client
  runs no model and downloads no weights. The one exception is a client
  opened with `SDMClient.in_process()`, which runs the model itself; see the
  [quickstart](quickstart.md#a-model-in-this-process).
- **No GPU** on the client. The NIM owns that. A client opened with
  `SDMClient.in_process()` uses one when there is one, and the CPU otherwise.

## Platforms

Both models use the native Rust core included in the client wheel.
Release builds target Linux x86-64 and aarch64, macOS Apple silicon and Intel, and 64-bit Windows.
A matching CPython wheel is required; PyPy, musl Linux, 32-bit interpreters and Windows on Arm are not supported release targets.
The API types and connector distributions are pure Python, but using the client still requires its native wheel.

## Getting a NIM

Use either option:

**A hosted endpoint.** Find the model in the NVIDIA API catalog, generate a
key, and pass the endpoint and key to `SDMClient`.

**The container, run yourself.** Pull the NIM and run it on your own GPU host,
then point the client at it. Use this when the data cannot leave your
environment.

Either way the client needs only the URL and, for a hosted endpoint, the key.
It reads nothing from the environment.
