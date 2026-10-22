# Prerequisites

- **Python** 3.10 to 3.14, for every distribution. The relational engine is a
  native build publishing CPython wheels and no source distribution, so it
  supports an interpreter only once it has built wheels for it, and PyPy not
  at all: on a Python newer than that,
  `pip install 'structured-data-models-client[relational]'` does not resolve while plain
  `pip install structured-data-models-client` does. It is reported as `No matching
  distribution found for structured-data-models-relational-engine`, which
  names the package and not the reason.
- **A reachable NIM** serving `kumo-tabular` or `kumo-relational`. The client
  runs no model and downloads no weights. The one exception is a client
  opened with `SDMClient.in_process()`, which runs the model itself; see the
  [quickstart](quickstart.md#a-model-in-this-process).
- **No GPU** on the client. The NIM owns that. A client opened with
  `SDMClient.in_process()` uses one when there is one, and the CPU otherwise.

## Platforms

The client, the language and the connectors are pure Python and install
anywhere. The relational engine is a native build: it compiles a neighbour
sampler, and published wheels are `manylinux_2_28` for x86-64 and aarch64,
macOS 11 or newer for Apple silicon and Intel, and 64-bit Windows, on CPython
only. Both models work on a laptop. What has no wheel has nothing to fall back
on, since the engine ships no source distribution: musl, a glibc older than
2.28, a 32-bit interpreter, or Windows on Arm.

## Getting a NIM

Use either option:

**A hosted endpoint.** Find the model in the NVIDIA API catalog, generate a
key, and pass the endpoint and key to `SDMClient`.

**The container, run yourself.** Pull the NIM and run it on your own GPU host,
then point the client at it. Use this when the data cannot leave your
environment.

Either way the client needs only the URL and, for a hosted endpoint, the key.
It reads nothing from the environment.
