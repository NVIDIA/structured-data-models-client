# Prerequisites

- **Python** 3.10 to 3.14 for the client, the language, the wire types and the
  connectors. **3.10 to 3.13 for the relational engine**, which is a native
  build with a shorter wheel matrix. `pip install 'structured-data-models-client[relational]'` does
  not resolve on 3.14, while plain `pip install structured-data-models-client` does. It is
  reported as `No matching distribution found for structured-data-models-relational-engine`,
  which names the package and not the reason.
- **A reachable NIM** serving `kumo-tabular` or `kumo-relational`. The client
  runs no model and downloads no weights.
- **No GPU** on the client. The NIM owns that.

## Platforms

The client does its work in its core, `sdm-core`, a native module that `pip install structured-data-models-client` brings with it.
Its wheels are built for:

- Linux, x86-64 and aarch64, glibc 2.28 or later;
- macOS, Apple silicon and Intel.

Windows is not built, so the client does not install there.
Every Python from 3.10 on uses the same wheel, and both models work with the base install, on a laptop as on a cluster.

The legacy relational engine (`structured-data-models-client[relational]`) publishes Linux wheels for CPython 3.10 through 3.13.
The client no longer needs it, and it is kept only until it is removed.

## Getting a NIM

Use either option:

**A hosted endpoint.** Find the model in the NVIDIA API catalog, generate a
key, and pass the endpoint and key to `SDMClient`.

**The container, run yourself.** Pull the NIM and run it on your own GPU host,
then point the client at it. Use this when the data cannot leave your
environment.

Either way the client needs only the URL and, for a hosted endpoint, the key.
It reads nothing from the environment.
