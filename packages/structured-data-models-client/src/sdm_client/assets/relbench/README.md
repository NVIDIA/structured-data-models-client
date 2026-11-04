<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# RelBench catalog

`registry.json` records immutable Hugging Face repository revisions, Parquet SHA-256 hashes and published table metadata.
Regenerate it with `scripts/pin_relbench.py`; pass the existing `repository=revision` pairs to preserve pins.
The initial restored registry is byte-identical to the previously shipped RelBench catalog.
`overrides.json` contains the retained HM/Amazon column exclusions and SALT semantic-type declarations.
These declarations are shared by the Python wheel and Java JAR; neither loader contains dataset-specific policy.
No dataset contents are shipped.
