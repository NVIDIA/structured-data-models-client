<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared RelBench loader fixture

Python and Java tests read these exact Parquet bytes through their maintained readers.
The two customers have IDs 1/2, names Ada/Bob and an excluded text column.
The two orders have IDs 10/11, customer IDs 1/2, timestamps 2024-01-01/02 and amounts 2.5/3.5.
The manifest explicitly gives orders no primary key, despite its unique ID column, to detect accidental key inference.
The override excludes the extra customer column and declares names as text.
`registry.json` hashes the files; these are synthetic test data, not downloaded benchmark rows.
