// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

/** Signed aggregate attribution for one table column. */
public record FeatureImportanceRow(String table, String column, double score) {}
