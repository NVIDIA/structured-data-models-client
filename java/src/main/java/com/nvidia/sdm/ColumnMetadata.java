// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

/** A column's existing type descriptors. */
public record ColumnMetadata(String name, String dtype, String stype) {}
