// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

/** A detached directed relationship. */
public record LinkMetadata(String sourceTable, String sourceColumn, String targetTable, String targetColumn) {}
