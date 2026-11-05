// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.util.List;

/**
 * What a model supports, as far as this client knows, as Python's {@code ModelCapabilities}: the
 * core's own table of the models ({@code source} is {@code sdk}: nothing here is read from the
 * endpoint).
 *
 * @param model the model's id
 * @param tasks the task kinds it answers, sorted
 * @param outputs the output fields it gives, sorted
 * @param singleTable whether it reads a schema of one table
 * @param multipleTables whether it reads several
 * @param relationships whether it reads relationships between them
 * @param explanationFormats the explanation shapes it gives, when it explains
 * @param sessions whether it pins a context in a session
 * @param source {@code sdk}: the client's description, not the endpoint's answer
 */
public record ModelCapabilities(
    String model,
    List<String> tasks,
    List<String> outputs,
    boolean singleTable,
    boolean multipleTables,
    boolean relationships,
    List<String> explanationFormats,
    boolean sessions,
    String source) {
  /** Copies the lists. */
  public ModelCapabilities {
    tasks = List.copyOf(tasks);
    outputs = List.copyOf(outputs);
    explanationFormats = List.copyOf(explanationFormats);
  }
}
