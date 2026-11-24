// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.provider;

import java.util.Objects;
import java.util.Optional;

/**
 * What a source's failure means, as the core reads it.
 *
 * @param notFound the table asked for does not exist under the name asked; a name tried in another
 *     spelling is tried next
 * @param refusesExpression the source refused an expression it was asked to read, for what it says
 *     or reads
 * @param unusable why a table that failed to be read cannot be used
 * @param denied the source denied the caller access to what was asked
 * @param reason what went wrong, where the failure says
 */
public record FailureFacts(
    boolean notFound,
    boolean refusesExpression,
    Optional<Unusable> unusable,
    boolean denied,
    Optional<Reason> reason) {

  public FailureFacts {
    Objects.requireNonNull(unusable, "unusable");
    Objects.requireNonNull(reason, "reason");
  }

  /** No fact: a failure that says nothing more than its codes. */
  public static FailureFacts none() {
    return new FailureFacts(false, false, Optional.empty(), false, Optional.empty());
  }

  /** Why a table the source failed to read cannot be used. */
  public enum Unusable {
    /** It does not exist, or the caller may not see it. */
    MISSING,
    /** The caller may not read it. */
    FORBIDDEN
  }

  /** What went wrong, where a source's failure says. */
  public enum Reason {
    VALUE_NOT_CONVERTED,
    DIVISION_BY_ZERO,
    OUT_OF_RANGE,
    COLUMN_UNRESOLVED,
    TABLE_UNRESOLVED,
    PERMISSION_DENIED,
    SYNTAX_REJECTED,
    TYPES_INCOMPATIBLE
  }
}
