// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.provider;

import java.sql.SQLException;
import java.util.Optional;

/**
 * What a provider's failure tells the core: each field where the failure says one.
 *
 * @param sqlState the SQLSTATE, as a JDBC driver gives it
 * @param vendorCode the database's own error code
 * @param errorClass the source's own error class name, where it has one
 * @param message the internal diagnostic; logged by the core, never shown to end users
 */
public record ProviderFailure(
    Optional<String> sqlState,
    Optional<String> vendorCode,
    Optional<String> errorClass,
    Optional<String> message) {

  /** No facts beyond the failure's class. */
  public static ProviderFailure none() {
    return new ProviderFailure(Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
  }

  /** A {@link SQLException}'s state, vendor code and message; for any other failure, its message. */
  public static ProviderFailure of(Exception failure) {
    if (failure instanceof SQLException sql) {
      return new ProviderFailure(
          Optional.ofNullable(sql.getSQLState()),
          sql.getErrorCode() == 0 ? Optional.empty() : Optional.of(Integer.toString(sql.getErrorCode())),
          Optional.empty(),
          Optional.ofNullable(sql.getMessage()));
    }
    return new ProviderFailure(
        Optional.empty(), Optional.empty(), Optional.empty(), Optional.ofNullable(failure.getMessage()));
  }
}
