// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.provider.DataProvider;
import java.util.Objects;

/**
 * A source whose data the caller's own {@link DataProvider} reads: no database connection, and no
 * SQL. Use it wherever the library takes a {@link Source}.
 *
 * <pre>{@code
 * Source source = ProviderSource.of(myProvider);
 * Schema schema = Schema.fromSource(source, options);
 * }</pre>
 *
 * <p>{@link Readers#readTable} and {@link Readers#readQuery}, which run SQL, refuse it with {@code
 * UNSUPPORTED_REQUEST}.
 *
 * @param provider the provider answering every data request; the library never closes it
 */
public record ProviderSource(DataProvider provider) implements Source {
  public ProviderSource {
    Objects.requireNonNull(provider, "provider");
  }

  public static ProviderSource of(DataProvider provider) {
    return new ProviderSource(provider);
  }
}
