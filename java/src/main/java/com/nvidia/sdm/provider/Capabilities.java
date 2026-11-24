// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.provider;

import java.math.BigInteger;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.Set;

/**
 * What a {@link com.nvidia.sdm.Source} answers, and how; built with {@link #builder()}.
 *
 * <p>Every source answers {@code describe} and {@code scan}. Each other request is answered by the
 * source when declared in {@link #optional}; otherwise the core answers it itself, in memory, from
 * the rows a {@code scan} reads, within {@link #fallback}'s limits, or refuses it with {@code
 * UNSUPPORTED_REQUEST} when no fallback is declared.
 *
 * <p>The rest describe the source to the core: how it names its tables, what its column types
 * mean, and how it reads what the core asks. Each default is a plain source of tables; the
 * built-in sources declare their engine's. Each accessor's documentation is its builder method's.
 */
public final class Capabilities {

  /** Whether a sample drawn with the same seed is the same rows every time. */
  public enum Sampling {
    SEEDED_STABLE,
    UNSEEDED
  }

  /** Where nulls go when rows are ordered. */
  public enum NullOrder {
    FIRST,
    LAST
  }

  /** How the core reads a time the source stores as text. */
  public enum TextTimes {
    /** Each value on its own, as pandas reads one ({@code to_datetime}). */
    VALUE_BY_VALUE,
    /** By the client's date rules, mixed offsets read at UTC. */
    CLIENT_RULES
  }

  /** Which of the source's indexes serve the sampler's reads, for its warning of missing ones. */
  public enum IndexAdvice {
    /** An index over exactly the one column read. */
    SINGLE_COLUMN,
    /** Any index whose first column is the read's first column. */
    LEADING_COLUMN
  }

  /** How the source compares text, where the core compares it for the source in memory. */
  public enum TextOrder {
    /** By the exact UTF-8 bytes: no locale, case folding or normalization. */
    BINARY_UTF8
  }

  /** The most rows and bytes the core reads to answer a request in memory. */
  public record Fallback(long maxRows, long maxBytes) {
    public Fallback {
      if (maxRows < 1 || maxBytes < 1) {
        throw new IllegalArgumentException("fallback limits must be positive");
      }
    }
  }

  /**
   * The semantic model a source is read through: its name, and whether its expressions calling
   * functions other than pure built-ins are kept (they run as SQL under the caller's credentials).
   */
  public record SemanticModel(String name, boolean trustExpressions) {
    public SemanticModel {
      if (name == null || name.isEmpty()) {
        throw new IllegalArgumentException("a semantic model needs its name");
      }
    }
  }

  private final Set<OptionalRequest> optional;
  private final OptionalLong maxKeysPerRequest;
  private final Sampling sampling;
  private final BigInteger maxSeed;
  private final Optional<NullOrder> nullOrder;
  private final Optional<Fallback> fallback;
  private final Set<DataRequest.FilterKind> filters;
  private final Optional<TextOrder> fallbackTextOrder;
  private final Optional<TextTimes> textTimes;
  private final Optional<IndexAdvice> indexAdvice;
  private final Optional<String> engine;
  private final List<String> namespace;
  private final boolean askNamespaceFirst;
  private final boolean retryUpperCaseNames;
  private final boolean columnTypesAreAffinities;
  private final boolean rereadTablesEveryRun;
  private final Optional<SemanticModel> semanticModel;
  private final boolean derivedRelations;
  private final boolean sampleByRowCount;
  private final boolean sampleByKeyType;
  private final boolean neighborLookback;
  private final boolean keysOptional;
  private final boolean keyEnforcement;
  private final boolean comparesStoredTimes;
  private final boolean nanIsMissing;
  private final Optional<String> opaqueVendor;

  private Capabilities(Builder b) {
    optional = Set.copyOf(b.optional);
    maxKeysPerRequest = b.maxKeysPerRequest;
    sampling = b.sampling;
    maxSeed = b.maxSeed;
    nullOrder = b.nullOrder;
    fallback = b.fallback;
    filters = Set.copyOf(b.filters);
    fallbackTextOrder = b.fallbackTextOrder;
    textTimes = b.textTimes;
    indexAdvice = b.indexAdvice;
    engine = b.engine;
    namespace = List.copyOf(b.namespace);
    askNamespaceFirst = b.askNamespaceFirst;
    retryUpperCaseNames = b.retryUpperCaseNames;
    columnTypesAreAffinities = b.columnTypesAreAffinities;
    rereadTablesEveryRun = b.rereadTablesEveryRun;
    semanticModel = b.semanticModel;
    derivedRelations = b.derivedRelations;
    sampleByRowCount = b.sampleByRowCount;
    sampleByKeyType = b.sampleByKeyType;
    neighborLookback = b.neighborLookback;
    keysOptional = b.keysOptional;
    keyEnforcement = b.keyEnforcement;
    comparesStoredTimes = b.comparesStoredTimes;
    nanIsMissing = b.nanIsMissing;
    opaqueVendor = b.opaqueVendor;
  }

  /** A builder starting from the defaults ({@link Builder}). */
  public static Builder builder() {
    return new Builder();
  }

  /** A builder starting from these capabilities. */
  public Builder toBuilder() {
    Builder b = new Builder();
    b.optional.addAll(optional);
    b.maxKeysPerRequest = maxKeysPerRequest;
    b.sampling = sampling;
    b.maxSeed = maxSeed;
    b.nullOrder = nullOrder;
    b.fallback = fallback;
    b.filters.addAll(filters);
    b.fallbackTextOrder = fallbackTextOrder;
    b.textTimes = textTimes;
    b.indexAdvice = indexAdvice;
    b.engine = engine;
    b.namespace = namespace;
    b.askNamespaceFirst = askNamespaceFirst;
    b.retryUpperCaseNames = retryUpperCaseNames;
    b.columnTypesAreAffinities = columnTypesAreAffinities;
    b.rereadTablesEveryRun = rereadTablesEveryRun;
    b.semanticModel = semanticModel;
    b.derivedRelations = derivedRelations;
    b.sampleByRowCount = sampleByRowCount;
    b.sampleByKeyType = sampleByKeyType;
    b.neighborLookback = neighborLookback;
    b.keysOptional = keysOptional;
    b.keyEnforcement = keyEnforcement;
    b.comparesStoredTimes = comparesStoredTimes;
    b.nanIsMissing = nanIsMissing;
    b.opaqueVendor = opaqueVendor;
    return b;
  }

  public Set<OptionalRequest> optional() {
    return optional;
  }

  public OptionalLong maxKeysPerRequest() {
    return maxKeysPerRequest;
  }

  public Sampling sampling() {
    return sampling;
  }

  /** The largest sampling seed the source takes, an unsigned 64-bit integer. */
  public BigInteger maxSeed() {
    return maxSeed;
  }

  public Optional<NullOrder> nullOrder() {
    return nullOrder;
  }

  public Optional<Fallback> fallback() {
    return fallback;
  }

  public Set<DataRequest.FilterKind> filters() {
    return filters;
  }

  public Optional<TextOrder> fallbackTextOrder() {
    return fallbackTextOrder;
  }

  public Optional<TextTimes> textTimes() {
    return textTimes;
  }

  public Optional<IndexAdvice> indexAdvice() {
    return indexAdvice;
  }

  public Optional<String> engine() {
    return engine;
  }

  public List<String> namespace() {
    return namespace;
  }

  public boolean askNamespaceFirst() {
    return askNamespaceFirst;
  }

  public boolean retryUpperCaseNames() {
    return retryUpperCaseNames;
  }

  public boolean columnTypesAreAffinities() {
    return columnTypesAreAffinities;
  }

  public boolean rereadTablesEveryRun() {
    return rereadTablesEveryRun;
  }

  public Optional<SemanticModel> semanticModel() {
    return semanticModel;
  }

  public boolean derivedRelations() {
    return derivedRelations;
  }

  public boolean sampleByRowCount() {
    return sampleByRowCount;
  }

  public boolean sampleByKeyType() {
    return sampleByKeyType;
  }

  public boolean neighborLookback() {
    return neighborLookback;
  }

  public boolean keysOptional() {
    return keysOptional;
  }

  public boolean keyEnforcement() {
    return keyEnforcement;
  }

  public boolean comparesStoredTimes() {
    return comparesStoredTimes;
  }

  public boolean nanIsMissing() {
    return nanIsMissing;
  }

  public Optional<String> opaqueVendor() {
    return opaqueVendor;
  }

  private List<Object> fields() {
    return List.of(
        optional, maxKeysPerRequest, sampling, maxSeed, nullOrder, fallback, filters,
        fallbackTextOrder, textTimes, indexAdvice, engine, namespace, askNamespaceFirst,
        retryUpperCaseNames, columnTypesAreAffinities, rereadTablesEveryRun, semanticModel,
        derivedRelations, sampleByRowCount, sampleByKeyType, neighborLookback, keysOptional,
        keyEnforcement, comparesStoredTimes, nanIsMissing, opaqueVendor);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof Capabilities that && fields().equals(that.fields());
  }

  @Override
  public int hashCode() {
    return fields().hashCode();
  }

  @Override
  public String toString() {
    return "Capabilities" + fields();
  }

  /**
   * Starts from: no optional requests, no key limit, seeded-stable sampling, seeds up to {@code
   * Long.MAX_VALUE}, nulls last, no fallback, no value filters, and a plain source of tables (every
   * other trait off or empty).
   */
  public static final class Builder {
    private final Set<OptionalRequest> optional = EnumSet.noneOf(OptionalRequest.class);
    private OptionalLong maxKeysPerRequest = OptionalLong.empty();
    private Sampling sampling = Sampling.SEEDED_STABLE;
    private BigInteger maxSeed = BigInteger.valueOf(Long.MAX_VALUE);
    private Optional<NullOrder> nullOrder = Optional.of(NullOrder.LAST);
    private Optional<Fallback> fallback = Optional.empty();
    private final Set<DataRequest.FilterKind> filters =
        EnumSet.noneOf(DataRequest.FilterKind.class);
    private Optional<TextOrder> fallbackTextOrder = Optional.empty();
    private Optional<TextTimes> textTimes = Optional.empty();
    private Optional<IndexAdvice> indexAdvice = Optional.empty();
    private Optional<String> engine = Optional.empty();
    private List<String> namespace = List.of();
    private boolean askNamespaceFirst;
    private boolean retryUpperCaseNames;
    private boolean columnTypesAreAffinities;
    private boolean rereadTablesEveryRun;
    private Optional<SemanticModel> semanticModel = Optional.empty();
    private boolean derivedRelations;
    private boolean sampleByRowCount;
    private boolean sampleByKeyType;
    private boolean neighborLookback;
    private boolean keysOptional;
    private boolean keyEnforcement;
    private boolean comparesStoredTimes;
    private boolean nanIsMissing;
    private Optional<String> opaqueVendor = Optional.empty();

    private Builder() {}

    /** Declares the optional requests the source answers itself. */
    public Builder answers(OptionalRequest... requests) {
      optional.addAll(List.of(requests));
      return this;
    }

    /**
     * The most seeds one {@code neighbors} or {@code lookup} may carry; more reach the source as
     * several requests, whose answers the core joins.
     */
    public Builder maxKeysPerRequest(long max) {
      if (max < 1) {
        throw new IllegalArgumentException("maxKeysPerRequest must be at least 1");
      }
      maxKeysPerRequest = OptionalLong.of(max);
      return this;
    }

    /** Whether a sample drawn with the same seed is the same rows every time. */
    public Builder sampling(Sampling value) {
      sampling = Objects.requireNonNull(value);
      return this;
    }

    /** The largest sampling seed the source takes, a non-negative {@code long}. */
    public Builder maxSeed(long value) {
      return maxSeed(BigInteger.valueOf(value));
    }

    /** The largest sampling seed the source takes, up to the largest unsigned 64-bit integer. */
    public Builder maxSeed(BigInteger value) {
      if (value.signum() < 0 || value.bitLength() > 64) {
        throw new IllegalArgumentException("maxSeed must be an unsigned 64-bit integer");
      }
      maxSeed = value;
      return this;
    }

    /** Where the source puts nulls when it orders rows; null, and it is never asked for an order. */
    public Builder nullOrder(NullOrder value) {
      nullOrder = Optional.ofNullable(value);
      return this;
    }

    /** The limits within which the core answers undeclared optional requests in memory. */
    public Builder fallback(long maxRows, long maxBytes) {
      fallback = Optional.of(new Fallback(maxRows, maxBytes));
      return this;
    }

    /**
     * Declares the value filters the source applies to a sample or scan; the core sends it no other
     * kind, and checks every row it keeps itself.
     */
    public Builder filters(DataRequest.FilterKind... kinds) {
      filters.addAll(List.of(kinds));
      return this;
    }

    /** How the source compares text, where the core's in-memory fallback compares text for it. */
    public Builder fallbackTextOrder(TextOrder value) {
      fallbackTextOrder = Optional.ofNullable(value);
      return this;
    }

    /** How the core reads a time the source stores as text; null, and it refuses to. */
    public Builder textTimes(TextTimes value) {
      textTimes = Optional.ofNullable(value);
      return this;
    }

    /** Which indexes serve the sampler's reads; null, and the core reads no index. */
    public Builder indexAdvice(IndexAdvice value) {
      indexAdvice = Optional.ofNullable(value);
      return this;
    }

    /**
     * The database engine the source is, by the name messages name it by ({@code duckdb}, {@code
     * sqlite}, {@code postgres}, {@code databricks}, {@code snowflake}); its semantic models are
     * read in that engine's SQL.
     */
    public Builder engine(String name) {
      engine = Optional.ofNullable(nonEmpty("engine", name));
      return this;
    }

    /** The namespace a table named by one part is read under, outermost first. */
    public Builder namespace(List<String> parts) {
      for (String part : parts) {
        if (part == null || part.isEmpty()) {
          throw new IllegalArgumentException("each namespace part is a non-empty string");
        }
      }
      namespace = List.copyOf(parts);
      return this;
    }

    /**
     * Whether the core asks the source its namespace (a {@code describe} of the {@code namespace}
     * part, answered in full) before it reads a table named by one part.
     */
    public Builder askNamespaceFirst(boolean value) {
      askNamespaceFirst = value;
      return this;
    }

    /**
     * Whether a table the catalog does not find as named is tried again upper-cased, as an engine
     * folding unquoted names does.
     */
    public Builder retryUpperCaseNames(boolean value) {
      retryUpperCaseNames = value;
      return this;
    }

    /** Whether a column's declared type is only an affinity, so its type is read from its values. */
    public Builder columnTypesAreAffinities(boolean value) {
      columnTypesAreAffinities = value;
      return this;
    }

    /** Whether each run reads the tables as they stand, rather than as a given schema holds them. */
    public Builder rereadTablesEveryRun(boolean value) {
      rereadTablesEveryRun = value;
      return this;
    }

    /** The semantic model the source is read through. */
    public Builder semanticModel(SemanticModel value) {
      semanticModel = Optional.ofNullable(value);
      return this;
    }

    /** Whether the source reads a derived {@link DataRequest.Relation}. */
    public Builder derivedRelations(boolean value) {
      derivedRelations = value;
      return this;
    }

    /**
     * Whether the source sizes a sample by the table's rows, which the core counts first and names
     * on each {@code sample_entities}.
     */
    public Builder sampleByRowCount(boolean value) {
      sampleByRowCount = value;
      return this;
    }

    /**
     * Whether the source orders a seeded sample by the key's data type, which the core names on each
     * {@code sample_entities}.
     */
    public Builder sampleByKeyType(boolean value) {
      sampleByKeyType = value;
      return this;
    }

    /**
     * Whether neighbor reads by a foreign key over a table with a time column look back from each
     * seed only as far as the table's rate of rows says, as warehouse samplers do.
     */
    public Builder neighborLookback(boolean value) {
      neighborLookback = value;
      return this;
    }

    /** Whether a failed read of a table's declared keys is read as no declared keys. */
    public Builder keysOptional(boolean value) {
      keysOptional = value;
      return this;
    }

    /** Whether the catalog answers which tables enforce their keys ({@code key_enforcement}). */
    public Builder keyEnforcement(boolean value) {
      keyEnforcement = value;
      return this;
    }

    /**
     * Whether the source compares a window's bounds with the stored text of the window's column,
     * which the core reads once and names on each bound's seed column.
     */
    public Builder comparesStoredTimes(boolean value) {
      comparesStoredTimes = value;
      return this;
    }

    /** Whether a NaN the source answers is read as missing, and {@code -0.0} as {@code 0.0}. */
    public Builder nanIsMissing(boolean value) {
      nanIsMissing = value;
      return this;
    }

    /**
     * The vendor whose opaque Arrow extension types ({@code arrow.opaque}) the source hands over,
     * read by that vendor's layout.
     */
    public Builder opaqueVendor(String vendor) {
      opaqueVendor = Optional.ofNullable(nonEmpty("opaqueVendor", vendor));
      return this;
    }

    private static String nonEmpty(String name, String given) {
      if (given != null && given.isEmpty()) {
        throw new IllegalArgumentException(name + " is a non-empty string or null");
      }
      return given;
    }

    public Capabilities build() {
      return new Capabilities(this);
    }
  }
}
