// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

/** File decoding and S3 access options. Credentials are never included in diagnostics. */
public final class ReadOptions {
  /** File format; omitted means CSV extension or a Parquet file/dataset. */
  public enum Format { CSV, PARQUET }
  /** Optional CSV column types; values never become executable SQL. */
  public enum CsvType { STRING, INT64, UINT64, FLOAT64, BOOLEAN }
  private final java.util.Map<String, CsvType> csvTypes;
  private final Format format;
  private final String profile;
  private final String region;
  private final String accessKey;
  private final String secretKey;
  private final String sessionToken;
  private final java.net.URI endpoint;

  private ReadOptions(Builder builder) {
    csvTypes = java.util.Map.copyOf(builder.csvTypes);
    format = builder.format;
    profile = builder.profile;
    region = builder.region;
    accessKey = builder.accessKey;
    secretKey = builder.secretKey;
    sessionToken = builder.sessionToken;
    endpoint = builder.endpoint;
  }
  public static Builder builder() { return new Builder(); }
  public java.util.Map<String, CsvType> csvTypes() { return csvTypes; }
  public Format format() { return format; }
  public String profile() { return profile; }
  public String region() { return region; }
  String accessKey() { return accessKey; }
  String secretKey() { return secretKey; }
  String sessionToken() { return sessionToken; }
  java.net.URI endpoint() { return endpoint; }
  public static final class Builder {
    private java.util.Map<String, CsvType> csvTypes = java.util.Map.of();
    private Format format;
    private String profile;
    private String region;
    private String accessKey;
    private String secretKey;
    private String sessionToken;
    private java.net.URI endpoint;
    private Builder() {}
    public Builder csvTypes(java.util.Map<String, CsvType> value) { csvTypes = java.util.Map.copyOf(value); return this; }
    public Builder format(Format value) { format = value; return this; }
    /** Profile understood by DuckDB's AWS SDK credential chain. */
    public Builder profile(String value) { profile = value; return this; }
    public Builder region(String value) { region = value; return this; }
    /** S3-compatible HTTP(S) service; custom endpoints use path-style bucket addressing. */
    public Builder endpoint(java.net.URI value) {
      java.util.Objects.requireNonNull(value, "endpoint");
      if (!("http".equals(value.getScheme()) || "https".equals(value.getScheme()))
          || value.getHost() == null || value.getUserInfo() != null
          || value.getQuery() != null || value.getFragment() != null
          || !(value.getPath().isEmpty() || value.getPath().equals("/"))) {
        throw new IllegalArgumentException("S3 endpoint must be an HTTP(S) origin without credentials, query, or path");
      }
      endpoint = value;
      return this;
    }
    /** Explicit temporary or long-lived credentials for this read only. */
    public Builder credentials(String key, String secret, String token) {
      accessKey = java.util.Objects.requireNonNull(key, "access key");
      secretKey = java.util.Objects.requireNonNull(secret, "secret key");
      sessionToken = token;
      return this;
    }
    public ReadOptions build() {
      if (profile != null && accessKey != null) {
        throw new IllegalArgumentException("choose a profile or explicit credentials");
      }
      return new ReadOptions(this);
    }
  }
}
