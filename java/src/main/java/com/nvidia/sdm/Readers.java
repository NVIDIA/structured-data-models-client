// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Providers;
import com.nvidia.sdm.internal.ReadTables;
import com.nvidia.sdm.internal.jdbc.JdbcProvider;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.Objects;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.table.Table;

/** Read one table into owned Arrow data for either model's in-memory API. */
public final class Readers {
  private Readers() {}

  /** Read a qualified table name; each dot-separated component is quoted as an identifier. */
  public static Table readTable(Source source, String name, BufferAllocator allocator)
      throws Exception {
    Objects.requireNonNull(name, "table");
    if (!name.contains(".")) {
      String catalog = null;
      String schema = null;
      if (source instanceof PostgresSource postgres) schema = postgres.schema().orElse(null);
      else if (source instanceof DatabricksSource databricks) {
        catalog = databricks.catalog().orElse(null);
        schema = databricks.schema().orElse(null);
      } else if (source instanceof DatabricksSparkSource spark) {
        catalog = spark.catalog().orElse(null);
        schema = spark.schema().orElse(null);
      }
      if (catalog != null && schema == null) {
        try (Table current = readQuery(source, "SELECT current_schema()", allocator);
            var column = current.getVectorCopy(0)) {
          schema = column.getObject(0).toString();
        }
      }
      if (schema != null) name = schema + "." + name;
      if (catalog != null) name = catalog + "." + name;
    }
    String quote = source instanceof DatabricksSource || source instanceof DatabricksSparkSource
        ? "`" : "\"";
    String sql = java.util.Arrays.stream(name.split("\\.", -1))
        .map(part -> {
          if (part.isEmpty()) throw new IllegalArgumentException("empty table-name component");
          return quote + part.replace(quote, quote + quote) + quote;
        }).collect(java.util.stream.Collectors.joining("."));
    return readQuery(source, "SELECT * FROM " + sql, allocator);
  }

  /** Execute caller SQL on the supplied connection/session without closing it. */
  public static Table readQuery(Source source, String sql, BufferAllocator allocator)
      throws Exception {
    Objects.requireNonNull(sql, "sql");
    var provider = Providers.of(Objects.requireNonNull(source, "source")).provider();
    if (provider instanceof JdbcProvider jdbc) {
      return ReadTables.collect(jdbc.readQuery(sql, allocator), allocator);
    }
    return ReadTables.collect(
        ((com.nvidia.sdm.internal.jdbc.SparkProvider) provider).readQuery(sql, allocator), allocator);
  }

  public static Table read(Path path, BufferAllocator allocator) throws Exception {
    return read(path.toUri(), ReadOptions.builder().build(), allocator);
  }

  public static Table read(URI uri, ReadOptions options, BufferAllocator allocator)
      throws Exception {
    Objects.requireNonNull(uri, "uri");
    Objects.requireNonNull(options, "options");
    String scheme = uri.getScheme();
    boolean s3 = "s3".equalsIgnoreCase(scheme);
    if (!s3 && scheme != null && !"file".equalsIgnoreCase(scheme)) {
      throw new IllegalArgumentException("supported file schemes are file and s3");
    }
    if (!s3 && (options.profile() != null || options.accessKey() != null || options.region() != null)) {
      throw new IllegalArgumentException("S3 access options require an s3 URI");
    }
    String path = s3 ? uri.toString() : (scheme == null ? Path.of(uri.toString()) : Path.of(uri)).toString();
    boolean directory = path.endsWith("/") || (!s3 && Files.isDirectory(Path.of(path)));
    String lower = path.toLowerCase(java.util.Locale.ROOT);
    if (ReaderFiles.needsDecode(lower)) {
      ReadOptions.Format decodedFormat = options.format();
      if (decodedFormat == null) {
        String base = lower;
        while (base.matches(".*\\.(?:zip|tar|gz|bz2|xz)$")) base = base.substring(0, base.lastIndexOf('.'));
        if (base.endsWith(".csv") || base.endsWith(".txt")) decodedFormat = ReadOptions.Format.CSV;
        else if (base.endsWith(".parquet") || base.endsWith(".parq") || base.endsWith(".pq")) decodedFormat = ReadOptions.Format.PARQUET;
        else throw new IllegalArgumentException("unknown archive file format; provide ReadOptions.format");
      }
      Path decoded = ReaderFiles.decode(uri, options);
      Table result = null;
      try {
        result = read(decoded.toUri(), ReadOptions.builder().format(decodedFormat).csvTypes(options.csvTypes()).build(), allocator);
        Files.deleteIfExists(decoded);
        return result;
      } catch (Exception | Error failure) {
        if (result != null) {
          try { result.close(); } catch (RuntimeException closing) { failure.addSuppressed(closing); }
        }
        try { Files.deleteIfExists(decoded); } catch (java.io.IOException cleanup) { failure.addSuppressed(cleanup); }
        throw failure;
      }
    }
    ReadOptions.Format format = options.format();
    if (format == null) {
      if (lower.matches(".*\\.(?:csv|txt)(?:\\.(?:gz|zst))?")) format = ReadOptions.Format.CSV;
      else if (directory || lower.endsWith(".parquet") || lower.endsWith(".parq") || lower.endsWith(".pq")) format = ReadOptions.Format.PARQUET;
      else if (s3) {
        // A suffix-free S3 location is resolved as a dataset prefix below.
        directory = true;
        format = ReadOptions.Format.PARQUET;
      } else throw new IllegalArgumentException("unknown file format; provide ReadOptions.format");
    }
    try (Connection connection = DriverManager.getConnection("jdbc:duckdb:")) {
      if (s3) configureS3(connection, options);
      String read;
      if (format == ReadOptions.Format.CSV) {
        read = "read_csv(" + literal(path) + ", header=true, all_varchar=true, nullstr="
            + "['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', "
            + "'1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None', 'n/a', 'nan', 'null'])";
      } else {
        String files = literal(path);
        if (directory) {
          java.util.List<String> paths = new java.util.ArrayList<>();
          try (Statement listing = connection.createStatement();
              var rows = listing.executeQuery("SELECT file FROM glob(" + literal(path.replaceAll("/+$", "") + "/**/*") + ") ORDER BY file")) {
            while (rows.next()) {
              String file = rows.getString(1);
              String relative = file.substring(Math.min(file.length(), path.replaceAll("/+$", "").length()));
              if (java.util.Arrays.stream(relative.split("/")).noneMatch(part -> part.startsWith("_") || part.startsWith("."))) paths.add(literal(file));
            }
          }
          if (paths.isEmpty()) throw new java.io.IOException("Parquet dataset contains no readable files");
          files = "[" + String.join(",", paths) + "]";
        }
        read = "read_parquet(" + files + ", hive_partitioning=true, union_by_name=true)";
      }
      Table raw = readQuery(DuckDbSource.of(connection), "SELECT * FROM " + read, allocator);
      if (format != ReadOptions.Format.CSV) return raw;
      try (raw) {
        return com.nvidia.sdm.internal.CsvTypes.infer(raw, options.csvTypes(), allocator);
      }
    }
  }

  private static void configureS3(Connection connection, ReadOptions options) throws Exception {
    StringBuilder secret = new StringBuilder("CREATE SECRET sdm_reader (TYPE s3");
    if (options.accessKey() == null) {
      secret.append(", PROVIDER credential_chain");
      if (options.profile() != null) secret.append(", PROFILE ").append(literal(options.profile()));
    } else {
      secret.append(", KEY_ID ").append(literal(options.accessKey()));
      secret.append(", SECRET ").append(literal(options.secretKey()));
      if (options.sessionToken() != null) secret.append(", SESSION_TOKEN ").append(literal(options.sessionToken()));
    }
    if (options.region() != null) secret.append(", REGION ").append(literal(options.region()));
    secret.append(')');
    try (Statement statement = connection.createStatement()) {
      statement.execute(secret.toString());
    } catch (java.sql.SQLException failure) {
      // Driver errors can include the complete secret statement. Do not retain that exception.
      throw new java.sql.SQLException("S3 credential setup failed", failure.getSQLState(), failure.getErrorCode());
    }
  }

  private static String literal(String value) { return "'" + value.replace("'", "''") + "'"; }
}
