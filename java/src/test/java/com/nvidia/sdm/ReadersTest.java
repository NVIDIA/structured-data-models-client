// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.DriverManager;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ReadersTest {
  @TempDir Path directory;

  @Test void jdbcReadsUseTheCallerConnectionAndKeepItOpen() throws Exception {
    for (String url : List.of("jdbc:duckdb:", "jdbc:sqlite::memory:")) {
      try (var connection = DriverManager.getConnection(url); var statement = connection.createStatement();
          RootAllocator allocator = new RootAllocator()) {
        Source source = url.contains("duckdb") ? DuckDbSource.of(connection) : SqliteSource.of(connection);
        statement.execute("CREATE TEMP TABLE \"private table\" (id BIGINT, label VARCHAR)");
        statement.execute("INSERT INTO \"private table\" VALUES (9007199254740993, 'a'), (NULL, NULL)");
        try (Table rows = Readers.readTable(source, "private table", allocator)) {
          assertEquals(2, rows.getRowCount());
          assertEquals(9007199254740993L, value(rows, 0, 0));
          assertNull(value(rows, 0, 1));
        }
        try (Table rows = Readers.readQuery(source, "SELECT label FROM \"private table\" WHERE id IS NOT NULL", allocator)) {
          assertEquals("a", value(rows, 0, 0).toString());
        }
        assertFalse(connection.isClosed());
      }
    }
  }

  @Test void csvKeepsDateStringsMissingTokensAndExactOversizedIntegers() throws Exception {
    Path path = directory.resolve("data.txt");
    Files.writeString(path, "id,date,wide,unsigned,flag,number\n9007199254740993,2024-01-01,-1,18446744073709551615,true,1.5\nNA,NA,18446744073709551617,1,false,2.5\n");
    try (RootAllocator allocator = new RootAllocator(); Table rows = Readers.read(path, allocator)) {
      assertEquals(9007199254740993L, value(rows, 0, 0));
      assertNull(value(rows, 0, 1));
      assertEquals("2024-01-01", value(rows, 1, 0).toString());
      assertNull(value(rows, 1, 1));
      assertEquals("18446744073709551617", value(rows, 2, 1).toString());
      assertEquals("18446744073709551615", Long.toUnsignedString((Long) value(rows, 3, 0)));
      assertEquals(true, value(rows, 4, 0));
      assertEquals(1.5d, value(rows, 5, 0));
    }
    try (RootAllocator allocator = new RootAllocator(); Table rows = Readers.read(path.toUri(),
        ReadOptions.builder().csvTypes(Map.of("id", ReadOptions.CsvType.STRING)).build(), allocator)) {
      assertEquals("9007199254740993", value(rows, 0, 0).toString());
    }
  }

  @Test void parquetDirectoryPreservesPartitionsDecimalsAndTimestamps() throws Exception {
    Path dataset = directory.resolve("dataset");
    Files.createDirectories(dataset.resolve("region=west"));
    Path path = dataset.resolve("region=west/part-data");
    try (var connection = DriverManager.getConnection("jdbc:duckdb:"); var statement = connection.createStatement()) {
      statement.execute("COPY (SELECT 9007199254740993::BIGINT AS id, 12.123456789::DECIMAL(20,9) AS amount, TIMESTAMP '2024-01-01 01:02:03.123456' AS time) TO '"+path+"' (FORMAT PARQUET)");
    }
    Files.writeString(dataset.resolve("_SUCCESS"), "");
    Path alias = directory.resolve("single.parq");
    Files.copy(path, alias);
    try (RootAllocator allocator = new RootAllocator(); Table rows = Readers.read(alias, allocator)) {
      assertEquals(1, rows.getRowCount());
    }
    try (RootAllocator allocator = new RootAllocator(); Table rows = Readers.read(dataset, allocator)) {
      assertEquals(9007199254740993L, value(rows, 0, 0));
      assertEquals(new java.math.BigDecimal("12.123456789"), value(rows, 1, 0));
      assertEquals(java.time.LocalDateTime.parse("2024-01-01T01:02:03.123456"), value(rows, 2, 0));
      assertEquals("west", value(rows, 3, 0).toString());
    }
  }

  @Test void compressionAndArchivesReuseTheSameLosslessCsvReader() throws Exception {
    byte[] data = "id,value\n9007199254740993,1.5\nNA,2.5\n".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    for (String suffix : List.of("bz2", "xz", "zip", "tar", "tar.gz")) {
      Path path = directory.resolve("data.csv." + suffix);
      writeCompressed(path, suffix, data, false);
      try (RootAllocator allocator = new RootAllocator(); Table rows = Readers.read(path, allocator)) {
        assertEquals(9007199254740993L, value(rows, 0, 0), suffix);
        assertNull(value(rows, 0, 1), suffix);
      }
    }
    for (String suffix : List.of("zip", "tar")) {
      Path path = directory.resolve("multiple.csv." + suffix);
      writeCompressed(path, suffix, data, true);
      try (RootAllocator allocator = new RootAllocator()) {
        assertThrows(java.io.IOException.class, () -> Readers.read(path, allocator));
      }
    }
  }

  @Test void csvRejectsNonDecimalGrammarAndPreservesFiniteOverflow() throws Exception {
    Path path = directory.resolve("numbers.csv");
    Files.writeString(path, "overflow,hex,arabic,inf\n1e400,0x1.0p3,١٢٣,inf\n2,1,2,-inf\n");
    try (RootAllocator allocator = new RootAllocator(); Table rows = Readers.read(path, allocator)) {
      assertEquals("1e400", value(rows, 0, 0).toString());
      assertEquals("0x1.0p3", value(rows, 1, 0).toString());
      assertEquals("١٢٣", value(rows, 2, 0).toString());
      assertEquals(Double.POSITIVE_INFINITY, value(rows, 3, 0));
      assertEquals(Double.NEGATIVE_INFINITY, value(rows, 3, 1));
    }
  }

  private static void writeCompressed(Path path, String format, byte[] data, boolean multiple) throws Exception {
    try (var file = Files.newOutputStream(path)) {
      switch (format) {
        case "bz2" -> { try (var output = new org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream(file)) { output.write(data); } }
        case "xz" -> { try (var output = new org.tukaani.xz.XZOutputStream(file, new org.tukaani.xz.LZMA2Options())) { output.write(data); } }
        case "zip" -> {
          try (var output = new java.util.zip.ZipOutputStream(file)) {
            output.putNextEntry(new java.util.zip.ZipEntry("../never-extracted.csv")); output.write(data); output.closeEntry();
            if (multiple) { output.putNextEntry(new java.util.zip.ZipEntry("second.csv")); output.write(data); output.closeEntry(); }
          }
        }
        default -> {
          try (var output = new org.apache.commons.compress.archivers.tar.TarArchiveOutputStream(
              format.endsWith(".gz") ? new java.util.zip.GZIPOutputStream(file) : file)) {
            var entry = new org.apache.commons.compress.archivers.tar.TarArchiveEntry("../never-extracted.csv");
            entry.setSize(data.length); output.putArchiveEntry(entry); output.write(data); output.closeArchiveEntry();
            if (multiple) {
              var second = new org.apache.commons.compress.archivers.tar.TarArchiveEntry("second.csv");
              second.setSize(data.length); output.putArchiveEntry(second); output.write(data); output.closeArchiveEntry();
            }
          }
        }
      }
    }
  }

  @Test void csvTablesFeedTheNativeTabularRequestAfterReaderResourcesClose() throws Exception {
    Path contextFile = directory.resolve("context.csv");
    Path scoringFile = directory.resolve("scoring.csv");
    Files.writeString(contextFile, "x,y\n1,2\n2,4\n");
    Files.writeString(scoringFile, "x\n3\n4\n");
    Request request;
    try (RootAllocator allocator = new RootAllocator(); Table context = Readers.read(contextFile, allocator);
        Table scoring = Readers.read(scoringFile, allocator)) {
      request = new Request(context, scoring, "y", "regression");
    }
    try (TabularTest.Endpoint endpoint = new TabularTest.Endpoint(); SdmClient client = SdmClient.connect(endpoint.url())) {
      PredictionResult result = client.predict(request, SdmClient.KUMO_TABULAR);
      assertEquals(2, result.rows());
      assertEquals(1, endpoint.calls.size());
    }
  }

  @Test void configuredS3FixtureUsesThePublicReader() throws Exception {
    String uri = System.getenv("SDM_TEST_S3_READER_URI");
    org.junit.jupiter.api.Assumptions.assumeTrue(uri != null, "S3 fixture not configured");
    ReadOptions options = ReadOptions.builder().region("us-west-2")
        .credentials(System.getenv("AWS_ACCESS_KEY_ID"), System.getenv("AWS_SECRET_ACCESS_KEY"),
            System.getenv("AWS_SESSION_TOKEN")).build();
    try (RootAllocator allocator = new RootAllocator(); Table rows = Readers.read(java.net.URI.create(uri), options, allocator)) {
      assertEquals(40, rows.getRowCount());
    }
  }

  @Test void ordinaryJavaWorksWithoutSparkOrSnowflakeRuntime() throws Exception {
    String classpath = java.util.Arrays.stream(System.getProperty("java.class.path").split(java.io.File.pathSeparator))
        .filter(entry -> !entry.contains("/org/apache/spark/") && !entry.contains("/org/scala-lang/") && !entry.contains("/net/snowflake/"))
        .collect(java.util.stream.Collectors.joining(java.io.File.pathSeparator));
    Process process = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
        "--add-opens=java.base/java.nio=ALL-UNNAMED", "-cp", classpath, NoSpark.class.getName()).redirectErrorStream(true).start();
    String output = new String(process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
    assertEquals(0, process.waitFor(), output);
    assertTrue(output.contains("no-spark-ok"), output);
  }

  public static class NoSpark {
    public static void main(String[] args) throws Exception {
      try (var connection = DriverManager.getConnection("jdbc:duckdb:"); RootAllocator allocator = new RootAllocator();
          Table rows = Readers.readQuery(DuckDbSource.of(connection), "SELECT 1 AS value", allocator)) {
        if (rows.getRowCount() != 1) throw new AssertionError();
        System.out.println("no-spark-ok");
      }
    }
  }

  static Object value(Table table, int column, int row) {
    try (FieldVector vector = table.getVectorCopy(column)) { return vector.getObject(row); }
  }
}
