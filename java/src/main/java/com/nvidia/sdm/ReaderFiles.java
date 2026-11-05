// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.io.BufferedInputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.archivers.zip.ZipFile;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.compress.compressors.xz.XZCompressorInputStream;
import software.amazon.awssdk.auth.credentials.*;
import software.amazon.awssdk.http.urlconnection.UrlConnectionHttpClient;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;

/** URI input and compressed single-file ownership; independent of model/schema policy. */
final class ReaderFiles {
  private ReaderFiles() {}

  static boolean needsDecode(String name) {
    String lower = name.toLowerCase(Locale.ROOT);
    return lower.matches(".*\\.(?:zip|tar|bz2|xz)$") || lower.matches(".*\\.tar\\.(?:gz|bz2|xz)$");
  }

  /** Caller closes the stream; it owns the temporary S3 client and credential provider too. */
  static InputStream open(URI uri, ReadOptions options) throws IOException {
    if (!"s3".equalsIgnoreCase(uri.getScheme())) {
      return Files.newInputStream(uri.getScheme() == null ? Path.of(uri.toString()) : Path.of(uri));
    }
    AwsCredentialsProvider credentials = options.accessKey() != null
        ? StaticCredentialsProvider.create(options.sessionToken() == null
            ? AwsBasicCredentials.create(options.accessKey(), options.secretKey())
            : AwsSessionCredentials.create(options.accessKey(), options.secretKey(), options.sessionToken()))
        : options.profile() == null ? DefaultCredentialsProvider.builder().build()
            : ProfileCredentialsProvider.builder().profileName(options.profile()).build();
    S3Client client = null;
    try {
      var builder = S3Client.builder().credentialsProvider(credentials)
          .httpClientBuilder(UrlConnectionHttpClient.builder());
      if (options.region() != null) builder.region(Region.of(options.region()));
      else if (options.profile() != null) builder.region(software.amazon.awssdk.regions.providers.DefaultAwsRegionProviderChain
          .builder().profileName(options.profile()).build().getRegion());
      client = builder.build();
      InputStream stream = client.getObject(request -> request.bucket(uri.getHost()).key(uri.getPath().substring(1)));
      S3Client owned = client;
      return new FilterInputStream(stream) {
        @Override public void close() throws IOException {
          try { super.close(); }
          finally {
            try { owned.close(); }
            finally { closeCredentials(credentials); }
          }
        }
      };
    } catch (RuntimeException failure) {
      if (client != null) client.close();
      closeCredentials(credentials);
      throw new IOException("Could not open S3 input", failure);
    }
  }

  private static void closeCredentials(AwsCredentialsProvider credentials) {
    if (credentials instanceof software.amazon.awssdk.utils.SdkAutoCloseable closeable) closeable.close();
  }

  /** Decode to a private bounded-memory spool. The caller owns/deletes the returned file. */
  static Path decode(URI uri, ReadOptions options) throws IOException {
    Path output = temporary();
    try {
      String lower = uri.getPath().toLowerCase(Locale.ROOT);
      if (lower.endsWith(".zip")) {
        Path encoded = temporary();
        try {
          try (InputStream input = open(uri, options)) { copy(input, encoded); }
          try (ZipFile archive = ZipFile.builder().setPath(encoded).get()) {
            var entries = archive.getEntries();
            org.apache.commons.compress.archivers.zip.ZipArchiveEntry selected = null;
            while (entries.hasMoreElements()) {
              var entry = entries.nextElement();
              if (entry.isDirectory()) continue;
              int kind = entry.getUnixMode() & 0170000;
              if (entry.isUnixSymlink() || (kind != 0 && kind != 0100000)) {
                throw new IOException("Archive must contain exactly one regular data file");
              }
              if (selected != null) throw new IOException("Archive must contain exactly one data file");
              selected = entry;
            }
            if (selected == null) throw new IOException("Archive contains no data file");
            if (!archive.canReadEntryData(selected)) throw new IOException("Unsupported ZIP entry encoding");
            try (InputStream input = archive.getInputStream(selected)) { copy(input, output); }
          }
        } finally { Files.deleteIfExists(encoded); }
      } else {
        try (InputStream input = new BufferedInputStream(open(uri, options));
            InputStream decoded = decompress(input, lower)) {
          String base = lower.replaceFirst("\\.(?:gz|bz2|xz)$", "");
          if (base.endsWith(".tar")) {
            try (TarArchiveInputStream archive = new TarArchiveInputStream(decoded)) {
              boolean copied = false;
              org.apache.commons.compress.archivers.tar.TarArchiveEntry entry;
              while ((entry = archive.getNextEntry()) != null) {
                if (entry.isDirectory()) continue;
                if (!entry.isFile() || copied) throw new IOException("Archive must contain exactly one regular data file");
                copy(archive, output);
                copied = true;
              }
              if (!copied) throw new IOException("Archive contains no data file");
            }
          } else copy(decoded, output);
        }
      }
      return output;
    } catch (IOException | RuntimeException | Error failure) {
      try { Files.deleteIfExists(output); } catch (IOException cleanup) { failure.addSuppressed(cleanup); }
      throw failure;
    }
  }

  private static void copy(InputStream input, Path path) throws IOException {
    // Truncate the existing private file without recreating it with broader default permissions.
    try (var output = Files.newOutputStream(path)) { input.transferTo(output); }
  }

  private static InputStream decompress(InputStream input, String lower) throws IOException {
    if (lower.endsWith(".bz2")) return new BZip2CompressorInputStream(input, true);
    if (lower.endsWith(".xz")) return XZCompressorInputStream.builder().setInputStream(input).setDecompressConcatenated(true).get();
    if (lower.endsWith(".gz")) return new java.util.zip.GZIPInputStream(input);
    return input;
  }

  private static Path temporary() throws IOException {
    try {
      return Files.createTempFile("sdm-reader-", ".data",
          java.nio.file.attribute.PosixFilePermissions.asFileAttribute(java.nio.file.attribute.PosixFilePermissions.fromString("rw-------")));
    } catch (UnsupportedOperationException noPosix) {
      return Files.createTempFile("sdm-reader-", ".data");
    }
  }
}
