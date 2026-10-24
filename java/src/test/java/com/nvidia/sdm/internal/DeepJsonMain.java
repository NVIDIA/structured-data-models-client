// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.FromSourceOptions;
import com.nvidia.sdm.PostgresSource;
import com.nvidia.sdm.Schema;
import com.nvidia.sdm.StructuredDataException;
import java.net.URI;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.util.UUID;

/**
 * Codex T3.12 r2 P1-2: a Postgres JSON value nested deep, described through the public {@code
 * Schema.fromSource} on a thread whose stack is small (256 KiB), as a JVM's may be: a value the
 * core reads (920 and 1,900 levels) is described, one past its depth (2,100) refused by name, and
 * the JVM runs on. The database is made on {@code SDM_POSTGRES_TEST_DSN}'s server and dropped.
 */
public final class DeepJsonMain {
  private DeepJsonMain() {}

  private static String nested(int depth) {
    return "[".repeat(depth) + "0" + "]".repeat(depth);
  }

  public static void main(String[] arguments) throws Exception {
    URI uri = URI.create(System.getenv("SDM_POSTGRES_TEST_DSN"));
    String[] user = uri.getUserInfo().split(":", 2);
    String server =
        "jdbc:postgresql://" + uri.getHost() + ":" + (uri.getPort() < 0 ? 5432 : uri.getPort()) + "/";
    String name = "sdm_t312_deep_" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
    try (Connection admin = DriverManager.getConnection(server + "postgres", user[0], user[1]);
        Statement statement = admin.createStatement()) {
      statement.execute("CREATE DATABASE " + name);
    }
    try {
      try (Connection connection = DriverManager.getConnection(server + name, user[0], user[1]);
          Statement statement = connection.createStatement()) {
        for (int depth : new int[] {920, 1900, 2100}) {
          statement.execute("CREATE TABLE deep" + depth + " (id bigint PRIMARY KEY, doc json)");
          statement.execute(
              "INSERT INTO deep" + depth + " VALUES (1, '" + nested(depth) + "'), (2, '[1]')");
        }
        Thread small =
            new Thread(
                null,
                () -> {
                  for (int depth : new int[] {920, 1900, 2100}) {
                    try {
                      Schema schema =
                          Schema.fromSource(
                              PostgresSource.of(connection),
                              FromSourceOptions.builder().table("deep" + depth).build());
                      System.out.println(
                          "described " + depth + " tables=" + schema.tables().size());
                    } catch (StructuredDataException refused) {
                      System.out.println("refused " + depth + " " + refused.code());
                    }
                  }
                },
                "small-stack",
                256 * 1024);
        small.start();
        small.join();
      }
      System.out.println("survived");
    } finally {
      try (Connection admin = DriverManager.getConnection(server + "postgres", user[0], user[1]);
          Statement statement = admin.createStatement()) {
        statement.execute("DROP DATABASE " + name + " WITH (FORCE)");
      }
    }
  }
}
