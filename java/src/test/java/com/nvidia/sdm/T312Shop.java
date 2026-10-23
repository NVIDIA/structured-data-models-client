// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.nvidia.sdm.internal.Json;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;

/**
 * T3.12's fixtures: the contract's shop base ({@code contract/bases.yaml}'s {@code shop}: eight
 * users and eighty orders, each order a week after the last) on DuckDB, SQLite and Postgres
 * through their JDBC drivers, and a model endpoint in the JVM answering by the scripted
 * endpoint's rule ({@code contract/runners/scripted_endpoint.py}) and keeping every body it
 * received, byte for byte.
 */
final class T312Shop {
  private T312Shop() {}

  /** The databases T3.12 reads. */
  static final List<String> DATABASES = List.of("duckdb", "sqlite", "postgres");

  /** A connection to a new database holding the shop base. */
  static Connection open(String database) throws SQLException {
    return switch (database) {
      case "duckdb" -> duckdb();
      case "sqlite" -> loaded(DriverManager.getConnection("jdbc:sqlite::memory:"), "INTEGER", "REAL", "TEXT", "TEXT");
      case "postgres" -> postgres();
      default -> throw new IllegalArgumentException(database);
    };
  }

  private static Connection duckdb() throws SQLException {
    Connection connection = DriverManager.getConnection("jdbc:duckdb:");
    try (Statement statement = connection.createStatement()) {
      statement.execute(
          "CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i)::INTEGER AS age, ['US',"
              + " 'DE', 'IN'][1 + i % 3] AS country FROM range(8) t(i)");
      statement.execute(
          "CREATE TABLE orders AS SELECT i::BIGINT AS order_id, (i % 8)::BIGINT AS user_id, (10 +"
              + " i * 1.5)::DOUBLE AS price, TIMESTAMP '2023-01-01' + to_days((i * 7)::INTEGER) AS"
              + " ts FROM range(80) t(i)");
    }
    return connection;
  }

  /**
   * The shop base loaded as the contract loader loads it: on SQLite a timestamp as its ISO text
   * (the type map's {@code iso}), on Postgres in its own types.
   */
  private static Connection loaded(
      Connection connection, String integer, String real, String text, String time)
      throws SQLException {
    boolean sqlite = "TEXT".equals(time);
    try (Statement statement = connection.createStatement()) {
      statement.execute(
          "CREATE TABLE users (user_id " + integer + ", age " + integer + ", country " + text + ")");
      statement.execute(
          "CREATE TABLE orders (order_id " + integer + ", user_id " + integer + ", price " + real
              + ", ts " + time + ")");
    }
    String[] countries = {"US", "DE", "IN"};
    try (PreparedStatement users = connection.prepareStatement("INSERT INTO users VALUES (?, ?, ?)")) {
      for (int i = 0; i < 8; i++) {
        users.setLong(1, i);
        users.setLong(2, 20 + i);
        users.setString(3, countries[i % 3]);
        users.executeUpdate();
      }
    }
    try (PreparedStatement orders =
        connection.prepareStatement("INSERT INTO orders VALUES (?, ?, ?, ?)")) {
      for (int i = 0; i < 80; i++) {
        orders.setLong(1, i);
        orders.setLong(2, i % 8);
        orders.setDouble(3, 10 + i * 1.5);
        LocalDateTime ts = LocalDateTime.of(2023, 1, 1, 0, 0).plusDays(i * 7L);
        if (sqlite) {
          orders.setString(4, ts.toString().replace('T', ' '));
        } else {
          orders.setObject(4, ts);
        }
        orders.executeUpdate();
      }
    }
    return connection;
  }

  /**
   * A new database on {@code SDM_POSTGRES_TEST_DSN}'s server holding the shop base, dropped when
   * the JVM exits; with no server named, the test is skipped, unless {@code SDM_REQUIRE_POSTGRES}
   * is set, when it fails (H26).
   */
  private static Connection postgres() throws SQLException {
    String dsn = System.getenv("SDM_POSTGRES_TEST_DSN");
    if (dsn == null || dsn.isEmpty()) {
      assertNotNull(
          "1".equals(System.getenv("SDM_REQUIRE_POSTGRES")) ? null : "skipped",
          "SDM_REQUIRE_POSTGRES is set and SDM_POSTGRES_TEST_DSN names no server");
      assumeTrue(false, "no Postgres server: set SDM_POSTGRES_TEST_DSN");
    }
    URI uri = URI.create(dsn);
    String[] user = uri.getUserInfo() == null ? new String[] {null, null} : uri.getUserInfo().split(":", 2);
    String server = "jdbc:postgresql://" + uri.getHost() + ":" + (uri.getPort() < 0 ? 5432 : uri.getPort()) + "/";
    String name = "sdm_t312_" + UUID.randomUUID().toString().replace("-", "").substring(0, 16);
    try (Connection admin = DriverManager.getConnection(server + "postgres", user[0], user.length > 1 ? user[1] : null);
        Statement statement = admin.createStatement()) {
      statement.execute("CREATE DATABASE " + name);
    }
    Runtime.getRuntime()
        .addShutdownHook(
            new Thread(
                () -> {
                  try (Connection admin = DriverManager.getConnection(server + "postgres", user[0], user.length > 1 ? user[1] : null);
                      Statement statement = admin.createStatement()) {
                    statement.execute("DROP DATABASE " + name + " WITH (FORCE)");
                  } catch (SQLException ignored) {
                    // Gone with the server.
                  }
                }));
    Connection connection =
        DriverManager.getConnection(server + name, user[0], user.length > 1 ? user[1] : null);
    return loaded(connection, "bigint", "double precision", "text", "timestamp");
  }

  /** The source of {@code connection} on {@code database}. */
  static Source source(String database, Connection connection) {
    return switch (database) {
      case "duckdb" -> DuckDbSource.of(connection);
      case "sqlite" -> SqliteSource.of(connection);
      default -> PostgresSource.of(connection);
    };
  }

  /** A model endpoint in this JVM: the scripted endpoint's rule, every body kept. */
  static final class Endpoint implements AutoCloseable {
    final List<byte[]> received = Collections.synchronizedList(new ArrayList<>());
    private final HttpServer server;

    /** Answers each call by {@code answer}: given the body, the status and the answer's bytes. */
    Endpoint(Function<byte[], Map.Entry<Integer, byte[]>> answer) throws IOException {
      server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
      server.createContext(
          "/",
          exchange -> {
            byte[] body;
            try (InputStream in = exchange.getRequestBody()) {
              body = in.readAllBytes();
            }
            received.add(body);
            Map.Entry<Integer, byte[]> answered = answer.apply(body);
            exchange.sendResponseHeaders(answered.getKey(), answered.getValue().length);
            try (OutputStream out = exchange.getResponseBody()) {
              out.write(answered.getValue());
            }
          });
      server.start();
    }

    /** By the scripted endpoint's rule. */
    Endpoint() throws IOException {
      this(body -> Map.entry(200, scripted(body)));
    }

    String url() {
      return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    @Override
    public void close() {
      server.stop(0);
    }
  }

  /**
   * The scripted endpoint's answer to a call: one prediction per row of {@code
   * predict.instance_table}, its id the row's {@code instance_id}, {@code 1.0} for a regression
   * and the first class otherwise.
   */
  @SuppressWarnings("unchecked")
  static byte[] scripted(byte[] body) {
    try {
      Map<String, Object> call = (Map<String, Object>) Json.read(body);
      Map<String, Object> task = (Map<String, Object>) call.get("task");
      Map<String, Object> table =
          (Map<String, Object>) ((Map<String, Object>) call.get("predict")).get("instance_table");
      int id = ((List<Object>) table.get("columns")).indexOf("instance_id");
      Object prediction =
          "regression".equals(task.get("kind"))
              ? new java.math.BigDecimal("1.0")
              : ((List<Object>) ((Map<String, Object>) task.get("target")).get("classes")).get(0);
      List<Object> predictions = new ArrayList<>();
      List<Object> rows = (List<Object>) table.get("rows");
      for (int at = 0; at < rows.size(); at++) {
        Map<String, Object> item = new LinkedHashMap<>();
        // The instance id as its text, as the scripted endpoint gives it.
        item.put("id", String.valueOf(((List<Object>) rows.get(at)).get(id)));
        item.put("row_index", (long) at);
        item.put("prediction", prediction);
        if (call.get("output") instanceof Map<?, ?> output
            && output.get("fields") instanceof List<?> fields
            && fields.contains("explanation")) {
          // As the scripted endpoint told to answer explanations gives one.
          Map<String, Object> explained = new LinkedHashMap<>();
          explained.put("format", "scripted");
          explained.put("summary", "row " + at + " of call 0");
          item.put("explanation", explained);
        }
        predictions.add(item);
      }
      Map<String, Object> answer = new LinkedHashMap<>();
      answer.put("object", "prediction.response");
      answer.put("id", "r1");
      answer.put("model", "kumo-relational");
      answer.put("metadata", Map.of());
      answer.put("predictions", predictions);
      return Json.write(answer);
    } catch (Json.JsonException | RuntimeException e) {
      return "{\"error\": \"not_read\"}".getBytes(StandardCharsets.UTF_8);
    }
  }
}
