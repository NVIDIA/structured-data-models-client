// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.internal.NativeCore;
import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.sql.*;
import java.time.*;
import java.util.*;
import java.util.zip.*;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.table.Table;
import org.apache.arrow.vector.types.pojo.*;

/** Public installed-client journeys; only observation formatting lives here. */
public final class ParityRunner {
  private static ReadOptions fileReadOptions(Map<String, Object> settings) {
    ReadOptions.Builder builder = ReadOptions.builder();
    if (settings.get("s3_endpoint") != null) {
      builder.endpoint(java.net.URI.create(text(settings.get("s3_endpoint"))))
          .region("us-east-1").credentials("testing", "testing", null);
    }
    return builder.build();
  }
  private ParityRunner() {}

  static final class Incomplete extends Exception {
    private static final long serialVersionUID = 1L;

    Incomplete(String reason) {
      super(reason);
    }
  }

  @SuppressWarnings("unchecked")
  static Map<String, Object> map(Object value) {
    return value == null ? Map.of() : (Map<String, Object>) value;
  }

  @SuppressWarnings("unchecked")
  static List<Object> list(Object value) {
    return (List<Object>) value;
  }

  static String text(Object value) {
    return value == null ? null : value.toString();
  }

  static Map<String, Object> read(Path path) throws Exception {
    return map(Json.read(Files.readAllBytes(path)));
  }

  static String hash(byte[] bytes) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
  }

  static String fileHash(Path path) throws Exception {
    MessageDigest hash = MessageDigest.getInstance("SHA-256");
    try (InputStream in = Files.newInputStream(path)) {
      byte[] buffer = new byte[65536];
      int n;
      while ((n = in.read(buffer)) != -1) hash.update(buffer, 0, n);
    }
    return HexFormat.of().formatHex(hash.digest());
  }

  static Object canonical(Object value) {
    if (value instanceof Map<?, ?> source) {
      var out =
          new TreeMap<String, Object>(
              (left, right) ->
                  Arrays.compare(left.codePoints().toArray(), right.codePoints().toArray()));
      source.forEach((k, v) -> out.put(k.toString(), canonical(v)));
      return out;
    }
    if (value instanceof List<?> values)
      return values.stream().map(ParityRunner::canonical).toList();
    return value;
  }

  static String digest(Object value) throws Exception {
    // Match Python json.dumps(sort_keys=True, ensure_ascii=True) exactly.
    String json = new String(Json.write(canonical(value)), StandardCharsets.UTF_8);
    var ascii = new StringBuilder();
    for (int at = 0; at < json.length(); at++) {
      char unit = json.charAt(at);
      if (unit >= 0x7f) ascii.append(String.format("\\u%04x", (int) unit));
      else ascii.append(unit);
    }
    return hash(ascii.toString().getBytes(StandardCharsets.UTF_8));
  }

  static boolean selected(String name, Object value) {
    String filter = System.getProperty("sdm.parity." + name);
    return filter == null
        || filter.isBlank()
        || Arrays.asList(filter.split(",")).contains(text(value));
  }

  static Map<String, Object> artifact() throws Exception {
    Path loaded =
        Path.of(SdmClient.class.getProtectionDomain().getCodeSource().getLocation().toURI())
            .toRealPath();
    if (!Files.isRegularFile(loaded)
        || !loaded.equals(Path.of(System.getProperty("sdm.parity.jar")).toRealPath()))
      throw new Incomplete("client is not loaded from requested packaged JAR");
    Path nativePath = Path.of(System.getProperty("sdm.native.library")).toRealPath();
    if (!System.getProperty("sdm.parity.commit", "").matches("[0-9a-f]{40}"))
      throw new Incomplete("full candidate source commit is required");
    Map<String, Object> proven = read(Path.of(System.getProperty("sdm.parity.provenance")));
    var actual = new LinkedHashMap<String, Object>();
    actual.put("commit", System.getProperty("sdm.parity.commit"));
    actual.put("package_hash", fileHash(loaded));
    actual.put("native_hash", fileHash(nativePath));
    actual.put("contract", NativeCore.load().get("contract"));
    for (String key : actual.keySet())
      if (!Objects.equals(text(actual.get(key)), text(proven.get(key))))
        throw new Incomplete("artifact provenance mismatch: " + key);
    return actual;
  }

  public static void main(String[] args) throws Exception {
    Path manifest = Path.of(System.getProperty("sdm.parity.manifest")).toRealPath();
    Map<String, Object> data = read(manifest),
        config =
            System.getProperty("sdm.parity.config") == null
                ? Map.of()
                : read(Path.of(System.getProperty("sdm.parity.config")));
    Map<String, Object> artifact = artifact();
    var records = new ArrayList<Map<String, Object>>();
    var expectations = new ArrayList<Map<String, Object>>();
    try (ParityRoute route = new ParityRoute()) {
      for (Object item : list(data.get("scenarios"))) {
        Map<String, Object> scenario = map(item);
        if (!selected("cases", scenario.get("id"))) continue;
        for (Object connector : list(scenario.get("connectors")))
          for (Object model : list(scenario.get("models"))) {
            if (!selected("connectors", connector) || !selected("models", model)) continue;
            boolean deferred =
                list(data.get("deferrals")).stream()
                    .map(ParityRunner::map)
                    .anyMatch(
                        d ->
                            "java".equals(d.get("language"))
                                && list(d.get("connectors")).contains(connector));
            if (!deferred) {
              var effective = new LinkedHashMap<>(scenario);
              effective.remove("variants");
              effective.putAll(map(map(scenario.get("variants")).get(connector)));
              Map<String, Object> record =
                  run(
                      manifest.getParent(),
                      effective,
                      text(connector),
                      text(model),
                      config,
                      artifact,
                      route);
              records.add(record);
              checkpoint(records, false);
              expectations.add(
                  map(
                      map(effective.get("expected_by_language"))
                          .getOrDefault(
                              "java",
                              map(effective.get("expected_by_model"))
                                  .getOrDefault(model, effective.get("expected")))));
              System.out.println(
                  "parity "
                      + scenario.get("id")
                      + " "
                      + connector
                      + " "
                      + model
                      + " "
                      + record.get("status"));
            }
          }
      }
      route.finish(records);
    }
    checkpoint(records, true);
    if (records.isEmpty()) System.exit(1);
    for (int at = 0; at < records.size(); at++)
      if (!accepted(records.get(at), expectations.get(at))) System.exit(1);
  }

  static void checkpoint(List<Map<String, Object>> records, boolean finished) throws Exception {
    Path output = Path.of(System.getProperty("sdm.parity.output")).toAbsolutePath();
    Path temporary = Files.createTempFile(output.getParent(), "parity-checkpoint-", ".jsonl");
    try {
      try (BufferedWriter out = Files.newBufferedWriter(temporary)) {
        for (var source : records) {
          var record = new LinkedHashMap<>(source);
          if (!finished) {
            record.put("status", "incomplete");
            record.put("reason", "journey complete; whole-run route observation pending");
            var pendingRoute = new LinkedHashMap<>(map(record.get("route")));
            pendingRoute.put("workers", List.of());
            pendingRoute.put("unobserved_subprocesses", List.of("route-observation-pending"));
            record.put("route", pendingRoute);
          }
          out.write(new String(Json.write(record), StandardCharsets.UTF_8));
          out.newLine();
        }
      }
      Files.move(
          temporary, output, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  static boolean accepted(Map<String, Object> record, Map<String, Object> expected) {
    if ("local".equals(record.get("execution"))
        && List.of("fit", "predict", "close").contains(expected.getOrDefault("stage", ""))) {
      return "passed".equals(record.get("status"));
    }
    Object wanted =
        expected.getOrDefault(
            "outcome", "refused".equals(expected.get("status")) ? "refused" : "passed");
    if (!Objects.equals(record.get("status"), wanted)) return false;
    if (!"refused".equals(expected.get("status"))) return true;
    Map<String, Object> error = map(record.get("error"));
    return List.of("stage", "code", "rule").stream()
        .allMatch(name -> Objects.equals(error.get(name), expected.get(name)));
  }

  static Map<String, Object> run(
      Path base,
      Map<String, Object> scenario,
      String connector,
      String model,
      Map<String, Object> config,
      Map<String, Object> artifact,
      ParityRoute route)
      throws Exception {
    Map<String, Object> fixture = read(base.resolve(text(scenario.get("fixture"))));
    var record = new LinkedHashMap<String, Object>();
    record.put("version", 1);
    record.put("case", scenario.get("id"));
    record.put("connector", connector);
    record.put("model", model);
    record.put("language", "java");
    record.put("phase", "candidate");
    record.put("status", "incomplete");
    record.put("execution", System.getProperty("sdm.parity.mode", "local"));
    record.put("fixture_hash", digest(Map.of("fixture", fixture, "case", scenario)));
    record.put("artifact", artifact);
    record.put("source_snapshot", null);
    record.put("endpoint", null);
    record.put("semantic", null);
    record.put("error", null);
    route.begin(record);
    try {
      journey(scenario, connector, model, config, fixture, record, route);
    } catch (Incomplete missing) {
      record.put("status", "incomplete");
      record.put("reason", missing.getMessage());
    } catch (Exception failure) {
      record.put("status", failure instanceof StructuredDataException ? "refused" : "failed");
      var error = new LinkedHashMap<String, Object>();
      error.put("stage", record.getOrDefault("stage", "setup"));
      error.put(
          "code",
          failure instanceof StructuredDataException e
              ? e.code()
              : failure.getClass().getSimpleName());
      error.put(
          "rule", failure instanceof StructuredDataException e ? e.details().get("rule") : null);
      record.put("error", error);
    } finally {
      route.end(record);
      record.remove("stage");
    }
    return record;
  }

  static void journey(
      Map<String, Object> scenario,
      String connector,
      String model,
      Map<String, Object> config,
      Map<String, Object> fixture,
      Map<String, Object> record,
      ParityRoute route)
      throws Exception {
    try (RootAllocator allocator = new RootAllocator();
        Inputs inputs = new Inputs()) {
      record.put("stage", "source");
      var scopedSources = new LinkedHashMap<>(map(config.get("sources")));
      scopedSources.putAll(
          map(map(config.get("fixture_sources")).get(text(scenario.get("fixture")))));
      var scopedConfig = new LinkedHashMap<>(config);
      scopedConfig.put("sources", scopedSources);
      inputs.open(connector, scopedConfig, fixture, allocator);
      var readerEvidence = new LinkedHashMap<String, Object>();
      inputs.tables.forEach(
          (name, value) ->
              readerEvidence.put(
                  name,
                  Map.of(
                      "schema",
                      value.getSchema().getFields().stream()
                          .map(field -> List.of(field.getName(), arrowType(field)))
                          .toList(),
                      "raw_schema",
                      value.getSchema().getFields().stream()
                          .map(field -> List.of(field.getName(), field.toString()))
                          .toList(),
                      "data",
                      frame(value))));
      record.put("reader", readerEvidence);
      boolean rawReader = "raw-reader".equals(scenario.get("input_mode"));
      if (!rawReader) inputs.restoreFixtureTypes(allocator, fixture);
      verify(inputs.tables, fixture, rawReader);
      record.put("source_snapshot", digest(fixture));
      record.put("stage", "schema");
      Schema schema;
      var keys = map(fixture.get("primary_keys"));
      String userKey = text(keys.get("users"));
      String targetColumn = fixtureColumn(fixture, "users", "value");
      String eventTime = fixtureColumn(fixture, "events", "event_time");
      boolean fullSchema = "pql-relational-schema".equals(scenario.get("entry"));
      if (model.equals("tabular") && !fullSchema)
        schema =
            Schema.fromData(Map.of("users", inputs.tables.get("users")))
                .withPrimaryKey("users", userKey);
      else {
        if (inputs.source == null
            || fullSchema
            || (rawReader && !"source".equals(scenario.get("schema_input"))))
          schema = Schema.fromData(inputs.tables);
        else {
          var options = FromSourceOptions.builder();
          inputs.names.forEach(options::table);
          keys.forEach((table, key) -> options.primaryKey(table, text(key)));
          options.timeColumn("events", eventTime);
          schema = Schema.fromSource(inputs.source, options.build());
        }
        for (var key : keys.entrySet())
          schema = schema.withPrimaryKey(key.getKey(), text(key.getValue()));
        schema = schema.withTimeColumn("events", eventTime);
        var links = new ArrayList<Relationship>();
        for (Object link : list(fixture.get("links"))) {
          var names = list(link);
          var from = text(names.get(0)).split("\\.");
          var to = text(names.get(1)).split("\\.");
          links.add(new Relationship(from[0], from[1], to[0], to[1]));
        }
        schema = schema.withRelationships(links);
      }
      schema = schema.withStype("users", targetColumn, "numerical");
      if (scenario.containsKey("semantic_types"))
        for (var declared : map(scenario.get("semantic_types")).entrySet()) {
          String[] name = declared.getKey().split("\\.", 2);
          schema = schema.withStype(name[0], name[1], text(declared.getValue()));
        }
      var semantic = new LinkedHashMap<String, Object>();
      var lifecycle = new ArrayList<String>();
      var results = new ArrayList<Object>();
      var rawResults = new ArrayList<Object>();
      record.put("raw_results", rawResults);
      semantic.put("schema", schema(schema));
      semantic.put("request", null);
      semantic.put("lifecycle", lifecycle);
      semantic.put("results", results);
      record.put("semantic", semantic);
      record.put("stage", "materialize");
      Request request;
      if ("direct".equals(scenario.get("entry"))) {
        Table context = inputs.tables.get("users");
        Map<String, Object> contextRows = frame(context);
        var columns = new ArrayList<>(list(contextRows.get("columns")));
        int target = columns.indexOf(targetColumn);
        columns.remove(target);
        var scoring = new ArrayList<Object>();
        for (Object row : list(contextRows.get("rows")).subList(0, 2)) {
          var cells = new ArrayList<>(list(row));
          cells.remove(target);
          scoring.add(cells);
        }
        Map<String, Object> predict = Map.of("columns", columns, "rows", scoring);
        try (Table values = table(predict, allocator)) {
          request =
              new Request(
                  context,
                  values,
                  targetColumn,
                  "regression",
                  RequestOptions.builder().schema(schema).entityTable("users").build());
        }
        semantic.put("request", requestEvidence(request, allocator));
      } else {
        var options =
            MaterializeOptions.builder()
                .model(
                    model.equals("tabular") ? SdmClient.KUMO_TABULAR : SdmClient.KUMO_RELATIONAL);
        map(scenario.get("options"))
            .forEach(
                (key, value) -> {
                  if (key.equals("indices")) options.indices(list(value));
                  else if (key.equals("candidates")) options.candidates(list(value));
                  else if (key.equals("anchor_time")) options.anchorTime(fixtureTime(value));
                  else if (key.equals("num_context_samples"))
                    options.numContextSamples(((Number) value).intValue());
                  else if (!model.equals("tabular") || !key.equals("random_seed"))
                    options.option(key, value);
                });
        request =
            PredictiveQuery.parse(text(scenario.get("pql"))).materialize(schema, options.build());
        semantic.put("request", requestEvidence(request, allocator));
      }
      lifecycle.add("materialize");
      if ("restore-predict".equals(scenario.get("journey"))) {
        record.put("stage", "restore");
        var output = new ByteArrayOutputStream();
        request.writeTo(output);
        request = Request.readFrom(new ByteArrayInputStream(output.toByteArray()));
        Object restored = requestEvidence(request, allocator);
        if (!digest(semantic.get("request")).equals(digest(restored))) {
          throw new IllegalStateException("restored request differs from its saved data");
        }
        semantic.put("request", restored);
        lifecycle.add("restore");
      }
      if (record.get("execution").equals("local")) {
        record.put("status", "passed");
        return;
      }
      Map<String, Object> endpoint =
          map(map(map(config.get("endpoints_by_case")).get(scenario.get("id"))).get(model));
      if (endpoint.isEmpty()) endpoint = map(map(config.get("endpoints")).get(model));
      if (!"live".equals(endpoint.get("kind")) || endpoint.get("version") == null)
        throw new Incomplete("live endpoint/model-version configuration missing");
      record.put(
          "endpoint",
          Map.of(
              "model",
              endpoint.getOrDefault("model", "kumo-" + model),
              "kind",
              "live",
              "version",
              endpoint.get("version"),
              "transport",
              endpoint.get("transport")));
      boolean supplemental = scenario.containsKey("supplement");
      List<String> outputs =
          scenario.containsKey("outputs")
              ? list(scenario.get("outputs")).stream().map(ParityRunner::text).toList()
              : null;
      var details = new ArrayList<Object>();
      var transport = new ArrayList<Map<String, Object>>();
      if (supplemental) {
        semantic.put("result_details", details);
        semantic.put("transport", transport);
        route.capture(transport);
      } else if (rawReader) {
        // Retain actual outgoing bodies even when the endpoint refuses prediction.
        // Successful fit responses retain the handle needed to prove session ownership.
        semantic.put("wire_calls", transport);
        route.capture(transport);
      }
      String before = digest(semantic.get("request"));
      java.util.function.Consumer<PredictionResult> observe =
          answer -> {
            try {
              rawResults.add(frame(answer, allocator));
            } catch (Exception error) {
              throw new IllegalStateException(error);
            }
            results.add(ParityResults.frame(answer));
            if (supplemental) {
              if (!answer.absent(outputs).isEmpty())
                throw new IllegalStateException("requested output is absent");
              details.add(ParityResults.details(answer));
            }
          };
      try (SdmClient client = client(endpoint)) {
        String chosen =
            model.equals("tabular") ? SdmClient.KUMO_TABULAR : SdmClient.KUMO_RELATIONAL;
        if ("fit-predict-close".equals(scenario.get("journey"))) {
          record.put("stage", "fit");
          List<?> held =
              supplemental ? List.of(ParityHoldout.key(request, allocator)) : List.of(1L, 2L);
          if (supplemental) semantic.put("holdout_keys", scalar(held));
          FittedModel fitted = client.fit(request, chosen, held, outputs);
          lifecycle.add("fit");
          try {
            for (int i = 0; i < 2; i++) {
              record.put("stage", "predict");
              observe.accept(fitted.predict(request));
              lifecycle.add("predict");
            }
          } finally {
            record.put("stage", "close");
            fitted.close();
            lifecycle.add("close");
          }
          if (supplemental) ParityHoldout.check(request, held, transport);
        } else {
          record.put("stage", "predict");
          observe.accept(client.predict(request, chosen, outputs));
          lifecycle.add("predict");
        }
      }
      if (supplemental) {
        if (!before.equals(digest(requestEvidence(request, allocator))))
          throw new IllegalStateException("prediction mutated request");
        if (!"holdout".equals(map(scenario.get("supplement")).get("kind"))) {
          int count = ((Number) map(scenario.get("supplement")).get("prediction_calls")).intValue();
          if (transport.size() != count
              || transport.stream().anyMatch(call -> !"predict".equals(call.get("operation"))))
            throw new IllegalStateException("unexpected prediction call count");
          if ("batch".equals(map(scenario.get("supplement")).get("kind"))
              && request.payloads().size() != count)
            throw new IllegalStateException("batch payload count differs");
        }
      }
      record.put("status", "passed");
    }
  }

  static Object requestEvidence(Request request, RootAllocator allocator) throws Exception {
    validateInspection(request.context(), request.contextRows(), allocator);
    validateInspection(request.predict(), request.predictRows(), allocator);
    var payloads = new ArrayList<Object>();
    for (String payload : request.payloads())
      payloads.add(Json.read(payload.getBytes(StandardCharsets.UTF_8)));
    return payloads.isEmpty() ? tabularSnapshot(request, allocator) : payloads;
  }

  /** Exercise public inspection without replacing the raw request evidence compared below. */
  static void validateInspection(RequestSplit split, Long expectedRows, RootAllocator allocator)
      throws Exception {
    long rows = consume(split.read(allocator));
    if (expectedRows == null || rows != expectedRows)
      throw new IllegalStateException("request inspection row count differs from its public count");
    for (String name : split.relatedTables()) consume(split.readRelated(name, allocator));
  }

  static long consume(ArrowReader reader) throws Exception {
    try (reader) {
      long rows = 0;
      while (reader.loadNextBatch())
        rows = Math.addExact(rows, reader.getVectorSchemaRoot().getRowCount());
      return rows;
    }
  }

  static LocalDateTime fixtureTime(Object value) {
    String text = text(value).replace(' ', 'T');
    return text.length() == 10 ? LocalDate.parse(text).atStartOfDay() : LocalDateTime.parse(text);
  }

  /** Inspect public persistence output until the full cross-model split API is delivered. */
  static Map<String, Object> tabularSnapshot(Request request, RootAllocator allocator)
      throws Exception {
    var saved = new ByteArrayOutputStream();
    request.writeTo(saved);
    var input = new DataInputStream(new ByteArrayInputStream(saved.toByteArray()));
    if (!Arrays.equals(input.readNBytes(8), new byte[] {'S', 'D', 'M', 'S', 'N', 'A', 'P', 0})
        || input.readInt() != 1)
      throw new Incomplete("unsupported public snapshot format for observation");
    Map<String, Object> manifest = map(Json.read(input.readNBytes(input.readInt())));
    int count = input.readInt();
    var blobs = new HashMap<Integer, byte[]>();
    for (int at = 0; at < count; at++) {
      int id = input.readInt();
      long size = input.readLong();
      if (size < 0 || size > Integer.MAX_VALUE) throw new Incomplete("invalid snapshot size");
      blobs.put(id, input.readNBytes((int) size));
    }
    if (!"direct_request".equals(manifest.get("kind")))
      throw new Incomplete("empty materialized payloads cannot establish request parity");
    var output = new LinkedHashMap<String, Object>();
    for (String name : List.of("context", "predict"))
      try (var reader =
          new org.apache.arrow.vector.ipc.ArrowStreamReader(
              new ByteArrayInputStream(blobs.get(((Number) manifest.get(name)).intValue())),
              allocator)) {
        var root = reader.getVectorSchemaRoot();
        var rows = new ArrayList<Object>();
        while (reader.loadNextBatch())
          for (int at = 0; at < root.getRowCount(); at++) {
            var values = new ArrayList<Object>();
            for (var vector : root.getFieldVectors()) values.add(scalar(vector.getObject(at)));
            rows.add(values);
          }
        output.put(
            name,
            Map.of(
                "columns",
                root.getSchema().getFields().stream().map(Field::getName).toList(),
                "rows",
                rows));
      }
    output.put("target", map(manifest.get("input")).get("target"));
    output.put("task", map(manifest.get("input")).get("task"));
    return output;
  }

  static SdmClient client(Map<String, Object> endpoint) throws Exception {
    Map<String, Object> cfg = map(endpoint.get("client"));
    String transport = text(endpoint.get("transport"));
    if (Set.of("nim", "private-bridge").contains(transport)) {
      var options = ClientOptions.builder();
      if (cfg.get("api_key") != null) options.apiKey(text(cfg.get("api_key")));
      if (cfg.get("timeout") instanceof Number n)
        options.timeout(Duration.ofMillis((long) (n.doubleValue() * 1000)));
      return SdmClient.connect(text(cfg.get("url")), options.build());
    }
    if (transport.equals("databricks")) {
      var config = new com.databricks.sdk.core.DatabricksConfig();
      if (cfg.get("profile") != null) config.setProfile(text(cfg.get("profile")));
      return SdmClient.forDatabricksServing(
          DatabricksServing.of(
              new com.databricks.sdk.WorkspaceClient(config), text(cfg.get("endpoint"))));
    }
    throw new Incomplete("unsupported public endpoint transport");
  }

  static String arrowType(Field field) {
    ArrowType type = field.getType();
    if (type instanceof ArrowType.List
        || type instanceof ArrowType.LargeList
        || type instanceof ArrowType.FixedSizeList) {
      if (field.getChildren().size() != 1)
        throw new IllegalArgumentException("list evidence requires one child field");
      Field child = field.getChildren().get(0);
      String nested =
          "item: " + arrowType(child) + (child.isNullable() ? "" : " not null");
      if (type instanceof ArrowType.FixedSizeList fixed)
        return "fixed_size_list<" + nested + ">[" + fixed.getListSize() + "]";
      return (type instanceof ArrowType.LargeList ? "large_list<" : "list<") + nested + ">";
    }
    return arrowType(type);
  }

  static String arrowType(ArrowType type) {
    if (type instanceof ArrowType.Int value)
      return (value.getIsSigned() ? "int" : "uint") + value.getBitWidth();
    if (type instanceof ArrowType.FloatingPoint value)
      return switch (value.getPrecision()) {
        case HALF -> "halffloat";
        case SINGLE -> "float";
        case DOUBLE -> "double";
      };
    if (type instanceof ArrowType.Utf8) return "string";
    if (type instanceof ArrowType.LargeUtf8) return "string";
    if (type instanceof ArrowType.Bool) return "bool";
    if (type instanceof ArrowType.Binary) return "binary";
    if (type instanceof ArrowType.LargeBinary) return "binary";
    if (type instanceof ArrowType.Null) return "null";
    if (type instanceof ArrowType.Timestamp value) {
      String unit =
          switch (value.getUnit()) {
            case SECOND -> "s";
            case MILLISECOND -> "ms";
            case MICROSECOND -> "us";
            case NANOSECOND -> "ns";
          };
      return "timestamp["
          + unit
          + (value.getTimezone() == null ? "" : ", tz=" + value.getTimezone())
          + "]";
    }
    if (type instanceof ArrowType.Date value)
      return value.getUnit() == org.apache.arrow.vector.types.DateUnit.DAY
          ? "date32[day]"
          : "date64[ms]";
    if (type instanceof ArrowType.Decimal value)
      return "decimal"
          + value.getBitWidth()
          + "("
          + value.getPrecision()
          + ", "
          + value.getScale()
          + ")";
    return type.toString();
  }

  static Object scalar(Object value) {
    if (value instanceof java.math.BigDecimal decimal) {
      try {
        return decimal.toBigIntegerExact();
      } catch (ArithmeticException fractional) {
        double converted = decimal.doubleValue();
        if (Double.isFinite(converted)
            && new java.math.BigDecimal(converted).compareTo(decimal) == 0) return converted;
        return Map.of("decimal", decimal.toPlainString());
      }
    }
    if (value == null || value instanceof Boolean || value instanceof Number) {
      if (value instanceof Double d && !Double.isFinite(d))
        return Map.of("float", d.isNaN() ? "nan" : d > 0 ? "inf" : "-inf");
      return value;
    }
    if (value instanceof LocalDateTime date)
      return date.format(java.time.format.DateTimeFormatter.ISO_LOCAL_DATE_TIME);
    if (value instanceof List<?> values) return values.stream().map(ParityRunner::scalar).toList();
    if (value instanceof Map<?, ?> values) {
      var out = new LinkedHashMap<String, Object>();
      values.forEach((k, v) -> out.put(text(k), scalar(v)));
      return out;
    }
    return value.toString();
  }

  static Map<String, Object> frame(Table table) {
    var columns = table.getSchema().getFields().stream().map(Field::getName).toList();
    var rows = new ArrayList<Object>();
    for (int row = 0; row < table.getRowCount(); row++) {
      var values = new ArrayList<Object>();
      for (int col = 0; col < columns.size(); col++) {
        var reader = table.getReader(col);
        reader.setPosition(row);
        Object value = reader.readObject();
        var type = table.getSchema().getFields().get(col).getType();
        if (value != null
            && type instanceof ArrowType.Date date
            && date.getUnit() == org.apache.arrow.vector.types.DateUnit.DAY)
          value = java.time.LocalDate.ofEpochDay(((Number) value).longValue()).toString();
        values.add(scalar(value));
      }
      rows.add(values);
    }
    return Map.of("columns", columns, "rows", rows);
  }

  static Map<String, Object> frame(PredictionResult result, RootAllocator allocator)
      throws Exception {
    try (ArrowReader reader = result.read(allocator)) {
      var root = reader.getVectorSchemaRoot();
      var columns = root.getSchema().getFields().stream().map(Field::getName).toList();
      var rows = new ArrayList<Object>();
      while (reader.loadNextBatch())
        for (int at = 0; at < root.getRowCount(); at++) {
          var values = new ArrayList<Object>();
          for (FieldVector vector : root.getFieldVectors())
            values.add(scalar(vector.getObject(at)));
          rows.add(values);
        }
      return Map.of("columns", columns, "rows", rows);
    }
  }

  static Map<String, Object> schema(Schema schema) {
    var output = new LinkedHashMap<String, Object>();
    for (com.nvidia.sdm.Table table : schema.tables()) {
      var item = new LinkedHashMap<String, Object>();
      item.put(
          "primary_key",
          table.primaryKey().size() == 1
              ? table.primaryKey().get(0)
              : table.primaryKey().isEmpty() ? null : table.primaryKey());
      item.put("time_column", table.timeColumn().orElse(null));
      item.put(
          "columns",
          table.columns().stream()
              .map(c -> Arrays.asList(c.name(), c.dtype(), c.stype()))
              .toList());
      output.put(table.name(), item);
    }
    return output;
  }

  static String fixtureColumn(Map<String, Object> fixture, String table, String name) {
    return list(map(map(fixture.get("tables")).get(table)).get("columns")).stream()
        .map(ParityRunner::text)
        .filter(column -> column.equalsIgnoreCase(name))
        .findFirst()
        .orElse(name);
  }

  static Field fixtureField(String name, String type) {
    ArrowType arrow;
    if (type.startsWith("decimal128(")) {
      String[] parts = type.substring(11, type.length() - 1).split(",");
      arrow = new ArrowType.Decimal(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), 128);
    } else if (type.startsWith("timestamp[")) {
      String[] parts = type.substring(10, type.length() - 1).split(",", 2);
      var unit =
          switch (parts[0]) {
            case "s" -> org.apache.arrow.vector.types.TimeUnit.SECOND;
            case "ms" -> org.apache.arrow.vector.types.TimeUnit.MILLISECOND;
            case "us" -> org.apache.arrow.vector.types.TimeUnit.MICROSECOND;
            case "ns" -> org.apache.arrow.vector.types.TimeUnit.NANOSECOND;
            default -> throw new IllegalArgumentException("unknown fixture time unit");
          };
      arrow = new ArrowType.Timestamp(unit, parts.length == 2 ? parts[1] : null);
    } else {
      arrow =
          switch (type) {
            case "date32" -> new ArrowType.Date(org.apache.arrow.vector.types.DateUnit.DAY);
            case "int16" -> new ArrowType.Int(16, true);
            case "int64" -> new ArrowType.Int(64, true);
            case "float32" ->
                new ArrowType.FloatingPoint(
                    org.apache.arrow.vector.types.FloatingPointPrecision.SINGLE);
            case "float64" ->
                new ArrowType.FloatingPoint(
                    org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE);
            case "string" -> ArrowType.Utf8.INSTANCE;
            default -> throw new IllegalArgumentException("unknown fixture type: " + type);
          };
    }
    return new Field(name, FieldType.nullable(arrow), null);
  }

  static Table table(Map<String, Object> fixture, RootAllocator allocator) {
    List<Object> columns = list(fixture.get("columns")), rows = list(fixture.get("rows"));
    var vectors = new ArrayList<FieldVector>();
    try {
      for (int col = 0; col < columns.size(); col++) {
        String name = text(columns.get(col));
        Object first = null;
        for (Object row : rows)
          if (list(row).get(col) != null) {
            first = list(row).get(col);
            break;
          }
        String declared = text(map(fixture.get("types")).get(name));
        if (declared != null && declared.startsWith("list<")) {
          ArrowType childType =
              switch (declared) {
                case "list<string>" -> ArrowType.Utf8.INSTANCE;
                case "list<float64>" ->
                    new ArrowType.FloatingPoint(
                        org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE);
                case "list<int64>" -> new ArrowType.Int(64, true);
                default ->
                    throw new IllegalArgumentException(
                        "unsupported fixture list type: " + declared);
              };
          var field =
              new Field(
                  name,
                  FieldType.nullable(ArrowType.List.INSTANCE),
                  List.of(new Field("item", FieldType.nullable(childType), null)));
          var vector = (org.apache.arrow.vector.complex.ListVector) field.createVector(allocator);
          vectors.add(vector);
          vector.allocateNew();
          FieldVector child = vector.getDataVector();
          int count = 0;
          for (int at = 0; at < rows.size(); at++) {
            Object value = list(rows.get(at)).get(col);
            if (value == null) {
              vector.setNull(at);
              continue;
            }
            var elements = list(value);
            int offset = vector.startNewValue(at);
            for (int element = 0; element < elements.size(); element++) {
              Object item = elements.get(element);
              int position = offset + element;
              if (item == null) child.setNull(position);
              else if (child instanceof VarCharVector v)
                v.setSafe(position, text(item).getBytes(StandardCharsets.UTF_8));
              else if (child instanceof Float8Vector v)
                v.setSafe(position, ((Number) item).doubleValue());
              else ((BigIntVector) child).setSafe(position, ((Number) item).longValue());
            }
            vector.endValue(at, elements.size());
            count = offset + elements.size();
          }
          child.setValueCount(count);
          vector.setValueCount(rows.size());
          continue;
        }
        FieldVector vector =
            declared != null
                ? fixtureField(name, declared).createVector(allocator)
                : name.equalsIgnoreCase("event_time")
                    ? new TimeStampMicroVector(name, allocator)
                    : first instanceof Boolean
                        ? new BitVector(name, allocator)
                        : first instanceof Float
                                || first instanceof Double
                                || first instanceof java.math.BigDecimal
                            ? new Float8Vector(name, allocator)
                            : first instanceof Number
                                ? new BigIntVector(name, allocator)
                                : new VarCharVector(name, allocator);
        vectors.add(vector);
        vector.allocateNew();
        for (int at = 0; at < rows.size(); at++) {
          Object value = list(rows.get(at)).get(col);
          if (value == null) continue;
          if (declared != null && vector instanceof TimeStampVector v) {
            Instant instant = OffsetDateTime.parse(text(value)).toInstant();
            var unit = ((ArrowType.Timestamp) vector.getField().getType()).getUnit();
            long factor =
                switch (unit) {
                  case SECOND -> 1L;
                  case MILLISECOND -> 1000L;
                  case MICROSECOND -> 1000000L;
                  case NANOSECOND -> 1000000000L;
                };
            v.setSafe(
                at,
                Math.addExact(
                    Math.multiplyExact(instant.getEpochSecond(), factor),
                    instant.getNano() / (1000000000L / factor)));
          } else if (vector instanceof DateDayVector v)
            v.setSafe(at, Math.toIntExact(java.time.LocalDate.parse(text(value)).toEpochDay()));
          else if (vector instanceof DecimalVector v)
            v.setSafe(at, new java.math.BigDecimal(text(value)));
          else if (vector instanceof SmallIntVector v) v.setSafe(at, ((Number) value).intValue());
          else if (vector instanceof Float4Vector v) v.setSafe(at, ((Number) value).floatValue());
          else if (vector instanceof TimeStampMicroVector v)
            v.setSafe(at, fixtureTime(value).toInstant(ZoneOffset.UTC).toEpochMilli() * 1000);
          else if (vector instanceof BitVector v) v.setSafe(at, (Boolean) value ? 1 : 0);
          else if (vector instanceof Float8Vector v) v.setSafe(at, ((Number) value).doubleValue());
          else if (vector instanceof BigIntVector v) v.setSafe(at, ((Number) value).longValue());
          else ((VarCharVector) vector).setSafe(at, text(value).getBytes(StandardCharsets.UTF_8));
        }
        vector.setValueCount(rows.size());
      }
      return new Table(vectors, rows.size());
    } finally {
      vectors.forEach(FieldVector::close);
    }
  }

  static void verify(Map<String, Table> actual, Map<String, Object> fixture, boolean rawReader)
      throws Exception {
    for (var entry : map(fixture.get("tables")).entrySet()) {
      String name = entry.getKey();
      if (!actual.containsKey(name)) throw new Incomplete("fixture table missing: " + name);
      Map<String, Object> expected = map(entry.getValue()), seen = frame(actual.get(name));
      Map<String, Object> normalized;
      // Construct an independent expected fixture, never replace reader output.
      try (RootAllocator allocator = new RootAllocator();
          Table reference = table(expected, allocator)) {
        normalized = frame(reference);
      }
      int key = list(expected.get("columns")).indexOf(map(fixture.get("primary_keys")).get(name));
      var ordered = new ArrayList<>(list(seen.get("rows")));
      ordered.sort(Comparator.comparingLong(row -> ((Number) list(row).get(key)).longValue()));
      if (rawReader) {
        // Only detached verification observations are adjusted. The reader's
        // Arrow table and captured reader evidence remain unchanged.
        var wantedRows = list(normalized.get("rows"));
        var columns = list(seen.get("columns"));
        for (int row = 0; row < Math.min(ordered.size(), wantedRows.size()); row++) {
          var values = new ArrayList<>(list(ordered.get(row)));
          var wanted = list(wantedRows.get(row));
          for (int col = 0; col < columns.size(); col++) {
            Object value = values.get(col), target = wanted.get(col);
            String column = text(columns.get(col));
            if (column.equalsIgnoreCase("class")
                && value instanceof Number
                && text(value).equals(target)) values.set(col, target);
            else if (column.equalsIgnoreCase("flag")
                && value instanceof Number number
                && target instanceof Boolean flag
                && new java.math.BigDecimal(number.toString())
                        .compareTo(java.math.BigDecimal.valueOf(flag ? 1 : 0))
                    == 0) values.set(col, target);
            else if (column.equalsIgnoreCase("event_time")
                && value instanceof String date
                && target instanceof String timestamp) {
              var parsed =
                  date.length() == 10
                      ? LocalDate.parse(date).atStartOfDay()
                      : LocalDateTime.parse(date);
              if (parsed.equals(LocalDateTime.parse(timestamp))) values.set(col, target);
            }
          }
          ordered.set(row, values);
        }
      }
      if (!digest(normalized)
          .equals(digest(Map.of("columns", seen.get("columns"), "rows", ordered))))
        throw new Incomplete("source values differ from shared fixture: " + name);
    }
  }

  static final class Inputs implements AutoCloseable {
    final Map<String, Table> tables = new LinkedHashMap<>();
    final Map<String, String> names = new LinkedHashMap<>();
    final List<AutoCloseable> owners = new ArrayList<>();
    Source source;
    Path temporary;

    void open(
        String connector,
        Map<String, Object> config,
        Map<String, Object> fixture,
        RootAllocator allocator)
        throws Exception {
      var descriptions = map(fixture.get("tables"));
      var settings = map(map(config.get("sources")).get(connector));
      descriptions
          .keySet()
          .forEach(
              name -> names.put(name, text(map(settings.get("tables")).getOrDefault(name, name))));
      if (connector.equals("memory")) {
        for (var entry : descriptions.entrySet())
          tables.put(entry.getKey(), table(map(entry.getValue()), allocator));
        return;
      }
      if (Set.of("csv", "parquet", "zip", "s3-csv", "s3-parquet").contains(connector)) {
        if (connector.startsWith("s3-") || !map(settings.get("paths")).isEmpty()) {
          var paths = map(settings.get("paths"));
          if (paths.isEmpty()) throw new Incomplete("missing configured S3 fixture paths");
          for (var entry : paths.entrySet())
            tables.put(
                entry.getKey(),
                Readers.read(
                    URI.create(text(entry.getValue())).getScheme() == null
                        ? Path.of(text(entry.getValue())).toUri()
                        : URI.create(text(entry.getValue())),
                    fileReadOptions(settings),
                    allocator));
        } else {
          if (!connector.equals("parquet")) {
            // pandas.to_csv renders this shared all-midnight fixture as date-only strings.
            // Assert the fixture assumption rather than silently truncate future time values.
            for (Object description : descriptions.values()) {
              var data = map(description);
              int time = list(data.get("columns")).indexOf("event_time");
              if (time >= 0)
                for (Object row : list(data.get("rows"))) {
                  Object value = list(row).get(time);
                  if (value != null
                      && !LocalDateTime.parse(text(value)).toLocalTime().equals(LocalTime.MIDNIGHT))
                    throw new IllegalStateException(
                        "CSV parity fixture requires midnight event_time values");
                }
            }
          }
          temporary = Files.createTempDirectory("sdm-parity-");
          try (Connection connection = DriverManager.getConnection("jdbc:duckdb:")) {
            create(connection, fixture, false);
            for (String name : descriptions.keySet()) {
              Path path =
                  temporary.resolve(name + (connector.equals("parquet") ? ".parquet" : ".csv"));
              try (Statement sql = connection.createStatement()) {
                sql.execute(
                    "COPY \""
                        + name
                        + "\" TO '"
                        + path.toString().replace("'", "''")
                        + "' (FORMAT "
                        + (connector.equals("parquet")
                            ? "PARQUET"
                            : "CSV, HEADER true, TIMESTAMPFORMAT '%Y-%m-%d'")
                        + ")");
              }
              if (connector.equals("zip")) {
                Path zipped = temporary.resolve(name + ".csv.zip");
                try (var zip = new ZipOutputStream(Files.newOutputStream(zipped))) {
                  zip.putNextEntry(new ZipEntry(name + ".csv"));
                  Files.copy(path, zip);
                  zip.closeEntry();
                }
                path = zipped;
              }
              tables.put(name, Readers.read(path, allocator));
            }
          }
        }
        return;
      }
      if (connector.equals("databricks-spark")) {
        if (settings.get("snapshot") == null)
          throw new Incomplete("missing verified Spark source configuration");
        Object builder;
        try {
          builder =
              Class.forName("com.databricks.connect.DatabricksSession")
                  .getMethod("builder")
                  .invoke(null);
        } catch (ClassNotFoundException missing) {
          throw new Incomplete("Databricks Connect Java runtime is not installed");
        }
        var sdk = new com.databricks.sdk.core.DatabricksConfig();
        var java = map(settings.get("java"));
        if (java.get("profile") != null) sdk.setProfile(text(java.get("profile")));
        if (java.get("cluster_id") != null) sdk.setClusterId(text(java.get("cluster_id")));
        builder =
            builder
                .getClass()
                .getMethod("sdkConfig", com.databricks.sdk.core.DatabricksConfig.class)
                .invoke(builder, sdk);
        var session =
            (org.apache.spark.sql.SparkSession)
                builder.getClass().getMethod("getOrCreate").invoke(builder);
        owners.add(session);
        var ns = map(settings.get("source"));
        source = DatabricksSparkSource.of(session, text(ns.get("catalog")), text(ns.get("schema")));
      } else {
        Connection connection;
        if (Set.of("duckdb", "sqlite").contains(connector)) {
          connection =
              DriverManager.getConnection(
                  connector.equals("duckdb") ? "jdbc:duckdb:" : "jdbc:sqlite::memory:");
          owners.add(connection);
          create(connection, fixture, connector.equals("sqlite"));
        } else {
          var java = map(settings.get("java"));
          if (settings.get("snapshot") == null || java.get("url") == null)
            throw new Incomplete("missing verified JDBC source configuration");
          var properties = new Properties();
          map(java.get("properties")).forEach((k, v) -> properties.setProperty(k, text(v)));
          connection = DriverManager.getConnection(text(java.get("url")), properties);
          owners.add(connection);
        }
        var ns = map(settings.get("source"));
        source =
            switch (connector) {
              case "duckdb" -> DuckDbSource.of(connection);
              case "sqlite" -> SqliteSource.of(connection);
              case "postgres", "lakebase" -> PostgresSource.of(connection, text(ns.get("schema")));
              case "snowflake", "snowflake-semantic" ->
                  SnowflakeSource.of(connection, text(ns.get("database")), text(ns.get("schema")));
              case "databricks-sql", "databricks-metric" ->
                  DatabricksSource.of(connection, text(ns.get("catalog")), text(ns.get("schema")));
              default -> throw new Incomplete("connector has no Java public adapter");
            };
        if (connector.equals("snowflake-semantic"))
          source =
              ((SnowflakeSource) source)
                  .semanticModel(
                      text(settings.get("view")),
                      Boolean.TRUE.equals(settings.get("trust_expressions")));
        if (connector.equals("databricks-metric"))
          source =
              ((DatabricksSource) source)
                  .semanticModel(
                      text(settings.get("view")),
                      Boolean.TRUE.equals(settings.get("trust_expressions")));
      }
      for (var name : names.entrySet())
        tables.put(name.getKey(), Readers.readTable(source, name.getValue(), allocator));
    }

    void restoreFixtureTypes(RootAllocator allocator, Map<String, Object> fixture) {
      for (String name : new ArrayList<>(tables.keySet())) {
        Table old = tables.get(name);
        if (old == null) continue;
        Map<String, Object> value = frame(old);
        var columns = list(value.get("columns"));
        var rows = new ArrayList<Object>();
        for (Object raw : list(value.get("rows"))) {
          var row = new ArrayList<>(list(raw));
          int category = columns.indexOf(fixtureColumn(fixture, name, "class")),
              flag = columns.indexOf(fixtureColumn(fixture, name, "flag"));
          var declared = list(list(map(map(fixture.get("tables")).get(name)).get("rows")).get(0));
          for (int col = 0; col < row.size(); col++) {
            if (row.get(col) instanceof java.math.BigDecimal decimal) {
              Object expected = declared.get(col);
              if (expected instanceof Byte
                  || expected instanceof Short
                  || expected instanceof Integer
                  || expected instanceof Long) row.set(col, decimal.longValueExact());
              else if (expected instanceof Float || expected instanceof Double)
                row.set(col, decimal.doubleValue());
            }
          }
          if (category >= 0 && row.get(category) != null)
            row.set(category, text(row.get(category)));
          if (flag >= 0 && row.get(flag) instanceof Number number) {
            long n = number.longValue();
            if (n != 0 && n != 1)
              throw new IllegalArgumentException("fixture boolean must be zero or one");
            row.set(flag, n == 1);
          }
          rows.add(row);
        }
        int key = columns.indexOf(map(fixture.get("primary_keys")).get(name));
        rows.sort(Comparator.comparingLong(row -> ((Number) list(row).get(key)).longValue()));
        Table replacement = table(Map.of("columns", columns, "rows", rows), allocator);
        old.close();
        tables.put(name, replacement);
      }
    }

    static void create(Connection connection, Map<String, Object> fixture, boolean sqlite)
        throws Exception {
      for (var entry : map(fixture.get("tables")).entrySet()) {
        Map<String, Object> data = map(entry.getValue());
        var columns = list(data.get("columns"));
        var rows = list(data.get("rows"));
        var fields = new ArrayList<String>();
        for (int col = 0; col < columns.size(); col++) {
          String name = text(columns.get(col));
          Object first = null;
          for (Object row : rows) {
            Object candidate = list(row).get(col);
            if (candidate != null) {
              first = candidate;
              break;
            }
          }
          String declared = text(map(data.get("types")).get(name));
          String type =
              name.equalsIgnoreCase("event_time")
                  ? (sqlite ? "TEXT" : "TIMESTAMP")
                  : first instanceof Boolean
                      ? "BOOLEAN"
                      : first instanceof Float
                              || first instanceof Double
                              || first instanceof java.math.BigDecimal
                          ? "DOUBLE"
                          : first instanceof Number ? "BIGINT" : "VARCHAR";
          if (declared != null) {
            if (sqlite) throw new IllegalArgumentException("typed fixture requires DuckDB");
            if (declared.startsWith("decimal128(")) type = "DECIMAL" + declared.substring(10);
            else if (declared.startsWith("timestamp["))
              type =
                  declared.contains(",")
                      ? "TIMESTAMPTZ"
                      : declared.startsWith("timestamp[ns") ? "TIMESTAMP_NS" : "TIMESTAMP";
            else
              type =
                  switch (declared) {
                    case "date32" -> "DATE";
                    case "string" -> "VARCHAR";
                    case "int16" -> "SMALLINT";
                    case "int64" -> "BIGINT";
                    case "float32" -> "FLOAT";
                    case "float64" -> "DOUBLE";
                    case "list<string>" -> "VARCHAR[]";
                    case "list<float64>" -> "DOUBLE[]";
                    case "list<int64>" -> "BIGINT[]";
                    default -> throw new IllegalArgumentException("unknown fixture SQL type");
                  };
          }
          fields.add("\"" + name + "\" " + type);
        }
        try (Statement sql = connection.createStatement()) {
          sql.execute("CREATE TABLE \"" + entry.getKey() + "\" (" + String.join(",", fields) + ")");
        }
        try (PreparedStatement insert =
            connection.prepareStatement(
                "INSERT INTO \""
                    + entry.getKey()
                    + "\" VALUES ("
                    + String.join(",", Collections.nCopies(columns.size(), "?"))
                    + ")")) {
          for (Object row : rows) {
            var values = list(row);
            var arrays = new ArrayList<java.sql.Array>();
            try {
              for (int col = 0; col < values.size(); col++) {
                Object value = values.get(col);
                if (value instanceof List<?> elements) {
                  String declared = text(map(data.get("types")).get(text(columns.get(col))));
                  String base =
                      declared.equals("list<string>")
                          ? "VARCHAR"
                          : declared.equals("list<float64>") ? "DOUBLE" : "BIGINT";
                  var array = connection.createArrayOf(base, elements.toArray());
                  arrays.add(array);
                  insert.setArray(col + 1, array);
                } else insert.setObject(col + 1, value);
              }
              insert.executeUpdate();
            } finally {
              for (var array : arrays) array.free();
            }
          }
        }
      }
    }

    @Override
    public void close() throws IOException {
      Exception failure = null;
      for (Table table : tables.values())
        try {
          table.close();
        } catch (Exception error) {
          if (failure == null) failure = error;
          else failure.addSuppressed(error);
        }
      for (int at = owners.size() - 1; at >= 0; at--)
        try {
          owners.get(at).close();
        } catch (Exception error) {
          if (failure == null) failure = error;
          else failure.addSuppressed(error);
        }
      if (temporary != null)
        try (var paths = Files.walk(temporary)) {
          for (Path path : paths.sorted(Comparator.reverseOrder()).toList())
            Files.deleteIfExists(path);
        }
      if (failure != null) throw new IOException("fixture resources could not be closed", failure);
    }
  }
}
