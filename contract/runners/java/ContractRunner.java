// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import com.nvidia.sdm.PredictiveQuery;
import com.nvidia.sdm.Profile;
import com.nvidia.sdm.QuerySchema;
import com.nvidia.sdm.StructuredDataException;
import com.nvidia.sdm.Validation;
import com.nvidia.sdm.internal.Contract;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.internal.NativeCore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The Java runner program of the contract cases (design choice 10, "Tests"), run by
 * contract/runners/run_java.py in source-file mode against the JAR under test:
 *
 * <pre>java FLAGS -cp JAR:DEPENDENCIES ContractRunner.java PLAN ANSWERS</pre>
 *
 * PLAN is the JSON run_java.py wrote: {"calls": [...]}, each call one case's, holding only what
 * to call and with what (no case file is read here). Each is made through the public API, and
 * ANSWERS gets one JSON line per call: the case, the operations the call started, what this host
 * sent the core (as the API wrote it, through the same conversions), the core's answer, and the
 * error's kind and code when the call raised. run_java.py turns them into observations. A runner
 * failure (anything but an answer the API gave) exits non-zero.
 *
 * <p>The calls it makes are S0's: {"op": "validate_query", "case", "query", "schema", "profile"}
 * through PredictiveQuery.parse(query).validate(schema, profile). The Java host has no provider
 * yet, so no case of S1 or S2 is handed over (T3.12, T3.13).
 */
public final class ContractRunner {
  private ContractRunner() {}

  public static void main(String[] arguments) throws Exception {
    if (arguments.length != 2) {
      System.err.println("usage: ContractRunner PLAN ANSWERS");
      System.exit(2);
    }
    Map<?, ?> plan = (Map<?, ?>) Json.read(Files.readAllBytes(Path.of(arguments[0])));
    List<Object> answers = new ArrayList<>();
    for (Object call : (List<?>) plan.get("calls")) {
      answers.add(answer((Map<?, ?>) call));
    }
    StringBuilder lines = new StringBuilder();
    for (Object answer : answers) {
      lines.append(new String(Json.write(answer), java.nio.charset.StandardCharsets.UTF_8));
      lines.append('\n');
    }
    Files.writeString(Path.of(arguments[1]), lines.toString());
  }

  /** What the library and the JVM are, for the observation's identity. */
  static Map<String, Object> identity() {
    Map<String, Object> identity = new LinkedHashMap<>();
    identity.put("platform", NativeCore.platform());
    identity.put("contract", Host.contract());
    identity.put("java", System.getProperty("java.version"));
    identity.put("test_ops", NativeCore.load().get("test_ops"));
    identity.put("providers", Host.PROVIDERS);
    return identity;
  }

  private static Map<String, Object> answer(Map<?, ?> call) throws IOException {
    String op = (String) call.get("op");
    if (!"validate_query".equals(op)) {
      throw new IllegalArgumentException("a call this runner does not make: " + op);
    }
    String query = (String) call.get("query");
    QuerySchema schema = schema((Map<?, ?>) call.get("schema"));
    Profile profile = Profile.valueOf((String) ((Map<?, ?>) call.get("profile")).get("name"));
    Map<String, Object> input = new LinkedHashMap<>();
    input.put("op", "validate_query");
    input.put("query", query);
    input.put("profile", Contract.profile(profile));
    input.put("schema", Contract.querySchema(schema));
    Map<String, Object> answer = new LinkedHashMap<>();
    answer.put("case", call.get("case"));
    answer.put("operations", List.of("validate_query"));
    answer.put("inputs", List.of(input));
    answer.put("identity", identity());
    try {
      Validation validation = PredictiveQuery.parse(query).validate(schema, profile);
      answer.put("accepted", validation.accepted());
      answer.put("error", null);
    } catch (StructuredDataException raised) {
      Map<String, Object> error = new LinkedHashMap<>();
      error.put("kind", raised.kind());
      error.put("code", raised.code());
      answer.put("accepted", null);
      answer.put("error", error);
    }
    return answer;
  }

  private static String text(Map<?, ?> map, String key) {
    Object value = map.get(key);
    return value == null ? null : (String) value;
  }

  /** The case's QuerySchema JSON as the API's plain data. */
  static QuerySchema schema(Map<?, ?> json) {
    List<QuerySchema.Table> tables = new ArrayList<>();
    for (Object item : (List<?>) json.get("tables")) {
      Map<?, ?> table = (Map<?, ?>) item;
      List<QuerySchema.Column> columns = new ArrayList<>();
      for (Object c : (List<?>) table.get("columns")) {
        Map<?, ?> column = (Map<?, ?>) c;
        columns.add(
            new QuerySchema.Column(
                text(column, "name"), text(column, "dtype"), text(column, "stype")));
      }
      tables.add(
          new QuerySchema.Table(
              text(table, "name"),
              columns,
              text(table, "primary_key"),
              text(table, "time_column"),
              text(table, "end_time_column")));
    }
    List<List<QuerySchema.ColumnRef>> links = new ArrayList<>();
    for (Object g : (List<?>) json.get("links")) {
      List<QuerySchema.ColumnRef> group = new ArrayList<>();
      for (Object r : (List<?>) g) {
        Map<?, ?> ref = (Map<?, ?>) r;
        group.add(new QuerySchema.ColumnRef(text(ref, "table"), text(ref, "column")));
      }
      links.add(group);
    }
    return new QuerySchema(tables, links);
  }
}
