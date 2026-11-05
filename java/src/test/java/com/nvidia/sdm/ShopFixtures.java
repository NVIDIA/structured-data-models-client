// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static com.nvidia.sdm.ParityRunner.*;

import java.nio.file.Path;
import java.util.*;

/** Shared shop data with explicit extra user columns or uppercase identifiers. */
final class ShopFixtures {
  private ShopFixtures() {}

  static Map<String, Object> load(Path path) throws Exception {
    var variant = read(path);
    if (!variant.containsKey("base")) return variant;
    if (!"shop.json".equals(variant.get("base")))
      throw new IllegalArgumentException("shop variants must use shop.json");
    var fixture = new LinkedHashMap<>(read(path.resolveSibling("shop.json")));
    var tables = new LinkedHashMap<String, Object>();
    for (var entry : map(fixture.get("tables")).entrySet()) {
      var table = new LinkedHashMap<>(map(entry.getValue()));
      table.put("columns", new ArrayList<>(list(table.get("columns"))));
      var rows = new ArrayList<Object>();
      for (Object row : list(table.get("rows"))) rows.add(new ArrayList<>(list(row)));
      table.put("rows", rows);
      tables.put(entry.getKey(), table);
    }
    fixture.put("tables", tables);
    fixture.put("primary_keys", new LinkedHashMap<>(map(fixture.get("primary_keys"))));
    if (variant.keySet().equals(Set.of("base", "uppercase_columns"))) {
      if (!Boolean.TRUE.equals(variant.get("uppercase_columns")))
        throw new IllegalArgumentException("uppercase_columns must be true");
      for (Object value : map(fixture.get("tables")).values()) {
        var table = map(value);
        table.put("columns", list(table.get("columns")).stream()
            .map(name -> text(name).toUpperCase(Locale.ROOT)).toList());
      }
      var keys = map(fixture.get("primary_keys"));
      keys.replaceAll((table, name) -> text(name).toUpperCase(Locale.ROOT));
      fixture.put("links", list(fixture.get("links")).stream().map(link ->
          list(link).stream().map(name -> {
            String[] parts = text(name).split("\\.", 2);
            return parts[0] + "." + parts[1].toUpperCase(Locale.ROOT);
          }).toList()).toList());
      return fixture;
    }
    if (!variant.keySet().equals(Set.of("base", "user_columns")))
      throw new IllegalArgumentException("unknown shop variant fields");
    var users = map(map(fixture.get("tables")).get("users"));
    var names = list(users.get("columns"));
    var rows = list(users.get("rows"));
    for (Object value : list(variant.get("user_columns"))) {
      var column = map(value);
      if (!column.keySet().containsAll(Set.of("name", "values"))
          || !Set.of("name", "values", "type").containsAll(column.keySet()))
        throw new IllegalArgumentException("invalid shop column fields");
      String name = text(column.get("name"));
      var values = list(column.get("values"));
      if (names.contains(name) || values.size() != rows.size())
        throw new IllegalArgumentException("shop columns need unique names and one value per row");
      names.add(name);
      for (int i = 0; i < rows.size(); i++) list(rows.get(i)).add(values.get(i));
      if (column.containsKey("type")) {
        users.computeIfAbsent("types", ignored -> new LinkedHashMap<String, Object>());
        map(users.get("types")).put(name, column.get("type"));
      }
    }
    return fixture;
  }
}
