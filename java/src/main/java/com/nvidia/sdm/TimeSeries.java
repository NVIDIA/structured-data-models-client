// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.ArrowFrames;
import com.nvidia.sdm.internal.ArrowTabular;
import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.ReadTables;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.apache.arrow.vector.table.Table;

/** Host recipe around shared Arrow history preparation and normal schema admission. */
public final class TimeSeries {
  private TimeSeries() {}

  private static StructuredDataException invalid(String reason) {
    return new StructuredDataException(reason, "INVALID_REQUEST", Map.of(), null);
  }

  /** Does not consume input; returned schema owns its buffers. Naive Arrow times mean UTC. */
  public static ForecastInput build(Table table, TimeSeriesOptions options) {
    if (table == null || options == null) throw invalid("table and options are required");
    if (options.history == null
        || options.history.isEmpty()
        || options.forecasts <= 0
        || options.timestamps != null && options.timestamps.isEmpty()
        || options.entity != null && options.entity.isEmpty())
      throw invalid("invalid time-series names or forecast count");
    Map<String, Object> input = new LinkedHashMap<>();
    input.put("history", options.history);
    input.put("timestamps", options.timestamps);
    input.put("entity_key", options.entity);
    try {
      input.put("step_ns", options.interval == null ? null : options.interval.toNanos());
      input.put(
          "anchor_ns",
          options.anchor == null
              ? null
              : Math.addExact(
                  Math.multiplyExact(options.anchor.getEpochSecond(), 1_000_000_000L),
                  options.anchor.getNano()));
    } catch (ArithmeticException overflow) {
      throw invalid("time-series duration or anchor exceeds nanosecond range");
    }
    input.put("forecasts", options.forecasts);
    try (Host.Finished finished =
        ArrowFrames.run(
            Map.of("op", "prepare_time_series", "input", 1, "table", 2),
            Map.of(1, Host.attachment(input)),
            Map.of(2, TimeSeriesArrow.snapshot(table, options.timestamps)),
            Host.Call.none())) {
      Map<String, Object> value = ArrowTabular.value(finished);
      try (Table entity =
              ReadTables.collect(
                  finished.arrowReader(
                      ((Number) value.get("entity")).intValue(), ArrowData.allocator()),
                  ArrowData.allocator());
          Table target =
              ReadTables.collect(
                  finished.arrowReader(
                      ((Number) value.get("target")).intValue(), ArrowData.allocator()),
                  ArrowData.allocator())) {
        String key = (String) value.get("key");
        Map<String, Table> tables = new LinkedHashMap<>();
        tables.put("entity", entity);
        tables.put("target", target);
        Schema schema =
            Schema.fromData(tables)
                .withRelationships(List.of())
                .withPrimaryKey("entity", key)
                .withoutPrimaryKey("target")
                .withoutTimeColumn("entity")
                .withTimeColumn("target", "timestamp")
                .withStype("target", "value", "numerical")
                .withRelationships(List.of(new Relationship("target", key, "entity", key)));
        long minutes = ((Number) value.get("step_ns")).longValue() / 60_000_000_000L;
        long amount = minutes;
        String unit = "minutes";
        if (minutes % 1440 == 0) {
          amount = minutes / 1440;
          unit = "days";
        } else if (minutes % 60 == 0) {
          amount = minutes / 60;
          unit = "hours";
        }
        String quoted = "`" + key.replace("`", "``") + "`";
        PredictiveQuery query =
            PredictiveQuery.parse(
                "PREDICT MAX(target.value, 0, "
                    + amount
                    + ", "
                    + unit
                    + ") FORECAST "
                    + options.forecasts
                    + " TIMEFRAMES FOR EACH entity."
                    + quoted);
        return new ForecastInput(schema, query, options.anchor);
      } catch (java.io.IOException error) {
        throw new StructuredDataException(
            "prepared Arrow tables could not be read", "INTERNAL_ERROR", Map.of(), error);
      }
    }
  }
}
