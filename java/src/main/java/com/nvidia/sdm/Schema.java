// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Json;
import com.nvidia.sdm.internal.NativeCore;
import com.nvidia.sdm.internal.Providers;
import com.nvidia.sdm.internal.Results;
import com.nvidia.sdm.internal.Wording;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * The tables a request is about, as Python's {@code Schema}: built from a source with {@link
 * #fromSource}, each table's columns with their data types and meanings, its key and time column,
 * and the relationships between the tables. The core describes the source and makes every edit,
 * as the reference's client does; this class holds the answer and decides nothing of it.
 *
 * <p>A schema is plain data and never changes: an edit ({@link #withPrimaryKey}, {@link
 * #withTimeColumn}, {@link #withStype}, {@link #withoutColumn}, {@link #withRelationships} and the
 * rest), checked and made by the core as Python's setters are, gives a new schema, or refuses as
 * Python's assignment refuses it. So a schema a request was gathered from stays as it was, which
 * Python's {@code snapshot()} is for.
 *
 * <pre>{@code
 * Schema schema = Schema.fromSource(DuckDbSource.of(connection))
 *     .withStype("users", "age", "categorical");
 * Request request = PredictiveQuery.parse(
 *         "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id")
 *     .materialize(schema, MaterializeOptions.builder().indices(List.of(1, 2)).build());
 * }</pre>
 */
public final class Schema {
  private final List<Table> tables;
  private final List<Relationship> relationships;
  private final Source source;
  private final List<String> conversionNotes;

  private Schema(List<Table> tables, List<Relationship> relationships, Source source) {
    this(tables, relationships, source, List.of());
  }

  private Schema(List<Table> tables, List<Relationship> relationships, Source source,
      List<String> conversionNotes) {
    this.tables = List.copyOf(tables);
    this.relationships = List.copyOf(relationships);
    this.source = source;
    this.conversionNotes = List.copyOf(conversionNotes);
  }

  /** Every table {@code source} holds, described on the core, keys checked. */
  public static Schema fromSource(Source source) {
    return fromSource(source, FromSourceOptions.defaults());
  }

  /**
   * The tables of {@code source}, described on the core as Python's {@code Schema.from_source}
   * describes them: each column's data type from the source's catalog and its meaning from a
   * preview, keys and relationships from what the catalog declares and else from names, each key
   * checked against every row unless {@code checkKeys(false)}, and what {@code options} declares
   * in place of what is inferred.
   *
   * @throws StructuredDataException the core's refusal ({@code INVALID_REQUEST} naming what it
   *     refuses), or {@code PROVIDER_ERROR} when the database failed a read
   */
  public static Schema fromSource(Source source, FromSourceOptions options) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(options, "options");
    Providers.Chosen chosen = Providers.of(source);
    Map<String, Object> operation = new LinkedHashMap<>();
    operation.put("op", "describe_source");
    operation.put("provider", chosen.choice());
    operation.put("source", 1);
    operation.put("check_keys", options.checkKeys());
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, Host.attachment(options.spec()));
    // The source, which only this host holds, as the core's sentence of a source with no table
    // names it.
    Map<String, Function<Map<String, Object>, Object>> held = Providers.held(source);
    try (Host.Finished finished =
        Host.run(operation, attachments, new Host.Call(chosen.provider(), null, held))) {
      Results.Described described =
          (Results.Described) Results.expect(finished.result(), "describe_source");
      return described(read(finished.bytes(described.description())), source)
          .withConversionNotes(described.notes());
    }
  }

  // --- what a schema holds ---------------------------------------------------------------

  /** Its tables, in order. */
  public List<Table> tables() {
    return tables;
  }

  /**
   * The table named {@code name}.
   *
   * @throws StructuredDataException {@code INVALID_REQUEST} when the schema has no such table
   */
  public Table table(String name) {
    for (Table table : tables) {
      if (table.name().equals(name)) {
        return table;
      }
    }
    throw new StructuredDataException(
        "the schema has no table " + name,
        "INVALID_REQUEST",
        Map.of("table", String.valueOf(name)),
        null);
  }

  /** The relationships between its tables, in order. */
  public List<Relationship> relationships() {
    return relationships;
  }

  /** The source its tables are in. */
  public Source source() {
    return source;
  }

  /** How a semantic model was converted, in the core's words, in source order. */
  public List<String> conversionNotes() {
    return conversionNotes;
  }

  /** Renders describe's notes after caller overrides, as Python's host does. */
  Schema withConversionNotes(List<Results.ViewNote> notes) {
    List<String> said = new ArrayList<>();
    for (Results.ViewNote note : notes) {
      if (note.table() != null) {
        Table named = tables.stream().filter(table -> table.name().equals(note.table()))
            .findFirst().orElse(null);
        if (named == null || (note.timeColumn() != null
            && !note.timeColumn().equals(named.timeColumn().orElse(null)))) {
          continue;
        }
      }
      said.add(Wording.said(Map.of("sentence", note.sentence(), "fields", note.fields())));
    }
    return new Schema(tables, relationships, source, said);
  }

  // --- edits ----------------------------------------------------------------------------

  /** This schema with {@code table}'s primary key set to {@code columns}, in order. */
  public Schema withPrimaryKey(String table, String... columns) {
    Map<String, Object> edit = new LinkedHashMap<>();
    edit.put("edit", "primary_key");
    edit.put("table", table);
    edit.put("key", FromSourceOptions.given(List.of(columns)));
    return edited(edit);
  }

  /** This schema with {@code table}'s primary key cleared. */
  public Schema withoutPrimaryKey(String table) {
    Map<String, Object> edit = new LinkedHashMap<>();
    edit.put("edit", "primary_key");
    edit.put("table", table);
    edit.put("key", FromSourceOptions.given(null));
    return edited(edit);
  }

  /** This schema with {@code table}'s time column set to {@code column}. */
  public Schema withTimeColumn(String table, String column) {
    Map<String, Object> edit = new LinkedHashMap<>();
    edit.put("edit", "time_column");
    edit.put("table", table);
    edit.put("column", FromSourceOptions.given(Objects.requireNonNull(column, "column")));
    return edited(edit);
  }

  /** This schema with {@code table}'s time column cleared. */
  public Schema withoutTimeColumn(String table) {
    Map<String, Object> edit = new LinkedHashMap<>();
    edit.put("edit", "time_column");
    edit.put("table", table);
    edit.put("column", FromSourceOptions.given(null));
    return edited(edit);
  }

  /** This schema with {@code table.column} meaning {@code stype}. */
  public Schema withStype(String table, String column, String stype) {
    Map<String, Object> edit = new LinkedHashMap<>();
    edit.put("edit", "stype");
    edit.put("table", table);
    edit.put("column", column);
    edit.put("stype", FromSourceOptions.given(Objects.requireNonNull(stype, "stype")));
    return edited(edit);
  }

  /** This schema with {@code table.column} left out. */
  public Schema withoutColumn(String table, String column) {
    Map<String, Object> edit = new LinkedHashMap<>();
    edit.put("edit", "leave_out");
    edit.put("table", table);
    edit.put("column", FromSourceOptions.given(Objects.requireNonNull(column, "column")));
    return edited(edit);
  }

  /** This schema with its relationships replaced by {@code relationships}, in order. */
  public Schema withRelationships(List<Relationship> relationships) {
    List<Object> links = new ArrayList<>();
    for (List<Object> link : links(relationships)) {
      links.add(FromSourceOptions.given(link));
    }
    Map<String, Object> edit = new LinkedHashMap<>();
    edit.put("edit", "relationships");
    edit.put("links", links);
    return edited(edit);
  }

  /** This schema with {@code relationship} added after its relationships. */
  public Schema withRelationship(Relationship relationship) {
    List<Relationship> more = new ArrayList<>(relationships);
    more.add(Objects.requireNonNull(relationship, "relationship"));
    return withRelationships(more);
  }

  /**
   * This schema as it stands, which a later edit cannot reach: the schema itself, since a schema
   * never changes; as Python's {@code snapshot()}.
   */
  public Schema snapshot() {
    return this;
  }

  /**
   * Refuses a column whose storage and meaning contradict, and a relationship that cannot join,
   * as Python's {@code check_consistent()} does: the core checks the schema as it stands, as it
   * checks an edit of its relationships, and changes nothing of it.
   *
   * @throws StructuredDataException {@code INVALID_REQUEST} naming what does not hold
   */
  public void checkConsistent() {
    withRelationships(relationships);
  }

  /** {@code edit} made by the core to this schema: the schema the core answered. */
  private Schema edited(Map<String, Object> edit) {
    Map<String, Object> operation = new LinkedHashMap<>();
    operation.put("op", "edit_schema");
    operation.put("schema", 1);
    operation.put("edit", 2);
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, Host.attachment(toHeld()));
    attachments.put(2, Host.attachment(edit));
    try (Host.Finished finished = Host.run(operation, attachments)) {
      Results.Edited edited = (Results.Edited) Results.expect(finished.result(), "edit_schema");
      Schema changed = held(read(finished.bytes(edited.schema())), source);
      return new Schema(changed.tables, changed.relationships, source, conversionNotes);
    }
  }

  // --- the schema as plain data -----------------------------------------------------------

  /**
   * This schema as the contract's {@code HeldSchema} (the structure design's fix A): each table by
   * its name, its source's relation, its columns' data types, meanings and what each meant before
   * the schema joined on it, with what the source said of each, its key, time column and repeated
   * keys, and the links. Nothing the core answered earlier is merged in.
   */
  Map<String, Object> toHeld() {
    List<Object> written = new ArrayList<>();
    for (Table table : tables) {
      List<Object> columns = new ArrayList<>();
      for (Column column : table.columns()) {
        Map<String, Object> each = new LinkedHashMap<>();
        each.put("name", column.name());
        each.put("dtype", column.dtype());
        each.put("stype", column.stype());
        each.put("stype_before_key", column.stypeBeforeKey());
        if (column.source() != null) {
          each.put("source", column.source());
        }
        columns.add(each);
      }
      Map<String, Object> held = new LinkedHashMap<>();
      held.put("name", table.name());
      held.put("source", Map.of("source", List.of(table.sourceName())));
      held.put("columns", columns);
      held.put("primary_key", table.primaryKey());
      held.put("time_column", table.timeColumn().orElse(null));
      if (table.duplicateKeysOrNull() != null) {
        Map<String, Object> count = new LinkedHashMap<>();
        count.put("t", "u64");
        count.put("v", Long.toUnsignedString(table.duplicateKeysOrNull()));
        held.put("duplicate_keys", count);
      }
      written.add(held);
    }
    Map<String, Object> schema = new LinkedHashMap<>();
    schema.put("tables", written);
    schema.put("links", links(relationships));
    return schema;
  }

  /** Relationships as the contract writes links, each {@code [foreign key, referenced key]}. */
  static List<List<Object>> links(List<Relationship> relationships) {
    List<List<Object>> links = new ArrayList<>();
    for (Relationship link : relationships) {
      Map<String, Object> foreign = new LinkedHashMap<>();
      foreign.put("table", link.sourceTable());
      foreign.put("column", link.sourceColumn());
      Map<String, Object> referenced = new LinkedHashMap<>();
      referenced.put("table", link.targetTable());
      referenced.put("column", link.targetColumn());
      links.add(List.of(foreign, referenced));
    }
    return links;
  }

  private static Object read(byte[] json) {
    try {
      return Json.read(json);
    } catch (Json.JsonException e) {
      throw new StructuredDataException(
          "the core answered JSON this host cannot read", "INTERNAL_ERROR", Map.of(), null);
    }
  }

  /**
   * The schema of a {@code SourceDescription}, as the core assembled it: each column's data type
   * as the client's wire names it, its meaning and what it meant before the schema joined on it,
   * and what the source said of it; each table's key, time column and repeated keys; the links.
   */
  @SuppressWarnings("unchecked")
  private static Schema described(Object read, Source source) {
    Map<String, Object> description = (Map<String, Object>) read;
    Map<String, Object> wireNames = (Map<String, Object>) NativeCore.load().get("wire_names");
    List<Table> tables = new ArrayList<>();
    for (Object item : (List<Object>) description.get("tables")) {
      Map<String, Object> table = (Map<String, Object>) item;
      Object relation = table.get("source");
      List<Column> columns = new ArrayList<>();
      for (Object c : (List<Object>) table.get("columns")) {
        Map<String, Object> column = (Map<String, Object>) c;
        Object dtype = wireNames.get(String.valueOf(column.get("dtype")));
        if (dtype == null) {
          throw new StructuredDataException(
              "the core described a column with no wire dtype", "INTERNAL_ERROR", Map.of(), null);
        }
        Map<String, Object> facts = new LinkedHashMap<>();
        facts.put("relation", relation);
        facts.put("dtype", column.get("dtype"));
        facts.put("received", column.get("received"));
        facts.put("nullable", column.get("nullable"));
        facts.put("unique", column.get("unique"));
        if (column.get("declared_type") != null) {
          facts.put("declared_type", column.get("declared_type"));
        }
        if (column.get("exact_number") != null) {
          facts.put("exact_number", column.get("exact_number"));
        }
        columns.add(
            new Column(
                (String) column.get("name"),
                (String) dtype,
                (String) column.get("stype"),
                (String) column.get("stype_before_key"),
                facts));
      }
      tables.add(table(table, columns));
    }
    return new Schema(tables, relationships(description.get("links")), source);
  }

  /** The schema of a {@code HeldSchema} the core answered an edit with. */
  @SuppressWarnings("unchecked")
  private static Schema held(Object read, Source source) {
    Map<String, Object> answer = (Map<String, Object>) read;
    List<Table> tables = new ArrayList<>();
    for (Object item : (List<Object>) answer.get("tables")) {
      Map<String, Object> table = (Map<String, Object>) item;
      List<Column> columns = new ArrayList<>();
      for (Object c : (List<Object>) table.get("columns")) {
        Map<String, Object> column = (Map<String, Object>) c;
        columns.add(
            new Column(
                (String) column.get("name"),
                (String) column.get("dtype"),
                (String) column.get("stype"),
                (String) column.get("stype_before_key"),
                (Map<String, Object>) column.get("source")));
      }
      tables.add(table(table, columns));
    }
    return new Schema(tables, relationships(answer.get("links")), source);
  }

  @SuppressWarnings("unchecked")
  private static Table table(Map<String, Object> table, List<Column> columns) {
    List<Object> parts = (List<Object>) ((Map<String, Object>) table.get("source")).get("source");
    if (parts == null || parts.size() != 1 || !(parts.get(0) instanceof String sourceName)) {
      throw new StructuredDataException(
          "the core described a relation of other than one part", "INTERNAL_ERROR", Map.of(), null);
    }
    List<String> key = new ArrayList<>();
    for (Object column : (List<Object>) table.get("primary_key")) {
      key.add((String) column);
    }
    Long duplicates = null;
    if (table.get("duplicate_keys") instanceof Map<?, ?> count) {
      duplicates = Long.parseUnsignedLong(String.valueOf(count.get("v")));
    }
    return new Table(
        (String) table.get("name"),
        sourceName,
        columns,
        key,
        (String) table.get("time_column"),
        duplicates);
  }

  @SuppressWarnings("unchecked")
  private static List<Relationship> relationships(Object links) {
    List<Relationship> read = new ArrayList<>();
    for (Object item : (List<Object>) links) {
      List<Object> link = (List<Object>) item;
      Map<String, Object> foreign = (Map<String, Object>) link.get(0);
      Map<String, Object> referenced = (Map<String, Object>) link.get(1);
      read.add(
          new Relationship(
              (String) foreign.get("table"),
              (String) foreign.get("column"),
              (String) referenced.get("table"),
              (String) referenced.get("column")));
    }
    return read;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof Schema schema
        && tables.equals(schema.tables)
        && relationships.equals(schema.relationships)
        && source == schema.source;
  }

  @Override
  public int hashCode() {
    return Objects.hash(tables, relationships, System.identityHashCode(source));
  }

  @Override
  public String toString() {
    return "Schema(" + tables + ", relationships=" + relationships + ")";
  }

  static String text(byte[] bytes) {
    return new String(bytes, StandardCharsets.UTF_8);
  }
}
