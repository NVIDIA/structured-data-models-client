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
  private Map<String, byte[]> data = Map.of();
  private Map<String, Object> inference;
  Map<String, byte[]> data() { return data; }

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

  /** Return an offline diagram of schema metadata, including all columns. */
  public String toMermaid() { return toMermaid(true); }

  /** Return an offline diagram; false shows only keys and time columns. */
  public String toMermaid(boolean showColumns) { return SchemaVisualization.mermaid(this, showColumns); }

  /** Return self-contained offline HTML including all columns. */
  public String toHtml() { return toHtml(true); }

  /** Return self-contained offline HTML with the bundled renderer. */
  public String toHtml(boolean showColumns) { return SchemaVisualization.html(toMermaid(showColumns)); }

  /** Export .mmd/.html offline; explicit .png/.svg sends metadata to https://mermaid.ink. */
  public void visualize(java.nio.file.Path path) throws java.io.IOException { visualize(path, true); }

  /** Export metadata only; no rows are read or transmitted. */
  public void visualize(java.nio.file.Path path, boolean showColumns) throws java.io.IOException {
    SchemaVisualization.export(this, path, showColumns);
  }

  /** Load a pinned RelBench dataset, using the default verified local cache. */
  public static Schema fromRelbench(String name) {
    return fromRelbench(name, RelBenchOptions.builder().build());
  }

  /** Load a pinned dataset with declared keys, times and links; offline mode never uses HTTP. */
  public static Schema fromRelbench(String name, RelBenchOptions options) {
    return RelBench.load(name, options);
  }

  /** Describe named in-memory Arrow tables, taking owned snapshots before returning. */
  public static Schema fromData(Map<String, org.apache.arrow.vector.table.Table> tables) {
    Objects.requireNonNull(tables, "tables");
    if (tables.isEmpty()) throw new StructuredDataException(
        "an in-memory schema needs at least one table", "INVALID_REQUEST", Map.of(), null);
    Map<String, byte[]> data = new LinkedHashMap<>();
    Map<Integer, byte[]> imports = new LinkedHashMap<>();
    List<Integer> ids = new ArrayList<>();
    for (Map.Entry<String, org.apache.arrow.vector.table.Table> table : tables.entrySet()) {
      if (table.getKey() == null || table.getKey().isEmpty()) throw new StructuredDataException(
          "table names must be non-empty strings", "INVALID_REQUEST", Map.of(), null);
      byte[] bytes = com.nvidia.sdm.internal.ArrowTabular.snapshot(table.getValue());
      data.put(table.getKey(), bytes);
      int id = imports.size() + 2;
      ids.add(id); imports.put(id, bytes);
    }
    Map<String, Object> operation = Map.of("op", "describe_frame_tables", "names", 1, "tables", ids);
    try (Host.Finished finished = com.nvidia.sdm.internal.ArrowFrames.run(operation,
        Map.of(1, Host.attachment(new ArrayList<>(data.keySet()))), imports, Host.Call.none())) {
      Schema schema = held(com.nvidia.sdm.internal.ArrowTabular.value(finished), null);
      schema.data = java.util.Collections.unmodifiableMap(data);
      return schema;
    }
  }

  /** Save this source-less schema and its owned frame snapshots; does not close/flush output. */
  public void writeTo(java.io.OutputStream output) throws java.io.IOException {
    Snapshots.Writer writer = new Snapshots.Writer();
    writer.write(Map.of("kind", "schema", "schema", saved(writer)), output);
  }

  /** Restore a data-only schema using the core's consistency checks; leaves input open. */
  public static Schema readFrom(java.io.InputStream input) throws java.io.IOException {
    Snapshots.Reader reader = Snapshots.read(input);
    if (!"schema".equals(reader.manifest.get("kind"))) throw Snapshots.invalid("snapshot is not a schema", null);
    Schema schema = restored(reader.manifest.get("schema"), reader);
    reader.complete();
    return schema;
  }

  Map<String, Object> saved(Snapshots.Writer writer) throws java.io.IOException {
    if (source != null) throw new StructuredDataException(
        "a schema with a live source cannot be saved", "UNSUPPORTED_REQUEST", Map.of(), null);
    List<Object> frames = new ArrayList<>();
    data.forEach((name, bytes) -> frames.add(Map.of("name", name, "blob", writer.blob(bytes))));
    return Map.of("held", toHeld(), "notes", conversionNotes, "frames", frames);
  }

  static Schema restored(Object value, Snapshots.Reader reader) throws java.io.IOException {
    Map<String, Object> saved = Snapshots.object(value);
    Map<String, byte[]> frames = new LinkedHashMap<>();
    for (Object item : Snapshots.list(saved.get("frames"))) {
      Map<String, Object> frame = Snapshots.object(item);
      String name = Snapshots.text(frame.get("name"));
      byte[] bytes = reader.blob(frame.get("blob"));
      if (frames.putIfAbsent(name, bytes) != null) throw Snapshots.invalid("duplicate frame name in snapshot", null);
      Snapshots.validateArrow(bytes);
    }
    Map<String, Object> operation = new LinkedHashMap<>();
    operation.put("op", "edit_schema"); operation.put("schema", 1); operation.put("edit", 2);
    Map<Integer, Object> attachments = new LinkedHashMap<>();
    attachments.put(1, Host.attachment(Snapshots.object(saved.get("held"))));
    attachments.put(2, Host.attachment(Map.of("edit", "check")));
    if (!frames.isEmpty()) {
      operation.put("context", 3);
      attachments.put(3, Host.attachment(Map.of("kind", "frames", "key_counts", Map.of())));
    }
    try (Host.Finished finished = Host.run(operation, attachments)) {
      Results.Edited edited = (Results.Edited) Results.expect(finished.result(), "edit_schema");
      Schema checked = held(read(finished.bytes(edited.schema())), null);
      if (!frames.isEmpty() && !frames.keySet().equals(checked.tables.stream().map(Table::name)
          .collect(java.util.stream.Collectors.toSet()))) {
        throw Snapshots.invalid("snapshot frames do not match its schema tables", null);
      }
      Schema result = new Schema(checked.tables, checked.relationships, null, Snapshots.texts(saved.get("notes")));
      result.inference = checked.inference;
      result.data = java.util.Collections.unmodifiableMap(frames);
      return result;
    } catch (ClassCastException | NullPointerException | IllegalArgumentException malformed) {
      throw Snapshots.invalid("invalid schema snapshot", malformed);
    }
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

  /** Snapshot existing descriptors without inference or source access. */
  public MetadataReport metadataReport() {
    return new MetadataReport(tables().stream().map(Table::metadataReport).toList());
  }

  /** Snapshot existing directed relationships. */
  public LinkReport linkReport() {
    return new LinkReport(relationships().stream().map(r -> new LinkMetadata(r.sourceTable(), r.sourceColumn(), r.targetTable(), r.targetColumn())).toList());
  }

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
    Schema result = new Schema(tables, relationships, source, said);
    result.inference = inference;
    result.data = data;
    return result;
  }

  /** Fill missing key/time choices without changing declared metadata or this schema. */
  public Schema inferMetadata() { return infer(false); }

  /** Infer missing links, preserving the links and key choices already held. */
  public Schema inferLinks() { return infer(true); }

  /** Return a schema without the named table and its incident links. */
  public Schema withoutTable(String name) {
    if (name == null) throw new StructuredDataException("table name is required", "INVALID_REQUEST", Map.of(), null);
    return edited(Map.of("edit", "remove_table", "table", name));
  }

  /** Remove an existing relationship atomically. */
  public Schema withoutRelationship(Relationship relationship) {
    if (relationship == null) throw new StructuredDataException(
        "relationship is required", "INVALID_REQUEST", Map.of(), null);
    return edited(Map.of("edit", "remove_link", "link", links(List.of(relationship)).get(0)));
  }

  /** Ordered immutable provenance; older saved records have unknown origins. */
  @SuppressWarnings("unchecked")
  public List<RelationshipOrigin> linkOrigins() {
    List<RelationshipOrigin> result = new ArrayList<>();
    for (Relationship relationship : relationships) {
      String origin = "unknown";
      if (inference != null) for (Object entry : (List<?>) inference.get("links")) {
        Map<String, Object> record = (Map<String, Object>) entry;
        if (relationships(List.of(record.get("link"))).get(0).equals(relationship)) origin = (String) record.get("origin");
      }
      result.add(new RelationshipOrigin(relationship, origin));
    }
    return List.copyOf(result);
  }

  private Schema infer(boolean links) {
    if (tables.isEmpty()) return this;
    Schema candidate;
    if (source != null) {
      FromSourceOptions.Builder options = FromSourceOptions.builder();
      for (Table table : tables) options.table(table.name(), table.sourceName());
      candidate = fromSource(source, options.build());
    } else {
      if (data.isEmpty()) throw new StructuredDataException("inference requires table data", "INVALID_REQUEST", Map.of(), null);
      Map<Integer, byte[]> imports = new LinkedHashMap<>();
      List<Integer> ids = new ArrayList<>();
      List<String> names = new ArrayList<>();
      for (Table table : tables) {
        int id = imports.size() + 2;
        ids.add(id); names.add(table.name()); imports.put(id, data.get(table.name()));
      }
      try (Host.Finished finished = com.nvidia.sdm.internal.ArrowFrames.run(
          Map.of("op", "describe_frame_tables", "names", 1, "tables", ids),
          Map.of(1, Host.attachment(names)), imports, Host.Call.none())) {
        candidate = held(com.nvidia.sdm.internal.ArrowTabular.value(finished), null);
      }
    }
    return edited(Map.of("edit", "infer", "candidate", candidate.toHeld(), "links", links));
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

  /** Declare the exclusive end of sampled entity validity. */
  public Schema withEndTimeColumn(String table, String column) {
    Map<String, Object> edit = new LinkedHashMap<>();
    edit.put("edit", "end_time_column");
    edit.put("table", table);
    edit.put("column", FromSourceOptions.given(Objects.requireNonNull(column, "column")));
    return edited(edit);
  }

  /** Clear the declared end-time role. */
  public Schema withoutEndTimeColumn(String table) {
    Map<String, Object> edit = new LinkedHashMap<>();
    edit.put("edit", "end_time_column");
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

  /** Gather a Relational request from caller-supplied ENTITY/ANCHOR_TIMESTAMP/TARGET examples. */
  public Request materializeExamples(RelationalTask task, org.apache.arrow.vector.table.Table context, org.apache.arrow.vector.table.Table predict) {
    return materializeExamples(task,context,predict,ExamplesOptions.builder().build());
  }

  /** Supplied examples with explicit sampling/inference options; Arrow inputs remain caller-owned. */
  public Request materializeExamples(RelationalTask task, org.apache.arrow.vector.table.Table context, org.apache.arrow.vector.table.Table predict, ExamplesOptions options) {
    return Examples.materialize(this,task,context,predict,options);
  }

  /** Revalidate these declarations on a replacement caller-owned source.
   * Backend, namespace, source kind and semantic settings must match.
   * Neither connection is closed; existing queries and requests remain unchanged.
   * This checks columns and types, not every row for key uniqueness.
   */
  public Schema withSource(Source replacement) {
    Objects.requireNonNull(replacement, "source");
    if (source == null) {
      throw new StructuredDataException("rebinding requires a source-backed schema", "UNSUPPORTED_REQUEST", Map.of(), null);
    }
    Providers.Chosen previous = Providers.of(source);
    Providers.Chosen chosen = Providers.of(replacement);
    Map<String, Object> operation = Map.of("op", "rebind_source", "provider", chosen.choice(),
        "previous_provider", previous.choice(), "source_kind", replacement.getClass().getSimpleName(),
        "previous_source_kind", source.getClass().getSimpleName(), "schema", 1);
    try (Host.Finished finished = Host.run(operation, Map.of(1, Host.attachment(toHeld())),
        new Host.Call(chosen.provider(), null, Providers.held(replacement)))) {
      Results.Described answer = (Results.Described) Results.expect(finished.result(), "describe_source");
      Schema result = described(read(finished.bytes(answer.description())), replacement);
      Schema noted = new Schema(result.tables, result.relationships, replacement, conversionNotes);
      noted.inference = result.inference;
      return noted;
    }
  }

  /** Return a schema with a new provider-dialect SQL column over physical source columns.
   * SQL is executed under the source credentials; no warehouse table is modified.
   */
  public Schema withSqlColumn(String table, String name, String expression) {
    if (table == null || name == null || expression == null) {
      throw new StructuredDataException("table, name and expression must be strings", "INVALID_REQUEST", Map.of(), null);
    }
    if (source == null) {
      throw new StructuredDataException("SQL columns require a source-backed schema", "UNSUPPORTED_REQUEST", Map.of(), null);
    }
    Providers.Chosen chosen = Providers.of(source);
    Map<String, Object> operation = Map.of("op", "add_sql_column", "provider", chosen.choice(), "schema", 1, "input", 2);
    Map<Integer, Object> attachments = Map.of(1, Host.attachment(toHeld()), 2, Host.attachment(Map.of("table", table, "name", name, "expression", expression)));
    try (Host.Finished finished = Host.run(operation, attachments, new Host.Call(chosen.provider(), null, Providers.held(source)))) {
      Results.Described answer = (Results.Described) Results.expect(finished.result(), "describe_source");
      Schema result = described(read(finished.bytes(answer.description())), source);
      Schema noted = new Schema(result.tables, result.relationships, source, conversionNotes);
      noted.inference = result.inference;
      return noted;
    }
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
    if (relationship == null) throw new StructuredDataException(
        "relationship is required", "INVALID_REQUEST", Map.of(), null);
    return edited(Map.of("edit", "add_link", "link", links(List.of(relationship)).get(0)));
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
    Map<Integer, byte[]> frames = new LinkedHashMap<>();
    if (!data.isEmpty()) {
      operation.put("context", 3);
      attachments.put(3, Host.attachment(Map.of("kind", "frames", "key_counts", Map.of())));
      if ("primary_key".equals(edit.get("edit")) && data.containsKey(edit.get("table"))) {
        operation.put("frame", 4);
        frames.put(4, data.get(edit.get("table")));
      }
    }
    try (Host.Finished finished = com.nvidia.sdm.internal.ArrowFrames.run(
        operation, attachments, frames, Host.Call.none())) {
      Results.Edited edited = (Results.Edited) Results.expect(finished.result(), "edit_schema");
      Schema changed = held(read(finished.bytes(edited.schema())), source);
      Schema result = new Schema(changed.tables, changed.relationships, source, conversionNotes);
      result.inference = changed.inference;
      Map<String, byte[]> kept = new LinkedHashMap<>();
      for (Table table : changed.tables) if (data.containsKey(table.name())) kept.put(table.name(), data.get(table.name()));
      result.data = java.util.Collections.unmodifiableMap(kept);
      return result;
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
      held.put("calculated_columns", table.calculatedColumns());
      held.put("primary_key", table.primaryKey());
      held.put("time_column", table.timeColumn().orElse(null));
      held.put("end_time_column", table.endTimeColumn().orElse(null));
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
    if (inference != null) schema.put("inference", inference);
    return schema;
  }

  @SuppressWarnings("unchecked")
  Map<String, Object> queryShape() {
    Map<String, String> dtypeNames = (Map<String, String>) NativeCore.load().get("dtype_names");
    List<Object> named = new ArrayList<>();
    for (Table table : tables) {
      Map<String, Object> value = new LinkedHashMap<>();
      value.put("name", table.name());
      value.put("columns", table.columns().stream().filter(c -> dtypeNames.containsKey(c.dtype())).map(c ->
          Map.of("name", c.name(), "dtype", dtypeNames.get(c.dtype()), "stype", c.stype())).toList());
      value.put("primary_key", table.primaryKey().size() == 1 ? table.primaryKey().get(0) : null);
      value.put("time_column", table.timeColumn().orElse(null));
      value.put("end_time_column", table.endTimeColumn().orElse(null));
      named.add(value);
    }
    return Map.of("tables", named, "links", links(relationships));
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
    Schema result = new Schema(tables, relationships(description.get("links")), source);
    result.inference = (Map<String, Object>) description.get("inference");
    return result;
  }

  /** The schema of a {@code HeldSchema} the core answered an edit with. */
  @SuppressWarnings("unchecked")
  static Schema held(Object read, Source source) {
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
    Schema result = new Schema(tables, relationships(answer.get("links")), source);
    result.inference = (Map<String, Object>) answer.get("inference");
    return result;
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
        (String) table.get("end_time_column"),
        duplicates,
        (List<Map<String, Object>>) table.getOrDefault("calculated_columns", List.of()));
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
