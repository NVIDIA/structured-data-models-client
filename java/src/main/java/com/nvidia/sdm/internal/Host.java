// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.ModelEndpointException;
import com.nvidia.sdm.StructuredDataException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The run loop: one core operation, from its {@code start} to its result, as the Python host's
 * {@code sdm_client._rust.run} drives it. It starts a run with this host's facts, asks it for
 * steps, answers each fetch through the call's provider and each model call through its model,
 * takes the run's warnings in the order given, and raises the run's error as {@link
 * StructuredDataException}, its code and fields the core's. Not part of the API.
 *
 * <p>A provider's failure reaches the core as the contract's {@code ProviderError}: its kind,
 * whether it may be retried, its SQLSTATE and vendor code and an internal driver diagnostic.
 * The diagnostic is never public (decision H17); the error the run then ends with names the driver's exception class ({@code
 * driver_class}). A model's {@link TransportFailed} is told to the core as the contract's
 * transport error; a {@link Refused} call is told as its status, and once the core ends the run,
 * the refusal is raised with what the core counted as answered before it.
 */
public final class Host {
  /** The databases the Java host has a provider for, and so the Java legs its cases run on. */
  public static final List<String> PROVIDERS =
      List.of("duckdb", "sqlite", "postgres", "databricks-sql");

  /** The largest control message, in bytes, in either direction. */
  public static final int CONTROL_CAP = 64 * 1024;

  /** Where the core's warnings are written, at {@code WARN}, as they are given. */
  private static final Logger WARNINGS = LoggerFactory.getLogger("com.nvidia.sdm.warnings");

  /** The core's fields that say what a failed request had answered (T2.8c). */
  private static final List<String> ANSWERED = List.of("calls_answered", "parts_answered");

  private Host() {}

  /**
   * Answers one fetch of the core's: given its request ({@code FetchStep}'s, with its {@code
   * plan}) and the key sets the plan names, by attachment id, the data, as a reader the host
   * exports to the core and closes.
   */
  @FunctionalInterface
  public interface Provider {
    ArrowReader answer(
        Map<String, Object> request, Map<Integer, ArrowReader> keys, BufferAllocator allocator)
        throws Exception;

    /**
     * The facts {@code failure} tells the core ({@code ProviderError}'s {@code sqlstate}, {@code
     * vendor_code}, {@code error_class} and internal {@code driver_message}), each where it says one: a driver's {@link
     * java.sql.SQLException}'s state and vendor code, by default.
     */
    default Map<String, Object> failure(Exception failure) {
      Map<String, Object> facts = new LinkedHashMap<>();
      if (failure instanceof java.sql.SQLException sql) {
        if (sql.getMessage() != null) {
          facts.put("driver_message", sql.getMessage());
        }
        if (sql.getSQLState() != null) {
          facts.put("sqlstate", sql.getSQLState());
        }
        if (sql.getErrorCode() != 0) {
          facts.put("vendor_code", Integer.toString(sql.getErrorCode()));
        }
      }
      return facts;
    }
  }

  /** Answers one model call: the core's body, as written, and the endpoint's answer's bytes. */
  @FunctionalInterface
  public interface Model {
    byte[] call(byte[] body) throws TransportFailed, Refused, EndpointFailed;
  }

  /**
   * The endpoint failed with no answer the core reads as one (T3.13): its HTTP status and the
   * body it sent, which the core reads only for codes and never repeats; the core names the
   * refusal. It carries no other text.
   */
  public static final class EndpointFailed extends Exception {
    private static final long serialVersionUID = 1L;
    private final int status;
    private final transient byte[] body;

    public EndpointFailed(int status, byte[] body) {
      super(null, null, false, false);
      this.status = status;
      this.body = body == null ? new byte[0] : body.clone();
    }

    public int status() {
      return status;
    }
  }

  /** The transport could not reach the endpoint or read its answer; it carries no text. */
  public static final class TransportFailed extends Exception {
    private static final long serialVersionUID = 1L;

    public TransportFailed() {
      super(null, null, false, false);
    }
  }

  /**
   * The call failed as the caller sees it: the endpoint's refusal as the transport reports it, or
   * the transport's own failure, raised by {@code raise} given what the core counted as answered
   * before it ({@code calls_answered}, {@code parts_answered}) to carry in its details; {@code
   * status} the endpoint's, or null.
   */
  public static final class Refused extends Exception {
    private static final long serialVersionUID = 1L;
    private final transient Function<Map<String, Object>, StructuredDataException> raise;
    private final Integer status;

    public Refused(
        Function<Map<String, Object>, StructuredDataException> raise, Integer status) {
      super(null, null, false, false);
      this.raise = raise;
      this.status = status;
    }

    /** The refusal, carrying {@code answered}. */
    public StructuredDataException raised(Map<String, Object> answered) {
      return raise.apply(answered);
    }

    public Integer status() {
      return status;
    }
  }

  /**
   * What one run is given beside its operation: its provider and model, or null for an operation
   * that reads no data and calls no model, and the values only this host holds that a sentence
   * may name ({@code {host.<name>}}), each read from the error's or warning's fields.
   */
  public record Call(
      Provider provider, Model model, Map<String, Function<Map<String, Object>, Object>> held) {
    public Call {
      held = held == null ? Map.of() : Map.copyOf(held);
    }

    /** A run with nothing beside its operation. */
    public static Call none() {
      return new Call(null, null, Map.of());
    }
  }

  /** A finished run: its result, the warnings it gave in order, and the run, to take from. */
  public static final class Finished implements AutoCloseable {
    private final Map<String, Object> result;
    private final List<Map<String, Object>> warnings;
    private final Run run;

    Finished(Map<String, Object> result, List<Map<String, Object>> warnings, Run run) {
      this.result = result;
      this.warnings = warnings;
      this.run = run;
    }

    public Map<String, Object> result() {
      return result;
    }

    public List<Map<String, Object>> warnings() {
      return warnings;
    }

    /** Takes the bytes attachment {@code id} the result names. */
    public byte[] bytes(int id) {
      Object taken = take(id);
      if (taken instanceof byte[] bytes) {
        return bytes;
      }
      if (taken instanceof NativeArrow arrow) {
        arrow.close();
      }
      throw internal("the core's result names a bytes attachment it did not give");
    }

    /** Takes the Arrow attachment {@code id} the result names, as an IPC stream's bytes. */
    public byte[] arrow(int id) {
      Object taken = take(id);
      if (!(taken instanceof NativeArrow arrow)) {
        throw internal("the core's result names an Arrow attachment it did not give");
      }
      try (arrow) {
        byte[] ipc = ArrowData.ipc(arrow.reader(ArrowData.allocator()));
        Trace trace = Trace.installed();
        if (trace != null) {
          trace.taken(id, ipc, true);
        }
        return ipc;
      }
    }

    private Object take(int id) {
      Object taken;
      try {
        taken = run.takeAttachment(id);
      } catch (NativeError e) {
        throw error(e, Map.of());
      }
      Trace trace = Trace.installed();
      if (trace != null && taken instanceof byte[] bytes) {
        trace.taken(id, bytes, false);
      }
      return taken;
    }

    @Override
    public void close() {
      run.close();
    }
  }

  /** The contract version of the library loaded: Java reads it from the crate, never its own. */
  public static int contract() {
    return ((Number) NativeCore.load().get("contract")).intValue();
  }

  /**
   * An input-sized field as a bytes attachment of its JSON; what JSON cannot hold (text with an
   * unpaired surrogate, a number that is not finite) is refused as the Python host's transport
   * refuses it, {@code INVALID_REQUEST}, in its words.
   */
  public static byte[] attachment(Object value) {
    try {
      return Json.write(value);
    } catch (Json.JsonException e) {
      throw new StructuredDataException(
          "this request carries a value that cannot be sent as JSON",
          "INVALID_REQUEST",
          Map.of(),
          null);
    }
  }

  /** Runs {@code operation}, which reads no data and calls no model, to its result. */
  public static Finished run(Map<String, Object> operation, Map<Integer, ?> attachments) {
    return run(operation, attachments, Call.none());
  }

  /**
   * Runs {@code operation} to its result, with the attachments its fields name, each a {@code
   * byte[]} or an {@link ArrowReader} this exports and closes.
   */
  public static Finished run(
      Map<String, Object> operation, Map<Integer, ?> attachments, Call call) {
    Map<String, Object> start = new LinkedHashMap<>();
    start.put("message", "start");
    start.put("contract", contract());
    start.put("facts", HostFacts.now());
    start.put("operation", operation);
    byte[] message = control(start);
    NativeCore.setLogLevel(NativeLog.level());
    Run run;
    List<ArrowArrayStream> exported = new ArrayList<>();
    try {
      Map<Integer, byte[]> told = new LinkedHashMap<>();
      Map<Integer, Object> sent = exportAll(attachments, exported, told);
      Trace trace = Trace.installed();
      if (trace != null) {
        trace.started(message, traced(attachments, told));
      }
      run = Run.start(message, sent);
    } catch (NativeError refused) {
      told(refused);
      throw error(refused, call.held());
    } finally {
      exported.forEach(ArrowArrayStream::close);
    }
    try {
      return steps(run, call);
    } catch (RuntimeException e) {
      run.close();
      throw e;
    }
  }

  /** Each attachment as the binding takes it: a reader exported to a stream it holds. */
  private static Map<Integer, Object> exportAll(
      Map<Integer, ?> attachments, List<ArrowArrayStream> exported, Map<Integer, byte[]> told) {
    Map<Integer, Object> sent = new LinkedHashMap<>();
    boolean traced = Trace.installed() != null;
    for (Map.Entry<Integer, ?> each : attachments.entrySet()) {
      Object value = each.getValue();
      if (value instanceof ArrowReader reader) {
        BufferAllocator allocator = ArrowData.allocator();
        if (traced) {
          // Read whole once, so the listener is told what is sent.
          byte[] ipc = ArrowData.ipc(reader);
          told.put(each.getKey(), ipc);
          reader = ArrowData.reader(ipc, allocator);
        }
        ArrowArrayStream stream = ArrowExchange.export(allocator, reader);
        exported.add(stream);
        sent.put(each.getKey(), stream);
      } else {
        sent.put(each.getKey(), value);
      }
    }
    return sent;
  }

  /** The attachments as the listener is told them: an Arrow one as its IPC bytes. */
  private static Map<Integer, Object> traced(Map<Integer, ?> given, Map<Integer, byte[]> ipcs) {
    Map<Integer, Object> told = new LinkedHashMap<>();
    for (Map.Entry<Integer, ?> each : given.entrySet()) {
      Object value = each.getValue();
      told.put(
          each.getKey(),
          value instanceof ArrowReader ? new ArrowIpc(ipcs.get(each.getKey())) : value);
    }
    return told;
  }

  /** An Arrow attachment as the listener is told it: an IPC stream's bytes. */
  public record ArrowIpc(byte[] bytes) {}

  /** The run's steps, each answered, to its result. */
  @SuppressWarnings("unchecked")
  private static Finished steps(Run run, Call call) {
    List<Map<String, Object>> warnings = new ArrayList<>();
    String driverClass = null;
    Refused refused = null;
    Trace trace = Trace.installed();
    while (true) {
      NativeMessage step;
      try {
        step = run.next();
      } catch (NativeError failed) {
        warn(run, warnings, call);
        if (trace != null) {
          trace.failed(failed.kind(), failed.code());
        }
        Map<String, Object> details = new LinkedHashMap<>();
        if (driverClass != null && "PROVIDER_ERROR".equals(failed.code())) {
          details.put("driver_class", driverClass);
        }
        StructuredDataException error = error(failed, call.held(), details);
        if (refused != null) {
          throw carried(refused, error);
        }
        throw error;
      }
      Map<String, Object> read;
      try (step) {
        warn(run, warnings, call);
        read = Messages.read(step);
      }
      if (!"step".equals(read.get("message"))) {
        throw internal("the core gave a message that is not a step");
      }
      if (trace != null) {
        trace.stepped(read);
      }
      Object kind = read.get("step");
      if ("done".equals(kind)) {
        Map<String, Object> result = (Map<String, Object>) read.get("result");
        if (trace != null) {
          trace.done(result);
        }
        return new Finished(result, Collections.unmodifiableList(warnings), run);
      }
      if ("call_model".equals(kind) && call.model() != null) {
        refused = model(run, read, call.model());
        continue;
      }
      if (!"fetch".equals(kind) || call.provider() == null) {
        cancel(run);
        throw internal("the core asked for a " + kind + " step this call cannot answer");
      }
      // A provider's failure is named by its driver's class only for the step that failed: the
      // core may carry on past one, and a later error is no other step's.
      driverClass = null;
      for (Object item : (List<Object>) read.get("requests")) {
        Map<String, Object> request = (Map<String, Object>) item;
        String failedClass = fetch(run, request, call.provider());
        if (failedClass != null) {
          driverClass = failedClass;
        }
      }
    }
  }

  /** Answers one fetch request; the class of the driver's exception when it failed, else null. */
  @SuppressWarnings("unchecked")
  private static String fetch(Run run, Map<String, Object> request, Provider provider) {
    Map<String, Object> outcome = new LinkedHashMap<>();
    Map<Integer, Object> attached = new LinkedHashMap<>();
    String driverClass = null;
    BufferAllocator allocator = ArrowData.allocator();
    Map<Integer, ArrowReader> keys = new LinkedHashMap<>();
    ArrowArrayStream stream = null;
    try {
      // A later page's step names the key sets the first took.
      if (request.get("continuation") == null) {
        keys = keySets(run, (Map<String, Object>) request.get("plan"), allocator);
      }
      ArrowReader answer = provider.answer(request, keys, allocator);
      stream = ArrowExchange.export(allocator, answer);
      outcome.put("outcome", "data");
      outcome.put("data", 1);
      attached.put(1, stream);
    } catch (StructuredDataException e) {
      // The host's own refusal (the flag Arrow needs), never a provider's failure.
      cancel(run);
      throw e;
    } catch (Exception failure) {
      Map<String, Object> provided = new LinkedHashMap<>();
      provided.put("kind", "data_source");
      provided.put("retryable", failure instanceof java.sql.SQLTransientException);
      // Internal facts go to the core; its public provider error sanitizes driver text.
      Map<String, Object> facts = provider.failure(failure);
      provided.put("sqlstate", facts.get("sqlstate"));
      provided.put("vendor_code", facts.get("vendor_code"));
      if (facts.get("error_class") != null) {
        provided.put("error_class", facts.get("error_class"));
      }
      if (facts.get("driver_message") != null) {
        provided.put("driver_message", facts.get("driver_message"));
      }
      outcome.put("outcome", "error");
      outcome.put("error", provided);
      driverClass = failure.getClass().getName();
    } finally {
      for (ArrowReader reader : keys.values()) {
        try {
          reader.close();
        } catch (Exception ignored) {
          // Released either way.
        }
      }
    }
    Map<String, Object> reply = new LinkedHashMap<>();
    reply.put("message", "fetch_reply");
    reply.put("id", request.get("id"));
    reply.put("outcome", outcome);
    try {
      run.reply(control(reply), attached);
    } catch (NativeError e) {
      told(e);
      throw error(e, Map.of(), driverClass == null ? Map.of() : Map.of("driver_class", driverClass));
    } finally {
      if (stream != null) {
        stream.close();
      }
    }
    return driverClass;
  }

  /**
   * The key sets a request's plan names, taken from the run by attachment id (decision Q1): each
   * a key set's Arrow, or the one text its dialect wrote of it (T3.28).
   */
  @SuppressWarnings("unchecked")
  private static Map<Integer, ArrowReader> keySets(
      Run run, Map<String, Object> plan, BufferAllocator allocator) throws NativeError {
    Map<Integer, ArrowReader> taken = new LinkedHashMap<>();
    Object statements = plan == null ? null : plan.get("statements");
    if (!(statements instanceof List<?> list)) {
      return taken;
    }
    for (Object statement : list) {
      for (Object param : (List<Object>) ((Map<String, Object>) statement).get("params")) {
        Map<String, Object> read = (Map<String, Object>) param;
        Object named;
        if ("key_set".equals(read.get("param"))) {
          named = ((Map<String, Object>) read.get("keys")).get("attachment");
        } else if ("text".equals(read.get("param"))) {
          named = read.get("attachment");
        } else {
          continue;
        }
        int id = ((Number) named).intValue();
        if (taken.containsKey(id)) {
          continue;
        }
        Object value = run.takeAttachment(id);
        if (value instanceof NativeArrow arrow) {
          try (arrow) {
            byte[] ipc = ArrowData.ipc(arrow.reader(allocator));
            taken.put(id, ArrowData.reader(ipc, allocator));
          }
        }
      }
    }
    return taken;
  }

  /**
   * Answers one model call: its answer as a 200, a failed transport as the contract's transport
   * error, a refused call as its status (or a transport error where it has none), returned for
   * the run's end to raise.
   */
  private static Refused model(Run run, Map<String, Object> step, Model model) {
    byte[] body;
    try {
      Object taken = run.takeAttachment(((Number) step.get("body")).intValue());
      if (!(taken instanceof byte[] bytes)) {
        throw internal("the core's model call carries no body");
      }
      body = bytes;
    } catch (NativeError e) {
      throw error(e, Map.of());
    }
    Trace trace = Trace.installed();
    if (trace != null) {
      trace.taken(((Number) step.get("body")).intValue(), body, false);
    }
    Map<String, Object> transportError = new LinkedHashMap<>();
    transportError.put("outcome", "transport_error");
    transportError.put("kind", "model_endpoint");
    // The transport has already retried what it retries.
    transportError.put("retryable", false);
    Map<String, Object> outcome;
    Map<Integer, Object> attached = new LinkedHashMap<>();
    Refused refused = null;
    try {
      byte[] answer = model.call(body);
      attached.put(1, answer);
      outcome = response(200);
    } catch (TransportFailed failed) {
      outcome = transportError;
    } catch (EndpointFailed failed) {
      // The endpoint's failure, its status and its body, for the core to name (T3.13).
      Map<String, Object> answer = new LinkedHashMap<>();
      answer.put("status", failed.status());
      if (failed.body.length > 0) {
        attached.put(1, failed.body);
        answer.put("detail", 1);
      }
      outcome = new LinkedHashMap<>();
      outcome.put("outcome", "failed");
      outcome.put("answer", answer);
    } catch (Refused failed) {
      refused = failed;
      Integer status = failed.status();
      if (status != null && status >= 300 && status <= 599) {
        attached.put(1, "{}".getBytes(java.nio.charset.StandardCharsets.UTF_8));
        outcome = response(status);
      } else {
        outcome = transportError;
      }
    } catch (RuntimeException e) {
      cancel(run);
      throw e;
    }
    Map<String, Object> reply = new LinkedHashMap<>();
    reply.put("message", "model_reply");
    reply.put("id", step.get("id"));
    reply.put("outcome", outcome);
    try {
      run.reply(control(reply), attached);
    } catch (NativeError e) {
      told(e);
      throw error(e, Map.of());
    }
    return refused;
  }

  /** The listener told of the core's error, when there is one. */
  private static void told(NativeError failed) {
    Trace trace = Trace.installed();
    if (trace != null) {
      trace.failed(failed.kind(), failed.code());
    }
  }

  private static Map<String, Object> response(int status) {
    Map<String, Object> outcome = new LinkedHashMap<>();
    outcome.put("outcome", "response");
    outcome.put("status", status);
    outcome.put("content_type", "application/json");
    outcome.put("body", 1);
    return outcome;
  }

  /** The refusal the caller sees, carrying what the core counted as answered before it. */
  private static StructuredDataException carried(Refused refused, StructuredDataException ended) {
    Map<String, Object> answered = new LinkedHashMap<>();
    for (String name : ANSWERED) {
      if (ended.details().containsKey(name)) {
        answered.put(name, ended.details().get(name));
      }
    }
    return refused.raised(answered);
  }

  private static void cancel(Run run) {
    try {
      run.cancel();
    } catch (NativeError ignored) {
      // The run ends here either way.
    }
  }

  /** A control message the host sends, within the control cap. */
  static byte[] control(Map<String, Object> message) {
    byte[] bytes;
    try {
      bytes = Json.write(message);
    } catch (Json.JsonException e) {
      throw internal("a control message cannot be written: " + e.getMessage());
    }
    if (bytes.length > CONTROL_CAP) {
      throw internal("a control message of " + bytes.length + " bytes is past the control cap");
    }
    return bytes;
  }

  /**
   * Takes the run's warnings, each with its words ({@link Wording}) as its {@code message}, writes
   * them to SLF4J, a group given one after another as one, and keeps each, in order.
   */
  @SuppressWarnings("unchecked")
  private static void warn(Run run, List<Map<String, Object>> into, Call call) {
    NativeMessage taken;
    try {
      taken = run.takeWarnings();
    } catch (NativeError e) {
      return;
    }
    if (taken == null) {
      return;
    }
    List<Map<String, Object>> given = new ArrayList<>();
    try (taken) {
      Object warnings = Messages.read(taken).get("warnings");
      if (warnings instanceof List<?> list) {
        for (Object item : list) {
          given.add((Map<String, Object>) item);
        }
      }
    }
    Trace trace = Trace.installed();
    if (trace != null) {
      trace.warned(given);
    }
    for (Wording.Said said : Wording.warnings(given, call.held())) {
      for (Map<String, Object> warning : said.warnings()) {
        Map<String, Object> kept = new LinkedHashMap<>(warning);
        kept.put("message", Wording.said(warning, call.held()));
        into.add(kept);
      }
      WARNINGS.warn("[{}] {}", said.code(), said.text());
    }
  }

  /**
   * The {@link StructuredDataException} a public call raises for the core's error: the same code,
   * fields, kind and retryability, its sentence filled with the values this call holds, holding
   * nothing native, the error's message closed.
   */
  public static StructuredDataException error(
      NativeError raised, Map<String, Function<Map<String, Object>, Object>> held) {
    return error(raised, held, Map.of());
  }

  private static StructuredDataException error(
      NativeError raised,
      Map<String, Function<Map<String, Object>, Object>> held,
      Map<String, Object> more) {
    Map<String, Object> given = Messages.fields(raised.message());
    raised.message().close();
    String said = held.isEmpty() ? raised.getMessage() : Wording.said(given, held);
    Map<String, Object> details = new LinkedHashMap<>(raised.details());
    details.putAll(more);
    if (ENDPOINT_REFUSED.equals(raised.code())) {
      // As Python raises NimRequestError(status, code=...): the endpoint's status and the code
      // the refusal is raised under, none where it has none, from the core's fields
      // (REFUSED_STATUS, REFUSED_CODE), which leave the details (T3.27).
      Object status = details.remove(REFUSED_STATUS);
      Object code = details.remove(REFUSED_CODE);
      return ModelEndpointException.refused(
          status instanceof Number number ? number.intValue() : 0,
          said,
          code == null ? null : String.valueOf(code),
          details,
          raised.kind(),
          raised.retryable());
    }
    return new StructuredDataException(
        said, raised.code(), details, raised.kind(), raised.retryable(), null);
  }

  /** The code of an endpoint's refusal, which Java raises as {@link ModelEndpointException}. */
  static final String ENDPOINT_REFUSED = "ENDPOINT_REFUSED";

  /** The field of an endpoint's refusal holding its HTTP status ({@code REFUSED_STATUS}). */
  static final String REFUSED_STATUS = "status";

  /** The field of an endpoint's refusal holding the code it is raised under ({@code REFUSED_CODE}). */
  static final String REFUSED_CODE = "refused_code";

  static StructuredDataException internal(String message) {
    return Messages.internal(message);
  }
}
