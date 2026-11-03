// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import com.nvidia.sdm.internal.Json;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;

class SettingsSummariesTest {
  @Test void summariesAreExplicitAndPreserveSuppliedProse() throws Exception {
    var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    List<Map<?,?>> bodies = new CopyOnWriteArrayList<>();
    List<String> auth = new CopyOnWriteArrayList<>();
    server.createContext("/v1/chat/completions", exchange -> {
      try { bodies.add((Map<?,?>)Json.read(exchange.getRequestBody().readAllBytes())); }
      catch (Json.JsonException e) { throw new java.io.IOException(e); }
      auth.add(Objects.toString(exchange.getRequestHeaders().getFirst("Authorization"), ""));
      byte[] response = "{\"choices\":[{\"message\":{\"content\":\" A summary. \"}}]}".getBytes(StandardCharsets.UTF_8);
      exchange.sendResponseHeaders(200,response.length); exchange.getResponseBody().write(response); exchange.close();
    });
    server.start();
    try {
      String url = "http://127.0.0.1:"+server.getAddress().getPort()+"/v1/chat/completions";
      var client = new ExplanationSummarizer(url,"synthetic-model","explicit-key",Duration.ofSeconds(2));
      var view = new ExplanationView(Map.of("details",Map.of("cohorts",List.of(Map.of("count",3)),"subgraphs",List.of())));
      assertTrue(bodies.isEmpty()); assertNull(view.summary());
      assertEquals("Original",client.summarize(new ExplanationView(Map.of("details",Map.of("summary","Original"))),"",null));
      assertTrue(bodies.isEmpty());
      assertEquals("A summary.",client.summarize(view,"PREDICT x",Map.of("prediction",2)));
      assertNull(view.summary()); assertEquals("Bearer explicit-key",auth.get(0));
      assertEquals("synthetic-model",bodies.get(0).get("model"));
      List<?> messages=(List<?>)bodies.get(0).get("messages");
      assertTrue(((Map<?,?>)messages.get(0)).get("content").toString().contains("do not claim causality"));
      Map<?,?> input=(Map<?,?>)Json.read(((Map<?,?>)messages.get(1)).get("content").toString().getBytes(StandardCharsets.UTF_8));
      assertEquals("PREDICT x", input.get("query"));
      assertEquals(view.raw().keySet(), ((Map<?,?>)input.get("explanation")).keySet());
      new ExplanationSummarizer(url,"synthetic-model",null,Duration.ofSeconds(2)).summarize(view,"",null);
      assertEquals("",auth.get(1));
    } finally { server.stop(0); }
  }

  @Test void errorsAreStructuredAndDoNotRetryOrLeakSecrets() throws Exception {
    var server = HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);
    java.util.concurrent.atomic.AtomicInteger calls=new java.util.concurrent.atomic.AtomicInteger();
    String[] body={"not json"}; int[] status={200}; long[] delay={0};
    server.createContext("/",exchange->{
      calls.incrementAndGet(); exchange.getRequestBody().readAllBytes();
      try { Thread.sleep(delay[0]); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
      try { byte[] data=body[0].getBytes(StandardCharsets.UTF_8);exchange.sendResponseHeaders(status[0],data.length);exchange.getResponseBody().write(data); }
      catch(java.io.IOException ignored) { /* Timed-out caller closed the socket. */ }
      finally { exchange.close(); }
    }); server.start();
    try {
      String url="http://127.0.0.1:"+server.getAddress().getPort()+"/";
      var client=new ExplanationSummarizer(url,"model","secret",Duration.ofSeconds(2));
      for(String malformed:List.of("not json","null","{}","{\"choices\":[]}","{\"choices\":[{\"message\":{\"content\":12}}]}","{\"choices\":[{\"message\":{\"content\":\" \"}}]}")) {
        body[0]=malformed;
        assertEquals("TRANSPORT_ERROR",assertThrows(StructuredDataException.class,()->client.summarize(new ExplanationView(Map.of()),"",null)).code());
      }
      status[0]=500;
      var error=assertThrows(StructuredDataException.class,()->client.summarize(new ExplanationView(Map.of()),"",null));
      assertFalse(error.toString().contains("secret"));assertEquals("http_status",error.details().get("reason"));
      status[0]=302;
      assertThrows(StructuredDataException.class,()->client.summarize(new ExplanationView(Map.of()),"",null));
      status[0]=200;delay[0]=600;
      var fast=new ExplanationSummarizer(url,"model",null,Duration.ofMillis(200));
      assertThrows(StructuredDataException.class,()->fast.summarize(new ExplanationView(Map.of()),"",null));
      assertEquals(9,calls.get());
    } finally { server.stop(0); }
  }

  @Test void configurationIsValidatedAndOptionsAreImmutable() {
    assertThrows(StructuredDataException.class,()->new ExplanationSummarizer("ftp://example.com","model",null,Duration.ofSeconds(1)));
    assertThrows(StructuredDataException.class,()->new ExplanationSummarizer("http://user:secret@example.com","model",null,Duration.ofSeconds(1)));
    assertThrows(StructuredDataException.class,()->new ExplanationSummarizer("http://example.com","model",null,Duration.ZERO));
    assertThrows(StructuredDataException.class,()->new ExplanationSummarizer("http://example.com:99999","model"));
    assertThrows(StructuredDataException.class,()->new ExplanationSummarizer("http://example.com","model","bad\nkey"));
    var cyclic=new ArrayList<Object>();cyclic.add(cyclic);
    var local=new ExplanationSummarizer("http://127.0.0.1:1/","model");
    var bad=assertThrows(StructuredDataException.class,()->local.summarize(new ExplanationView(Map.of()),"",cyclic));
    assertEquals("INVALID_REQUEST",bad.code());
    assertTrue(InferenceConfig.builder().build().toMap().isEmpty());
    List<String> transforms=new ArrayList<>(Arrays.asList("clip",null));
    var builder=InferenceConfig.builder().targetTransforms(transforms).outputType("mean");
    var config=builder.build(); transforms.clear();builder.outputType("median");
    assertEquals("mean",config.toMap().get("output_type"));
    assertEquals(Arrays.asList("clip",null),config.toMap().get("target_transforms"));
    assertThrows(UnsupportedOperationException.class,()->config.toMap().clear());
    assertEquals(config.toMap(),MaterializeOptions.builder().inferenceConfig(config).build().options().get("inference_config"));
  }

  @Test void typedClassificationUsesRustValidation() throws Exception {
    var fixture=CustomExamplesTest.fixture();
    var test=CustomExamplesTest.cases(fixture).stream().filter(c->CustomExamplesTest.object(c.get("task")).get("kind").equals("binary_classification")).findFirst().orElseThrow();
    var schema=CustomExamplesTest.memory(fixture,"timeless".equals(test.get("schema_variant")));
    var config=InferenceConfig.builder().classShuffle(true).build();
    try(var allocator=new RootAllocator();var context=CustomExamplesTest.arrow(allocator,CustomExamplesTest.object(test.get("context")),"binary_classification");var predict=CustomExamplesTest.arrow(allocator,CustomExamplesTest.object(test.get("predict")),"")) {
      var typed=schema.materializeExamples(CustomExamplesTest.task(test),context,predict,ExamplesOptions.builder().inference(config).build());
      var mapped=schema.materializeExamples(CustomExamplesTest.task(test),context,predict,ExamplesOptions.builder().inference(config.toMap()).build());
      assertEquals(CustomExamplesTest.payload(typed,0),CustomExamplesTest.payload(mapped,0));
    }
  }

  @Test void typedAndMappingPqlAreIdentical() throws Exception {
    var schema=CustomExamplesTest.memory(CustomExamplesTest.fixture(), true);
    var query=PredictiveQuery.parse("PREDICT accounts.feature FOR EACH accounts.id");
    var config=InferenceConfig.builder().numEstimators(2).outputType("mean").build();
    var typed=query.materialize(schema,MaterializeOptions.builder().model(SdmClient.KUMO_RELATIONAL).indices(List.of(1L)).inferenceConfig(config).build());
    var mapping=query.materialize(schema,MaterializeOptions.builder().model(SdmClient.KUMO_RELATIONAL).indices(List.of(1L)).option("inference_config",config.toMap()).build());
    assertEquals(CustomExamplesTest.payload(typed,0),CustomExamplesTest.payload(mapping,0));
  }

  @Test void typedAndMappingExamplesHaveIdenticalPayloadsAndRustErrors() throws Exception {
    var fixture=CustomExamplesTest.fixture();
    var test=CustomExamplesTest.cases(fixture).stream().filter(c->CustomExamplesTest.object(c.get("task")).get("kind").equals("regression")).findFirst().orElseThrow();
    Schema schema=CustomExamplesTest.memory(fixture,"timeless".equals(test.get("schema_variant")));
    var task=CustomExamplesTest.task(test);
    try(var allocator=new RootAllocator();var context=CustomExamplesTest.arrow(allocator,CustomExamplesTest.object(test.get("context")),"regression");
        var predict=CustomExamplesTest.arrow(allocator,CustomExamplesTest.object(test.get("predict")),"")) {
      for(var config:List.of(InferenceConfig.builder().build(),InferenceConfig.builder().numEstimators(2).columnShuffle(true).categoryShuffle(true).hopShuffle(true).targetTransforms(Arrays.asList("clip",null)).outputType("mean").build())) {
        Request typed=schema.materializeExamples(task,context,predict,ExamplesOptions.builder().inference(config).build());
        Request mapped=schema.materializeExamples(task,context,predict,ExamplesOptions.builder().inference(config.toMap()).build());
        assertEquals(CustomExamplesTest.payload(typed,0),CustomExamplesTest.payload(mapped,0));
      }
      for(var config:List.of(InferenceConfig.builder().numEstimators(0).build(),InferenceConfig.builder().classShuffle(true).build(),InferenceConfig.builder().outputType("invalid").build())) {
        var typed=assertThrows(StructuredDataException.class,()->schema.materializeExamples(task,context,predict,ExamplesOptions.builder().inference(config).build()));
        var mapped=assertThrows(StructuredDataException.class,()->schema.materializeExamples(task,context,predict,ExamplesOptions.builder().inference(config.toMap()).build()));
        assertEquals(typed.code(),mapped.code());assertEquals(typed.details(),mapped.details());
      }
    }
  }
}
