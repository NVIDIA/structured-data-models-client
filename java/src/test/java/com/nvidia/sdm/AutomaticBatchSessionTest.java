// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import static com.nvidia.sdm.CustomExamplesTest.*;
import com.nvidia.sdm.internal.Json;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.file.*;
import java.util.*;
import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Test;

class AutomaticBatchSessionTest {
  static Request request() throws Exception {
    Map<String,Object> fixture=object(Json.read(Files.readAllBytes(Path.of("../tests/fixtures/batch-sessions/scenarios.json"))));
    try(RootAllocator allocator=new RootAllocator();
        var users=arrow(allocator,object(object(fixture.get("tables")).get("users")),"");
        var context=arrow(allocator,object(fixture.get("context")),"regression");
        var predict=arrow(allocator,object(fixture.get("predict")),"regression")) {
      Schema schema=Schema.fromData(Map.of("users",users)).withPrimaryKey("users","id").withStype("users","value","numerical");
      return schema.materializeExamples(RelationalTask.regression("users"),context,predict,ExamplesOptions.builder().rowsPerCall(1).build());
    }
  }

  static final class Endpoint implements AutoCloseable {
    final HttpServer server;
    final String mode;
    volatile int creates,scores,deletes;
    final List<Map<String,Object>> scoredBodies=Collections.synchronizedList(new ArrayList<>());
    Endpoint(String mode) throws Exception {
      this.mode=mode;
      server=HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);
      server.createContext("/", exchange->{
        try {
          if(exchange.getRequestMethod().equals("DELETE")) { deletes++; exchange.sendResponseHeaders((mode.equals("malformed") || mode.equals("transport"))?400:204,-1); return; }
          var payload=object(Json.read(exchange.getRequestBody().readAllBytes()));
          int status=200; Object answer=Map.of();
          if(exchange.getRequestURI().getPath().equals("/v1/sessions")) {
            creates++;
            if(mode.equals("unsupported")) status=405;
            else if(mode.equals("uncertain")) status=503;
            else if(mode.equals("alias")) { var fields=new LinkedHashMap<String,Object>();fields.put("session_id",null);fields.put("id","s"+creates);answer=fields; }
            else answer=Map.of("session_id","s"+creates);
          } else {
            scores++; scoredBodies.add(payload);
            if(mode.equals("transport") && scores>=2) { exchange.close();return; }
            if(mode.equals("expired") && scores==2) status=404;
            else if(mode.equals("retry") && scores==1) status=408;
            else if(mode.equals("fail") && scores==2) status=422;
            else if(mode.equals("malformed") && scores==2) answer=Map.of("predictions","invalid");
            else {
              var table=object(object(payload.get("predict")).get("instance_table"));
              int id=((List<?>)table.get("columns")).indexOf("instance_id");
              List<Object> predictions=new ArrayList<>();
              for(var row:rows(table)) predictions.add(Map.of("id",row.get(id).toString(),"row_index",predictions.size(),"prediction",5.0));
              answer=Map.of("model","kumo-relational","id","answer","metadata",Map.of(),"predictions",predictions);
            }
          }
          byte[] bytes=Json.write(answer); exchange.sendResponseHeaders(status,bytes.length);exchange.getResponseBody().write(bytes);
        } catch (Exception error) { throw new java.io.IOException(error); } finally { exchange.close(); }
      }); server.start();
    }
    SdmClient client() { return SdmClient.connect("http://127.0.0.1:"+server.getAddress().getPort(), ClientOptions.builder().maxRetries(1).build()); }
    public void close() { server.stop(0); }
  }

  @Test void sharedLifecycleCases() throws Exception {
    var fixture=object(Json.read(Files.readAllBytes(Path.of("../tests/fixtures/batch-sessions/scenarios.json"))));
    for(Object value:(List<?>)fixture.get("cases")) {
      var scenario=object(value);
      try(Endpoint endpoint=new Endpoint(scenario.get("mode").toString()); SdmClient client=endpoint.client()) {
        var result=client.predict(request(),SdmClient.KUMO_RELATIONAL);
        assertEquals(3,result.rows());
        assertEquals(((Number)scenario.get("creates")).intValue(),endpoint.creates,scenario.toString());
        assertEquals(((Number)scenario.get("scores")).intValue(),endpoint.scores);
        assertEquals(((Number)scenario.get("deletes")).intValue(),endpoint.deletes);
      }
    }
  }
  @Test void malformedAnswerSurvivesFailedCleanup() throws Exception {
    try(Endpoint endpoint=new Endpoint("malformed"); SdmClient client=endpoint.client()) {
      var error=assertThrows(StructuredDataException.class,()->client.predict(request(),SdmClient.KUMO_RELATIONAL));
      assertEquals("INVALID_RESPONSE",error.code()); assertEquals(1,endpoint.deletes);
      assertNotNull(error.details().get("answered"));
    }
  }
  @Test void transportFailureSurvivesFailedCleanup() throws Exception {
    try(Endpoint endpoint=new Endpoint("transport"); SdmClient client=endpoint.client()) {
      var error=assertThrows(StructuredDataException.class,()->client.predict(request(),SdmClient.KUMO_RELATIONAL));
      assertEquals("TRANSPORT_ERROR",error.code()); assertEquals(1,endpoint.deletes);
      assertNotNull(error.details().get("answered"));
    }
  }
  @Test void scoringOptionsMatchStatelessFallback() throws Exception {
    Request request=request(); List<List<Map<String,Object>>> sent=new ArrayList<>();
    for(String mode:List.of("ok","unsupported")) {
      try(Endpoint endpoint=new Endpoint(mode); SdmClient client=endpoint.client()) {
        client.predict(request,SdmClient.KUMO_RELATIONAL);
        sent.add(endpoint.scoredBodies.stream().map(body->{
          Map<String,Object> fields=new LinkedHashMap<>();
          for(String key:List.of("predict","output","inference")) fields.put(key,body.get(key));
          return fields;
        }).toList());
      }
    }
    assertEquals(sent.get(0),sent.get(1));
  }
  @Test void differentContextsStayStateless() throws Exception {
    Schema schema=RelationalTest.fixture(false,false);
    Request request=PredictiveQuery.parse(RelationalTest.SCALAR).materialize(schema,
      RelationalTest.options(1L,2L,3L).rowsPerCall(1).numContextSamples(100).build());
    assertNotEquals(object(Json.read(request.payloads().get(0).getBytes())).get("context"),object(Json.read(request.payloads().get(1).getBytes())).get("context"));
    try(Endpoint endpoint=new Endpoint("ok"); SdmClient client=endpoint.client()) {
      assertEquals(3,client.predict(request,SdmClient.KUMO_RELATIONAL).rows());
      assertEquals(0,endpoint.creates);assertEquals(3,endpoint.scores);
    }
  }
  @Test @org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable(named="SDM_RELATIONAL_URL",matches=".+")
  void livePreparedBatchPrediction() throws Exception {
    try(SdmClient client=SdmClient.connect(System.getenv("SDM_RELATIONAL_URL"),
        ClientOptions.builder().timeout(java.time.Duration.ofSeconds(180)).build())) {
      assertEquals(3,client.predict(request(),SdmClient.KUMO_RELATIONAL).rows());
    }
  }
  @Test void uncertainCreationNeverRetries() throws Exception {
    try(Endpoint endpoint=new Endpoint("uncertain"); SdmClient client=endpoint.client()) {
      assertThrows(StructuredDataException.class,()->client.predict(request(),SdmClient.KUMO_RELATIONAL));
      assertEquals(1,endpoint.creates);assertEquals(0,endpoint.scores);
    }
  }
  @Test void failedBatchKeepsPaidAnswersAndCloses() throws Exception {
    try(Endpoint endpoint=new Endpoint("fail"); SdmClient client=endpoint.client()) {
      var error=assertThrows(ModelEndpointException.class,()->client.predict(request(),SdmClient.KUMO_RELATIONAL));
      assertEquals(422,error.status()); assertEquals(2,endpoint.scores);assertEquals(1,endpoint.deletes);
      assertNotNull(error.details().get("answered"));
    }
  }
}
