// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import com.nvidia.sdm.internal.ArrowData;
import com.nvidia.sdm.internal.Json;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.sql.Connection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

class ResultInspectionTest {
  @Test void endpointDiscoveryUsesTheServerAndEnforcesClosedState() throws Exception {
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    AtomicInteger ready = new AtomicInteger(200);
    server.createContext("/v1/models", exchange -> {
      byte[] bytes = "{\"data\":[{\"id\":\"endpoint-model\"},{},null]}".getBytes();
      exchange.sendResponseHeaders(200, bytes.length); exchange.getResponseBody().write(bytes); exchange.close();
    });
    server.createContext("/v1/health/ready", exchange -> {exchange.sendResponseHeaders(ready.get(), -1); exchange.close();});
    server.createContext("/api/2.0/serving-endpoints/model", exchange -> {
      byte[] bytes = ("{\"state\":{\"ready\":\""+(ready.get()==200?"READY":"NOT_READY")+"\"}}").getBytes();
      exchange.sendResponseHeaders(200, bytes.length); exchange.getResponseBody().write(bytes); exchange.close();
    });
    server.start();
    String url = "http://127.0.0.1:"+server.getAddress().getPort();
    SdmClient client = SdmClient.connect(url, ClientOptions.builder().maxRetries(0).build());
    try (SdmClient serving = SdmClient.forDatabricksServing(DatabricksServing.of(url,"model",()->"test-token"))) {
      assertEquals(List.of("endpoint-model"),client.models());
      assertTrue(client.healthReady()); assertTrue(serving.healthReady());
      ready.set(503); assertFalse(client.healthReady()); assertFalse(serving.healthReady());
      assertThrows(StructuredDataException.class,serving::models);
      client.close(); assertThrows(StructuredDataException.class,client::models);
      assertThrows(StructuredDataException.class,client::healthReady);
    } finally {client.close();server.stop(0);}
  }

  @Test void directResultsMergeWithoutAnotherPredictionAndRejectDifferentOutputs() throws Exception {
    PredictionResult first, second;
    try (RootAllocator allocator = new RootAllocator(); Table context = TabularTest.table(allocator,true,1,2);
        Table scoring = TabularTest.table(allocator,false,3,4); TabularTest.Endpoint endpoint = new TabularTest.Endpoint();
        SdmClient client = SdmClient.connect(endpoint.url())) {
      Request request = new Request(context,scoring,"y","regression");
      assertTrue(request.parts().isEmpty()); assertTrue(request.asked().isEmpty());
      first=client.predict(request,SdmClient.KUMO_TABULAR);
      second=client.predict(request,SdmClient.KUMO_TABULAR);
      assertSame(first,PredictionResult.merge(List.of(first)));
      PredictionResult merged=PredictionResult.merge(List.of(first,second));
      assertEquals(4,merged.rows()); assertEquals(2L,merged.metadata().get("batches"));
      assertEquals(List.of(first.requestId(),second.requestId()),merged.metadata().get("batch_request_ids"));
      assertEquals(2,endpoint.calls.size());
      assertThrows(UnsupportedOperationException.class,()->merged.metadata().put("x",1));
      PredictionResult incompatible = new PredictionResult(first.model(),first.requestId(),first.metadata(),List.of("embeddings"),
          bytes(first));
      assertThrows(StructuredDataException.class,()->PredictionResult.merge(List.of(first,incompatible)));
      PredictionResult otherModel = new PredictionResult(SdmClient.KUMO_RELATIONAL,first.requestId(),first.metadata(),first.outputs(),bytes(first));
      assertThrows(StructuredDataException.class,()->PredictionResult.merge(List.of(first,otherModel)));
    }
    assertThrows(StructuredDataException.class,()->PredictionResult.merge(List.of()));
    assertEquals(4,PredictionResult.merge(List.of(first,second)).rows());
  }

  private static byte[] bytes(PredictionResult result) throws Exception {
    try (ArrowReader reader=result.read(ArrowData.allocator());java.io.ByteArrayOutputStream out=new java.io.ByteArrayOutputStream()) {
      try (org.apache.arrow.vector.ipc.ArrowStreamWriter writer=new org.apache.arrow.vector.ipc.ArrowStreamWriter(reader.getVectorSchemaRoot(),null,out)) {
        writer.start();while(reader.loadNextBatch())writer.writeBatch();writer.end();
      }
      return out.toByteArray();
    }
  }

  @Test void rankingPartsRemainUsableAndUnscoredCandidatesSurviveMerge() throws Exception {
    Request request, otherRequest;
    try (Connection connection=WarehouseSessionTest.open("duckdb")) {
      Schema schema=ModelConnectorCoverageTest.linkSchema("duckdb",connection);
      request=PredictiveQuery.parse(RelationalTest.RANK).materialize(schema,
          WarehouseSessionTest.asked(1L,2L).candidates(List.of(1L,2L,999L)).build());
      otherRequest=PredictiveQuery.parse(RelationalTest.RANK).materialize(schema,
          WarehouseSessionTest.asked(1L,2L).candidates(List.of(2L,3L,999L)).build());
    }
    assertEquals(RelationalTest.RANK,request.asked().orElseThrow());
    List<Request> parts=request.parts().orElseThrow();assertEquals(3,parts.size());
    assertTrue(parts.get(0).parts().isEmpty());assertTrue(parts.get(0).asked().isEmpty());
    try (RelationalTest.Endpoint endpoint=new RelationalTest.Endpoint();SdmClient client=endpoint.client()) {
      assertEquals(2,client.predict(parts.get(0),SdmClient.KUMO_RELATIONAL).rows());
      PredictionResult result=client.predict(request,SdmClient.KUMO_RELATIONAL);
      assertEquals(List.of(999L),result.unscoredCandidates());
      PredictionResult other=client.predict(otherRequest,SdmClient.KUMO_RELATIONAL);
      PredictionResult merged=PredictionResult.merge(List.of(result,other));
      assertEquals(List.of(999L),merged.unscoredCandidates());assertEquals(4,merged.rows());
      try (ArrowReader reader=merged.read(ArrowData.allocator())) {
        assertTrue(reader.loadNextBatch());assertNotNull(reader.getVectorSchemaRoot().getVector("candidates"));
        assertNotNull(reader.getVectorSchemaRoot().getVector("rankings"));
        java.util.Set<List<Object>> ranked = new java.util.HashSet<>();
        for (int row=0; row<reader.getVectorSchemaRoot().getRowCount(); row++) {
          List<Object> values = new java.util.ArrayList<>();
          for (Object value : (List<?>) reader.getVectorSchemaRoot().getVector("rankings").getObject(row))
            values.add(Json.read(value.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8)));
          ranked.add(values);
        }
        assertEquals(java.util.Set.of(List.of(1L,2L),List.of(2L,3L)),ranked);
      }
    }
  }
}
