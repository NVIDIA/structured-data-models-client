// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import java.util.Map;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

class GradientExplanationsTest {
  @Test void publicGradientCollectionSurvivesNestedMergesAndSessions() throws Exception {
    try (RootAllocator allocator=new RootAllocator();
        Table context=TabularTest.table(allocator,true,1,2,3);
        Table predict=TabularTest.table(allocator,false,4,5);
        TabularTest.Endpoint endpoint=new TabularTest.Endpoint();
        SdmClient client=SdmClient.connect(endpoint.url())) {
      endpoint.gradients=true;
      Request request=new Request(context,predict,"y","regression");
      PredictionResult result=client.predict(request,SdmClient.KUMO_TABULAR,List.of("prediction","explanation"));
      assertEquals(2,result.explanations().gradients().entries().size());
      assertNotNull(result.explanation());
      PredictionResult merged=PredictionResult.merge(List.of(PredictionResult.merge(List.of(result,result)),result));
      assertEquals(6,merged.explanations().gradients().entries().size());
      for(long index=0;index<6;index++) assertNotNull(merged.explanations().gradients().get(new GradientExplanations.Key(GradientExplanations.Kind.ROW_INDEX,index,null)));
      try(VectorSchemaRoot root=merged.explanations().gradients().toArrow(allocator)) {
        assertEquals(6,root.getRowCount());
        assertInstanceOf(BigIntVector.class,root.getVector("identity"));
      }
      try(FittedModel fitted=client.fit(request,SdmClient.KUMO_TABULAR,List.of(),List.of("prediction","explanation"))) {
        assertEquals(2,fitted.predict(request).explanations().gradients().entries().size());
      }
    }
  }
  @Test void immutableViewsRetainWideKey() {
    GradientExplanations values=new GradientExplanations(Map.of("keyed_by","id","rows",List.of(Map.of(
        "id","9007199254740993","row_index",0,"answer_index",0,"forecast_step",0,
        "explanation","{\"outputs\":[],\"related\":{},\"hops\":{}}"))));
    assertEquals(9007199254740993L,((Number)values.entries().keySet().iterator().next().value()).longValue());
    assertThrows(UnsupportedOperationException.class,()->values.entries().clear());
  }
  @Test void realModelEndpointGradients() throws Exception {
    String url=System.getenv("SDM_GRADIENT_URL");
    org.junit.jupiter.api.Assumptions.assumeTrue(url!=null,"set SDM_GRADIENT_URL to a gradient-capable model runtime");
    for(String model:List.of(SdmClient.KUMO_TABULAR,SdmClient.KUMO_RELATIONAL)) {
      for(String target:List.of("value","group")) {
        Schema schema;
        try(RootAllocator allocator=new RootAllocator();Table users=RelationalTest.users(allocator)) {
          schema=model.equals(SdmClient.KUMO_RELATIONAL) ? RelationalTest.fixture(true,false)
              : Schema.fromData(Map.of("users",users)).withPrimaryKey("users","user_id").withStype("users","value","numerical");
        }
        MaterializeOptions.Builder options=MaterializeOptions.builder().model(model).indices(1L,2L).numContextSamples(12);
        if(model.equals(SdmClient.KUMO_RELATIONAL)) options.randomSeed(7).option("num_neighbors",List.of(2,2));
        Request request=PredictiveQuery.parse("EXPLAIN PREDICT users."+target+" FOR EACH users.user_id").materialize(schema,options.build());
        try(SdmClient client=SdmClient.connect(url)) {
          PredictionResult result=client.predict(request,model,List.of("prediction","explanation",target.equals("value")?"quantiles":"probabilities"));
          assertEquals(2,result.explanations().gradients().entries().size());
          ColumnExplanations columns=result.explanations().columns();
          assertNotNull(columns); assertFalse(columns.rows().isEmpty());
          for(ColumnExplanations.Row row:columns.rows()) {
            assertTrue(Double.isFinite(row.mean()));
            assertTrue(row.share()>=0 && row.share()<=1);
            assertTrue(row.score()>=0 && row.score()<=1);
            if(target.equals("value")) {
              assertNull(row.classLabel());
              assertTrue(row.mean()>=0 && row.mean()<=23.75);
            } else {
              assertTrue(List.of("a","b").contains(row.classLabel()));
              assertTrue(row.mean()>=0 && row.mean()<=1);
            }
          }
          try(RootAllocator allocator=new RootAllocator();VectorSchemaRoot root=columns.toArrow(allocator)) {
            assertEquals(columns.rows().size(),root.getRowCount());
          }
          try(RootAllocator allocator=new RootAllocator();VectorSchemaRoot root=result.explanations().gradients().toArrow(allocator)) {
            assertTrue(root.getRowCount()>0);
          }
        }
      }
    }
  }

}
