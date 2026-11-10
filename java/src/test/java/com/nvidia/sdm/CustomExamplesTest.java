// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;

import com.nvidia.sdm.internal.Json;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDate;
import java.util.*;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.*;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.table.Table;
import org.junit.jupiter.api.Test;

@SuppressWarnings("unchecked")
class CustomExamplesTest {
  static Map<String,Object> object(Object value) { return (Map<String,Object>) value; }
  static List<List<Object>> rows(Map<String,Object> table) { return (List<List<Object>>) table.get("rows"); }
  static Map<String,Object> fixture() throws Exception {
    return object(Json.read(Files.readAllBytes(Path.of("../tests/fixtures/custom-examples/examples.json"))));
  }
  static List<Map<String,Object>> cases(Map<String,Object> fixture) {
    return ((List<Object>) fixture.get("cases")).stream().map(CustomExamplesTest::object).toList();
  }
  static Table arrow(RootAllocator allocator, Map<String,Object> data, String kind) {
    List<String> names=(List<String>)data.get("columns");
    List<List<Object>> rows=rows(data);
    List<FieldVector> columns=new ArrayList<>();
    try {
      for(int col=0;col<names.size();col++) {
        String name=names.get(col);
        boolean target=name.equals("TARGET");
        FieldVector vector = name.equals("ANCHOR_TIMESTAMP") || List.of("opened","closed","time").contains(name)
            ? new TimeStampMilliVector(name,allocator)
            : target && kind.equals("temporal_link_prediction") ? ListVector.empty(name,allocator)
            : target && kind.equals("binary_classification") ? new BitVector(name,allocator)
            : target && kind.equals("multiclass_classification") ? new VarCharVector(name,allocator)
            : target || List.of("feature","price").contains(name) ? new Float8Vector(name,allocator)
            : new BigIntVector(name,allocator);
        columns.add(vector); vector.allocateNew();
        for(int row=0;row<rows.size();row++) {
          Object value=rows.get(row).get(col);
          if(vector instanceof TimeStampMilliVector time) {
            if(value==null) time.setNull(row);
            else time.setSafe(row,LocalDate.parse(value.toString()).toEpochDay()*86_400_000L);
          } else if(vector instanceof ListVector list) {
            var writer=list.getWriter(); writer.setPosition(row); writer.startList();
            for(Object item:(List<?>)value) writer.bigInt().writeBigInt(((Number)item).longValue());
            writer.endList();
          } else if(vector instanceof BitVector bits) bits.setSafe(row,Boolean.TRUE.equals(value)?1:0);
          else if(vector instanceof VarCharVector text) text.setSafe(row,value.toString().getBytes(StandardCharsets.UTF_8));
          else if(vector instanceof Float8Vector floats) floats.setSafe(row,((Number)value).doubleValue());
          else ((BigIntVector)vector).setSafe(row,((Number)value).longValue());
        }
        vector.setValueCount(rows.size());
      }
      return new Table(columns);
    } finally { columns.forEach(FieldVector::close); }
  }
  static Schema memory(Map<String,Object> fixture, boolean timeless) {
    try(RootAllocator allocator=new RootAllocator()) {
      Map<String,Table> tables=new LinkedHashMap<>();
      try {
        object(fixture.get(timeless?"timeless_tables":"tables")).forEach((name,value)->tables.put(name,arrow(allocator,object(value),"")));
        Schema schema=Schema.fromData(tables);
        for(String name:tables.keySet()) schema=schema.withPrimaryKey(name,"id");
        schema=schema.withStype("accounts","feature","numerical");
        if(!timeless) schema=schema.withTimeColumn("accounts","opened").withEndTimeColumn("accounts","closed")
            .withTimeColumn("events","time").withStype("items","price","numerical")
            .withRelationships(List.of(new Relationship("events","account","accounts","id"),new Relationship("events","item","items","id")));
        return schema;
      } finally { tables.values().forEach(Table::close); }
    }
  }
  static RelationalTask task(Map<String,Object> test) {
    Map<String,Object> task=object(test.get("task"));
    return new RelationalTask(RelationalTask.Kind.valueOf(task.get("kind").toString().toUpperCase(Locale.ROOT)),
        task.get("entity_table").toString(),(String)task.get("link_target_table"),
        task.get("step_size_ns")==null?null:((Number)task.get("step_size_ns")).longValue(),
        task.get("num_forecasts")==null?null:((Number)task.get("num_forecasts")).intValue());
  }
  static Request prepare(Schema schema,Map<String,Object> test,Map<String,Object> prediction) {
    try(RootAllocator allocator=new RootAllocator();Table context=arrow(allocator,object(test.get("context")),object(test.get("task")).get("kind").toString());
        Table predict=arrow(allocator,prediction,"")) {
      return schema.materializeExamples(task(test),context,predict,ExamplesOptions.builder().numNeighbors(List.of(10,10)).build());
    }
  }
  static Map<String,Object> payload(Request request,int index) throws Exception { return object(Json.read(request.payloadBytes().get(index))); }
  static Map<String,Object> instance(Map<String,Object> payload,String side) { return object(object(payload.get(side)).get("instance_table")); }
  static List<Object> column(Map<String,Object> table,String name) {
    int at=((List<String>)table.get("columns")).indexOf(name); assertTrue(at>=0,name);
    return rows(table).stream().map(row->row.get(at)).toList();
  }
  static Object normalized(Object value) {
    if(value instanceof Number number) return number.doubleValue();
    if(value instanceof List<?> list) return list.stream().map(CustomExamplesTest::normalized).toList();
    return value;
  }
  static byte[] saved(Request request) throws Exception { var out=new ByteArrayOutputStream();request.writeTo(out);return out.toByteArray(); }

  @Test void sharedLabelsOverlapsMaskingOwnershipAndPersistence() throws Exception {
    Map<String,Object> fixture=fixture();
    for(Map<String,Object> test:cases(fixture)) {
      String name=test.get("id").toString();
      Schema schema=memory(fixture,"timeless".equals(test.get("schema_variant")));
      if(name.equals("temporal_link_prediction")) {
        StructuredDataException refused=assertThrows(StructuredDataException.class,()->prepare(schema,test,object(test.get("predict"))));
        assertEquals("UNSUPPORTED_REQUEST",refused.code(),name);
        continue;
      }
      Request request=prepare(schema,test,object(test.get("predict")));
      // Inputs and their allocator are already closed: the prepared request owns its data.
      Map<String,Object> payload=payload(request,0),expected=object(test.get("expected"));
      List<Object> labels=column(instance(payload,"context"),"TARGET");
      List<?> wanted=(List<?>)expected.get("context_targets");
      assertEquals(normalized(wanted),normalized(labels),name);
      assertEquals(rows(object(test.get("context"))).size(),rows(instance(payload,"context")).size(),name+" overlaps retained");
      assertEquals(expected.get("use_prediction_time"),object(payload.get("inference")).get("use_prediction_time"));
      if(name.equals("regression") || name.equals("forecasting"))
        assertEquals("float64",object(object(payload.get("task")).get("target")).get("dtype"));
      if(expected.containsKey("closed_by_context_row")) {
        Map<String,Object> accounts=object(object(object(payload.get("context")).get("related_tables")).get("accounts"));
        List<Object> batch=column(accounts,"instance_id"),closed=column(accounts,"closed"),ids=column(accounts,"id");
        List<?> entityIds=(List<?>)expected.get("context_entities"),ends=(List<?>)expected.get("closed_by_context_row");
        for(int i=0;i<entityIds.size();i++) {
          int position=-1;
          for(int j=0;j<batch.size();j++) if(((Number)batch.get(j)).intValue()==i && normalized(ids.get(j)).equals(normalized(entityIds.get(i)))) {position=j;break;}
          assertTrue(position>=0,name+" context row "+i);
          if(ends.get(i)==null) assertNull(closed.get(position));
          else assertTrue(closed.get(position).toString().startsWith(ends.get(i).toString()));
        }
      }
      Request restored=Request.readFrom(new ByteArrayInputStream(saved(request)));
      assertEquals(request.payloadBytes().size(),restored.payloadBytes().size());
      for(int i=0;i<request.payloadBytes().size();i++) assertArrayEquals(request.payloadBytes().get(i),restored.payloadBytes().get(i));
    }
  }

  @Test void localSqlSourcesPreserveSuppliedLabelsAndRepeatedEntities() throws Exception {
    for(String backend:List.of("sqlite","duckdb")) {
      try(var connection=Shop.open(backend);RootAllocator allocator=new RootAllocator()) {
        Schema schema=Schema.fromSource(Shop.source(backend,connection))
            .withPrimaryKey("users","user_id").withPrimaryKey("orders","order_id")
            .withRelationships(List.of(new Relationship("orders","user_id","users","user_id")));
        Map<String,Object> contextData=Map.of("columns",List.of("ENTITY","TARGET"),
            "rows",List.of(List.of(3,1.25),List.of(1,8.5),List.of(3,-2.75)));
        Map<String,Object> predictData=Map.of("columns",List.of("ENTITY"),
            "rows",List.of(List.of(3),List.of(0),List.of(3)));
        Request request;
        try(Table context=arrow(allocator,contextData,"regression");Table predict=arrow(allocator,predictData,"")) {
          request=schema.materializeExamples(RelationalTask.regression("users"),context,predict,
              ExamplesOptions.builder().numNeighbors(List.of(10)).randomSeed(7).build());
        }
        Map<String,Object> payload=payload(request,0);
        assertEquals(List.of(1.25,8.5,-2.75),normalized(column(instance(payload,"context"),"TARGET")),backend);
        assertEquals(3,rows(instance(payload,"context")).size(),backend);
        assertEquals(3,rows(instance(payload,"predict")).size(),backend);
        Map<String,Object> users=object(object(object(payload.get("predict")).get("related_tables")).get("users"));
        List<Object> batches=column(users,"instance_id"),ids=column(users,"user_id");
        List<Object> actual=new ArrayList<>();
        for(Object batch:column(instance(payload,"predict"),"instance_id")) {
          int position=-1;
          for(int i=0;i<batches.size();i++) if(normalized(batch).equals(normalized(batches.get(i)))) {position=i;break;}
          assertTrue(position>=0,backend); actual.add(ids.get(position));
        }
        assertEquals(List.of(3.0,0.0,3.0),normalized(actual),backend);
      }
    }
  }

  static byte[] forecastAnswer(byte[] bytes) {
    try {
      Map<String,Object> call=object(Json.read(bytes)),table=instance(call,"predict");
      List<Object> ids=column(table,"instance_id"); assertEquals(1,ids.size());
      List<Object> predictions=new ArrayList<>();
      for(int step:List.of(3,1,2)) predictions.add(Map.of("id",ids.get(0).toString(),"row_index",0,"forecast_step",step,"prediction",step*1.25));
      return Json.write(Map.of("id","test-response","model","kumo-relational","metadata",Map.of("task_kind","forecasting"),"predictions",predictions));
    } catch(Exception error) { throw new IllegalStateException(error); }
  }

  @Test void nativeForecastBatchesKeepBothAnchorsAndAlignAllSteps() throws Exception {
    Map<String,Object> fixture=fixture();
    Map<String,Object> test=cases(fixture).stream().filter(c->c.get("id").equals("forecasting")).findFirst().orElseThrow();
    Request request=prepare(memory(fixture,false),test,object(test.get("additional_prediction_examples")));
    assertEquals(2,request.payloadBytes().size());
    try(Shop.Endpoint endpoint=new Shop.Endpoint(body->Map.entry(200,forecastAnswer(body)));
        SdmClient client=SdmClient.connect(endpoint.url())) {
      PredictionResult result=client.predict(request,SdmClient.KUMO_RELATIONAL);
      assertEquals(6,result.rows()); assertEquals(2,endpoint.received.size());
      List<PredictionItem> items=result.items();
      for(int row=0;row<2;row++) {
        Map<String,Object> call=object(Json.read(endpoint.received.get(row)));
        String anchor=object(call.get("task")).get("anchor_time_column").toString();
        assertTrue(column(instance(call,"predict"),anchor).get(0).toString().startsWith(row==0?"2024-01-07":"2024-01-09"));
        for(int step=1;step<=3;step++) {
          final int expectedRow=row,expectedStep=step;
          assertTrue(items.stream().anyMatch(item->item.rowIndex()==expectedRow && item.forecastStep()==expectedStep && normalized(item.id()).equals(3.0)));
        }
      }
    }
  }

  static byte[] corruptFacts(byte[] bytes,String mutation) throws Exception {
    var input=new DataInputStream(new ByteArrayInputStream(bytes));
    byte[] magic=input.readNBytes(8);int version=input.readInt();byte[] metadata=input.readNBytes(input.readInt());
    Map<String,Object> manifest=new LinkedHashMap<>(object(Json.read(metadata)));
    Map<String,Object> facts=new LinkedHashMap<>(object(manifest.get("example_facts")));
    if(mutation.equals("missing")) facts.remove("entity_table");
    else if(mutation.equals("kind")) facts.put("kind","made_up_task");
    else if(mutation.equals("fraction")) facts.put("num_forecasts",3.5);
    else if(mutation.equals("window")) facts.put("windowed",false);
    else facts.put("num_forecasts",99);
    manifest.put("example_facts",mutation.equals("null") ? null : mutation.equals("shape") ? "bad" : facts);
    byte[] changed=Json.write(manifest);
    var output=new ByteArrayOutputStream();var data=new DataOutputStream(output);
    data.write(magic);data.writeInt(version);data.writeInt(changed.length);data.write(changed);input.transferTo(data);
    return output.toByteArray();
  }

  @Test void corruptSuppliedSnapshotFactsAreCallerErrors() throws Exception {
    Map<String,Object> fixture=fixture();
    Map<String,Object> test=cases(fixture).stream().filter(c->c.get("id").equals("forecasting")).findFirst().orElseThrow();
    byte[] snapshot=saved(prepare(memory(fixture,false),test,object(test.get("predict"))));
    for(String mutation:List.of("missing","kind","horizon","fraction","window","null","shape")) {
      byte[] corrupt=corruptFacts(snapshot,mutation);
      Exception error=assertThrows(Exception.class,()->Request.readFrom(new ByteArrayInputStream(corrupt)),mutation);
      assertTrue(error instanceof IOException || error instanceof StructuredDataException,error.getClass().toString());
    }
  }
}
