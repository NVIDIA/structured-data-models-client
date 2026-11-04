// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.*;
import static com.nvidia.sdm.CustomExamplesTest.object;
import static com.nvidia.sdm.CustomExamplesTest.rows;

import com.nvidia.sdm.internal.Json;
import java.sql.Connection;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Same public PostgreSQL JSON/XML/domain sampling contract as Python. */
class PostgresKeylessOrderTest {
  static Schema prepare(Connection c, String kind) throws Exception {
    String ns="sdm_keyless_"+UUID.randomUUID().toString().replace("-", "");
    try(var s=c.createStatement()) {
      s.execute("CREATE SCHEMA "+ns);
      s.execute("CREATE DOMAIN "+ns+".payload AS JSON");
      s.execute("CREATE TABLE "+ns+".users(id BIGINT PRIMARY KEY,value DOUBLE PRECISION)");
      s.execute("INSERT INTO "+ns+".users SELECT i,i*1.5 FROM generate_series(1,20)i");
      String type=kind.equals("domain")?ns+".payload":kind;
      s.execute("CREATE TABLE "+ns+".notes(user_id BIGINT REFERENCES "+ns+".users(id),noted_at TIMESTAMP,a_number INTEGER,\"body quoted\" "+type+")");
      String expr=kind.equals("xml")?"xmlelement(name v,v)":"json_build_object('v',v)";
      s.execute("INSERT INTO "+ns+".notes SELECT i,TIMESTAMP '2024-01-01',n,CASE WHEN v IS NULL THEN NULL ELSE "+expr+" END FROM generate_series(1,20)i CROSS JOIN (VALUES(2,'z'),(10,'a'),(2,'a'),(2,NULL)) a(n,v)");
    }
    return Schema.fromSource(PostgresSource.of(c,ns),FromSourceOptions.builder()
        .table("users").table("notes").primaryKey("users","id").timeColumn("notes","noted_at").build())
        .withStype("notes","body quoted","categorical");
  }

  static Request materialize(Schema schema,int size) {
    return PredictiveQuery.parse("PREDICT users.value FOR EACH users.id").materialize(schema,
        MaterializeOptions.builder().model(SdmClient.KUMO_RELATIONAL).indices(List.of(1L,2L))
        .numContextSamples(10).option("num_neighbors",List.of(size)).option("random_seed",7).build());
  }

  @Test void jsonXmlDomainsKeepNumericAndNullOrdering() throws Exception {
    for(String kind:List.of("json","xml","domain")) {
      try(Connection c=T312Shop.open("postgres")) {
        Schema schema=prepare(c,kind);
        var expected=kind.equals("xml")?List.of("<v>a</v>","<v>z</v>"):List.of("{\"v\" : \"a\"}","{\"v\" : \"z\"}");
        for(int pass=0;pass<2;pass++) {
          var request=materialize(schema,2);
          var payload=object(Json.read(request.payloads().get(0).getBytes(java.nio.charset.StandardCharsets.UTF_8)));
          var table=object(object(object(payload.get("predict")).get("related_tables")).get("notes"));
          var columns=(List<?>)table.get("columns");
          assertEquals(List.of(2,2,2,2),rows(table).stream().map(r->((Number)r.get(columns.indexOf("a_number"))).intValue()).toList());
          var twice=new ArrayList<>(expected);twice.addAll(expected);
          assertEquals(twice,rows(table).stream().map(r->r.get(columns.indexOf("body quoted"))).toList());
        }
        var payload=object(Json.read(materialize(schema,3).payloads().get(0).getBytes(java.nio.charset.StandardCharsets.UTF_8)));
        var table=object(object(object(payload.get("predict")).get("related_tables")).get("notes"));
        int body=((List<?>)table.get("columns")).indexOf("body quoted");
        var expectedNull=new ArrayList<Object>(expected);expectedNull.add(null);expectedNull.addAll(new ArrayList<>(expectedNull));
        assertEquals(expectedNull,rows(table).stream().map(r->r.get(body)).toList());
        try(var s=c.createStatement();var result=s.executeQuery("SELECT 1")){assertTrue(result.next());}
      }
    }
  }

  @Test @org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable(named="SDM_RELATIONAL_URL",matches=".+")
  void livePostgresReadToPrediction() throws Exception {
    try(Connection c=T312Shop.open("postgres");SdmClient client=SdmClient.connect(System.getenv("SDM_RELATIONAL_URL"),ClientOptions.builder().timeout(java.time.Duration.ofSeconds(180)).build())) {
      var result=client.predict(materialize(prepare(c,"json"),2),SdmClient.KUMO_RELATIONAL);
      assertEquals(List.of(1,2),result.items().stream().map(i->((Number)i.id()).intValue()).toList());
    }
  }
}
