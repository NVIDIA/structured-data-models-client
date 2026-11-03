// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.sdm;

import com.nvidia.sdm.internal.*;
import java.util.LinkedHashMap;
import java.util.Map;

/** Thin Arrow/ownership adapter; task admission and graph sampling stay in Rust. */
final class Examples {
  private Examples() {}
  static Request materialize(Schema schema,RelationalTask task,org.apache.arrow.vector.table.Table context,org.apache.arrow.vector.table.Table predict,ExamplesOptions options) {
    Map<String,Object> input=new LinkedHashMap<>(); input.put("task",task.contract()); input.put("num_neighbors",options.neighbors); input.put("random_seed",PredictiveQuery.seed(options.seed));
    Map<Integer,Object> attachments=new LinkedHashMap<>(); attachments.put(2,Host.attachment(input)); attachments.put(5,Host.optionAttachment(options.inference));
    Map<Integer,byte[]> tables=Map.of(3,ArrowTabular.snapshot(context),4,ArrowTabular.snapshot(predict));
    RelationalMaterializer owner=null;
    try {
      Map<String,Object> source;
      Host.Call call;
      if(schema.source()==null) {
        owner=new RelationalMaterializer(schema); attachments.put(1,owner.graph());
        source=Map.of("source","frames","graph",1); call=Host.Call.none();
      } else {
        Providers.Chosen chosen=Providers.of(schema.source()); attachments.put(1,Host.attachment(schema.toHeld()));
        source=Map.of("source","sql","provider",chosen.choice(),"schema",1); call=new Host.Call(chosen.provider(),null,Providers.held(schema.source()));
      }
      Map<String,Object> operation=Map.of("op","materialize_examples","source",source,"input",2,"context",3,"predict",4,"inference",5);
      try(Host.Finished finished=ArrowFrames.run(operation,attachments,tables,call)) {
        Results.Gathered gathered=(Results.Gathered)Results.expect(finished.result(),"materialize");
        @SuppressWarnings("unchecked") Map<String,Object> state=(Map<String,Object>)finished.result().get("state");
        Request result=new Request(schema,new MaterializedRequest(state,finished),"",gathered.examples(),PredictiveQuery.inferenceAsked(options.inference));
        return owner==null?result:result.withSanitizationReport(owner.sanitization());
      }
    } finally { if(owner!=null) owner.close(); }
  }
}
