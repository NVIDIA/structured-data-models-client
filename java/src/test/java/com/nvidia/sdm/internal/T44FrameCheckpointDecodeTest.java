// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The existing Java decoder keeps the new opaque state beside an unchanged result. */
class T44FrameCheckpointDecodeTest {
  @Test
  void warningBoundaryRemainsAReplyFreeControlStep() {
    try (NativeMessage message = new NativeMessage(
        "{\"message\":\"step\",\"step\":\"warnings\"}".getBytes(StandardCharsets.UTF_8),
        new int[0], new Object[0])) {
      assertEquals(Map.of("message", "step", "step", "warnings"), Messages.read(message));
    }
  }
  @Test
  @SuppressWarnings("unchecked")
  void attachedMaterializedResultStillDecodesWithAnOpaqueFrameCheckpoint() {
    byte[] whole = """
        {"message":"step","step":"done","result":{"op":"materialize",
         "state":{"version":1,"payload":4,"query":5},
         "query":{"entity_table":"users","problem_type":"regression",
                  "windowed":false,"num_forecasts":1,"text":"scalar"}},
         "sampler_state":{"version":1,"memo":20,
                          "frame_checkpoint":{"version":1,"manifest":30}}}
        """.getBytes(StandardCharsets.UTF_8);
    try (NativeMessage nativeMessage = new NativeMessage(
        "{\"message\":\"attached\",\"attachment\":1}".getBytes(StandardCharsets.UTF_8),
        new int[] {1}, new Object[] {whole})) {
      Map<String, Object> read = Messages.read(nativeMessage);
      Map<String, Object> state = (Map<String, Object>) read.get("sampler_state");
      Map<String, Object> checkpoint = (Map<String, Object>) state.get("frame_checkpoint");
      assertEquals(1, ((Number) checkpoint.get("version")).intValue());
      assertEquals(30, ((Number) checkpoint.get("manifest")).intValue());
      Results.Gathered result = (Results.Gathered) Results.expect(
          (Map<String, Object>) read.get("result"), "materialize");
      assertEquals(4, result.state().payload());
      assertEquals("users", result.query().entityTable());
    }
  }
}
