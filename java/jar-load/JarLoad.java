// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import com.nvidia.sdm.PredictiveQuery;
import com.nvidia.sdm.Profile;
import com.nvidia.sdm.QuerySchema;
import com.nvidia.sdm.internal.NativeCore;
import java.util.List;
import java.util.Map;

/**
 * The JAR-load test (T3.11, locked check 14): the JAR and its runtime dependencies alone, on a JDK
 * with no Rust toolchain beside it, load this platform's native from the JAR (never one named by
 * sdm.native.library) and answer a query's validation. Run in source-file mode:
 *
 * <pre>java FLAGS -cp JAR:LIB/* JarLoad.java</pre>
 *
 * It exits 0 printing {@code loaded <platform> contract <n>}, else non-zero.
 */
public final class JarLoad {
  private JarLoad() {}

  public static void main(String[] arguments) {
    if (System.getProperty(NativeCore.LIBRARY_PROPERTY) != null) {
      throw new IllegalStateException("the JAR-load test loads the JAR's own native");
    }
    Map<String, Object> info = NativeCore.load();
    if (!Boolean.FALSE.equals(info.get("test_ops"))) {
      throw new IllegalStateException("the JAR holds a native built with the test-only operations");
    }
    QuerySchema schema =
        new QuerySchema(
            List.of(
                new QuerySchema.Table(
                    "users",
                    List.of(
                        new QuerySchema.Column("id", "int", "ID"),
                        new QuerySchema.Column("age", "float", "numerical")),
                    "id",
                    null,
                    null)),
            List.of());
    boolean accepted =
        PredictiveQuery.parse("PREDICT users.age FOR EACH users.id")
            .validate(schema, Profile.RELATIONAL)
            .accepted();
    if (!accepted) {
      throw new IllegalStateException("the core refused a valid query");
    }
    System.out.println("loaded " + NativeCore.platform() + " contract " + info.get("contract"));
  }
}
