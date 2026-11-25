// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Opt-in launcher: the observed JVM loads the packaged client, never target/classes. */
class ParityRunnerTest {
  @Test
  @EnabledIfSystemProperty(named = "sdm.parity.output", matches = ".+")
  void requestedManifestScope() throws Exception {
    Path jar = Path.of(System.getProperty("sdm.parity.jar")).toRealPath();
    Path classes =
        Path.of(SdmClient.class.getProtectionDomain().getCodeSource().getLocation().toURI())
            .toRealPath();
    String cp =
        String.join(
            java.io.File.pathSeparator,
            Arrays.stream(System.getProperty("java.class.path").split(java.io.File.pathSeparator))
                .map(
                    part ->
                        Path.of(part).toAbsolutePath().normalize().equals(classes)
                            ? jar.toString()
                            : part)
                .toList());
    String connect = System.getProperty("sdm.parity.connect.classpath");
    if (connect != null) {
      cp =
          String.join(
                  java.io.File.pathSeparator,
                  Arrays.stream(cp.split(java.io.File.pathSeparator))
                      .filter(
                          path ->
                              !path.replace('\\', '/').contains("/org/apache/spark/")
                                  && !path.replace('\\', '/').contains("/org/scala-lang/"))
                      .toList())
              + java.io.File.pathSeparator
              + Files.readString(Path.of(connect)).trim();
    }
    var command = new ArrayList<String>();
    command.add(
        Path.of(
                System.getProperty("sdm.parity.java.home", System.getProperty("java.home")),
                "bin",
                "java")
            .toString());
    command.addAll(
        Arrays.asList(
            System.getProperty("sdm.java.flags", "--add-opens=java.base/java.nio=ALL-UNNAMED")
                .split(" ")));
    for (String name : System.getProperties().stringPropertyNames()) {
      if (name.startsWith("sdm.parity.") || name.equals("sdm.native.library"))
        command.add("-D" + name + "=" + System.getProperty(name));
    }
    command.add("-cp");
    command.add(cp);
    command.add(ParityRunner.class.getName());
    assertEquals(
        0,
        new ProcessBuilder(command).inheritIO().start().waitFor(),
        "Requested parity scope has incomplete/failed cells; inspect JSONL");
  }
}
