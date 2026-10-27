// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

/**
 * Exit 3: a panic inside the core reaches Java as an exception and leaves
 * the JVM running, at each place the core can panic under a binding entry
 * (the test-only {@code panic} operation: in {@code start}, in the run's
 * body at {@code next}, and on the PQL adapter's grown stack), each in a
 * JVM of its own ({@link T311Probe}), as the brief's threat model asks: a
 * panic unwinding across JNI kills the process, which this sees as the
 * process dying, where an in-process test would lose the whole run.
 *
 * <p>And the brief's resource axis, also in a JVM of its own: a run
 * released twice and used after its release, and one run used from two
 * threads at once.
 */
class T311PanicTest {
    private static final String[] SCENARIOS = {
        "panic-start", "panic-body", "panic-adapter-stack", "released", "threads",
    };

    /** Where this JVM found the probe and the host's classes, then its own
     * class path: a launcher may load the tests through a class loader of
     * its own. */
    private static String classPath() {
        List<String> parts = new ArrayList<>();
        for (String name : new String[] {T311Probe.class.getName(), T311Names.RUN, T311Names.QUERY}) {
            try {
                java.security.CodeSource source =
                        Class.forName(name).getProtectionDomain().getCodeSource();
                if (source != null) {
                    parts.add(Path.of(source.getLocation().toURI()).toString());
                }
            } catch (ClassNotFoundException | java.net.URISyntaxException absent) {
                // Named by the probe's own failure.
            }
        }
        parts.add(System.getProperty("java.class.path"));
        return String.join(java.io.File.pathSeparator, parts);
    }

    private static String run(String scenario, Path scratch) throws Exception {
        String java = ProcessHandle.current().info().command()
                .orElseThrow(() -> new IllegalStateException("no java command"));
        List<String> command = new ArrayList<>();
        command.add(java);
        command.addAll(Arrays.asList(T311Names.FLAGS));
        command.add("-Dsdm.repo=" + T311Names.root());
        command.add("-cp");
        command.add(classPath());
        command.add(T311Probe.class.getName());
        command.add(scenario);
        Process process = new ProcessBuilder(command)
                .directory(scratch.toFile())
                .redirectInput(ProcessBuilder.Redirect.from(new java.io.File("/dev/null")))
                .redirectError(scratch.resolve("stderr.txt").toFile())
                .redirectOutput(scratch.resolve("stdout.txt").toFile())
                .start();
        boolean ended = process.waitFor(300, TimeUnit.SECONDS);
        if (!ended) {
            process.destroyForcibly();
        }
        String out = Files.readString(scratch.resolve("stdout.txt"), StandardCharsets.UTF_8);
        String err = Files.readString(scratch.resolve("stderr.txt"), StandardCharsets.UTF_8);
        String said = scenario + "\n--- stdout\n" + out + "\n--- stderr\n" + err;
        assertTrue(ended, "the probe did not end in 300 s: " + said);
        List<Path> crashes;
        try (Stream<Path> files = Files.list(scratch)) {
            crashes = files.filter(file -> file.getFileName().toString().startsWith("hs_err"))
                    .collect(Collectors.toList());
        }
        assertEquals(List.of(), crashes, "the JVM crashed: " + said);
        assertEquals(0, process.exitValue(), "the probe failed: " + said);
        assertTrue(out.contains("ALIVE " + scenario), "the JVM ran no core call after: " + said);
        return err;
    }

    @TestFactory
    Stream<DynamicTest> aPanicReachesJavaAsAnExceptionAndTheJvmRuns() {
        return Arrays.stream(SCENARIOS).map(scenario -> DynamicTest.dynamicTest(scenario, () -> {
            Path scratch = Files.createTempDirectory("t311-" + scenario);
            String err = run(scenario, scratch);
            if (scenario.startsWith("panic-")) {
                // sdm-host's contain reports where the panic happened, and
                // only that (runs.rs, contain).
                assertTrue(err.contains("sdm core: a contained panic at "), scenario + ": " + err);
                assertTrue(!err.contains(T311Probe.PAYLOAD), scenario + ": the payload was written: " + err);
            }
        }));
    }
}
