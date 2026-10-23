// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

/**
 * The program {@link T311PanicTest} runs in a JVM of its own, one scenario
 * per process, so a crash shows as the process dying rather than taking
 * the test run with it. It prints {@code CHECK <what>} for each check that
 * holds and, last, {@code ALIVE <scenario>} once the JVM has run the core
 * again after the scenario; any check failing exits 1, naming it.
 */
public final class T311Probe {
    /** A panic's payload: the panic's message, which no error shows. */
    static final String PAYLOAD = "SENTINEL-t311-panic-5c1e";

    private T311Probe() {}

    private static void check(boolean holds, String what) {
        if (!holds) {
            System.out.println("FAILED " + what);
            System.exit(1);
        }
        System.out.println("CHECK " + what);
    }

    private static String panic(String at) {
        return "{\"op\":\"test\",\"test\":\"panic\",\"at\":\"" + at + "\",\"payload\":\"" + PAYLOAD + "\"}";
    }

    /** A panic's error: an Exception (never an Error), the one
     * StructuredDataException, INTERNAL_ERROR, panic true, and no word of
     * the payload. */
    private static void panicked(Throwable raised, String where) throws Exception {
        check(raised instanceof Exception, where + ": the panic reached Java as an Exception, not "
                + raised.getClass().getName());
        check(T311Names.isStructuredDataException(raised), where + ": as StructuredDataException");
        check("INTERNAL_ERROR".equals(T311Names.code(raised)), where + ": with INTERNAL_ERROR, not "
                + T311Names.code(raised));
        check("true".equals(String.valueOf(T311Names.detail(raised, "panic"))),
                where + ": with details panic = true");
        check(!String.valueOf(raised).contains(PAYLOAD)
                        && !String.valueOf(raised.getMessage()).contains(PAYLOAD),
                where + ": showing nothing of the panic's message");
    }

    /** The core answers again: a validation as S0's first case asks it,
     * accepted, and a test-only run to its end. */
    private static void stillAnswers(String scenario) throws Exception {
        Map<String, Object> vector = T311Json.object(T311Json.object(T311Json.object(
                T311Names.vectors().get("binding")).get("validate")).get("s0.accept.binary-classification"));
        Object run = T311Names.startRun((String) vector.get("operation"),
                T311Names.attachments(T311Json.object(vector.get("attachments"))));
        String done = new String(T311Names.next(run), StandardCharsets.UTF_8);
        T311Names.close(run);
        check(done.contains("\"accepted\":true"), scenario + ": a later validation is answered");
        Object log = T311Names.startRun("{\"op\":\"test\",\"test\":\"log\",\"hold_ms\":0}", Map.of());
        check(new String(T311Names.next(log), StandardCharsets.UTF_8).contains("\"step\":\"done\""),
                scenario + ": a later run finishes");
        T311Names.close(log);
        System.gc();
        System.out.println("ALIVE " + scenario);
    }

    private interface Call {
        Object call() throws Exception;
    }

    private static Throwable raises(Call call) {
        try {
            call.call();
        } catch (Throwable raised) {
            return raised;
        }
        return null;
    }

    public static void main(String[] args) throws Exception {
        String scenario = args[0];
        switch (scenario) {
            case "panic-start": {
                Throwable raised = raises(() -> T311Names.startRun(panic("start"), Map.of()));
                check(raised != null, "a panic in start raised");
                panicked(raised, "start");
                break;
            }
            case "panic-body":
            case "panic-adapter-stack": {
                String at = scenario.equals("panic-body") ? "body" : "adapter_stack";
                Object run = T311Names.startRun(panic(at), Map.of());
                Throwable raised = raises(() -> T311Names.next(run));
                check(raised != null, "a panic in next raised");
                panicked(raised, at);
                Throwable again = raises(() -> T311Names.next(run));
                check(again != null && "RUN_POISONED".equals(T311Names.code(again)),
                        at + ": the poisoned run answers RUN_POISONED");
                T311Names.close(run);
                break;
            }
            case "released": {
                Object run = T311Names.startRun("{\"op\":\"test\",\"test\":\"log\",\"hold_ms\":0}", Map.of());
                T311Names.close(run);
                Throwable twice = raises(() -> {
                    T311Names.close(run);
                    return null;
                });
                check(twice == null, "a run released twice is no error");
                Throwable after = raises(() -> T311Names.next(run));
                check(after instanceof Exception, "a run used after its release raises an Exception, not "
                        + (after == null ? "nothing" : after.getClass().getName()));
                break;
            }
            case "threads": {
                Object run = T311Names.startRun("{\"op\":\"test\",\"test\":\"log\",\"hold_ms\":500}", Map.of());
                List<String> seen = Collections.synchronizedList(new ArrayList<>());
                CountDownLatch go = new CountDownLatch(1);
                List<Thread> threads = new ArrayList<>();
                for (int i = 0; i < 2; i++) {
                    Thread thread = new Thread(() -> {
                        try {
                            go.await();
                            seen.add(new String(T311Names.next(run), StandardCharsets.UTF_8).contains("\"done\"")
                                    ? "done" : "step");
                        } catch (Throwable raised) {
                            String code = null;
                            try {
                                code = T311Names.code(raised);
                            } catch (Throwable unreadable) {
                                code = raised.getClass().getName();
                            }
                            seen.add(String.valueOf(code));
                        }
                    });
                    thread.start();
                    threads.add(thread);
                }
                go.countDown();
                for (Thread thread : threads) {
                    thread.join(60_000);
                }
                check(seen.size() == 2, "both threads came back: " + seen);
                check(seen.contains("done"), "one thread finished the run: " + seen);
                check(List.of("done", "RUN_BUSY", "RUN_FINISHED").containsAll(seen),
                        "the other was told RUN_BUSY or RUN_FINISHED: " + seen);
                T311Names.close(run);
                break;
            }
            default:
                System.out.println("FAILED no scenario " + scenario);
                System.exit(2);
        }
        stillAnswers(scenario);
        System.exit(0);
    }
}
