// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The host facts every run starts with (contract {@code HostFacts}), read here as the Python host
 * reads them, so the core answers the same on either host:
 *
 * <ul>
 *   <li>{@code now}: the clock, a {@code ts} in microseconds, in UTC;
 *   <li>{@code local_zone}: by the core's one rule, which the Python host reads too ({@link
 *       NativeCore#localZone}), from {@code TZ} and the lookup roots;
 *   <li>{@code zone_roots}: the directories of Python's default {@code zoneinfo.TZPATH} that
 *       exist, each with the database version read there;
 *   <li>{@code platform}: the operating system as Python's {@code sys.platform} names it, and
 *       whether its C library pads a year below 1000 to four digits, which macOS's does and
 *       glibc's does not.
 * </ul>
 *
 * Java holds a date of any year, as pandas 3 does, so no {@code date_literal_range} is declared.
 * Not part of the API.
 */
public final class HostFacts {
  /** Python's default {@code zoneinfo.TZPATH}. */
  static final List<String> TZPATH =
      List.of("/usr/share/zoneinfo", "/usr/lib/zoneinfo", "/usr/share/lib/zoneinfo", "/etc/zoneinfo");

  private static final Pattern VERSION = Pattern.compile("#\\s*version\\s+(\\S+)");

  private HostFacts() {}

  /** The facts of a run starting now. */
  public static Map<String, Object> now() {
    List<Map<String, Object>> roots = zoneRoots();
    Map<String, Object> facts = new LinkedHashMap<>();
    Instant now = Instant.now();
    long micros = Math.addExact(Math.multiplyExact(now.getEpochSecond(), 1_000_000L), now.getNano() / 1_000L);
    Map<String, Object> clock = new LinkedHashMap<>();
    clock.put("t", "ts");
    clock.put("v", Long.toString(micros));
    clock.put("unit", "us");
    clock.put("zone", "UTC");
    facts.put("now", clock);
    facts.put("local_zone", localZone(roots, System.getenv("TZ")));
    facts.put("zone_roots", roots);
    Map<String, Object> platform = new LinkedHashMap<>();
    String os = NativeCore.platform().startsWith("macos") ? "darwin" : osName();
    platform.put("os", os);
    platform.put("strftime_pads_year", "darwin".equals(os));
    facts.put("platform", platform);
    return facts;
  }

  private static String osName() {
    String platform = NativeCore.platform();
    int dash = platform.indexOf('-');
    return dash < 0 ? platform : platform.substring(0, dash);
  }

  /** The zone database version in {@code root}: macOS's {@code +VERSION}, else {@code tzdata.zi}'s. */
  static String version(Path root) {
    try {
      String found = Files.readString(root.resolve("+VERSION"), StandardCharsets.US_ASCII).strip();
      if (!found.isEmpty()) {
        return found;
      }
    } catch (IOException | RuntimeException e) {
      // Not there, or not text: try the next.
    }
    try (BufferedReader zi = Files.newBufferedReader(root.resolve("tzdata.zi"), StandardCharsets.US_ASCII)) {
      String first = zi.readLine();
      Matcher matched = first == null ? null : VERSION.matcher(first);
      return matched != null && matched.lookingAt() ? matched.group(1) : null;
    } catch (IOException | RuntimeException e) {
      return null;
    }
  }

  /** The lookup roots' paths ({@link #zoneRoots}), in order, as a zone is looked up in them. */
  public static List<String> zoneRootPaths() {
    List<String> paths = new ArrayList<>();
    for (Map<String, Object> root : zoneRoots()) {
      paths.add((String) root.get("path"));
    }
    return paths;
  }

  /** The lookup roots that exist, in {@code TZPATH}'s order, with their versions. */
  static List<Map<String, Object>> zoneRoots() {
    List<Map<String, Object>> roots = new ArrayList<>();
    for (String path : TZPATH) {
      Path root = Path.of(path);
      if (Files.isDirectory(root)) {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("path", path);
        entry.put("version", version(root));
        roots.add(entry);
      }
    }
    return roots;
  }

  /**
   * The local zone, by the core's one rule ({@link NativeCore#localZone}, which the Python host
   * reads too), from {@code setting} ({@code TZ}, or null when unset) and the lookup roots; null
   * when it names none the core reads.
   */
  static Map<String, Object> localZone(List<Map<String, Object>> roots, String setting) {
    List<String> paths = new ArrayList<>();
    for (Map<String, Object> root : roots) {
      paths.add((String) root.get("path"));
    }
    return NativeCore.localZone(setting, paths);
  }
}
