// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The native library, {@code sdm-jni}: its entries, and its loading from the JAR at first use
 * (design choice 10). Control messages cross as {@code byte[]}, never as {@code String}; an
 * attachment in is {@code bytes[i]} when it is not null, else the {@code ArrowArrayStream} struct
 * at address {@code streams[i]}, which the library moves out of. Every entry but {@link #release}
 * and {@link #releaseArrow} throws {@link NativeError} for the core's error. Not part of the API.
 *
 * <p>The library is the one in this JAR for the platform the JVM runs on, under {@code
 * com/nvidia/sdm/native/<os>-<arch>/}; the system property {@code sdm.native.library} names
 * another file to load instead (the tests' build with the test-only operations). A platform the
 * JAR holds no library for is refused by name, as {@code UNSUPPORTED_REQUEST}.
 */
public final class NativeCore {
  /** The platforms this JAR is built with a library for. */
  public static final List<String> PLATFORMS =
      List.of("linux-x86_64", "linux-aarch64", "macos-aarch64", "macos-x86_64");

  /** The system property naming a library file to load in place of the JAR's. */
  public static final String LIBRARY_PROPERTY = "sdm.native.library";

  private static final Object LOCK = new Object();
  private static volatile Map<String, Object> info;
  private static StructuredDataException failed;

  private NativeCore() {}

  /** The platform the JVM runs on, as this JAR names it: {@code <os>-<arch>}. */
  public static String platform() {
    String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
    String osName = os.startsWith("linux") ? "linux" : os.startsWith("mac") ? "macos" : os;
    String archName =
        switch (arch) {
          case "amd64", "x86_64", "x64" -> "x86_64";
          case "aarch64", "arm64" -> "aarch64";
          default -> arch;
        };
    return osName + "-" + archName;
  }

  /**
   * Loads the library, once; later calls give what the first found: the library's description
   * ({@code contract}, {@code test_ops}, {@code providers}, {@code error_codes}), or the same
   * refusal.
   *
   * @throws StructuredDataException {@code UNSUPPORTED_REQUEST} when this JAR holds no library for
   *     the platform, or the platform cannot load it
   */
  @SuppressWarnings("unchecked")
  public static Map<String, Object> load() {
    Map<String, Object> known = info;
    if (known != null) {
      return known;
    }
    synchronized (LOCK) {
      if (info != null) {
        return info;
      }
      if (failed != null) {
        throw failed;
      }
      try {
        loadLibrary();
        init(NativeMessage.class, NativeArrow.class, NativeError.class, NativeLog.class);
        info = (Map<String, Object>) Json.read(info0());
        return info;
      } catch (StructuredDataException e) {
        failed = e;
      } catch (UnsatisfiedLinkError | IOException | Json.JsonException e) {
        failed =
            new StructuredDataException(
                "the native library for " + platform() + " could not be loaded: " + e,
                "UNSUPPORTED_REQUEST",
                Map.of("platform", platform()),
                e);
      }
      throw failed;
    }
  }

  private static void loadLibrary() throws IOException {
    String named = System.getProperty(LIBRARY_PROPERTY);
    if (named != null && !named.isEmpty()) {
      System.load(Path.of(named).toAbsolutePath().toString());
      return;
    }
    String platform = platform();
    String file = platform.startsWith("macos") ? "libsdm_jni.dylib" : "libsdm_jni.so";
    String resource = "/com/nvidia/sdm/native/" + platform + "/" + file;
    try (InputStream in = NativeCore.class.getResourceAsStream(resource)) {
      if (in == null) {
        throw new StructuredDataException(
            "this JAR holds no native library for "
                + platform
                + "; it holds one for each of "
                + String.join(", ", PLATFORMS),
            "UNSUPPORTED_REQUEST",
            Map.of("platform", platform, "platforms", PLATFORMS),
            null);
      }
      Path directory = Files.createTempDirectory("sdm-native-");
      Path library = directory.resolve(file);
      try {
        Files.copy(in, library, StandardCopyOption.REPLACE_EXISTING);
        System.load(library.toString());
      } finally {
        // A loaded library stays mapped: the file is not needed past the load.
        try {
          Files.deleteIfExists(library);
          Files.deleteIfExists(directory);
        } catch (IOException e) {
          library.toFile().deleteOnExit();
          directory.toFile().deleteOnExit();
        }
      }
    }
  }

  private static native void init(
      Class<?> message, Class<?> arrow, Class<?> error, Class<?> log);

  private static native byte[] info0();

  /**
   * The table of what every error's and warning's sentence says, as its JSON bytes ({@code
   * sdm_host::templates}), which {@link Wording} reads once.
   */
  static byte[] templates() {
    load();
    return templates0();
  }

  private static native byte[] templates0();

  /** Whether the UTF-8 text is a POSIX TZ string a run takes as its local zone. */
  public static native boolean isPosixZone(byte[] text);

  /**
   * The local zone a run's facts name, by the core's one rule ({@code sdm_host::local_zone},
   * which the Python host reads too): {@code setting} is {@code TZ}, or null when unset, and
   * {@code roots} the zone lookup roots in this host's order. {@code {"iana": name}}, {@code
   * {"posix": text}}, or null when neither names one.
   */
  @SuppressWarnings("unchecked")
  public static Map<String, Object> localZone(String setting, List<String> roots) {
    load();
    try {
      byte[] answer =
          localZone0(
              setting == null ? null : setting.getBytes(StandardCharsets.UTF_8),
              Json.write(roots));
      Object read = Json.read(answer);
      return read == null ? null : Map.copyOf((Map<String, Object>) read);
    } catch (Json.JsonException | ClassCastException e) {
      throw new StructuredDataException(
          "the native library's local zone cannot be read", "INTERNAL_ERROR", Map.of(), e);
    }
  }

  private static native byte[] localZone0(byte[] setting, byte[] roots);

  /**
   * Whether the UTF-8 path names a regular file, links followed: the rule of {@link #localZone}
   * asks it of this host, which reads the machine's files (the native library reads none).
   */
  static boolean isRegularFile(byte[] path) {
    try {
      return Files.isRegularFile(Path.of(new String(path, StandardCharsets.UTF_8)));
    } catch (RuntimeException e) {
      return false;
    }
  }

  /**
   * The UTF-8 path with every link resolved, or null when it does not resolve: the rule of {@link
   * #localZone} asks it of this host.
   */
  static byte[] realPath(byte[] path) {
    try {
      return Path.of(new String(path, StandardCharsets.UTF_8))
          .toRealPath()
          .toString()
          .getBytes(StandardCharsets.UTF_8);
    } catch (IOException | RuntimeException e) {
      return null;
    }
  }

  /**
   * What each model supports, as the client describes it to a caller, as the core's JSON bytes
   * ({@code sdm_host::capabilities}): {@code {"models": [...]}}.
   */
  public static byte[] capabilities() {
    load();
    return capabilities0();
  }

  private static native byte[] capabilities0();


  /** Sets the most verbose level the core's events pass at, 0 none to 5 trace. */
  public static native void setLogLevel(int level);

  /** Starts a run; its handle, which the caller owns and releases once. */
  public static native long start(byte[] message, int[] ids, byte[][] bytes, long[] streams)
      throws NativeError;

  /** The run's next step. */
  public static native NativeMessage next(long run) throws NativeError;

  /** Hands the run a reply. */
  public static native void reply(
      long run, byte[] message, int[] ids, byte[][] bytes, long[] streams) throws NativeError;

  /** Cancels the run. */
  public static native void cancel(long run) throws NativeError;

  /** Takes an attachment the run made: {@code byte[]}, a {@link NativeArrow}, or null. */
  public static native Object takeAttachment(long run, int id) throws NativeError;

  /** The warnings given since the last call, or null. */
  public static native NativeMessage takeWarnings(long run) throws NativeError;

  /** The sampler state the run left, or null. */
  public static native NativeMessage takeSamplerState(long run) throws NativeError;

  /** Releases a run's handle. */
  public static native void release(long run);

  /** Exports an Arrow handle's data into the empty stream struct at {@code address}. */
  static native void exportArrow(long arrow, long address);

  /** Releases an Arrow handle. */
  static native void releaseArrow(long arrow);
}
