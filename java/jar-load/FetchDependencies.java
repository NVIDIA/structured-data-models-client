// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.List;

/**
 * Fetches the third-party JARs that CI's java-jar artifact lists but does not carry. Each line of
 * {@code ROOT/java-dependencies.tsv} is a path under ROOT, a SHA-256 and a Maven Central URL. Run
 * in source-file mode:
 *
 * <pre>java FetchDependencies.java ROOT</pre>
 *
 * It exits 0 once every listed JAR is in place with its listed digest, else non-zero.
 */
public final class FetchDependencies {
  private static final int ATTEMPTS = 4;

  private FetchDependencies() {}

  public static void main(String[] arguments) throws Exception {
    Path root = Path.of(arguments[0]);
    HttpClient client =
        HttpClient.newBuilder()
            .followRedirects(HttpClient.Redirect.NORMAL)
            .connectTimeout(Duration.ofSeconds(30))
            .build();
    List<String> lines = Files.readAllLines(root.resolve("java-dependencies.tsv"));
    for (String line : lines) {
      String[] fields = line.split("\t");
      Path target = root.resolve(fields[0]);
      Files.createDirectories(target.getParent());
      Files.write(target, fetch(client, URI.create(fields[2]), fields[1]));
    }
    System.out.println("fetched " + lines.size() + " dependencies");
  }

  private static byte[] fetch(HttpClient client, URI uri, String sha256)
      throws IOException, InterruptedException, NoSuchAlgorithmException {
    HttpRequest request = HttpRequest.newBuilder(uri).timeout(Duration.ofMinutes(2)).build();
    String failure = "";
    for (int attempt = 1; attempt <= ATTEMPTS; attempt++) {
      try {
        HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
        if (response.statusCode() == 200) {
          byte[] body = response.body();
          String digest = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(body));
          if (!digest.equals(sha256)) {
            throw new IllegalStateException(uri + " has SHA-256 " + digest + ", not " + sha256);
          }
          return body;
        }
        failure = "HTTP " + response.statusCode();
      } catch (IOException error) {
        failure = error.toString();
      }
      Thread.sleep(2000L * attempt);
    }
    throw new IOException(uri + " failed " + ATTEMPTS + " times: " + failure);
  }
}
