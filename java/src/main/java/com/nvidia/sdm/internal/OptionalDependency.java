// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import com.nvidia.sdm.StructuredDataException;
import java.util.Map;

/**
 * The library's optional dependencies: each feature that needs one checks it is on the class path
 * first, and refuses, {@code INVALID_CONFIGURATION} naming the Maven artifact to add, when it is
 * not. Not part of the API.
 */
public enum OptionalDependency {
  /** {@code s3://} input. */
  AWS_S3("software.amazon.awssdk.services.s3.S3Client", "software.amazon.awssdk:s3 and software.amazon.awssdk:url-connection-client", "reading s3:// files"),
  /** {@code .zip}, {@code .tar}, {@code .bz2} input. */
  COMMONS_COMPRESS("org.apache.commons.compress.archivers.zip.ZipFile", "org.apache.commons:commons-compress", "reading compressed or archived files"),
  /** {@code .xz} input. */
  XZ("org.tukaani.xz.XZInputStream", "org.tukaani:xz", "reading .xz files"),
  /** {@code DatabricksServing} with the Databricks SDK's authentication. */
  DATABRICKS_SDK("com.databricks.sdk.WorkspaceClient", "com.databricks:databricks-sdk-java", "DatabricksServing with Databricks SDK authentication");

  private final String probe;
  private final String artifact;
  private final String feature;

  OptionalDependency(String probe, String artifact, String feature) {
    this.probe = probe;
    this.artifact = artifact;
    this.feature = feature;
  }

  /** The Maven coordinates ({@code group:artifact}) to add. */
  public String artifact() {
    return artifact;
  }

  /** Whether the dependency is on the class path. */
  public boolean present() {
    try {
      Class.forName(probe, false, OptionalDependency.class.getClassLoader());
      return true;
    } catch (ClassNotFoundException | LinkageError e) {
      return false;
    }
  }

  /** Refuses, naming what to add, when the dependency is not on the class path. */
  public void require() {
    if (!present()) {
      throw new StructuredDataException(
          feature + " needs " + artifact + " on the class path; add it to your build",
          "INVALID_CONFIGURATION",
          Map.of("dependency", artifact),
          null);
    }
  }
}
