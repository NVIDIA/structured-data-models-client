// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.nvidia.sdm.internal.Host;
import com.nvidia.sdm.internal.Profile;
import com.nvidia.sdm.internal.QueryValidation;
import com.nvidia.sdm.internal.Validation;
import com.nvidia.sdm.internal.NativeCore;
import java.util.List;
import org.junit.jupiter.api.Test;

/** S0 through the public API: the core's validation, as Python's core route asks it. */
class PredictiveQueryTest {
  private static final String COUNT =
      "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id";

  @Test
  void anAcceptedQueryIsAnsweredAccepted() {
    Validation validation =
        QueryValidation.validate(COUNT, Schemas.shop(), Profile.RELATIONAL);
    assertTrue(validation.accepted());
    assertEquals(List.of(), validation.reasons());
  }

  @Test
  void aRefusedQueryIsAnsweredWithItsReasonsNotThrown() {
    Validation validation =
        QueryValidation.validate("PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.nope", Schemas.shop(), Profile.RELATIONAL);
    assertFalse(validation.accepted());
    assertFalse(validation.reasons().isEmpty());
    assertTrue(validation.diagnostics().stream().anyMatch(Validation.Diagnostic::isError));
  }

  @Test
  void anEmptyQueryIsRefusedAtParseAsPythonsConstructorRefusesIt() {
    for (String text : new String[] {null, "", "   \n"}) {
      StructuredDataException refused =
          assertThrows(StructuredDataException.class, () -> PredictiveQuery.parse(text));
      assertEquals("INVALID_REQUEST", refused.code());
    }
  }

  @Test
  void textOutsideAsciiCrossesWhole() {
    // Non-ASCII, a character outside the BMP and a NUL in a string literal:
    // control messages and attachments cross JNI as UTF-8 bytes, never as
    // JNI's modified UTF-8.
    for (String value : new String[] {"Zürich", "😀", "a\u0000b", "中文"}) {
      Validation validation =
          QueryValidation.validate(
                  "PREDICT users.country = '" + value + "' FOR EACH users.user_id", Schemas.shop(), Profile.RELATIONAL);
      assertTrue(validation.accepted(), value);
    }
  }

  @Test
  void anUnpairedSurrogateIsRefusedAsPythonsTransportRefusesIt() {
    StructuredDataException refused =
        assertThrows(
            StructuredDataException.class,
            () ->
                QueryValidation.validate("PREDICT users.country = '\uD83D' FOR EACH users.user_id", Schemas.shop(), Profile.RELATIONAL));
    assertEquals("INVALID_REQUEST", refused.code());
  }

  @Test
  void aQueryPastTheControlCapTravelsAsAnAttachment() {
    String padded = COUNT + " ".repeat(70 * 1024);
    assertTrue(padded.length() > Host.CONTROL_CAP);
    assertTrue(QueryValidation.validate(padded, Schemas.shop(), Profile.RELATIONAL).accepted());
  }

  @Test
  void theContractVersionIsTheLibrarys() {
    assertEquals(
        ((Number) NativeCore.load().get("contract")).intValue(), Host.contract());
    assertTrue(Host.contract() > 0);
  }

  @Test
  void everyErrorCodeOfTheContractIsKnown() {
    @SuppressWarnings("unchecked")
    List<String> codes = (List<String>) NativeCore.load().get("error_codes");
    assertTrue(codes.contains("INVALID_REQUEST"));
    assertTrue(codes.contains("INTERNAL_ERROR"));
  }

  @Test
  void anExceptionPrintsItsCode() {
    StructuredDataException error =
        new StructuredDataException("it failed", "INVALID_REQUEST", null, null);
    assertTrue(error.toString().endsWith(": [INVALID_REQUEST] it failed"));
    assertEquals("it failed", error.getMessage());
  }
}
