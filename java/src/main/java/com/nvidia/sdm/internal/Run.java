// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.sdm.internal;

import java.lang.ref.Cleaner;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.apache.arrow.c.ArrowArrayStream;

/**
 * One core run, the Java twin of Python's {@code sdm_client._native.Run}: control messages as
 * {@code byte[]} both ways, with the attachments each names by id, a bytes attachment a {@code
 * byte[]} and an Arrow one an {@link ArrowArrayStream} the caller exported its data into (the
 * library moves the stream out during the call; the caller closes the struct after). The core's
 * error is thrown as {@link NativeError}, a {@link com.nvidia.sdm.StructuredDataException} with
 * the core's code and fields, holding the {@code error} control message.
 *
 * <p>The run is the one owner of its native handle, released once by {@link #close} (and by a
 * {@link Cleaner}, should it never be closed). Calls may come from any thread; one made while
 * another is in the run is refused by the core with {@code RUN_BUSY}. A call after {@link #close}
 * throws {@link IllegalStateException} before it reaches the library, and {@link #close} waits
 * for the calls in flight, so no call meets a released handle. Not part of the API.
 */
public final class Run implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();

  /** The handle and its guard, shared with the cleaning action. */
  private static final class Handle implements Runnable {
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private long run;

    Handle(long run) {
      this.run = run;
    }

    @Override
    public void run() {
      lock.writeLock().lock();
      try {
        if (run != 0) {
          NativeCore.release(run);
          run = 0;
        }
      } finally {
        lock.writeLock().unlock();
      }
    }
  }

  /** A call into the run, given its handle. */
  @FunctionalInterface
  public interface Call<T> {
    T call(long run) throws NativeError;
  }

  private final Handle handle;
  private final Cleaner.Cleanable cleanable;

  private Run(long run) {
    this.handle = new Handle(run);
    this.cleanable = CLEANER.register(this, handle);
  }

  /**
   * Starts the run a {@code start} message asks for, with the attachments it names, loading the
   * native library first.
   *
   * @throws com.nvidia.sdm.StructuredDataException {@code UNSUPPORTED_REQUEST} when this JAR
   *     holds no native library for the platform
   */
  public static Run start(byte[] message, Map<Integer, ?> attachments) throws NativeError {
    NativeCore.load();
    Attachments sent = Attachments.of(attachments);
    return new Run(NativeCore.start(message, sent.ids(), sent.bytes(), sent.streams()));
  }

  /** Makes {@code call} on the run's handle, while the run is open. */
  public <T> T enter(Call<T> call) throws NativeError {
    handle.lock.readLock().lock();
    try {
      if (handle.run == 0) {
        throw new IllegalStateException("this run was closed");
      }
      return call.call(handle.run);
    } finally {
      handle.lock.readLock().unlock();
    }
  }

  /** The run's next step. */
  public NativeMessage next() throws NativeError {
    return enter(NativeCore::next);
  }

  /** Hands the run a {@code fetch_reply} or {@code model_reply}, with the attachments it names. */
  public void reply(byte[] message, Map<Integer, ?> attachments) throws NativeError {
    Attachments sent = Attachments.of(attachments);
    enter(
        run -> {
          NativeCore.reply(run, message, sent.ids(), sent.bytes(), sent.streams());
          return null;
        });
  }

  /** Cancels the run. */
  public void cancel() throws NativeError {
    enter(
        run -> {
          NativeCore.cancel(run);
          return null;
        });
  }

  /** Takes an attachment the run made: {@code byte[]}, a {@link NativeArrow}, or null. */
  public Object takeAttachment(int id) throws NativeError {
    return enter(run -> NativeCore.takeAttachment(run, id));
  }

  /** The warnings given since the last call, or null. */
  public NativeMessage takeWarnings() throws NativeError {
    return enter(NativeCore::takeWarnings);
  }

  /** The sampler state the run left, or null. */
  public NativeMessage takeSamplerState() throws NativeError {
    return enter(NativeCore::takeSamplerState);
  }

  /** Releases the run, once the calls in flight return; closing again does nothing. */
  @Override
  public void close() {
    cleanable.clean();
  }
}
