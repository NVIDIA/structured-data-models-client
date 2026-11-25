// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Records the name of every environment variable read through getenv while
// recording is on, for the clean-import test: preloaded with LD_PRELOAD on
// Linux, where it defines getenv and secure_getenv itself, and with
// DYLD_INSERT_LIBRARIES on macOS, where dyld's interposing replaces getenv
// in every other image. shim_start() clears the record and starts it;
// shim_stop() stops it and returns the names, one per line.

#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#ifndef __APPLE__
#include <dlfcn.h>
#endif

static int recording;
static int overflowed;
static char names[16384];
static size_t used;

void shim_start(void) {
    used = 0;
    names[0] = 0;
    overflowed = 0;
    recording = 1;
}

// A record too long for the buffer ends with a name no test excuses, so
// the test fails rather than miss what was dropped.
const char *shim_stop(void) {
    recording = 0;
    if (overflowed) {
        // note() keeps 32 bytes free for this.
        strcpy(names + used, "SHIM_RECORD_OVERFLOWED\n");
    }
    return names;
}

static void note(const char *name) {
    if (!recording || !name) {
        return;
    }
    size_t n = strlen(name);
    if (used + n + 2 >= sizeof names - 32) {
        overflowed = 1;
        return;
    }
    memcpy(names + used, name, n);
    used += n;
    names[used++] = '\n';
    names[used] = 0;
}

#ifdef __APPLE__
static char *noted_getenv(const char *name) {
    note(name);
    return getenv(name);
}

__attribute__((used)) static struct {
    const void *replacement;
    const void *original;
} interposers[] __attribute__((section("__DATA,__interpose"))) = {
    {(const void *)noted_getenv, (const void *)getenv},
};
#else
char *getenv(const char *name) {
    static char *(*real)(const char *);
    if (!real) {
        real = (char *(*)(const char *))dlsym(RTLD_NEXT, "getenv");
    }
    note(name);
    return real(name);
}

char *secure_getenv(const char *name) {
    static char *(*real)(const char *);
    if (!real) {
        real = (char *(*)(const char *))dlsym(RTLD_NEXT, "secure_getenv");
    }
    note(name);
    return real(name);
}
#endif
