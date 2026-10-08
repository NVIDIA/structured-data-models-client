// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The reference P7.rng compares the Rust core's std::mt19937 with: the C++
// standard library's own generator, drawn the way relationallib's
// NeighborSampler draws it (csrc/neighbor_sampler.cpp, `_sample`, case 2).
//
// Each input line is a job: a seed, then operations.
//
//   SEED u K f P C u K ...
//
// `u K` draws K raw values with `rng()`; `f P C` picks C of a population of
// P without replacement, as `_sample` does, and writes the picked offsets.
// Each output line holds every number the job's operations wrote, in order.

#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    unsigned long long seed = 0;
    if (!(in >> seed)) {
      std::cout << "\n";
      continue;
    }
    std::mt19937 rng;
    rng.seed(static_cast<std::mt19937::result_type>(seed));
    std::string op;
    bool first = true;
    auto put = [&](unsigned long long value) {
      if (!first) std::cout << ' ';
      std::cout << value;
      first = false;
    };
    while (in >> op) {
      if (op == "u") {
        long long count = 0;
        in >> count;
        for (long long i = 0; i < count; ++i) put(rng());
      } else if (op == "f") {
        int64_t population = 0, count = 0;
        in >> population >> count;
        std::unordered_set<int64_t> seen;
        for (auto i = population - count; i < population; ++i) {
          int64_t rnd = rng() % (i + 1);
          if (!seen.insert(rnd).second) {
            rnd = i;
            seen.insert(i);
          }
          put(static_cast<unsigned long long>(rnd));
        }
      }
    }
    std::cout << "\n";
  }
  return 0;
}
