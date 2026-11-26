// Copyright 2026 N Dev (ndevcode)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "base/ForkJoinPool.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace ml::detail {

// Smallest amount of work (multiply-adds or element operations) worth one more
// thread. Below it a dispatch costs more than the loop it splits.
constexpr std::size_t MIN_WORK_PER_TASK = 1u << 14;

// Tasks worth running for totalWork units spread over at most maxTasks owners.
inline unsigned taskCount(const ndp::ForkJoinPool& pool, std::size_t totalWork,
                          std::size_t maxTasks) {
    const std::size_t useful = std::max<std::size_t>(1, totalWork / MIN_WORK_PER_TASK);
    return static_cast<unsigned>(
        std::max<std::size_t>(1, std::min({useful, maxTasks, std::size_t(pool.size())})));
}

inline std::size_t saturatingProduct(std::size_t a, std::size_t b) {
    return b && a > std::numeric_limits<std::size_t>::max() / b
        ? std::numeric_limits<std::size_t>::max() : a * b;
}

// Task t's slice [begin, end) of count items split as evenly as possible over tasks.
// The slices are disjoint and cover [0, count), so each index has exactly one owner.
struct Range { std::size_t begin, end; };
inline Range chunk(std::size_t count, unsigned task, unsigned tasks) {
    const std::size_t base = count / tasks, extra = count % tasks;
    const std::size_t begin = task*base + std::min<std::size_t>(task, extra);
    return {begin, begin + base + (task < extra)};
}

// Runs task(t) for t in [0, tasks) on the pool. Only tasks - 1 workers are signalled.
template<class Task>
void runTasks(ndp::ForkJoinPool& pool, unsigned tasks, Task task) {
    pool.run(tasks, [&](unsigned t, unsigned) { task(t); });
}

// run(begin, end) once per task over disjoint slices covering [0, count). For kernels
// that set up per-slice state (scratch rows, a broadcast cursor) once, not per index.
template<class Function>
void parallelRanges(ndp::ForkJoinPool& pool, std::size_t count, std::size_t workPerItem,
                    Function run) {
    if (!count) return;
    const unsigned tasks = taskCount(pool, saturatingProduct(count, workPerItem), count);
    runTasks(pool, tasks, [&](unsigned t) {
        const auto [begin, end] = chunk(count, t, tasks);
        run(begin, end);
    });
}

// run(i) for every i in [0, count). Each i must write only outputs that i owns: then no
// lock is needed and the result is the same for every thread count.
template<class Function>
void parallelFor(ndp::ForkJoinPool& pool, std::size_t count, std::size_t workPerItem,
                 Function run) {
    parallelRanges(pool, count, workPerItem, [&](std::size_t begin, std::size_t end) {
        for (auto i = begin; i < end; ++i) run(i);
    });
}

} // namespace ml::detail
