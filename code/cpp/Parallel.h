#pragma once

#include "base/ForkJoinPool.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace ml::detail {

// Smallest amount of work (multiply-adds or element operations) worth one more
// thread. Below it a dispatch costs more than the loop it splits.
constexpr std::size_t kMinWorkPerTask = 1u << 14;

// Tasks worth running for total_work units spread over at most max_tasks owners.
inline unsigned taskCount(const ndp::ForkJoinPool& pool, std::size_t total_work,
                          std::size_t max_tasks) {
    const std::size_t useful = std::max<std::size_t>(1, total_work / kMinWorkPerTask);
    return static_cast<unsigned>(std::min({useful, max_tasks, std::size_t(pool.size())}));
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

// run(i) for every i in [0, count). Each i must write only outputs that i owns: then no
// lock is needed and the result is the same for every thread count.
template<class Function>
void parallel_for(ndp::ForkJoinPool& pool, std::size_t count, std::size_t work_per_item,
                  Function run) {
    if (!count) return;
    const unsigned tasks = taskCount(pool, saturatingProduct(count, work_per_item), count);
    runTasks(pool, tasks, [&](unsigned t) {
        const auto [begin, end] = chunk(count, t, tasks);
        for (auto i = begin; i < end; ++i) run(i);
    });
}

} // namespace ml::detail
