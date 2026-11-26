#pragma once

#include "base/ThreadPool.h"
#include <algorithm>
#include <future>

namespace ml::detail {

class RangeJob final : public ndp::ThreadJob {
public:
    explicit RangeJob(std::packaged_task<void()> task) : task_(std::move(task)) {}
    void execute() override { task_(); }
private:
    std::packaged_task<void()> task_;
};

// Each iteration owns its destination. Keep tiny loops local and wait for
// only this kernel's tasks, so independent graphs can share a pool.
template<class Function>
void parallel_for(ndp::ThreadPool& pool, std::size_t count, std::size_t work_per_item,
                  Function run) {
    const auto grain = std::max<std::size_t>(1, 4096 / work_per_item);
    const auto tasks = count ? std::min(pool.workerCount(), 1 + (count - 1) / grain) : 0;
    if (tasks <= 1 || pool.onWorkerThread()) {
        for (std::size_t i = 0; i < count; ++i) run(i);
        return;
    }
    std::vector<std::future<void>> done;
    done.reserve(tasks);
    try {
        std::size_t begin = 0;
        for (std::size_t t = 0; t < tasks; ++t) {
            const auto end = begin + count / tasks + (t < count % tasks);
            std::packaged_task<void()> task([&run, begin, end] {
                for (auto i = begin; i < end; ++i) run(i);
            });
            auto future = task.get_future();
            pool.queueJob(std::make_unique<RangeJob>(std::move(task)));
            done.push_back(std::move(future));
            begin = end;
        }
        for (auto& f : done) f.get();
    } catch (...) {
        for (auto& f : done) if (f.valid()) f.wait();
        throw;
    }
}

} // namespace ml::detail
