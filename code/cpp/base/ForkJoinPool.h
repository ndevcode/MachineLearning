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

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

namespace ndp {

// Fork-join pool for statically partitioned data-parallel loops. No mutex anywhere.
//
// run(tasks, f) calls f(task, tasks) exactly once for every task in [0, tasks). The
// calling thread runs task 0 and worker w runs task w. Each task derives the output
// indexes it owns from its task number, so there is no job queue to protect:
//   dispatch   = write the callable, then bump the ticket of each worker that has a task;
//   completion = each of those workers decrements remaining_; the caller waits for zero.
// The ticket store (release) and the remaining_ decrement / caller load (acquire) are the
// barrier: every write made by a task happens-before run() returns. Workers without a
// task are not signalled and not waited for. Idle threads spin briefly, then sleep on a
// futex via C++20 atomic wait (libstdc++ on Linux: SYS_futex, no lock).
//
// One run() owns the workers at a time. A run() that finds them owned (a second caller
// thread, or a nested call from inside a task) runs its tasks serially on its own
// thread. That is correct because a task's results depend only on its number, not on
// the thread that runs it.
class ForkJoinPool {
public:
    // nThreads counts the calling thread, so nThreads - 1 workers are started.
    explicit ForkJoinPool(unsigned nThreads);
    ~ForkJoinPool();

    ForkJoinPool(const ForkJoinPool&) = delete;
    ForkJoinPool& operator=(const ForkJoinPool&) = delete;
    ForkJoinPool(ForkJoinPool&&) = delete;
    ForkJoinPool& operator=(ForkJoinPool&&) = delete;

    unsigned size() const { return size_; }

    // Requires 1 <= tasks <= size(). Blocks until all tasks finish and rethrows the
    // first exception a task threw.
    template<class Function> void run(unsigned tasks, Function&& f) {
        using F = std::remove_reference_t<Function>;
        dispatch(tasks,
                 [](const void* fn, unsigned task, unsigned count) {
                     (*static_cast<F*>(const_cast<void*>(fn)))(task, count);
                 },
                 std::addressof(f));
    }
    template<class Function> void run(Function&& f) { run(size_, f); }

private:
    using Invoke = void (*)(const void*, unsigned, unsigned);

    // A value one thread sleeps on and another changes. `sleeping` lets the changer skip
    // the futex syscall when the waiter is still spinning.
    struct alignas(64) Signal {
        std::atomic<std::uint32_t> value{0};
        std::atomic<bool> sleeping{false};
    };
    static std::uint32_t awaitChange(Signal& signal, std::uint32_t old);
    static void wake(Signal& signal);

    void dispatch(unsigned tasks, Invoke invoke, const void* fn);
    void work(unsigned task);
    void record(std::exception_ptr error) noexcept;
    void shutdown() noexcept;

    const unsigned size_;
    std::unique_ptr<Signal[]> tickets_;  // tickets_[w] is worker w's; index 0 is unused
    std::vector<std::thread> workers_;

    // Plain fields: written by the owning caller before the tickets are released and read
    // only by signalled workers, which the caller waits for before writing them again.
    Invoke invoke_ = nullptr;
    const void* fn_ = nullptr;
    unsigned tasks_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;  // written only by the thread that wins failed_

    Signal remaining_;  // workers of the current run still working; the caller waits on it
    alignas(64) std::atomic<bool> owned_{false};
    std::atomic<bool> failed_{false};
};

} // namespace ndp
