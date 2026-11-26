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
// run(tasks, f) calls f(task, tasks) exactly once for every task in [0, tasks). Each task
// derives the output indexes it owns from its task number, so there is no job queue:
//   dispatch   = claim an idle worker (compare-and-swap on its slot), write the task into
//                the slot, bump the slot's ticket;
//   completion = the worker bumps the slot's done counter; the caller waits for it, then
//                releases the slot.
// The ticket store (release) and the done counter (release, then acquire by the caller)
// are the barrier: every write made by a task happens-before run() returns. The calling
// thread runs task 0, plus any task for which no worker was idle. Idle threads spin
// briefly, then sleep on a futex via C++20 atomic wait (libstdc++ on Linux: SYS_futex,
// no lock).
//
// Any number of callers can run at once, nested runs included: each claims whichever
// workers are idle, so a second caller or an inner run uses the workers an outer run left
// free rather than falling back to serial. Waits form a tree (a thread waits only on
// workers it claimed), so they cannot deadlock. Which thread runs a task never changes
// its result. If tasks throw, run() rethrows the exception of the lowest-numbered one.
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

    // Requires 1 <= tasks <= size(). Blocks until all tasks finish, then rethrows the
    // exception of the lowest-numbered task that threw.
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

    // One per worker. A caller claims an idle worker by setting `claimed`, writes the task
    // into the plain fields and bumps `ticket`. The worker runs it, stores any exception
    // and bumps `done`. The caller sees `done` change, takes the exception and clears
    // `claimed`. Each handoff is a release/acquire pair, so every plain field is touched
    // by one thread at a time.
    struct Slot {
        Signal ticket;
        Signal done;
        alignas(64) std::atomic<bool> claimed{false};
        Invoke invoke = nullptr;
        const void* fn = nullptr;
        unsigned task = 0, tasks = 0;
        std::uint32_t doneBefore = 0;  // done.value when claimed; read only by the claimer
        std::exception_ptr error;
    };

    void dispatch(unsigned tasks, Invoke invoke, const void* fn);
    void work(unsigned worker);
    void shutdown() noexcept;

    const unsigned size_;
    std::unique_ptr<Slot[]> slots_;  // slots_[w] belongs to worker w
    std::vector<std::thread> workers_;
    bool stop_ = false;  // written before the shutdown tickets, read after them
};

} // namespace ndp
