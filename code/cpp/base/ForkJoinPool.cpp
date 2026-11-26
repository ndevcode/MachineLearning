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

#include "base/ForkJoinPool.h"

#include <cassert>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace ndp {
namespace {

void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

// How long an idle thread polls before it sleeps on the futex. Waking a sleeping thread
// cost 40-260 us on the WSL2 machine this was measured on (i7-13650HX), against ~1.5 us
// to dispatch to a spinning one, and the serial gaps inside a training step (zero_grad,
// gradient-buffer allocation) exceed 100 us. So spin through a step and sleep between
// runs of work. OpenMP runtimes make the same trade (Intel's KMP_BLOCKTIME is 200 ms).
#ifndef NDP_SPIN_MICROSECONDS
#define NDP_SPIN_MICROSECONDS 2000
#endif
constexpr std::chrono::microseconds SPIN_TIME{NDP_SPIN_MICROSECONDS};

} // namespace

// Waiter half of a Dekker handshake with wake(): the waiter publishes `sleeping` and then
// re-reads the value; the changer publishes the value and then reads `sleeping`. Under
// seq_cst at least one sees the other's store, so a wake-up is never lost.
std::uint32_t ForkJoinPool::awaitChange(Signal& signal, std::uint32_t old) {
    std::chrono::steady_clock::time_point deadline{};
    for (unsigned spin = 0;; ++spin) {
        const auto now = signal.value.load(std::memory_order_acquire);
        if (now != old) return now;
        cpuRelax();
        if (spin % 64 != 0) continue;  // read the clock rarely; one PAUSE is ~35 ns here
        const auto time = std::chrono::steady_clock::now();
        if (spin == 0) deadline = time + SPIN_TIME;
        if (time < deadline) continue;
        signal.sleeping.store(true, std::memory_order_seq_cst);
        if (signal.value.load(std::memory_order_seq_cst) == old)
            signal.value.wait(old, std::memory_order_acquire);
        signal.sleeping.store(false, std::memory_order_relaxed);
    }
}

// Call after changing signal.value with a seq_cst operation.
void ForkJoinPool::wake(Signal& signal) {
    if (signal.sleeping.load(std::memory_order_seq_cst)) signal.value.notify_one();
}

ForkJoinPool::ForkJoinPool(unsigned nThreads)
    : size_(nThreads), slots_(std::make_unique<Slot[]>(nThreads ? nThreads - 1 : 0)) {
    if (nThreads == 0) throw std::invalid_argument("ForkJoinPool: thread count must be positive");
    workers_.reserve(nThreads - 1);
    try {
        for (unsigned w = 0; w + 1 < nThreads; ++w) workers_.emplace_back(&ForkJoinPool::work, this, w);
    } catch (...) {
        shutdown();
        throw;
    }
}

ForkJoinPool::~ForkJoinPool() {
    for (std::size_t w = 0; w < workers_.size(); ++w)
        assert(!slots_[w].claimed.load() && "ForkJoinPool destroyed while a run() is in progress");
    shutdown();
}

void ForkJoinPool::shutdown() noexcept {
    stop_ = true;
    for (std::size_t w = 0; w < workers_.size(); ++w) {
        slots_[w].ticket.value.fetch_add(1, std::memory_order_seq_cst);
        slots_[w].ticket.value.notify_one();
    }
    for (auto& worker : workers_) worker.join();
}

void ForkJoinPool::work(unsigned worker) {
    Slot& slot = slots_[worker];
    // Only a claimer bumps this ticket, and it waits for `done` before releasing the slot,
    // so each new ticket differs from `seen` and none is missed.
    std::uint32_t seen = 0;
    for (;;) {
        seen = awaitChange(slot.ticket, seen);
        if (stop_) return;
        try {
            slot.invoke(slot.fn, slot.task, slot.tasks);
        } catch (...) {
            slot.error = std::current_exception();
        }
        // Release publishes the task's output writes (and slot.error) to the claimer. After
        // this the worker touches only the slot's atomics, so the claimer may reuse it.
        slot.done.value.fetch_add(1, std::memory_order_seq_cst);
        wake(slot.done);
    }
}

void ForkJoinPool::dispatch(unsigned tasks, Invoke invoke, const void* fn) {
    if (tasks == 0 || tasks > size_)
        throw std::invalid_argument("ForkJoinPool::run: " + std::to_string(tasks) +
                                    " tasks requested from a pool of " + std::to_string(size_));
    if (tasks == 1) {
        invoke(fn, 0, 1);
        return;
    }
    // Step 1: hand tasks 1, 2, ... to workers no run has claimed. Usually few enough to
    // record on the stack.
    constexpr unsigned INLINE_SLOTS = 64;
    unsigned inlineSlots[INLINE_SLOTS];
    std::unique_ptr<unsigned[]> heapSlots;
    unsigned* claimed = inlineSlots;
    if (tasks - 1 > INLINE_SLOTS) claimed = (heapSlots = std::make_unique<unsigned[]>(tasks - 1)).get();
    unsigned placed = 0, next = 1;
    for (unsigned w = 0; w < workers_.size() && next < tasks; ++w) {
        Slot& slot = slots_[w];
        if (slot.claimed.load(std::memory_order_relaxed) ||
            slot.claimed.exchange(true, std::memory_order_acquire))
            continue;
        slot.invoke = invoke;
        slot.fn = fn;
        slot.task = next++;
        slot.tasks = tasks;
        slot.error = nullptr;
        slot.doneBefore = slot.done.value.load(std::memory_order_relaxed);
        slot.ticket.value.fetch_add(1, std::memory_order_seq_cst);
        wake(slot.ticket);
        claimed[placed++] = w;
    }

    // Step 2: task 0, and every task no idle worker took, run on this thread.
    std::exception_ptr firstError;
    unsigned firstErrorTask = tasks;
    auto keep = [&](std::exception_ptr error, unsigned task) {
        if (error && task < firstErrorTask) {
            firstError = std::move(error);
            firstErrorTask = task;
        }
    };
    auto runHere = [&](unsigned task) {
        try {
            invoke(fn, task, tasks);
        } catch (...) {
            keep(std::current_exception(), task);
        }
    };
    runHere(0);
    for (unsigned task = next; task < tasks; ++task) runHere(task);

    // Step 3: wait for every claimed worker (they reference fn even if a task threw), take
    // its exception, release it.
    for (unsigned p = 0; p < placed; ++p) {
        Slot& slot = slots_[claimed[p]];
        awaitChange(slot.done, slot.doneBefore);
        keep(std::exchange(slot.error, nullptr), slot.task);
        slot.claimed.store(false, std::memory_order_release);
    }
    if (firstError) std::rethrow_exception(firstError);
}

} // namespace ndp
