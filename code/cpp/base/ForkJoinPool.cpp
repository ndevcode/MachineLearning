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
    : size_(nThreads), tickets_(nThreads ? new Signal[nThreads] : nullptr) {
    if (nThreads == 0) throw std::invalid_argument("ForkJoinPool: thread count must be positive");
    workers_.reserve(nThreads - 1);
    try {
        for (unsigned task = 1; task < nThreads; ++task)
            workers_.emplace_back(&ForkJoinPool::work, this, task);
    } catch (...) {
        shutdown();
        throw;
    }
}

ForkJoinPool::~ForkJoinPool() {
    assert(!owned_.load() && "ForkJoinPool destroyed while a run() is in progress");
    shutdown();
}

void ForkJoinPool::shutdown() noexcept {
    stop_ = true;
    for (unsigned w = 1; w <= workers_.size(); ++w) {
        tickets_[w].value.fetch_add(1, std::memory_order_seq_cst);
        tickets_[w].value.notify_one();
    }
    for (auto& worker : workers_) worker.join();
}

void ForkJoinPool::record(std::exception_ptr error) noexcept {
    if (!failed_.exchange(true, std::memory_order_relaxed)) error_ = std::move(error);
}

void ForkJoinPool::work(unsigned task) {
    // Only the caller bumps this ticket, and only after this worker finished the previous
    // run it was given, so each new ticket differs from `seen` and none is missed.
    std::uint32_t seen = 0;
    for (;;) {
        seen = awaitChange(tickets_[task], seen);
        if (stop_) return;
        try {
            invoke_(fn_, task, tasks_);
        } catch (...) {
            record(std::current_exception());
        }
        // Release publishes this task's output writes (and error_) to the caller.
        if (remaining_.value.fetch_sub(1, std::memory_order_seq_cst) == 1) wake(remaining_);
    }
}

void ForkJoinPool::dispatch(unsigned tasks, Invoke invoke, const void* fn) {
    if (tasks == 0 || tasks > size_)
        throw std::invalid_argument("ForkJoinPool::run: " + std::to_string(tasks) +
                                    " tasks requested from a pool of " + std::to_string(size_));
    if (tasks == 1 || owned_.exchange(true, std::memory_order_acquire)) {
        for (unsigned task = 0; task < tasks; ++task) invoke(fn, task, tasks);
        return;
    }
    invoke_ = invoke;
    fn_ = fn;
    tasks_ = tasks;
    remaining_.value.store(tasks - 1, std::memory_order_relaxed);
    for (unsigned w = 1; w < tasks; ++w) {
        tickets_[w].value.fetch_add(1, std::memory_order_seq_cst);
        wake(tickets_[w]);
    }

    try {
        invoke(fn, 0, tasks);
    } catch (...) {
        record(std::current_exception());
    }
    // Workers still reference fn, so wait for them even if task 0 threw.
    for (auto left = remaining_.value.load(std::memory_order_acquire); left != 0;)
        left = awaitChange(remaining_, left);

    auto error = std::exchange(error_, nullptr);
    failed_.store(false, std::memory_order_relaxed);
    owned_.store(false, std::memory_order_release);
    if (error) std::rethrow_exception(error);
}

} // namespace ndp
