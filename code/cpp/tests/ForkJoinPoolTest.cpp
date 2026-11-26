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

#ifdef NDEBUG
#undef NDEBUG
#endif
// Tests for ndp::ForkJoinPool and the partitioning helpers in Parallel.h built on it.
// CMake also builds this file as ForkJoinPoolSleepTest with NDP_SPIN_MICROSECONDS=0,
// so that every wait goes through the futex sleep/wake handshake. Run both under TSan.
#include "base/ForkJoinPool.h"
#include "Parallel.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef FJP_TEST_RUNS
#define FJP_TEST_RUNS 20000
#endif

using ndp::ForkJoinPool;
namespace detail = ml::detail;

// Longer than the pool's default 2 ms spin, so idle threads are asleep on the futex.
constexpr auto IDLE_TIME = std::chrono::milliseconds(10);

// Every task index runs exactly once per run(), and its plain (non-atomic) writes to
// its own slot are visible to the caller when run() returns.
void testEachTaskOncePerRun() {
    std::cout << "testEachTaskOncePerRun... ";
    for (unsigned n : {1u, 2u, 3u, 8u}) {
        ForkJoinPool pool(n);
        assert(pool.size() == n && "size() counts the calling thread");
        std::vector<long> hits(n, 0);
        constexpr int runs = FJP_TEST_RUNS;
        for (int r = 0; r < runs; ++r) {
            pool.run([&](unsigned task, unsigned tasks) {
                assert(tasks == n && "run() passes the pool size as the task count");
                ++hits[task];
            });
            for (unsigned t = 0; t < n; ++t)
                assert(hits[t] == r + 1 && "a task was skipped, repeated, or not yet visible");
        }
    }
    std::cout << "PASSED\n";
}

// Waits up to 10 s for `flag`; a hang would otherwise only show as a CTest timeout.
void awaitFlag(const std::atomic<bool>& flag, const char* what) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!flag.load(std::memory_order_acquire)) {
        assert(std::chrono::steady_clock::now() < deadline && what);
        std::this_thread::yield();
    }
}

// Nested runs cover every task once. With every worker busy the inner runs execute in
// place; with workers left idle by the outer run, the inner runs claim them.
void testNestedRun() {
    std::cout << "testNestedRun... ";
    ForkJoinPool pool(4);
    std::vector<std::vector<int>> inner(4, std::vector<int>(4, 0));
    pool.run([&](unsigned outer, unsigned) {
        pool.run([&](unsigned task, unsigned) { ++inner[outer][task]; });
    });
    for (const auto& row : inner)
        for (int hit : row) assert(hit == 1 && "nested run must cover every task once");

    // Outer run of 2 leaves 2 of 3 workers idle; each inner run of 2 claims one of them.
    // The inner task 1s rendezvous so both inner runs hold their worker at once; otherwise
    // the second could legitimately reuse the worker the first had already released.
    std::vector<std::thread::id> innerCaller(2), innerWorker(2);
    std::atomic<int> started{0};
    std::atomic<bool> bothStarted{false};
    pool.run(2u, [&](unsigned outer, unsigned) {
        innerCaller[outer] = std::this_thread::get_id();
        pool.run(2u, [&](unsigned task, unsigned) {
            if (task != 1) return;
            innerWorker[outer] = std::this_thread::get_id();
            if (started.fetch_add(1) + 1 == 2) bothStarted.store(true, std::memory_order_release);
            awaitFlag(bothStarted, "the two inner runs never held workers at the same time");
        });
    });
    for (unsigned outer = 0; outer < 2; ++outer)
        assert(innerWorker[outer] != innerCaller[outer] && "an inner run must use an idle worker");
    assert(innerWorker[0] != innerWorker[1] && "two inner runs must not share a worker");
    std::cout << "PASSED\n";
}

// A second caller is not serialized behind the first: it claims the workers the first
// run left idle while that run is still in progress.
void testSecondCallerGetsIdleWorkers() {
    std::cout << "testSecondCallerGetsIdleWorkers... ";
    ForkJoinPool pool(5);
    std::atomic<bool> firstRunning{false}, secondDone{false};
    std::thread first([&] {
        pool.run(2u, [&](unsigned task, unsigned) {
            if (task != 1) return;
            firstRunning.store(true, std::memory_order_release);
            awaitFlag(secondDone, "second caller did not finish while the first run was held");
        });
    });
    awaitFlag(firstRunning, "first run did not start");
    std::vector<std::thread::id> ids(3);
    pool.run(3u, [&](unsigned task, unsigned) { ids[task] = std::this_thread::get_id(); });
    secondDone.store(true, std::memory_order_release);
    first.join();
    assert(ids[1] != ids[0] && ids[2] != ids[0] && ids[1] != ids[2] &&
           "the second caller must run its tasks on idle workers, not serially");
    std::cout << "PASSED\n";
}

// Two caller threads share one pool, each claiming whatever workers are idle. Both must
// see complete, correct results.
void testConcurrentCallers() {
    std::cout << "testConcurrentCallers... ";
    ForkJoinPool pool(4);
    constexpr long runs = FJP_TEST_RUNS / 4;
    auto caller = [&](std::vector<long>& sums) {
        for (long r = 0; r < runs; ++r)
            pool.run([&](unsigned task, unsigned) { sums[task] += task + 1; });
    };
    std::vector<long> a(4, 0), b(4, 0);
    std::thread first(caller, std::ref(a)), second(caller, std::ref(b));
    first.join();
    second.join();
    for (unsigned t = 0; t < 4; ++t)
        assert(a[t] == runs*(t+1) && b[t] == runs*(t+1) && "concurrent callers lost work");
    std::cout << "PASSED\n";
}

void testExceptionsPropagate() {
    std::cout << "testExceptionsPropagate... ";
    ForkJoinPool pool(4);
    for (unsigned thrower : {0u, 3u}) {  // caller's own task, then a worker's
        bool caught = false;
        try {
            pool.run([&](unsigned task, unsigned) {
                if (task == thrower) throw std::runtime_error("task " + std::to_string(task));
            });
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()) == "task " + std::to_string(thrower);
        }
        assert(caught && "run() must rethrow the task's exception");
        std::vector<int> hits(4, 0);
        pool.run([&](unsigned task, unsigned) { ++hits[task]; });
        for (int hit : hits) assert(hit == 1 && "pool must stay usable after an exception");
    }
    // Several tasks throw: the lowest-numbered task's exception surfaces, whichever finished first.
    auto surfaced = [&](auto throws) {
        try {
            pool.run([&](unsigned task, unsigned) {
                if (throws(task)) throw std::runtime_error(std::to_string(task));
            });
        } catch (const std::runtime_error& e) {
            return std::string(e.what());
        }
        return std::string("none");
    };
    assert(surfaced([](unsigned task) { return task == 1 || task == 3; }) == "1" &&
           "the lowest-numbered throwing task must win");
    assert(surfaced([](unsigned) { return true; }) == "0" && "task 0 wins when every task throws");
    std::vector<int> hits(4, 0);
    pool.run([&](unsigned task, unsigned) { ++hits[task]; });
    for (int hit : hits) assert(hit == 1 && "pool must stay usable after concurrent throws");
    std::cout << "PASSED\n";
}

// Workers and caller asleep on the futex between runs must be woken exactly once each:
// a lost wake-up hangs here (CTest times out), a spurious one double-counts.
void testWakeAfterSleep() {
    std::cout << "testWakeAfterSleep... ";
    ForkJoinPool pool(4);
    std::vector<int> hits(4, 0);
    for (int r = 1; r <= 20; ++r) {
        std::this_thread::sleep_for(IDLE_TIME);
        // On odd rounds a worker's task is the slow one, so the caller sleeps on the futex
        // while waiting for it and must be woken by that worker.
        pool.run([&](unsigned task, unsigned) {
            if (task == 3 && r % 2) std::this_thread::sleep_for(IDLE_TIME);
            ++hits[task];
        });
        for (int hit : hits) assert(hit == r && "a task was lost or repeated after a sleep");
    }
    std::cout << "PASSED\n";
}

// Destruction must wake and join workers whether they never ran, are spinning, or are
// asleep. A worker that is not woken makes the destructor hang.
void testShutdown() {
    std::cout << "testShutdown... ";
    { ForkJoinPool fresh(4); }
    { ForkJoinPool spinning(4); spinning.run([](unsigned, unsigned) {}); }
    {
        ForkJoinPool sleeping(4);
        sleeping.run([](unsigned, unsigned) {});
        std::this_thread::sleep_for(IDLE_TIME);
    }
    { ForkJoinPool partial(4); partial.run(2u, [](unsigned, unsigned) {}); }  // workers 2, 3 never ran
    std::cout << "PASSED\n";
}

// chunk() splits [0, count) into contiguous, disjoint slices whose sizes differ by at most 1.
void testChunkPartition() {
    std::cout << "testChunkPartition... ";
    for (std::size_t count = 0; count <= 64; ++count)
        for (unsigned tasks = 1; tasks <= 9; ++tasks) {
            std::size_t next = 0, smallest = count, largest = 0;
            for (unsigned t = 0; t < tasks; ++t) {
                const auto [begin, end] = detail::chunk(count, t, tasks);
                assert(begin == next && end >= begin && "slices must be contiguous and in order");
                smallest = std::min(smallest, end - begin);
                largest = std::max(largest, end - begin);
                next = end;
            }
            assert(next == count && "slices must cover [0, count)");
            assert(largest - smallest <= 1 && "slices must be balanced");
        }
    std::cout << "PASSED\n";
}

// taskCount() stays serial below MIN_WORK_PER_TASK and never exceeds the pool or max_tasks.
void testTaskCount() {
    std::cout << "testTaskCount... ";
    ForkJoinPool pool(8);
    const auto grain = detail::MIN_WORK_PER_TASK;
    assert(detail::taskCount(pool, 0, 100) == 1 && "no work runs serially");
    assert(detail::taskCount(pool, grain - 1, 100) == 1 && "work below one grain runs serially");
    assert(detail::taskCount(pool, 3*grain, 100) == 3 && "one task per grain of work");
    assert(detail::taskCount(pool, 100*grain, 100) == 8 && "capped at the pool size");
    assert(detail::taskCount(pool, 100*grain, 5) == 5 && "capped at the number of owners");
    assert(detail::saturatingProduct(~std::size_t(0), 2) == ~std::size_t(0) && "work estimate saturates");
    std::cout << "PASSED\n";
}

// parallel_for writes every index exactly once, the caller sees all of the writes, and the
// result does not depend on the thread count (serial, split, or a busy pool's fallback).
void testParallelFor() {
    std::cout << "testParallelFor... ";
    constexpr std::size_t n = 1 << 20;
    auto fill = [&](ForkJoinPool& pool, std::size_t work_per_item) {
        std::vector<float> out(n, -1.f);
        std::vector<unsigned char> writes(n, 0);
        detail::parallelFor(pool, n, work_per_item, [&](std::size_t i) {
            ++writes[i];
            out[i] = static_cast<float>(i % 1000) * 0.5f;
        });
        assert(std::all_of(writes.begin(), writes.end(), [](auto w) { return w == 1; }) &&
               "each index must be written exactly once");
        return out;
    };
    ForkJoinPool one(1), four(4), seven(7);
    const auto serial = fill(one, 1);
    assert(fill(four, 1) == serial && fill(seven, 64) == serial && "thread count changed the result");
    std::vector<float> nested;
    // Inside a 4-task run the pool is owned, so the inner parallel_for takes the serial fallback.
    four.run([&](unsigned task, unsigned) { if (task == 0) nested = fill(four, 64); });
    assert(nested == serial && "fallback inside a run changed the result");
    std::size_t calls = 0;
    detail::parallelFor(four, 0, 1, [&](std::size_t) { ++calls; });
    assert(calls == 0 && "an empty range runs nothing");
    std::cout << "PASSED\n";
}

// run(k, f) with k < size() uses only workers 1..k-1. Alternating k leaves some workers
// idle for many runs; they must still pick up their next ticket exactly once.
void testPartialRuns() {
    std::cout << "testPartialRuns... ";
    ForkJoinPool pool(6);
    std::vector<long> hits(6, 0), expected(6, 0);
    for (int r = 0; r < FJP_TEST_RUNS; ++r) {
        const unsigned k = 1 + (r * 7) % 6;
        pool.run(k, [&](unsigned task, unsigned tasks) {
            assert(tasks == k && task < k && "task outside the requested range");
            ++hits[task];
        });
        for (unsigned t = 0; t < k; ++t) ++expected[t];
        assert(hits == expected && "partial run skipped or repeated a task");
    }
    std::cout << "PASSED\n";
}

void testRejectsBadCounts() {
    std::cout << "testRejectsBadCounts... ";
    bool zeroPool = false, zeroTasks = false, tooMany = false;
    try { ForkJoinPool pool(0); } catch (const std::invalid_argument&) { zeroPool = true; }
    ForkJoinPool pool(2);
    try { pool.run(0u, [](unsigned, unsigned) {}); } catch (const std::invalid_argument&) { zeroTasks = true; }
    try { pool.run(3u, [](unsigned, unsigned) {}); } catch (const std::invalid_argument&) { tooMany = true; }
    assert(zeroPool && zeroTasks && tooMany && "pool and task counts must be in range");
    std::cout << "PASSED\n";
}

int main() {
    testEachTaskOncePerRun();
    testNestedRun();
    testSecondCallerGetsIdleWorkers();
    testConcurrentCallers();
    testExceptionsPropagate();
    testPartialRuns();
    testWakeAfterSleep();
    testShutdown();
    testRejectsBadCounts();
    testChunkPartition();
    testTaskCount();
    testParallelFor();
    std::cout << "\nAll ForkJoinPool tests PASSED.\n";
}
