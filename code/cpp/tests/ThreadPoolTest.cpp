#ifdef NDEBUG
#undef NDEBUG
#endif
#include "base/ThreadPool.h"
#include "base/ThreadJob.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace ndp;

// ---------- Test jobs ----------

// Increments an atomic counter.
class IncrementJob : public ThreadJob {
public:
    explicit IncrementJob(std::atomic<int>& counter)
        : counter_(counter) {}
    void execute() override {
        counter_.fetch_add(1, std::memory_order_relaxed);
    }
private:
    std::atomic<int>& counter_;
};

// Appends a string to a shared vector under a mutex.
class AppendJob : public ThreadJob {
public:
    AppendJob(std::mutex& mtx, std::vector<std::string>& out,
              std::string value)
        : mtx_(mtx), out_(out), value_(std::move(value)) {}
    void execute() override {
        std::lock_guard<std::mutex> lock(mtx_);
        out_.push_back(value_);
    }
private:
    std::mutex& mtx_;
    std::vector<std::string>& out_;
    std::string value_;
};

// Sleeps briefly to test concurrency.
class SleepJob : public ThreadJob {
public:
    SleepJob(std::atomic<int>& counter, int sleepMs)
        : counter_(counter), sleepMs_(sleepMs) {}
    void execute() override {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(sleepMs_));
        counter_.fetch_add(1, std::memory_order_relaxed);
    }
private:
    std::atomic<int>& counter_;
    int sleepMs_;
};

// ---------- Tests ----------

void testBasicExecution() {
    std::cout << "testBasicExecution... ";
    std::atomic<int> counter{0};
    {
        ThreadPool pool(4);
        for (int i = 0; i < 100; ++i) {
            pool.queueJob(std::make_unique<IncrementJob>(counter));
        }
        pool.waitAll();
    }
    assert(counter.load() == 100 && "expected 100 increments");
    std::cout << "PASSED\n";
}

void testPendingJobsCounter() {
    std::cout << "testPendingJobsCounter... ";
    ThreadPool pool(2);
    std::atomic<int> counter{0};

    // Queue slow jobs so we can observe pendingJobs > 0
    for (int i = 0; i < 4; ++i) {
        pool.queueJob(std::make_unique<SleepJob>(counter, 50));
    }
    // Right after queuing, some jobs should still be pending.
    // (Not strictly guaranteed, but with 50ms sleep it's reliable.)
    assert(pool.busy() && "pool should be busy right after queuing");

    pool.waitAll();
    assert(!pool.busy() && "pool should not be busy after waitAll");
    assert(pool.pendingJobs() == 0 && "pendingJobs should be 0 after waitAll");
    assert(counter.load() == 4 && "expected 4 completions");
    std::cout << "PASSED\n";
}

void testWaitAllOnEmpty() {
    std::cout << "testWaitAllOnEmpty... ";
    ThreadPool pool(2);
    // waitAll on an empty pool should return immediately.
    pool.waitAll();
    assert(!pool.busy());
    std::cout << "PASSED\n";
}

void testMultipleWaitAll() {
    std::cout << "testMultipleWaitAll... ";
    std::atomic<int> counter{0};
    ThreadPool pool(4);

    for (int i = 0; i < 50; ++i) {
        pool.queueJob(std::make_unique<IncrementJob>(counter));
    }
    pool.waitAll();
    assert(counter.load() == 50);

    for (int i = 0; i < 50; ++i) {
        pool.queueJob(std::make_unique<IncrementJob>(counter));
    }
    pool.waitAll();
    assert(counter.load() == 100 && "expected 100 total after two batches");
    std::cout << "PASSED\n";
}

void testStringAppend() {
    std::cout << "testStringAppend... ";
    std::mutex mtx;
    std::vector<std::string> results;
    std::vector<std::string> names{
        "Tom", "Dick", "Harry", "Ram", "Shyam",
        "Jodu", "Modhu", "Alice", "Bob", "Alpha",
        "Bravo", "Charlie", "Delta"};

    {
        ThreadPool pool(4);
        for (const auto& name : names) {
            pool.queueJob(
                std::make_unique<AppendJob>(mtx, results, name));
        }
        pool.waitAll();
    }
    assert(results.size() == names.size()
           && "all names should have been appended");
    std::cout << "PASSED\n";
}

void testSingleThread() {
    std::cout << "testSingleThread... ";
    std::atomic<int> counter{0};
    {
        ThreadPool pool(1);
        for (int i = 0; i < 20; ++i) {
            pool.queueJob(std::make_unique<IncrementJob>(counter));
        }
        pool.waitAll();
    }
    assert(counter.load() == 20 && "single-thread pool should work");
    std::cout << "PASSED\n";
}

void testDestructorDrainsQueue() {
    std::cout << "testDestructorDrainsQueue... ";
    std::atomic<int> counter{0};
    {
        ThreadPool pool(2);
        for (int i = 0; i < 50; ++i) {
            pool.queueJob(std::make_unique<IncrementJob>(counter));
        }
        // No waitAll — destructor should join after draining.
    }
    assert(counter.load() == 50
           && "destructor should drain remaining jobs");
    std::cout << "PASSED\n";
}

int main() {
    testBasicExecution();
    testPendingJobsCounter();
    testWaitAllOnEmpty();
    testMultipleWaitAll();
    testStringAppend();
    testSingleThread();
    testDestructorDrainsQueue();

    std::cout << "\nAll ThreadPool tests PASSED.\n";
    return 0;
}
