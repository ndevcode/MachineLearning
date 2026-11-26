#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "base/ThreadJob.h"

namespace ndp {

// Fixed-size thread pool.  Workers block on an empty queue via
// condition_variable — no busy-waiting.  Pending job count is tracked
// with an atomic<int> so callers can wait for all submitted work to drain.
class ThreadPool {
public:
    // Create a pool with at most nThreads workers (capped at
    // hardware_concurrency).  nThreads must be > 0.
    explicit ThreadPool(unsigned nThreads);

    // Non-copyable, non-movable — threads hold a back-pointer to `this`.
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // Signals termination and joins all threads.  Any jobs still in
    // the queue are executed before the threads exit.
    ~ThreadPool();

    // Submit a job.  Ownership transfers to the pool.
    void queueJob(std::unique_ptr<ThreadJob> job);

    // Block until every submitted job has finished executing.
    void waitAll();

    // True if any jobs are queued or in flight.
    bool busy() const;

    // Number of jobs queued or in flight.
    int pendingJobs() const;
    bool onWorkerThread() const;
    std::size_t workerCount() const { return threads_.size(); }

private:
    void run();
    void stop();

    bool terminate_{false};
    std::mutex queueMutex_;
    std::condition_variable queueCond_;
    std::condition_variable doneCond_;
    std::vector<std::thread> threads_;
    std::deque<std::unique_ptr<ThreadJob>> jobQueue_;
    std::atomic<int> pendingJobs_{0};
};

} // namespace ndp
