#include "base/ThreadPool.h"

#include <algorithm>
#include <stdexcept>

namespace ndp {
namespace {
thread_local ThreadPool* currentPool = nullptr;
}

bool ThreadPool::onWorkerThread() const { return currentPool == this; }

ThreadPool::ThreadPool(unsigned nThreads) {
    if (nThreads == 0) throw std::invalid_argument("thread count must be positive");
    unsigned numThreads = std::min(nThreads,
                                   std::max(1u, std::thread::hardware_concurrency()));
    threads_.reserve(numThreads);
    try {
        for (unsigned i = 0; i < numThreads; ++i)
            threads_.emplace_back(&ThreadPool::run, this);
    } catch (...) {
        stop();
        throw;
    }
}

ThreadPool::~ThreadPool() { stop(); }

void ThreadPool::stop() {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        terminate_ = true;
    }
    queueCond_.notify_all();
    for (auto& thr : threads_) {
        thr.join();
    }
}

void ThreadPool::queueJob(std::unique_ptr<ThreadJob> job) {
    if (!job) throw std::invalid_argument("cannot queue a null job");
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        jobQueue_.push_back(std::move(job));
        pendingJobs_.fetch_add(1, std::memory_order_relaxed);
    }
    queueCond_.notify_one();
}

void ThreadPool::waitAll() {
    std::unique_lock<std::mutex> lock(queueMutex_);
    doneCond_.wait(lock, [this] {
        return pendingJobs_.load(std::memory_order_relaxed) == 0;
    });
}

bool ThreadPool::busy() const {
    return pendingJobs_.load(std::memory_order_relaxed) > 0;
}

int ThreadPool::pendingJobs() const {
    return pendingJobs_.load(std::memory_order_relaxed);
}

void ThreadPool::run() {
    currentPool = this;
    while (true) {
        std::unique_ptr<ThreadJob> job;
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCond_.wait(lock, [this] {
                return !jobQueue_.empty() || terminate_;
            });
            if (terminate_ && jobQueue_.empty()) {
                return;
            }
            job = std::move(jobQueue_.front());
            jobQueue_.pop_front();
        }
        job->execute();
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (--pendingJobs_ == 0) doneCond_.notify_all();
        }
    }
}

} // namespace ndp
