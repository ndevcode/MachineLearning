#pragma once

namespace ndp {

// Abstract base class for work items submitted to ThreadPool.
// Clients derive from ThreadJob and implement execute().
class ThreadJob {
public:
    virtual ~ThreadJob() = default;
    virtual void execute() = 0;
};

} // namespace ndp
