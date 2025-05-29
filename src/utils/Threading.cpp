/**
 * @file Threading.cpp
 * @brief Threading utilities implementation
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "../../include/LwTT/utils/Threading.hpp"
#include <algorithm>

namespace LwTT {
namespace Utils {

// ThreadPool implementation
ThreadPool::ThreadPool(size_t num_threads) 
    : num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads),
      stop_(false), active_threads_(0) {
    
    // Create worker threads
    workers_.reserve(num_threads_);
    for (size_t i = 0; i < num_threads_; ++i) {
        workers_.emplace_back(&ThreadPool::WorkerThread, this);
    }
}

ThreadPool::~ThreadPool() {
    // Signal all threads to stop
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    
    // Wait for all threads to finish
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::WaitForCompletion() {
    // Wait until all tasks are completed
    while (IsBusy()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void ThreadPool::WorkerThread() {
    for (;;) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !task_queue_.empty(); });
            
            if (stop_ && task_queue_.empty()) {
                return;
            }
            
            task = std::move(task_queue_.front());
            task_queue_.pop();
            active_threads_++;
        }
        
        // Execute the task
        try {
            task();
        } catch (...) {
            // Swallow exceptions to prevent thread termination
        }
        
        active_threads_--;
    }
}

} // namespace Utils
} // namespace LwTT
