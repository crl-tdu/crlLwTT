/**
 * @file Threading.hpp
 * @brief Threading utilities for LwTT
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_UTILS_THREADING_HPP
#define LWTT_UTILS_THREADING_HPP

#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <chrono>

namespace LwTT {
namespace Utils {

/**
 * @brief Thread pool for parallel execution
 */
class ThreadPool {
public:
    /**
     * @brief Constructor
     * @param num_threads Number of worker threads (0 = auto-detect)
     */
    explicit ThreadPool(size_t num_threads = 0);

    /**
     * @brief Destructor
     */
    ~ThreadPool();

    /**
     * @brief Submit a task to the thread pool
     * @param task Task to execute
     * @return Future for the task result
     */
    template<typename F, typename... Args>
    auto Submit(F&& task, Args&&... args) -> std::future<decltype(task(args...))> {
        using return_type = decltype(task(args...));
        
        auto packaged_task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(task), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = packaged_task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("Submit called on stopped ThreadPool");
            }
            
            task_queue_.emplace([packaged_task]() { (*packaged_task)(); });
        }
        
        condition_.notify_one();
        return result;
    }

    /**
     * @brief Submit multiple tasks and wait for all to complete
     * @param tasks Vector of tasks
     */
    template<typename F>
    void SubmitBatch(const std::vector<F>& tasks) {
        std::vector<std::future<void>> futures;
        futures.reserve(tasks.size());
        
        for (const auto& task : tasks) {
            futures.emplace_back(Submit(task));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
    }

    /**
     * @brief Get number of worker threads
     * @return Number of threads
     */
    size_t GetNumThreads() const { return num_threads_; }

    /**
     * @brief Get number of pending tasks
     * @return Task count
     */
    size_t GetPendingTasks() const {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return task_queue_.size();
    }

    /**
     * @brief Check if thread pool is busy
     * @return true if any thread is working
     */
    bool IsBusy() const {
        return active_threads_.load() > 0 || GetPendingTasks() > 0;
    }

    /**
     * @brief Wait for all tasks to complete
     */
    void WaitForCompletion();

private:
    size_t num_threads_;
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> task_queue_;
    
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    std::atomic<size_t> active_threads_;
    
    // Worker thread function
    void WorkerThread();
};

/**
 * @brief Thread-safe queue
 */
template<typename T>
class ConcurrentQueue {
public:
    /**
     * @brief Push element to queue
     * @param item Item to push
     */
    void Push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        condition_.notify_one();
    }

    /**
     * @brief Push element to queue (move version)
     * @param item Item to push
     */
    void Push(T&& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        condition_.notify_one();
    }

    /**
     * @brief Pop element from queue (blocking)
     * @return Popped element
     */
    T Pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        T result = std::move(queue_.front());
        queue_.pop();
        return result;
    }

    /**
     * @brief Try to pop element from queue (non-blocking)
     * @param item Reference to store popped element
     * @return true if element was popped
     */
    bool TryPop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    /**
     * @brief Check if queue is empty
     * @return true if empty
     */
    bool Empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /**
     * @brief Get queue size
     * @return Number of elements
     */
    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;
};

/**
 * @brief Parallel for loop execution
 */
class ParallelFor {
public:
    /**
     * @brief Execute parallel for loop
     * @param start Start index
     * @param end End index (exclusive)
     * @param func Function to execute for each index
     * @param thread_pool Thread pool to use (optional)
     * @param chunk_size Chunk size for load balancing (0 = auto)
     */
    template<typename Func>
    static void Execute(size_t start, size_t end, Func&& func, 
                       ThreadPool* thread_pool = nullptr, size_t chunk_size = 0) {
        if (start >= end) return;
        
        // Use global thread pool if none provided
        static ThreadPool default_pool;
        ThreadPool* pool = thread_pool ? thread_pool : &default_pool;
        
        size_t total_work = end - start;
        size_t num_threads = pool->GetNumThreads();
        
        // Calculate optimal chunk size
        if (chunk_size == 0) {
            chunk_size = std::max(size_t(1), total_work / (num_threads * 4));
        }
        
        std::vector<std::future<void>> futures;
        
        for (size_t i = start; i < end; i += chunk_size) {
            size_t chunk_end = std::min(i + chunk_size, end);
            
            futures.emplace_back(pool->Submit([func, i, chunk_end]() {
                for (size_t idx = i; idx < chunk_end; ++idx) {
                    func(idx);
                }
            }));
        }
        
        // Wait for all chunks to complete
        for (auto& future : futures) {
            future.wait();
        }
    }
};

} // namespace Utils
} // namespace LwTT

#endif // LWTT_UTILS_THREADING_HPP
