//
// Created by creeper on 23-5-17.
//

#ifndef SPMX_CONCURRENCY_UTILS_H
#define SPMX_CONCURRENCY_UTILS_H

#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T> class ThreadSafeQueue {
public:

private:
  std::queue<T> q;
  std::mutex mtx;
  std::condition_variable cond;
};

#endif // SPMX_CONCURRENCY_UTILS_H
