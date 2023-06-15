//
// Created by creeper on 23-5-17.
//

#ifndef SPMX_PARALLEL_H
#define SPMX_PARALLEL_H

#include <condition_variable>
#include <mutex>
#include <queue>
#include <spmx-types.h>
#include <thread>
#include <type-utils.h>

namespace spmx {
template <typename T> class ThreadSafeQueue {
public:
  ThreadSafeQueue() = default;
  bool empty() {
    std::lock_guard<std::mutex> lk(mtx);
    return q.empty();
  }
  void push(T v) {
    std::lock_guard<std::mutex> lk(mtx);
    q.push(v);
  }
  void ImmPush(T v) { q.push(v); }
  bool TryPop(T &v) {
    std::lock_guard<std::mutex> lk(mtx);
    if (q.empty())
      return false;
    v = q.front();
    q.pop();
    return true;
  }
  void WaitPop(T &v) {
    std::unique_lock<std::mutex> lk(mtx);
    cond.wait(lk, [this]() -> bool { return !q.empty(); });
    v = q.front();
    q.pop();
    lk.unlock();
  }

private:
  std::mutex mtx;
  std::condition_variable cond;
  std::queue<T> q;
};

template <typename Kernel, typename... Ts> class ParallelExecuter {
public:
  explicit ParallelExecuter(int num_threads, Ts... arg) : args(arg...) {
    auto thread_work = [this]() -> void {
      int task_id = -1;
      while (true) {
        WaitSignal();
        [[unlikely]] if (destruct)
          break;
        while (task_queue.TryPop(task_id)) {
          kernel(task_id, args);
          DecCounter();
        }
      }
    };
    nthreads = num_threads;
    //    threads = new std::thread[nthreads];
    for (int i = 0; i < nthreads; i++)
      threads[i] = std::thread(thread_work);
  }
  ParallelExecuter(const Kernel &ker, int num_threads, Ts... arg)
      : ParallelExecuter(num_threads, arg...) {
    kernel = ker;
  }
  ParallelExecuter(Kernel &&ker, int num_threads, Ts... arg)
      : ParallelExecuter(num_threads, arg...) {
    kernel = std::move(ker);
  }
  ~ParallelExecuter() {
    StartDestruct();
    for (int i = 0; i < nthreads; i++)
      if (threads[i].joinable())
        threads[i].join();
    //    delete[] threads;
  }
  void run(uint num_total_tasks) {
    StartRun(num_total_tasks);
    WaitDone();
    can_run_tasks = false;
  }
  void StartRun(int num_total_tasks) {
    std::lock_guard<std::mutex> lk(mtx);
    for (int i = 0; i < num_total_tasks; i++)
      task_queue.ImmPush(i);
    task_counter = num_total_tasks;
    can_run_tasks = true;
    work_cond.notify_all();
  }
  void StartDestruct() {
    std::lock_guard<std::mutex> lk(mtx);
    destruct = true;
    work_cond.notify_all();
  }
  void WaitSignal() {
    std::unique_lock<std::mutex> lk(mtx);
    work_cond.wait(lk, [this]() -> bool { return can_run_tasks || destruct; });
    lk.unlock();
  }
  void DecCounter() {
    std::lock_guard<std::mutex> lk(mtx);
    task_counter--;
    if (task_counter == 0)
      end_cond.notify_one();
  }
  void WaitDone() {
    std::unique_lock<std::mutex> lk(mtx);
    end_cond.wait(lk);
    lk.unlock();
  }

private:
  int nthreads = 0;
  std::mutex mtx;
  Kernel kernel;
  bool destruct = false;
  bool can_run_tasks = false;
  uint task_counter = 0;
  std::tuple<Ts...> args;
  std::thread threads[NUM_MAX_THREADS];
  ThreadSafeQueue<int> task_queue;
  std::condition_variable work_cond;
  std::condition_variable end_cond;
};

template <typename Derived> class ParallelKernelBase {
  Derived &derived() { return *static_cast<Derived *>(this); }
  const Derived &derived() const { return *static_cast<Derived *>(this); }
  void operator()(uint id) const { derived().operator()(id); }
};

template <typename Lhs, typename Rhs, uint BlockSize = BLOCK_SIZE>
class ParallelSpmvKernel
    : ParallelKernelBase<ParallelSpmvKernel<Lhs, Rhs, BlockSize>> {
public:
  using Base = ParallelKernelBase<ParallelSpmvKernel<Lhs, Rhs, BlockSize>>;
  static_assert((traits<Lhs>::major == RowMajor ||
                 traits<Lhs>::major == Symmetric) &&
                traits<Lhs>::storage == Sparse &&
                is_supported_vector_of_major_v<Rhs, ColMajor> &&
                traits<Rhs>::storage == Dense);
  ParallelSpmvKernel() = default;
  using ProdRet = typename ProductReturnType<Lhs, Rhs>::type;
  using Args = std::tuple<const Lhs &, const Rhs &, ProdRet &>;

  void operator()(uint id, Args &arg) const {
    uint upper = (id + 1) * BlockSize;
    [[unlikely]] if (upper > std::get<0>(arg).Rows())
      upper = std::get<0>(arg).Rows();
    for (uint i = id * BlockSize; i < upper; i++) {
      Real sum = 0.0;
      for (uint j = std::get<0>(arg).OuterIdx(i);
           j < std::get<0>(arg).OuterIdx(i + 1); j++)
        sum += std::get<0>(arg).Data(j) *
               std::get<1>(arg)(std::get<0>(arg).InnerIdx(j));
      std::get<2>(arg)(i) = sum;
    }
  }
};

//template <typename Vec, uint BlockSize = BLOCK_SIZE>
//class ParallelDotKernel
//    : ParallelKernelBase<ParallelDotKernel<Vec, BlockSize>> {
//public:
//  using Base = ParallelKernelBase<ParallelDotKernel<Vec, BlockSize>>;
//  static_assert(traits<Vec>::storage == Dense);
//  using Args = std::tuple<const Vec &, Real&>;
//  [[gnu::noinline]]void operator()(uint id, Args &arg) const {
//    uint upper = (id + 1) * BlockSize;
//    [[unlikely]] if (upper > std::get<0>(arg).Rows())
//      upper = std::get<0>(arg).Rows();
//    for (uint i = id * BlockSize; i < upper; i++) {
//      Real sum = 0.0;
//      for (uint j = std::get<0>(arg).OuterIdx(i);
//           j < std::get<0>(arg).OuterIdx(i + 1); j++)
//        sum += std::get<0>(arg).Data(j) *
//               std::get<1>(arg)(std::get<0>(arg).InnerIdx(j));
//      std::get<2>(arg)(i) = sum;
//    }
//  }
//};
} // namespace spmx

#endif // SPMX_PARALLEL_H
