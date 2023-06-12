//
// Created by creeper on 23-5-17.
//

#ifndef SPMX_PARALLEL_H
#define SPMX_PARALLEL_H

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

template<typename T> class ThreadSafeQueue {
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
  void ImmPush(T v) {
    q.push(v);
  }
  bool TryPop(T& v) {
    std::lock_guard<std::mutex> lk(mtx);
    if(q.empty()) return false;
    v = q.front();
    q.pop();
    return true;
  }
  void WaitPop(T& v) {
    std::unique_lock<std::mutex> lk(mtx);
    cond.wait(lk, [this]() -> bool {return !q.empty();});
    v = q.front();
    q.pop();
    lk.unlock();
  }
private:
  std::mutex mtx;
  std::condition_variable cond;
  std::queue<T> q;
};

template <typename Task>
class TheadPoolSystem {
public:
  explicit TheadPoolSystem(int num_threads) {
    auto thread_work = [this] ()-> void {
      int task_id = -1;
      while(true) {
        WaitSignal();
        if(destruct) break;
        while(task_queue.TryPop(task_id))
          irun(task_id, num_tot_tasks);
      }
    };
    nthreads = num_threads;
    threads = new std::thread[nthreads];
    for(int i = 0; i < nthreads; i++)
      threads[i] = std::thread(thread_work);
  }
  ~TheadPoolSystem() {
    StartDestruct();
    for(int i = 0; i < nthreads; i++)
      if(threads[i].joinable()) threads[i].join();
    delete[] threads;
  }
  void run(Task&& runnable, int num_total_tasks) {
    StartRun(runnable, num_total_tasks);
    while(!task_queue.empty());
    EndRun();
  }
  void StartRun(Task&& runnable, int num_total_tasks) {
    std::lock_guard<std::mutex> lk(mtx);
    irun = runnable;
    num_tot_tasks = num_total_tasks;
    for(int i = 0; i < num_total_tasks; i++)
      task_queue.ImmPush(i);
    can_run_tasks = true;
    work_cond.notify_all();
  }
  void EndRun() {
    std::lock_guard<std::mutex> lk(mtx);
    can_run_tasks = false;
  }
  void StartDestruct() {
    std::lock_guard<std::mutex> lk(mtx);
    destruct = true;
    work_cond.notify_all();
  }
  void WaitSignal() {
    std::unique_lock<std::mutex> lk(mtx);
    work_cond.wait(lk, [this]() -> bool {return can_run_tasks || destruct;});
    lk.unlock();
  }
private:
  int nthreads = 0;
  std::mutex mtx;
  Task irun;
  int num_tot_tasks = 0;
  bool destruct = false;
  bool can_run_tasks = false;
  std::thread* threads = nullptr;
  ThreadSafeQueue<int> task_queue;
  std::condition_variable work_cond;
};

#endif // SPMX_PARALLEL_H
