//
// Created by creeper on 23-3-20.
//

#ifndef SPMX_MY_STL_H
#define SPMX_MY_STL_H

#include "spmx-types.h"
#include <algorithm>
#include <iostream>

namespace spmx {
/**
 * Linked-List class
 * @tparam T the type of the elements
 */
template <typename T> class List {
private:
  struct Node {
    T data;
    Node *nxt = nullptr;
    explicit Node(const T &_data) : data(_data) {}
    explicit Node(T &&_data) : data(std::move(_data)) {}
  };
  Node *head = nullptr, *tail = nullptr;
  uint sz = 0;

public:
  class iterator {
  private:
    Node *ptr = nullptr;

  public:
    iterator() = default;
    explicit iterator(Node *_ptr) : ptr(_ptr) {}
    iterator &operator++() {
      ptr = ptr->nxt;
      return *this;
    }
    T operator*() const { return ptr->data; }
    T *operator->() const { return &(ptr->data); }
    bool operator==(const iterator &it) { return ptr == it.ptr; }
    bool operator!=(const iterator &it) { return ptr != it.ptr; }
  };
  uint size() const { return sz; }
  void push_front(const T &ele) {
    Node *newNode = new Node(ele);
    newNode->nxt = head;
    head = newNode;
    sz++;
  }
  void push_back(const T &ele) {
    if (!sz)
      head = tail = new Node(ele);
    else {
      tail->nxt = new Node(ele);
      tail = tail->nxt;
    }
    sz++;
  }
  iterator begin() const { return iterator(head); }
  constexpr iterator end() const { return iterator(); }
  void clear() {
    if (!sz)
      return;
    Node *p = head, *pn = head->nxt;
    while (true) {
      delete p;
      p = pn;
      if (!p)
        break;
      pn = p->nxt;
    }
    head = tail = nullptr;
    sz = 0;
  }
  bool empty() const { return sz == 0; }
  ~List() { clear(); }
};

class BitSet {
public:
  using ull = unsigned long long;
  explicit BitSet(uint size) : size_(size) {
    bits_ = new ull[(size >> 6) + 1];
    allocated_ = (size >> 6) + 1;
  }
  void Set(uint i) { bits_[i >> 6] |= (1ull << (i & sizeof(ull))); }
  void Erase(uint i) { bits_[i >> 6] &= ~(1ull << (i & sizeof(ull))); }
  bool operator()(uint i) const {
    return bits_[i >> 6] & (1ll << (i & sizeof(ull)));
  }
  uint BitCnt() {
    uint sum = 0;
    for (uint i = 0; i < allocated_; i++) {
      ull x = bits_[i];
      while (x) {
        x ^= (x & (-x));
        sum++;
      }
    }
    return sum;
  }
  void Clear() {
    for (uint i = 0; i < allocated_; i++)
      bits_[i] = 0ull;
  }
  ~BitSet() { delete[] bits_; }

private:
  unsigned long long *bits_ = nullptr;
  uint size_ = 0;
  uint allocated_ = 0;
};

/**
 * a RAII Array class for trivially copyable type.
 * This is used for temporary arrays in various kinds of functions.
 * There is an option to construct these arrays on a specified buffer,
 * so that we can avoid the work of allocating and freeing memory frequently.
 * @note it is not thread-safe, so please avoid using it in multi-threaded programmes.
 * @tparam T
 */
template <typename T> class Array {
public:
  Array(T *data, uint n) : n_(n), data_(new T[n]) {
    memcpy(data_, data, sizeof(T) * n);
#ifdef MEMORY_TRACING
    MEMORY_LOG_ALLOC(Array, n);
#endif
  }
  explicit Array(uint n) : n_(n), data_(new T[n]) {
#ifdef MEMORY_TRACING
    MEMORY_LOG_ALLOC(Array, n);
#endif
  }
  T operator[](uint i) const {
#ifdef MEMORY_TRACING
    if (i >= n_)
      MEMORY_LOG_INVALID_ACCESS(Array, i);
#endif
    return data_[i];
  }
  T &operator[](uint i) {
#ifdef MEMORY_TRACING
    if (i >= n_)
      MEMORY_LOG_INVALID_ACCESS(Array, i);
#endif
    return data_[i];
  }
  uint Dim() const { return n_; }
  void Fill(T val) {
    // if this can be easily assigned, I expect compiler to optimize this
    for (uint i = 0; i < n_; i++)
      data_[i] = val;
  }
  T* Data() const { return data_; }
  T* operator+(uint offset) const { return data_ + offset; }
  T operator()(uint i) const {
#ifdef MEMORY_TRACING
    if (i >= n_)
      MEMORY_LOG_INVALID_ACCESS(Array, i);
#endif
    return data_[i];
  }
  T& operator()(uint i) {
#ifdef MEMORY_TRACING
    if (i >= n_)
      MEMORY_LOG_INVALID_ACCESS(Array, i);
#endif
    return data_[i];
  }
  ~Array() {
    delete[] data_;
#ifdef MEMORY_TRACING
    MEMORY_LOG_ALLOC(Array, n_);
#endif
  }

private:
  uint n_ = 0;
  T *data_;
};

} // namespace spmx

#endif // SPMX_MY_STL_H
