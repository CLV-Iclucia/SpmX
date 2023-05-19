//
// Created by creeper on 23-3-20.
//

#ifndef SPMX_MY_STL_H
#define SPMX_MY_STL_H

#include "spmx-types.h"
#include <algorithm>

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
} // namespace spmx

#endif // SPMX_MY_STL_H
