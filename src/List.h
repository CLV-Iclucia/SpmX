//
// Created by creeper on 23-3-20.
//

#ifndef SPMX_LIST_H
#define SPMX_LIST_H

#include <algorithm>
#include <spmx_types.h>

namespace SpmX
{
    /**
     * Linked-List class
     * @tparam T the type of the elements
     */
    template<typename T>
    class List
    {
        private:
            struct Node
            {
                T data;
                Node *nxt = nullptr;
                explicit Node(const T& _data) : data(_data) {}
                explicit Node(T&& _data) : data(std::move(_data)) {}
            };
            Node *head = nullptr, *tail = nullptr;
            uint sz = 0;
            ~List()
            {
                if(!head) return;
                clear();
            }
        public:
            class iterator
            {
                private:
                    Node* ptr;
                public:
                    explicit iterator(Node* _ptr) : ptr(_ptr) {}
                    iterator operator ++()
                    {
                        ptr = ptr->nxt;
                        return *this;
                    }
                    T operator *() const { return ptr->data; }
                    T* operator ->() const { return &(ptr->data); }
                    bool operator==(const iterator& it) { return ptr == it.ptr; }
                    bool operator!=(const iterator& it) { return ptr != it.ptr; }
            };
            uint size() const { return sz; }
            void push_front(const T& ele)
            {
                Node *newNode = new Node(ele);
                newNode->nxt = head;
                head = newNode;
                sz++;
            }
            void push_back(const T& ele)
            {
                if(!tail) head = new Node(ele);
                else
                {
                    tail->nxt = new Node(ele);
                    tail = tail->nxt;
                }
                sz++;
            }
            iterator begin() const { return iterator(head); }
            iterator end() const { return iterator(nullptr); }
            void clear()
            {
                if(!head) return ;
                for(Node *p = head, *pn = head->nxt; pn != nullptr; p = p->nxt, pn = p->nxt)
                {
                    free(p);
                    p = pn;
                }
            }
            bool empty() const { return sz == 0; }
    };
}



#endif //SPMX_LIST_H
