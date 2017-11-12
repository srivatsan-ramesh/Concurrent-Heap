#include "constants.h"
#include "utils.cu"

template <class T, typename P = less<T> >
class SLHeap {
    T *list;
    int size;
    int mutex;

    __device__ void bubbleUp(P comp=P());
    __device__ void bubbleDown(P comp=P());
    __device__ void swap(int child, int parent);
    __device__ int getLeftChild(int parent);
    __device__ int getRightChild(int parent);
    __device__ int getParent(int child);

    public:
        __device__ void init(int s);
        __device__ void insert(T);
        __device__ T remove();
        __device__ int getSize();
};

template <class T, typename P >
__device__ void SLHeap<T, P> :: init(int s){
   list = (T *)malloc(s*sizeof(T));
   mutex = 0;
   size = 0;
}

template <class T, typename P >
__device__ int SLHeap<T, P> :: getSize(){
    return size;
}

template <class T, typename P >
__device__ void SLHeap<T, P>::swap(int child, int parent) {
    T temp;
    temp = list[child];
    list[child] = list[parent];
    list[parent] = temp;
}

template <class T, typename P >
__device__ int SLHeap<T, P> :: getParent(int child) {
    if (child % 2 == 0)
        return (child /2 ) -1;
    else 
        return child/2;
}

template <class T, typename P >
__device__ int SLHeap<T, P> :: getLeftChild(int parent){
    return 2*parent +1;
}

template <class T, typename P >
__device__ int SLHeap<T, P> :: getRightChild(int parent){
    return 2 * parent + 2;
}

template <class T, typename P >
__device__ void SLHeap<T, P> :: insert(T value) {
    lock(&mutex);
    list[size++] = value;
    bubbleUp();
    unlock(&mutex);
}

template <class T, typename P >
__device__ void SLHeap <T, P>:: bubbleUp(P comp) {
    int child = size - 1;
    int parent = getParent(child);
    while (comp(list[child], list[parent]) && child >=0 && parent >= 0) {
        swap(child, parent);
        child = parent;
        parent = getParent(child);
    }
}

template <class T, typename P >
__device__ T SLHeap<T, P> :: remove() {
    lock(&mutex);
    int child = size - 1;
    swap(child, 0);
    T value = list[size-1];
    size--;
    bubbleDown();
    unlock(&mutex);
    return value;
}

template <class T, typename P >
__device__ void SLHeap<T, P> :: bubbleDown(P comp) {
    int parent = 0;
    while (1) {
        int left = getLeftChild(parent);
        int right = getRightChild(parent);
        int length = size;
        int largest = parent;

        if (left < length && comp(list[left], list[largest]))
            largest = left;

        if (right < length && comp(list[right], list[largest]))
            largest = right;

        if (largest != parent) {
            swap(largest, parent);
            parent = largest;
        }
        else 
            break;
    }
}

template <class T>
SLHeap<T> *createSLHeap() {
    SLHeap<T> *heap_h, *heap_d;
    heap_h = (SLHeap<T> *)malloc(sizeof(SLHeap<T>));
    cudaMalloc(&heap_d, sizeof(SLHeap<T>));
    cudaMemcpy(heap_d, heap_h, sizeof(SLHeap<T>), cudaMemcpyHostToDevice);
    return heap_d;
}

template <class T >
__global__ void Init(SLHeap<T> *heap, T l) {
    heap->init(l);
}


template <class T >
void initSLHeap(SLHeap<T> *heap, int l) {
    Init<<<1, 1>>>(heap, l);
    cudaThreadSynchronize();
}

template <class T, typename P=less<T> >
class Heap {
    T *list;
    int size;
    int max_size;
    int mutex_s;
    int *mutex;
    long int *tag;

    __device__ void bubbleUp(int pos, long tid, P comp=P());
    __device__ void bubbleDown(P comp=P());
    __device__ void swap(int child, int parent);
    __device__ int getLeftChild(int parent);
    __device__ int getRightChild(int parent);
    __device__ int getParent(int child);

    public:
        __device__ void init(int s);
        __device__ void insert(T);
        __device__ T remove();
        __device__ int getSize();
};

template <class T, typename P >
__device__ void Heap<T, P> :: init(int s){
   list = (T *)malloc(s*sizeof(T));
   mutex = (int *)malloc(s*sizeof(int));
   tag = (long int *)malloc(s*sizeof(long int));
   max_size = s;
   memset(list, 0, sizeof(list));
   memset(mutex, 0, sizeof(mutex));
   memset(tag, EMPTY, sizeof(tag));
   mutex_s = 0;
   size = 0;
}

template <class T, typename P >
__device__ int Heap<T, P> :: getSize(){
    lock(&mutex_s);
    int s = size;
    unlock(&mutex_s);
    return s;
}

template <class T, typename P >
__device__ void Heap<T, P>::swap(int child, int parent) {
    T temp;
    temp = list[child];
    list[child] = list[parent];
    list[parent] = temp;
    int t;
    t = tag[child];
    tag[child] = tag[parent];
    tag[parent] = t;
}

template <class T, typename P >
__device__ int Heap<T, P> :: getParent(int child) {
    if (child % 2 == 0)
        return (child /2 ) -1;
    else 
        return child/2;
}

template <class T, typename P >
__device__ int Heap<T, P> :: getLeftChild(int parent){
    return 2*parent +1;
}

template <class T, typename P >
__device__ int Heap<T, P> :: getRightChild(int parent){
    return 2 * parent + 2;
}

template <class T, typename P >
__device__ void Heap<T, P> :: insert(T value) {
    lock(&mutex_s);
    size++;
    int last = size - 1;
    lock(&mutex[last]);
    unlock(&mutex_s);
    list[last] = value;
    if(last > 0)
        tag[last] = getThreadID();
    else
        tag[last] = AVAILABLE;
    unlock(&mutex[last]);
    bubbleUp(last, tag[last]);
}

template <class T, typename P >
__device__ void Heap <T, P>:: bubbleUp(int pos, long tid, P comp) {
    int child = pos;
    int parent = getParent(child), old_child = child;
    while(child > 0) {
        lock(&mutex[parent]);
        lock(&mutex[child]);
        old_child = child;
        if(tag[parent] == AVAILABLE && tag[child] == tid) {
            if(comp(list[child], list[parent])) {
                swap(child, parent);
                child = parent;
            }
            else {
                tag[child] = AVAILABLE;
                child = 0;
            }
        }
        else if(tag[parent] == EMPTY) {
            child = 0;
        }
        else if(tag[child] != tid) {
            child = parent;
        }
        unlock(&mutex[old_child]);
        unlock(&mutex[parent]);
        parent = getParent(child);
    }
    if(child == 0) {
        lock(&mutex[child]);
        if(tag[child] == tid) {
            tag[child] = AVAILABLE;
        }
        unlock(&mutex[child]);
    }
}

template <class T, typename P >
__device__ T Heap<T, P> :: remove() {
    lock(&mutex_s);
    if(size == 0) {
        unlock(&mutex_s);
        return list[0];
    }
    int bottom = size - 1;
    size--;

    lock(&mutex[bottom]);
    unlock(&mutex_s);
    T bottom_val = list[bottom];
    tag[bottom] = EMPTY;
    unlock(&mutex[bottom]);

    lock(&mutex[0]);
    T value = list[0];
    if(tag[0] == EMPTY) {
        unlock(&mutex[0]);
        return value;
    }

    list[0] = bottom_val;
    tag[0] = AVAILABLE;

    bubbleDown();
    return value;
}

template <class T, typename P >
__device__ void Heap<T, P> :: bubbleDown(P comp) {
    int parent = 0;
    while (parent < max_size/2 - 1) {
        int left = getLeftChild(parent);
        lock(&mutex[left]);
        int right = getRightChild(parent);
        lock(&mutex[right]);
        int largest = parent;

        if(tag[left] == EMPTY) {
            unlock(&mutex[left]);
            unlock(&mutex[right]);
            break;
        }
        else if(tag[right] == EMPTY || comp(list[left], list[right])) {
            unlock(&mutex[right]);
            largest = left;
        }
        else {
            unlock(&mutex[left]);
            largest = right;
        }
        if (comp(list[largest], list[parent])) {
            swap(largest, parent);
            unlock(&mutex[parent]);
            parent = largest;
        }
        else {
            unlock(&mutex[largest]);
            break;
        }
    }
    unlock(&mutex[parent]);
}

template <class T>
Heap<T> *createHeap() {
    Heap<T> *heap_h, *heap_d;
    heap_h = (Heap<T> *)malloc(sizeof(Heap<T>));
    cudaMalloc(&heap_d, sizeof(Heap<T>));
    cudaMemcpy(heap_d, heap_h, sizeof(Heap<T>), cudaMemcpyHostToDevice);
    return heap_d;
}

template<class T>
__global__ void Init(Heap<T> *heap, T l) {
    heap->init(l);
}


template <class T >
void initHeap(Heap<T> *heap, int l) {
    Init<T><<<1, 1>>>(heap, l);
    cudaThreadSynchronize();
}
