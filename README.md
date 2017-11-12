# Concurrent Heap

## Usage
### On the Host side
```cpp
Heap<int> *heap = createHeap<int>();
initHeap(heap, size);
```
### On the Device side
```cpp
__global__ void ParallelInsert(Heap<int> *heap, int *a, int n) {
    int id = getThreadID();
    if(id % 32 == 0)
            heap->insert(a[id/32]);
}

__global__ void ParallelRemove(Heap<int> *heap) {
    int id = getThreadID();
    if(id % 32 == 0)
            heap->remove(); // Returns the topmost element of the heap
}
```

## Limitation
More than one thread of a warp can not insert or remove elements parallely from the heap, it will get into a deadlock situation.

## TODO
Support for custom comparators.
