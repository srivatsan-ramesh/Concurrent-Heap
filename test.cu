#include <stdio.h>
#include <cuda.h>

#include <algorithm>
#include <cstdlib> 
#include <ctime>

#include "heap.cu"

#define N 100000
#define B 10
#define S 320

using namespace std;

template <class T >
__global__ void ParallelInsert(T *heap, int *a, int n) {
    int id = getThreadID();
    if(id % 32 == 0)
        for(int i = 0; i < n*32/(B*S); i++)
            heap->insert(a[id/32 + (B*S/32)*i]);
}

template <class T >
__global__ void ParallelRemove(T *heap, int n) {
    int id = getThreadID();
    if(id % 32 == 0)
        for(int i = 0; i < n*32/(B*S); i++)
            heap->remove();
}

template <class T >
__global__ void ParallelInsertAndRemove(T *heap, int *a, int n) {
    int id = getThreadID();
    if(id % 32 == 0)
        for(int i = 0; i < n*32/(B*S); i++) {
            if(i % 2)
                heap->insert(a[id/32 + (B*S/32)*i]);
            else
                heap->remove();
        }    
}

template <class T >
__global__ void Remove(T *heap, int *a, int *size) {
    int i = 0;
    while(heap->getSize() > 0)
        a[i++] = heap->remove();
    *size = i;
}

int isSorted(int *a, int *n) {
    for(int i=0;i<(*n)-1;i++) {
        if(a[i] > a[i+1]) return 0;
    }
    return 1;
}

int cmpArr(int *a, int *b, int n) {
    for(int i=0;i<n;i++) {
        if(a[i] != b[i]) return 0;
    }
    return 1;
}

int main(){
    srand((unsigned)time(0));
    int a[N], a_temp[N];
    for(int i=0; i<N; i++) {
        a[i] = (rand()%100)+1;
        a_temp[i] = a[i];
    }
    int b[N], c[N];
    int *da;
    int size[1], *dsize;

    cudaMalloc(&dsize, sizeof(int));

    // Insertion test for Single Lock Heap
    SLHeap<int> *heap1 = createSLHeap<int>();
    initSLHeap(heap1, N);

    cudaMalloc(&da, N*sizeof(int));
    cudaMemcpy(da, a, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);

    ParallelInsert<SLHeap<int> ><<<B, S>>>(heap1, da, N);
    cudaEventRecord(stop1, 0);
    cudaThreadSynchronize();

    float elapsedtime;
    cudaEventElapsedTime(&elapsedtime, start1, stop1);
    printf("Insertion - Single Lock Time = %f ms\n", elapsedtime);

    Remove<SLHeap<int> ><<<1,1>>>(heap1, da, dsize);
    cudaThreadSynchronize();
    cudaMemcpy(b, da, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Insertion test for Node Lock heap
    Heap<int> *heap2 = createHeap<int>();
    initHeap(heap2, N);

    cudaMemcpy(da, a, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    ParallelInsert<Heap<int> ><<<B, S>>>(heap2, da, N);
    cudaEventRecord(stop2, 0);
    cudaThreadSynchronize();

    cudaEventElapsedTime(&elapsedtime, start2, stop2);
    printf("Insertion - Node Lock Time = %f ms\n", elapsedtime);

    Remove<Heap<int> ><<<1,1>>>(heap2, da, dsize);
    cudaThreadSynchronize();
    cudaMemcpy(c, da, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // Print Insertion test results
    sort(a_temp, a_temp+N);
    printf("Insertion - Single Lock : %s\n", (cmpArr(a_temp, b, N)?"Success":"Failure"));
    printf("Insertion - Node Lock : %s\n", (cmpArr(a_temp, c, N)?"Success":"Failure"));

    // Deletion test for Single Lock Heap
    initSLHeap(heap1, N);

    cudaMalloc(&da, N*sizeof(int));
    cudaMemcpy(da, a, N * sizeof(int), cudaMemcpyHostToDevice);
    ParallelInsert<SLHeap<int> ><<<B, S>>>(heap1, da, N);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);

    ParallelRemove<SLHeap<int> ><<<B, S>>>(heap1, N);
    cudaEventRecord(stop1, 0);
    cudaThreadSynchronize();

    cudaEventElapsedTime(&elapsedtime, start1, stop1);
    printf("Deletion - Single Lock Time = %f ms\n", elapsedtime);

    // Deletion test for Node Lock heap
    initHeap(heap2, N);

    ParallelInsert<Heap<int> ><<<B, S>>>(heap2, da, N);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    ParallelRemove<Heap<int> ><<<B, S>>>(heap2, N);
    cudaEventRecord(stop2, 0);
    cudaThreadSynchronize();

    cudaEventElapsedTime(&elapsedtime, start2, stop2);
    printf("Deletion - Node Lock Time = %f ms\n", elapsedtime);

    // Deletion stability chefck
    initHeap(heap2, N);
    ParallelInsert<Heap<int> ><<<B, S>>>(heap2, da, N);
    ParallelRemove<Heap<int> ><<<B, S>>>(heap2, N/2);

    cudaThreadSynchronize();
    Remove<Heap<int> ><<<1,1>>>(heap2, da, dsize);
    cudaMemcpy(c, da, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(size, dsize, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Deletion Stability Check : %s\n", (isSorted(c, size)?"Success":"Failure"));
    printf("Size of final array = %d\n", *size);
    return 0;
}
