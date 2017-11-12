#include <cuda.h>

__device__ void lock(int *mutex) {
	while (atomicCAS(mutex, 0, 1));
}

__device__ void unlock(int *mutex) {
	atomicExch(mutex, 0);
}

__device__ long getThreadID() {
    int blockId = blockIdx.x 
			 + blockIdx.y * gridDim.x 
             + gridDim.x * gridDim.y * blockIdx.z;

	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			  + (threadIdx.z * (blockDim.x * blockDim.y))
			  + (threadIdx.y * blockDim.x)
              + threadIdx.x;

    return threadId;
}

template<class T>
struct greater
{
    __device__ bool operator()(T a, T b)
    {
        return a > b;
    }
};

template<class T>
struct less
{
    __device__ bool operator()(T a, T b)
    {
        return a < b;
    }
};
