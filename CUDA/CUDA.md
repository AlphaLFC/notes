# CUDA

## Concepts

1. A GPU consists of $U$ multiprocessors, and each multiprocessor can
   run $T$ threads.
1. Threads are organized in blocks.  Blocks are indivisibly assigned to
   multiprocessors during scheduling.
1. What we need to know about scheduling are

   1. gridDim: number of blocks
   1. blockDim: number of threeads per block
   1. blockIdx: index of a block in the grid
   1. threadIdx: index of a thread in a block

1. Above Dim's and Idx's can be represented by either `unsigned int`
   or CUDA native type `dim3`.  Consider `dim3` a C struct with three
   `unsigned int` fields `x`, `y`, and `z`.

1. `__global__` keyword in CUDA indicates a function as a *kernels*,
   the entry point to programs executed by GPU threads.

1. CUDA operator `<<<` and `>>>` runs kernels in blocks. For example:

        dim3 gridDim(2,2)
        dim3 blockDim(64,64)
		kernel<<<gridDim, blockDim>>>(parameter)

1. `__host__` indicates a function is going to run on host (CPU).
   `__device__` indicates a function is to run on the device (GPU).

## Tutorial

This is a great
[tutorial](https://code.google.com/p/stanford-cs193g-sp2010/wiki/GettingStartedWithCUDA).

## Questions:

1. Could a kernel change the value of a global variable?
