# Optimizing SGEMM from Scratch in CUDA

Modern AI is powered not only by better models, but by how efficiently we can execute them on hardware. At companies like NVIDIA, AMD, Google, Intel, OpenAI and Anthropic, GPU performance engineering sits at the intersection of **hardware, compilers, kernels and large-scale ML systems**. Small improvements in kernel efficiency can translate into faster training, lower inference latency and significantly better hardware utilization.

In this project, I’ll iteratively optimize matrix multiplication written in CUDA, not to replace cuBLAS, but to understand *why* high-performance GPU kernels are fast. Along the way, I’ll explore concepts such as **Global memory coalescing, shared-memory tiling, register reuse, thread/block tiling, arithmetic intensity, occupancy, and instruction-level parallelism**.

Matrix multiplication is an ideal place to learn these ideas because it sits at the heart of modern deep learning. A huge portion of the computation in transformers and other neural networks eventually reduces to matrix multiplications.

Here, we will focus on **SGEMM** : single-precision(FP32) general matrix multiplication:

`C = αAB + βC`

I’ll start with the CPU implementation to understand why mat-mul operations are done on GPUs and then move to simplest possible CUDA implementation and progressively optimize it, measuring performance after every major change and comparing the results against NVIDIA’s highly optimized **cuBLAS** implementation.

The interesting question is not simply:

**“Can we make matrix multiplication fast?”**

It is:

**“What exactly does the GPU need from us to make it fast?”**
