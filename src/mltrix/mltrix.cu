#include "kernel.cuh"
#include <cuda_runtime.h>
#define E 2.71828182845904523536
#include <math.h>
#include "stdio.h"

__device__ double sigmoidOp(double a)
{
    return pow(E, a) / (pow(E, a) + 1);
}

__global__ void kernelSigmoid(double *numberToTransform)
{
    *numberToTransform = sigmoidOp(*numberToTransform);
}

void mltrix::sigmoid(double *numberToTransform)
{
    double *a_d;
    cudaMallocManaged(&a_d, 1 * sizeof(double));
    cudaMemcpy(a_d, numberToTransform, 1 * sizeof(double), cudaMemcpyHostToDevice);
    kernelSigmoid<<<1, 1>>>(a_d);
    cudaDeviceSynchronize();
    *numberToTransform = *a_d;
    cudaFree(a_d);
}
