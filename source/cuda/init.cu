/**
 * init.cu
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * Code that will trigger loading of CUDA libraries.
 *
 * Version: 0.1
 */

#include <cassert>


namespace cuda
{

__global__ void
initKernel(float *devTable)
{
    float lovelyAcc = 0;
    for (int i = 0; i < 4; ++i) lovelyAcc += *(devTable + sizeof(float) * i);
    lovelyAcc = pow(lovelyAcc, 2.0);
}

void
deviceInit()
{
    float someData[] = { 1.0, 3.0, 5.0, 10.0 };

    float *devTable;

    cudaError_t cudaStat = cudaMalloc(&devTable, sizeof(float) * 4);
    assert(cudaSuccess == cudaStat);
    cudaStat = cudaMemcpy(devTable, someData, 4, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);

    initKernel<<<8, 32>>>(devTable);

    cudaStat = cudaFree(devTable);
    assert(cudaSuccess == cudaStat);
}

}
