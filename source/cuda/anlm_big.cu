/**
 * anlm_big.cu
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * A CUDA implementation of Adaptive Non-Local Means algorithm with moderate
 * GPU DRAM consumption. It can be used for relative big images, as long as
 * GPU is able to hold 5-6 copies of the image in its memory.
 *
 * The logic behind this implementation is the creation of a linked list like
 * structure for the pixels of each region in the image. That is done in place,
 * so memory consumption stays low. The downside is that memory accessing by
 * each CUDA thread is unpredictable and depends on given data. So, coalescion
 * can be pretty bad and L1 cache misses amount can also be really high.
 *
 * Version: 0.1
 *
 * License: GNU GPL v3 (see project's license).
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <cuda_profiler_api.h>
#include "DMat.hpp"


#define BLOCK_SIZE 256
// #define MIN_BLOCKS_PER_SM 4

namespace cuda
{

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <class T>
__global__ void
cudaFindNextEqualKernel(DMat<int> dIds, DMat<DMatPos> dNext, DMat<DMatPos> dPrev);

template <class T>
__global__ void
cudaSetPrevEqualKernel(DMat<DMatPos> dNext, DMat<DMatPos> dPrev);

template <class T>
__global__ void
cudaFindRegionHeads(DMat<int> dIds, DMat<DMatPos> dPrev, DMat<DMatPos> dHeads);

template <class T>
std::vector<T>
calculateGaussianFilter(int m, int n, T sigma);

template<class T>
__global__ void
cudaAnlmKernel(DMatExpanded<T> dSrc, DMat<T> dDst, DMat<int> dIds,
               DMat<DMatPos> dNext, DMat<DMatPos> dPrev, DMat<DMatPos> dHeads,
               DMat<T> dFilterSigma, DMat<T> dPatchBlur, int patchH, int patchW);

template <class T> __device__ inline T __anlm_exp(T x) { return (T) exp(x); }
template <> __device__ inline float __anlm_exp<float>(float x) { return expf(x); }
template <class T> __device__ inline T __anlm_pow(T x, T y) { return (T) pow(x, y); }
template <> __device__ inline float __anlm_pow(float x, float y) { return powf(x, y); }


template <class T>
void
adaptiveNonLocalMeansBigData(T *src, T *dst, int *ids, T *filterSigma,
                             int imgH, int imgW, int patchH, int patchW,
                             T patchSigma, int regions)
{
    cudaProfilerStart();

    // Create matrices on device.
    DMatExpanded<T> dSrc(src, imgW, imgH, patchW, patchH);
    DMat<T>         dDst(imgW, imgH);
    DMat<int>       dIds(ids, imgW, imgH);
    DMat<DMatPos>   dNext(imgW, imgH);
    DMat<DMatPos>   dPrev(imgW, imgH);
    DMat<DMatPos>   dHeads(1, 6);
    DMat<T>         dFilterSigma(filterSigma, imgW, imgH);
    DMat<T>         dPatchBlur(patchW, patchH);

    int gridW = imgW / 32;
    if ((imgW % 32) > 0) gridW++;
    int gridH = imgH / (BLOCK_SIZE / 32);
    if ((imgH % (BLOCK_SIZE / 32)) > 0) gridH++;
    dim3 blockDim(32, BLOCK_SIZE / 32);
    dim3 gridDim(gridW, gridH);

    // Precompute the pixels belonging to each search area.
    cudaFindNextEqualKernel<T><<<gridDim, blockDim>>>(dIds, dNext, dPrev);
    cudaSetPrevEqualKernel<T><<<gridDim, blockDim>>>(dNext, dPrev);
    cudaFindRegionHeads<T><<<gridDim, blockDim>>>(dIds, dPrev, dHeads);

    // While computing search areas on GPU, calculate a gaussian filter on CPU.
    std::vector<T> patchBlur = calculateGaussianFilter<T>(
            patchH, patchW, patchSigma
    );
    dPatchBlur.copyFromHost(patchBlur.data(), patchW, patchH);
    cudaDeviceSynchronize();

    cudaError_t cudaStat = cudaFuncSetCacheConfig(
            cudaAnlmKernel<T>, cudaFuncCachePreferL1);
    assert(cudaSuccess == cudaStat);

    // Apply anlm to each pixel separately.
    cudaAnlmKernel<<<gridDim, blockDim>>>(
            dSrc, dDst, dIds, dNext, dPrev, dHeads, dFilterSigma, dPatchBlur, patchH, patchW
    );
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    dDst.copyToHost(dst);

    cudaProfilerStop();
}

template<class T>
__global__ void
__launch_bounds__(BLOCK_SIZE)
cudaAnlmKernel(DMatExpanded<T> dSrc, DMat<T> dDst, DMat<int> dIds,
               DMat<DMatPos> dNext, DMat<DMatPos> dPrev, DMat<DMatPos> dHeads,
               DMat<T> dFilterSigma, DMat<T> dPatchBlur, int patchH, int patchW)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dSrc.width || y >= dSrc.height) return;

    const int boundH = (patchH - 1) / 2;
    const int boundW = (patchW - 1) / 2;
    const T fSigma   = dFilterSigma(y, x) * dFilterSigma(y, x);

    __shared__ T patchBlur[5][5];

    if (threadIdx.y < 5 && threadIdx.x < 5)
        patchBlur[threadIdx.y][threadIdx.x] = dPatchBlur(threadIdx.y, threadIdx.x);
    __syncthreads();

    T nom = 0;
    T denom = 0;
    T maxWeight = 0;

    DMatPos curCell = dHeads(0, dIds(y, x));

    // Iterate over all pixels in the region of (y, x) pixel.
    while (curCell.x != -1) {

        T weight = 0;

        for (int i = -boundH; i < boundH+1; ++i) {
            for (int j = -boundW; j < boundW+1; ++j) {
                T d = dSrc(y+i, x+j) * patchBlur[boundH+i][boundW+j] -
                      dSrc(curCell.y+i, curCell.x+j) * patchBlur[boundH+i][boundW+j];
                weight += d*d;
            }
        }

        weight = __anlm_exp<T>(-weight/fSigma);

        if (!(curCell.y == y && curCell.x == x)) {
            nom += dSrc.at(curCell) * weight;
            denom += weight;
        }

        if (weight > maxWeight && !(curCell.x == x && curCell.y == y)) maxWeight = weight;

        curCell = dNext.at(curCell);
    }

    if (maxWeight < __anlm_pow<T>(2.0, -52.0)) maxWeight = __anlm_pow<T>(2.0, -52.0);

    nom   += dSrc(y, x) * maxWeight;
    denom += maxWeight;

    if (denom != 0) dDst(y, x) = nom / denom;
    else dDst(y, x) = dSrc(y, x);
}

/**
 * A kernel that finds the next element that belongs to the same region. It
 * actually creates a forward linkage between each element of regions.
 *
 * Also initializes backward linkage matrix.
 */
template <class T>
__global__ void
cudaFindNextEqualKernel(DMat<int> dIds, DMat<DMatPos> dNext, DMat<DMatPos> dPrev)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dIds.width || y >= dIds.height) return;

    T self = dIds(y, x);
    DMatPos nextPos(-1, -1);

    // Search the remnants of current row.
    for (int j = x+1; j < dIds.width; ++j) {
        if (dIds(y, j) == self) {
            nextPos.x = j;
            nextPos.y = y;
            goto eqFound;
        }
    }

    // Search all next rows.
    for (int i = y+1; i < dIds.height; ++i) {
        for (int j = 0; j < dIds.width; ++j) {
            if (dIds(i, j) == self) {
                nextPos.x = j;
                nextPos.y = i;
                goto eqFound;
            }
        }
    }
eqFound:
    dNext(y, x) = nextPos;
    // Initialize all previous positions.
    dPrev(y, x) = DMatPos(-1, -1);
}

/**
 * Creates a backwards linkage between elements of each region, expecting
 * an already formed forward linkage.
 */
template <class T>
__global__ void
cudaSetPrevEqualKernel(DMat<DMatPos> dNext, DMat<DMatPos> dPrev)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dNext.width || y >= dNext.height) return;
    DMatPos next = dNext(y, x);
    if (next.y != -1 && next.x != -1) dPrev.at(next) = DMatPos(y, x);
}

/**
 * Finds the root pixels of the linked list like structure of each region.
 */
template <class T>
__global__ void
cudaFindRegionHeads(DMat<int> dIds, DMat<DMatPos> dPrev, DMat<DMatPos> dHeads)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dPrev.width || y >= dPrev.height) return;
    if (dPrev(y, x).x == -1) dHeads(0, dIds(y, x)) = DMatPos(y, x);
}

/**
 * Calculates a gaussian filter matrix.
 *
 * Parameters:
 *  -m : Height of filter matrix.
 *  -n : Width of filter matrix.
 *  -sigma : Sigma to be used for gaussian filter computation.
 *
 * Returns:
 *  A vector containing the gaussian filter in row major order.
 */
template <class T>
std::vector<T>
calculateGaussianFilter(int m, int n, T sigma)
{
    std::vector<T> filter(m*n);
    T sum = 0;
    T mean = (m - 1) / 2;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T a = (T) i - mean;
            T b = (T) j - mean;
            T val = exp(-(a*a + b*b) / (2*sigma*sigma));
            filter[i*m+j] = val;
            sum += val;
        }
    }

    // Normalize kernels.
    T max = filter[m*n/2] / sum;  // greatest kernel will always be on center
    for (int i = 0; i < m*n; ++i) filter[i] /= (sum * max);

    return filter;
}


// ----------- Declarations for pregenerating code by compiler -------------
template
void
adaptiveNonLocalMeansBigData<float> (float *, float *, int *, float *,
                              int, int, int, int, float, int);
template
void
adaptiveNonLocalMeansBigData<double>(double *, double *, int *, double *,
                                     int, int, int, int, double, int);

}
