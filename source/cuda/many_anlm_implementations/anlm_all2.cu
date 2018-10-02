/**
 * anlm_all.cu
 *
 * Version: 0.1
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include "DMat.hpp"

#define BLOCK_SIZE 256


namespace cuda
{

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
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
std::vector<T>
calculateGaussianFilter(int m, int n, T sigma);

template<class T>
__global__ void
cudaAnlmKernel(DMatExpanded<T> dSrc, DMat<T> dDst, DMat<int> dIds,
               DMat<T> dFilterSigma, DMat<T> dPatchBlur,
               int patchH, int patchW);


template <class T>
void
adaptiveNonLocalMeans(T *src, T *dst, int *ids, T *filterSigma,
                     int imgH, int imgW, int patchH, int patchW,
                     T patchSigma)
{
    std::cout << "Entering anlm" << std::endl;
    std::cout << "imgH:" << imgH << " imgW: " << imgW << " patchH:" << patchH
              << " patchW:" << patchW << " patchSigma:" << patchSigma << std::endl;

    // Create matrices on device.
    DMatExpanded<T> dSrc(src, imgW, imgH, patchW, patchH);
    DMat<T> dDst(imgW, imgH);
    DMat<int> dIds(ids, imgW, imgH);
    DMat<T> dFilterSigma(filterSigma, imgW, imgH);
    DMat<T> dPatchBlur(patchW, patchH);

    std::cout << "Matrices created" << std::endl;

    int gridW = imgW / 32;
    if ((imgW % 32) > 0) gridW++;
    int gridH = imgH / (BLOCK_SIZE / 32);
    if ((imgH % (BLOCK_SIZE / 32)) > 0) gridH++;
    dim3 blockDim(32, BLOCK_SIZE / 32);
    dim3 gridDim(gridW, gridH);

    std::cout << "Block size calculated" << std::endl;

    // While computing search areas on GPU, calculate a gaussian filter on CPU.
    std::vector<T> patchBlur = calculateGaussianFilter<T>(
            patchH, patchW, patchSigma
    );
    dPatchBlur.copyFromHost(patchBlur.data(), patchW, patchH);
    cudaDeviceSynchronize();

    std::cout << "Computed gauss blur" << std::endl;

    cudaError_t cudaStat = cudaFuncSetCacheConfig(
            cudaAnlmKernel<T>, cudaFuncCachePreferL1);
    assert(cudaSuccess == cudaStat);
    // Apply anlm to each pixel separately.
    cudaAnlmKernel<<<gridDim, blockDim>>>(
            dSrc, dDst, dIds, dFilterSigma, dPatchBlur, patchH, patchW
    );
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();

    std::cout << "Computed anlm" << std::endl;

    dDst.copyToHost(dst);

    std::cout << "Returning anlm" << std::endl;
}

template<class T>
__global__ void
cudaAnlmKernel(DMatExpanded<T> dSrc, DMat<T> dDst, DMat<int> dIds,
               DMat<T> dFilterSigma, DMat<T> dPatchBlur,
               int patchH, int patchW)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dSrc.width || y >= dSrc.height) return;

    const int boundH = (patchH - 1) / 2;
    const int boundW = (patchW - 1) / 2;
    const T fSigma = dFilterSigma(y, x) * dFilterSigma(y, x);
    const T id = dIds(y, x);
    const int width = dSrc.width;
    const int height = dSrc.height;

    T nom = 0;
    T denom = 0;
    T maxWeight = 0;

    // if (x == 0 && y == 0) {
    //     for (int i = 0; i < dNext.height; ++i) {
    //         for (int j = 0; j < dNext.width; ++j) {
    //             DMatPos n = dNext(i, j);
    //             DMatPos p = dPrev(i, j);
    //             printf("%d,%d: %d,%d %d,%d\n", i, j, p.y, p.x, n.y, n.x);
    //         }
    //     }
    // }

    // if (x == 0 && y == 0) {
    //     for (int i = 0; i < dFilterSigma.width; ++i) {
    //         for (int j = 0; j < dFilterSigma.height; ++j) {
    //             printf("%f ", dFilterSigma(i, j));
    //         }
    //         printf("\n");
    //     }
    // }

    // int l = 0;

    // Iterate over all pixels in the region of (y, x) pixel.
    // for (int l = 0; l < 5000; ++l) {
    for (int p = 0; p < width*y+x; ++p) {
        int pI = p / width;
        int pJ = p % width;

        if (id != dIds(pI, pJ)) continue;
        // if (x == 0 && y == 0) {
        //     printf("%d cur: %d,%d\n", l, curCell.y, curCell.x);
        // }

        T weight = 0;

        for (int i = -boundH; i < boundH+1; ++i) {
            for (int j = -boundW; j < boundW+1; ++j) {
                // if (x == 0 && y == 0)
                //     printf("%d %d %f %f  ", i, j, dSrc(y+i, x+j)*dPatchBlur(boundH+i, boundW+j), dSrc(curCell.y+i, curCell.x+j)*dPatchBlur(boundH+i, boundW+j));

                T d = dSrc(y+i, x+j) * dPatchBlur(boundH+i, boundW+j) -
                      dSrc(pI+i, pJ+j) * dPatchBlur(boundH+i, boundW+j);
                weight += d*d;

                // if (x== 0 && y == 0)
                //     printf("%f %f |", d*d, weight);
            }
            // if (x == 0 && y == 0) printf("\n");
        }

        // if (y == 0 && x == 0) {
        //     printf("%d: %.10f - %f\n", l, -weight/fSigma, fSigma);
        //     ++l;
        // }

        weight = exp(-weight/fSigma);
        nom += dSrc(pI, pJ) * weight;
        denom += weight;
        if (weight > maxWeight) maxWeight = weight;
    }

    for (int p = width*y+x+1; p < width*height; ++p) {
        int pI = p / width;
        int pJ = p % width;

        if (id != dIds(pI, pJ)) continue;
        // if (x == 0 && y == 0) {
        //     printf("%d cur: %d,%d\n", l, curCell.y, curCell.x);
        // }

        T weight = 0;

        for (int i = -boundH; i < boundH+1; ++i) {
            for (int j = -boundW; j < boundW+1; ++j) {
                // if (x == 0 && y == 0)
                //     printf("%d %d %f %f  ", i, j, dSrc(y+i, x+j)*dPatchBlur(boundH+i, boundW+j), dSrc(curCell.y+i, curCell.x+j)*dPatchBlur(boundH+i, boundW+j));

                T d = dSrc(y+i, x+j) * dPatchBlur(boundH+i, boundW+j) -
                      dSrc(pI+i, pJ+j) * dPatchBlur(boundH+i, boundW+j);
                weight += d*d;

                // if (x== 0 && y == 0)
                //     printf("%f %f |", d*d, weight);
            }
            // if (x == 0 && y == 0) printf("\n");
        }

        // if (y == 0 && x == 0) {
        //     printf("%d: %.10f - %f\n", l, -weight/fSigma, fSigma);
        //     ++l;
        // }

        weight = exp(-weight/fSigma);
        nom += dSrc(pI, pJ) * weight;
        denom += weight;
        if (weight > maxWeight) maxWeight = weight;
    }

    if (maxWeight < pow(2.0, -52.0)) maxWeight = pow(2.0, -52.0);

    // if (y == 0 && x == 0)
    //     printf("Max weight: %.20f\n", maxWeight);

    // Calculate the weight with itself.
    nom += dSrc(y, x) * maxWeight;
    denom += maxWeight;

    // if (y == 0 && x == 0)
    //     printf("Orig: %.10f Result: %.10f - Denom: %f\n", dSrc(y, x), nom / denom, denom);

    if (denom != 0) dDst(y, x) = nom / denom;
    else dDst(y, x) == dSrc(y, x);
}

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


template
void
adaptiveNonLocalMeans<float> (float *, float *, int *, float *,
                              int, int, int, int, float);
template
void
adaptiveNonLocalMeans<double>(double *, double *, int *, double *,
                              int, int, int, int, double);

}
