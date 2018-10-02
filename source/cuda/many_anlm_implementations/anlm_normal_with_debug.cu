/**
 * anlm_normal.cu
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * A CUDA implementation of Adaptive Non-Local Means(ANLM) algorithm that
 * trades-off memory for computation time. It is suitable for small and
 * moderate sized images.
 *
 * The GPU is expected to be able to hold at least patchH*patchW+8 copies of
 * the image.
 *
 * Version: 0.1
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda_profiler_api.h>
#include "DMat.hpp"


#define BLOCK_SIZE 256
// #define MIN_BLOCKS_PER_SM 6

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
std::vector<T>
calculateGaussianFilter(int m, int n, T sigma);

std::vector<int>
divideInRegions(int *ids, int imgH, int imgW, std::vector<int> &regionSizes);

template <class T>
void
fillRegionalMatrices(const DMatExpanded<T> &dSrc,
                     const DMat<T> &dFiltersSigma,
                     const DMat<int> &dIds,
                     std::vector<int> &regionalIndices,
                     std::vector<int> &regionSizes,
                     int patchH,
                     int patchW,
                     std::vector<DMat<T>> &rSrc,
                     std::vector<DMat<T>> &rPatches,
                     std::vector<DMat<T>> &rFiltersSigma,
                     std::vector<DMat<DMatPos>> &origPos);

template<class T>
__global__ void
cudaAnlmKernel(DMat<T> dRSrc, DMat<T> dRPatches, DMat<T> dDst,
               DMat<DMatPos> dOriginalPos, DMat<T> dRFiltersSigma,
               int patchH, int patchW);

template <class T>
__global__ void
cudaDivideRegions(DMatExpanded<T> dSrc,
                  DMat<T> dFiltersSigma,
                  DMat<int> dIds,
                  DMat<int> dRegionalIndices,
                  int patchH,
                  int patchW,
                  DMat<T*> dRSrc,
                  DMat<T*> dRPatches,
                  DMat<int> dRPatchesPitch,
                  DMat<T*> dRFiltersSigma,
                  DMat<DMatPos*> dOrigPos);

template <class T>
__global__ void
cudaApplyBlur(DMat<T> dRSrc, DMat<T> dPatchBlur);

template <class T> __device__ inline T __anlm_exp(T x) { return (T) exp(x); }
template <> __device__ inline float __anlm_exp<float>(float x) { return expf(x); }
template <class T> __device__ inline T __anlm_pow(T x, T y) { return (T) pow(x, y); }
template <> __device__ inline float __anlm_pow(float x, float y) { return powf(x, y); }

// template <class T>
// __global__ void
// printMatrices(DMat<T> patches, DMat<T> src, DMat<T> sigmas, DMat<DMatPos> pos)
// {
//     printf("===============Patches Matrix=============\n");
//     for (int i = 0; i < patches.width; ++i) {
//         for (int j = 0; j < patches.height; ++j) {
//             printf("%f ", patches(j, i));
//         }
//         printf("\n");
//     }
//
//     printf("===============Src Matrix=============\n");
//     for (int i = 0; i < src.width; ++i) {
//         printf("%f ", src(0, i));
//     }
//     printf("\n");
//
//     printf("===============Sigmas Matrix=============\n");
//     for (int i = 0; i < sigmas.width; ++i) {
//         printf("%f ", sigmas(0, i));
//     }
//     printf("\n");
//
//     printf("===============Pos Matrix=============\n");
//     for (int i = 0; i < sigmas.width; ++i) {
//         printf("(%d %d)", pos(0, i).y, pos(0, i).x);
//     }
//     printf("\n");
// }

template <class T>
void
adaptiveNonLocalMeansNormalData(T *src, T *dst, int *ids, T *filterSigma,
                                int imgH, int imgW, int patchH, int patchW,
                                T patchSigma, int regions)
{
    using std::vector;

    cudaProfilerStart();

    // std::cout << "Entering anlm" << std::endl;
    // std::cout << "imgH:" << imgH << " imgW: " << imgW << " patchH:" << patchH
    //           << " patchW:" << patchW << " patchSigma:" << patchSigma
    //           << " regions:" << regions << std::endl;

    // Create generic matrices on device.
    DMatExpanded<T> dSrc(src, imgW, imgH, patchW, patchH);
    DMat<T>         dDst(imgW, imgH);
    DMat<int>       dIds(ids, imgW, imgH);
    DMat<T>         dFilterSigma(filterSigma, imgW, imgH);
    DMat<T>         dPatchBlur(patchW, patchH);

    // std::cout << "Created generic matrices" << std::endl;

    // DEBUG - Print src matrix
    // std::cout << "Src matrix" << std::endl;
    // for (int i = 0; i < dSrc.height; ++i) {
    //     for (int j = 0; j < dSrc.width; ++j) {
    //         std::cout << src[i*dSrc.width+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // DEBUG - Print ids matrix
    // std::cout << "Ids matrix" << std::endl;
    // for (int i = 0; i < dSrc.height; ++i) {
    //     for (int j = 0; j < dSrc.width; ++j) {
    //         std::cout << ids[i*dSrc.width+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Find the index of each pixel in its own region and also get the size
    // of each region.
    vector<int> regionSizes(regions);
    vector<int> regionalIndices = divideInRegions(ids, imgH, imgW, regionSizes);

    // DEBUG - Print region sizes.
    // std::cout << "regionSizes" << std::endl;
    // for (int i = 0; i < regionSizes.size(); ++i) {
    //     std::cout << regionSizes[i] << " ";
    // }
    // std::cout << std::endl;
    // DEBUG - Print regionalIndices
    // std::cout << "regionalIndices" << std::endl;
    // for (int i = 0; i < dSrc.height; ++i) {
    //     for (int j = 0; j < dSrc.width; ++j) {
    //         std::cout << regionalIndices[i*dSrc.width+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Found regions of each pixel" << std::endl;

    // Create separate matrices for each region on device.
    vector<DMat<T>>       dRegionsPatches(regions);
    vector<DMat<T>>       dRegionsSrc(regions);
    vector<DMat<T>>       dRegionsFiltersSigma(regions);
    vector<DMat<DMatPos>> dOriginalPos(regions);
    for (int i = 0; i < regions; ++i) {
        // Pixels of each region horizontally, pixels of the patch vertically.
        dRegionsPatches[i]      = DMat<T>(regionSizes[i], patchH*patchW);
        dRegionsSrc[i]          = DMat<T>(regionSizes[i], 1);
        dRegionsFiltersSigma[i] = DMat<T>(regionSizes[i], 1);
        dOriginalPos[i]         = DMat<DMatPos>(regionSizes[i], 1);
    }

    // std::cout << "Created special matrices" << std::endl;

    fillRegionalMatrices<T>(dSrc, dFilterSigma, dIds, regionalIndices, regionSizes,
                            patchH, patchW, dRegionsSrc, dRegionsPatches,
                            dRegionsFiltersSigma, dOriginalPos
    );

    // std::cout << "Matrices divided and copied" << std::endl;

    // Create a separate stream for each region.
    vector<cudaStream_t> streams(regions);
    for (int i = 0; i < streams.size(); ++i) {
        cudaError_t cudaStatus = cudaStreamCreate(&streams[i]);
        assert(cudaSuccess == cudaStatus);
    }

    // std::cout << "Streams created" << std::endl;

    // Calculate block and grid dims for each region.
    std::vector<dim3> rBlockDims(regions);
    std::vector<dim3> rGridDims(regions);
    for (int i = 0; i < regions; ++i) {
        int gridW = regionSizes[i] / BLOCK_SIZE;
        if ((regionSizes[i] % BLOCK_SIZE) > 0) ++gridW;
        rBlockDims[i] = dim3(BLOCK_SIZE);
        rGridDims[i]  = dim3(gridW);
    }

    // std::cout << "Block size calculated" << std::endl;

    // Calculate and apply a gaussian filter.
    std::vector<T> patchBlur = calculateGaussianFilter<T>(
            patchH, patchW, patchSigma);
    dPatchBlur.copyFromHost(patchBlur.data(), patchW, patchH);
    for (int i = 0; i < regions; ++i) {
        cudaApplyBlur<<<rGridDims[i], rBlockDims[i], 0, streams[i]>>>(
            dRegionsPatches[i], dPatchBlur);
    }

    // std::cout << "Applied gauss blur" << std::endl;

    // for (int i = 0; i < regions; ++i) {
    //     printMatrices<<<1, 1>>>(dRegionsPatches[i], dRegionsSrc[i], dRegionsFiltersSigma[i], dOriginalPos[i]);
    // }

    cudaError_t cudaStat = cudaFuncSetCacheConfig(
            cudaAnlmKernel<T>, cudaFuncCachePreferL1);
    assert(cudaSuccess == cudaStat);

    // Apply anlm to each region separately.
    for (int i = 0; i < regions; ++i) {
        cudaAnlmKernel<<<rGridDims[i], rBlockDims[i], 0, streams[i]>>>(
                dRegionsSrc[i], dRegionsPatches[i], dDst, dOriginalPos[i],
                dRegionsFiltersSigma[i], patchH, patchW
        );
        // gpuErrchk( cudaPeekAtLastError() );
    }

    // std::cout << "Computed anlm" << std::endl;

    dDst.copyToHost(dst);

    for (int i = 0; i < streams.size(); ++i) cudaStreamDestroy(streams[i]);

    // DEBUG - Print dst matrix
    // std::cout << "Dst matrix" << std::endl;
    // for (int i = 0; i < dSrc.height; ++i) {
    //     for (int j = 0; j < dSrc.width; ++j) {
    //         std::cout << dst[i*dSrc.width+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "Returning anlm" << std::endl;

    cudaProfilerStop();
}

template<class T>
__global__ void
// __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)
cudaAnlmKernel(DMat<T> dRSrc, DMat<T> dRPatches, DMat<T> dDst,
               DMat<DMatPos> dOriginalPos, DMat<T> dRFiltersSigma,
               int patchH, int patchW)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    const int patchSize  = patchH * patchW;
    const int regionSize = dRPatches.width;
    const T fSigma       = dRFiltersSigma(0, x) * dRFiltersSigma(0, x);

    T nom = 0;
    T denom = 0;
    T maxWeight = 0;

    __shared__ T sPatches[BLOCK_SIZE];
    const int patchesC = BLOCK_SIZE / patchSize;   // Patches that fit into shared memory.
    const int chunks   = regionSize / patchesC;    // Iterations needed to load all patches (without division remnants).
    const int targetY  = threadIdx.x % patchSize;  // Pixel in patch that current thread will load.
    const int targetX  = threadIdx.x / patchSize;  // Patch whose pixel current thread will load.

    // Calculate weights excluding the remainder patches (of chunks division).
    for (int cI = 0; cI < chunks; ++cI) {
        __syncthreads();
        sPatches[threadIdx.x] = dRPatches(targetY, cI*patchesC+targetX);
        __syncthreads();
        if (x >= regionSize) continue;

        // Iterate over all patches in shared memory.
        for (int i = 0; i < patchesC; ++i) {
            if (patchesC*cI+i == x) continue;  // Do not count the weight against itself.

            T weight = 0;
            for (int j = 0; j < patchSize; ++j) {
                T d = dRPatches(j, x) - sPatches[i*patchSize+j];
                weight += d*d;
            }
            weight = __anlm_exp<T>(-weight/fSigma);
            if (weight > maxWeight) maxWeight = weight;  // will use max weight for itself

            nom   += dRSrc(0, patchesC*cI+i) * weight;
            denom += weight;
        }
    }

    const int rem = regionSize % patchesC;  // Remainder of patches.

    // Load the remainder patches (if any).
    __syncthreads();
    if (threadIdx.x < rem*patchSize) {
        sPatches[threadIdx.x] = dRPatches(targetY, chunks*patchesC+targetX);
    }
    __syncthreads();
    if (x >= regionSize) return;  // Shared memory loads are over - thread no longer needed.

    // Calculate weights for the remainder patches.
    for (int i = 0; i < rem; ++i) {
        if (chunks*patchesC+i == x) continue;  // Do not count the weight against itself.

        T weight = 0;
        for (int j = 0; j < patchSize; ++j) {
            T d = dRPatches(j, x) - sPatches[i*patchSize+j];
            weight += d*d;
        }
        weight = __anlm_exp<T>(-weight/fSigma);
        if (weight > maxWeight) maxWeight = weight;  // will use max weight for itself

        nom   += dRSrc(0, patchesC*chunks+i) * weight;
        denom += weight;
    }

    // Guarantee that maxWeight will not be 0.
    if (maxWeight < __anlm_pow<T>(2.0, -52.0)) maxWeight = __anlm_pow<T>(2.0, -52.0);

    // Use maxWeight as the weight for itself.
    nom   += dRSrc(0, x) * maxWeight;
    denom += maxWeight;

    // Write back to the complete matrix.
    if (denom != 0) dDst.at(dOriginalPos(0, x)) = nom / denom;
    else dDst.at(dOriginalPos(0, x)) = dRSrc(0, x);
}

// template<class T>
// __global__ void
// // __launch_bounds__(BLOCK_SIZE)
// cudaAnlmKernel(DMat<T> dRSrc, DMat<T> dRPatches, DMat<T> dDst,
//                DMat<DMatPos> dOriginalPos, DMat<T> dRFiltersSigma,
//                int patchH, int patchW)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//
//     T fSigma = dRFiltersSigma(0, x);
//     fSigma = fSigma * fSigma;
//
//     T nom = 0;
//     T denom = 0;
//     T maxWeight = 0;
//
//     __shared__ T sPatches[BLOCK_SIZE];
//     int patchesC = BLOCK_SIZE / dRPatches.height;   // Patches that fit into shared memory.
//     int chunks   = dRPatches.width / patchesC;      // Iterations needed to load all patches (without division remnants).
//     int targetY  = threadIdx.x % dRPatches.height;  // Pixel in patch that current thread will load.
//     int targetX  = threadIdx.x / dRPatches.height;  // Patch whose pixel current thread will load.
//
//     // Calculate weights excluding the remainder patches of
//     // dRPatches.width/patchesC division.
//     for (int cI = 0; cI < chunks; ++cI) {
//         __syncthreads();
//         sPatches[threadIdx.x] = dRPatches(targetY, cI*patchesC+targetX);
//         __syncthreads();
//         if (x >= dRSrc.width) continue;
//
//         // Iterate over all patches in shared memory.
//         for (int i = 0; i < patchesC; ++i) {
//             if (patchesC*cI+i == x) continue;  // Do not count the weight against itself.
//
//             T weight = 0;
//             for (int j = 0; j < dRPatches.height; ++j) {
//                 T d = dRPatches(j, x) - sPatches[i*dRPatches.height+j];
//                 weight += d*d;
//             }
//             weight = __anlm_exp<T>(-weight/fSigma);
//             if (weight > maxWeight) maxWeight = weight;  // will use max weight for itself
//
//             nom += dRSrc(0, patchesC*cI+i) * weight;
//             denom += weight;
//         }
//     }
//
//     // Load the remainder patches (if any).
//     int rem = dRPatches.width % patchesC;
//     __syncthreads();
//     if (threadIdx.x < rem*dRPatches.height) {
//         sPatches[threadIdx.x] = dRPatches(targetY, chunks*patchesC+targetX);
//     }
//     __syncthreads();
//     if (x >= dRSrc.width) return;
//
//     // Calculate weights for the remainder patches of
//     // dRPatches.width/patchesC division.
//     for (int i = 0; i < rem; ++i) {
//         if (chunks*patchesC+i == x) continue;  // Do not count the weight against itself.
//
//         T weight = 0;
//         for (int j = 0; j < dRPatches.height; ++j) {
//             T d = dRPatches(j, x) - sPatches[i*dRPatches.height+j];
//             weight += d*d;
//         }
//         weight = __anlm_exp<T>(-weight/fSigma);
//         if (weight > maxWeight) maxWeight = weight;  // will use max weight for itself
//
//         nom += dRSrc(0, patchesC*chunks+i) * weight;
//         denom += weight;
//     }
//
//     // Guarantee that maxWeight will not be 0.
//     if (maxWeight < __anlm_pow<T>(2.0, -52.0)) maxWeight = __anlm_pow<T>(2.0, -52.0);
//
//     // Use maxWeight as the weight for itself.
//     nom += dRSrc(0, x) * maxWeight;
//     denom += maxWeight;
//
//     // Write back to the complete matrix.
//     if (denom != 0) dDst.at(dOriginalPos(0, x)) = nom / denom;
//     else dDst.at(dOriginalPos(0, x)) = dRSrc(0, x);
// }

// template<class T>
// __global__ void
// //__launch_bounds__(BLOCK_SIZE, MIN_BLOCKS_PER_SM)
// cudaAnlmKernel(DMat<T> dRSrc, DMat<T> dRPatches, DMat<T> dDst,
//                DMat<DMatPos> dOriginalPos, DMat<T> dRFiltersSigma,
//                int patchH, int patchW)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     if (x >= dRSrc.width) return;
//
//     T fSigma = dRFiltersSigma(0, x);
//     fSigma = fSigma * fSigma;
//
//     T nom = 0;
//     T denom = 0;
//     T maxWeight = 0;
//
//     // Iterate over all pixels in the region to
//     for (int i = 0; i < dRPatches.width; ++i) {
//         if (i == x) continue;  // Do not count the weight against itself.
//
//         T weight = 0;
//         for (int j = 0; j < dRPatches.height; ++j) {
//             T d = dRPatches(j, x) - dRPatches(j, i);
//             weight += d*d;
//         }
//         weight = __anlm_exp<T>(-weight/fSigma);
//         if (weight > maxWeight) maxWeight = weight;  // will use max weight for itself
//
//         nom += dRSrc(0, i) * weight;
//         denom += weight;
//     }
//
//     // Guarantee that maxWeight will not be 0.
//     if (maxWeight < __anlm_pow<T>(2.0, -52.0)) maxWeight = __anlm_pow<T>(2.0, -52.0);
//
//     // Use maxWeight as the weight for itself.
//     nom += dRSrc(0, x) * maxWeight;
//     denom += maxWeight;
//
//     // Write back to the complete matrix.
//     if (denom != 0) dDst.at(dOriginalPos(0, x)) = nom / denom;
//     else dDst.at(dOriginalPos(0, x)) = dRSrc(0, x);
// }


/**
 * Kernel that applies given gaussian blur to patches contained in dRSrc.
 */
template <class T>
__global__ void
cudaApplyBlur(DMat<T> dRSrc, DMat<T> dPatchBlur)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= dRSrc.width) return;

    const int patchW = dPatchBlur.width;

    for (int i = 0; i < dPatchBlur.height; ++i) {
        for (int j = 0; j < patchW; ++j) {
            dRSrc(i*patchW+j, x) *= dPatchBlur(i, j);
        }
    }
}

/**
 * Kernel that copies data from given uniform matrices into given region
 * specific matrices.
 */
template <class T>
__global__ void
cudaDivideRegions(DMatExpanded<T> dSrc,
                  DMat<T> dFiltersSigma,
                  DMat<int> dIds,
                  DMat<int> dRegionalIndices,
                  int patchH,
                  int patchW,
                  DMat<T*> dRSrc,
                  DMat<T*> dRPatches,
                  DMat<int> dRPatchesPitch,
                  DMat<T*> dRFiltersSigma,
                  DMat<DMatPos*> dOrigPos)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dSrc.width || y >= dSrc.height) return;

    const int currentId = dIds(y, x);  // id of region
    const int rIndex    = dRegionalIndices(y, x);  // location in region

    const int boundH = (patchH - 1) / 2;
    const int boundW = (patchW - 1) / 2;

    // Keep a reference of pixel's original position in dSrc.
    *(dOrigPos(0, currentId) + rIndex) = DMatPos(y, x);

    // Copy filter sigma.
    *(dRFiltersSigma(0, currentId) + rIndex) = dFiltersSigma(y, x);

    // Copy the current pixel.
    *(dRSrc(0, currentId) + rIndex) = dSrc(y, x);

    const int patchesPitch = dRPatchesPitch(0, currentId);

    // Copy the entire neighborhood of current pixel.
    for (int i = -boundH; i < boundH+1; ++i) {
        for (int j = -boundW; j < boundW+1; ++j) {
            int row = (i+boundH)*patchW + (j+boundW);
            T *loc = (T*) ((char*) dRPatches(0, currentId) + row*patchesPitch) + rIndex;
            *loc = dSrc(y+i, x+j);
        }
    }
}

/**
 * Returns a vector, that contains for each element in ids the index of that
 * pixel in its own region.
 *
 * The region of a pixel is consisted of the pixels having the same id.
 *
 * Parameters:
 *  -ids : Pointer to an array of size imgH*imgW containing the id of each pixel
 *          in row major order.
 *  -imgH : Height of the image in pixels (number of rows in ids).
 *  -imgW : Width of the image in pixels (number of columns in ids).
 *  -regionSizes : A vector of size equal to the number of different regions
 *          (i.e. different ID values) contained in ids. On return, each
 *          item in this array will contain the number of elements in the
 *          corresponding region.
 *
 * Returns:
 *  A vector of size imgH*imgW containing the region-wise indices for each pixel.
 *  Also, in regionSizes returns the number of elements in each region.
 */
std::vector<int>
divideInRegions(int *ids, int imgH, int imgW, std::vector<int> &regionSizes)
{
    std::vector<int> indices(imgH*imgW);

    for (int i = 0; i < regionSizes.size(); ++i) regionSizes[i] = 0;

    for (int i = 0; i < imgH; ++i) {
        for (int j = 0; j < imgW; ++j) {
            int index = imgW * i + j;
            int id = *(ids+index);
            indices[index] = regionSizes[id];
            ++regionSizes[id];
        }
    }

    return indices;
}

/**
 * Divides given uniform matrices into region specific matrices.
 */
template <class T>
void
fillRegionalMatrices(const DMatExpanded<T> &dSrc,
                     const DMat<T> &dFiltersSigma,
                     const DMat<int> &dIds,
                     std::vector<int> &regionalIndices,
                     std::vector<int> &regionSizes,
                     int patchH,
                     int patchW,
                     std::vector<DMat<T>> &rSrc,
                     std::vector<DMat<T>> &rPatches,
                     std::vector<DMat<T>> &rFiltersSigma,
                     std::vector<DMat<DMatPos>> &origPos)
{
    using std::vector;

    // Objects cannot be copied directly to device, so the only way to pass
    // an array of objects is to pass their raw device pointers.
    vector<T*>       rawSrc(rSrc.size());
    vector<T*>       rawPatches(rPatches.size());
    vector<int>      rawPatchesPitch(rPatches.size());
    vector<T*>       rawFiltersSigma(rFiltersSigma.size());
    vector<DMatPos*> rawOrigPos(origPos.size());

    for (int i = 0; i < rSrc.size(); ++i) {
        rawSrc[i]          = rSrc[i].data;
        rawPatches[i]      = rPatches[i].data;
        rawPatchesPitch[i] = rPatches[i].pitch;
        rawFiltersSigma[i] = rFiltersSigma[i].data;
        rawOrigPos[i]      = origPos[i].data;
    }

    // Wrap the pointers into DMats, so they can be easily passed to device.
    DMat<T*>       dRPatches(rawPatches.data(), rawPatches.size(), 1);
    DMat<int>      dRPatchesPitch(rawPatchesPitch.data(), rawPatches.size(), 1);
    DMat<T*>       dRSrc(rawSrc.data(), rawSrc.size(), 1);
    DMat<T*>       dRFiltersSigma(rawFiltersSigma.data(), rawFiltersSigma.size(), 1);
    DMat<DMatPos*> dOrigPos(rawOrigPos.data(), rawOrigPos.size(), 1);

    DMat<int> dRegionalIndices(regionalIndices.data(), dSrc.width, dSrc.height);

    int gridW = dSrc.width / 32;
    if ((dSrc.width % 32) > 0) gridW++;
    int gridH = dSrc.height / (BLOCK_SIZE / 32);
    if ((dSrc.height % (BLOCK_SIZE / 32)) > 0) gridH++;
    dim3 blockDim(32, BLOCK_SIZE / 32);
    dim3 gridDim(gridW, gridH);

    cudaDivideRegions<<<gridDim, blockDim>>>(
        dSrc, dFiltersSigma, dIds, dRegionalIndices, patchH, patchW,
        dRSrc, dRPatches, dRPatchesPitch, dRFiltersSigma, dOrigPos
    );
    gpuErrchk( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
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
adaptiveNonLocalMeansNormalData<float>(float *, float *, int *, float *,
                                       int, int, int, int, float, int);
template
void
adaptiveNonLocalMeansNormalData<double>(double *, double *, int *, double *,
                                        int, int, int, int, double, int);

}
