/**
 * anlm.hpp
 *
 * Version: 0.1
 */

#ifndef __cuda_anlm_hpp__
#define __cuda_anlm_hpp__

// DRAM of GPU in MB that should not be exceeded.
#define MAX_DEVICE_DRAM_USAGE 1480  // Currently targeting GTX 480.


namespace cuda
{
    template <class T>
    void
    adaptiveNonLocalMeans(T *src, T *dst, int *ids, T *filterSigma,
                          int imgH, int imgW, int patchH, int patchW,
                          T patchSigma, int regions);

    template <class T>
    void
    adaptiveNonLocalMeansNormalData(T *src, T *dst, int *ids, T *filterSigma,
                                    int imgH, int imgW, int patchH, int patchW,
                                    T patchSigma, int regions);

    template <class T>
    void
    adaptiveNonLocalMeansBigData(T *src, T *dst, int *ids, T *filterSigma,
                                 int imgH, int imgW, int patchH, int patchW,
                                 T patchSigma, int regions);

    template <class T>
    void
    adaptiveNonLocalMeans(T *src, T *dst, int *ids, T *filterSigma,
                          int imgH, int imgW, int patchH, int patchW,
                          T patchSigma, int regions)
    {
        long imgSize = sizeof(T) * imgH * imgW;

        // Make a rough estimation about the maximum dram that may be needed
        // for the normal algorithm.
        long normalRamUsage = imgSize * (patchH*patchW + 8);

        if (normalRamUsage <= MAX_DEVICE_DRAM_USAGE*1024*1024) {
            adaptiveNonLocalMeansNormalData(
                src, dst, ids, filterSigma, imgH, imgW, patchH, patchW,
                patchSigma, regions
            );
        } else {
            adaptiveNonLocalMeansBigData(
                src, dst, ids, filterSigma, imgH, imgW, patchH, patchW,
                patchSigma, regions
            );
        }
    }
}

#endif
