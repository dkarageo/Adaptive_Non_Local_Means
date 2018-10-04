/**
 * anlm.hpp
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * Definitions for Adaptive Non Local Means (ANLM) implementations provided
 * under current project. All implementations require a CUDA enabled GPU.
 *
 * Macros defined in anlm.hpp:
 *  -MAX_DEVICE_DRAM_USAGE
 *
 * Routines defined in anlm.hpp:
 *  -adaptiveNonLocalMeans
 *  -adaptiveNonLocalMeansNormalData
 *  -adaptiveNonLocalMeansBigData
 *
 * Version: 0.1
 */

#ifndef __cuda_anlm_hpp__
#define __cuda_anlm_hpp__


// DRAM of GPU in MB that should not be exceeded.
#define MAX_DEVICE_DRAM_USAGE 1480  // Currently targeting GTX 480.


namespace cuda
{

/**
 * Applies Adaptive Non-Local Means to the given image using a CUDA enabled GPU.
 *
 * While the routine is fully templated, it is expected to be used with FP32/64
 * data.
 *
 * This routine is actually a proxy to adaptiveNonLocalMeansNormalData or
 * adaptiveNonLocalMeansBigData. Selection is done dynamically based on
 * the value MAX_DEVICE_DRAM_USAGE macro.
 *
 * Parameters:
 *  -src : An imgH*imgW matrix stored in row major order that contains the
            image to be denoised by ANLM.
 *  -dst : An imgH*imgW matrix stored in row major order where the result of
 *          ANLM will be returned.
 *  -ids : An imgH*imgW matrix in row major order containing the id of the
 *          region where the corresponding pixel belongs to. Ids are expected
 *          to be in the range [0, regions).
 *  -filterSigma : An imgH*imgW matrix in row major order containing the sigma
 *          values for each pixel. Each pixel can have its own unique sigma
 *          value to be used by ANLM. Usually the 'std' of each region is used.
 *  -imgH : Height of image in pixels.
 *  -imgW : Width of image in pixels.
 *  -patchH : Patch height, a.k.a. height of the areas close to each pixel that
 *          will be compared. Usually 5 or 7.
 *  -patchW : Patch width, a.k.a. width of the areas close to each pixel that
 *          will be compared. Usually 5 or 7.
 *  -patchSigma : Sigma that will be used to create a gaussian blur filter
 *          for each patch.
 *  -regions : Number of regions contained in ids matrix.
 *
 * Returns:
 *  Filtered image in dst matrix.
 */
template <class T>
void
adaptiveNonLocalMeans(T *src, T *dst, int *ids, T *filterSigma,
                      int imgH, int imgW, int patchH, int patchW,
                      T patchSigma, int regions);

/**
 * Applies Adaptive Non-Local Means to the given image using a CUDA enabled GPU.
 *
 * It is expected to be used for small and moderate sized images. The gpu
 * should have enough DRAM to hold about patchH*patchW+8 copies of the image
 * provided in src.
 *
 * Use adaptiveNonLocalMeans() for a safe call.
 *
 * Parameters:
 *  see adaptiveNonLocalMeans
 */
template <class T>
void
adaptiveNonLocalMeansNormalData(T *src, T *dst, int *ids, T *filterSigma,
                                int imgH, int imgW, int patchH, int patchW,
                                T patchSigma, int regions);

/**
 * Applies Adaptive Non-Local Means to the given image using a CUDA enabled GPU.
 *
 * It can be used for much larger images compared to
 * adaptiveNonLocalMeansNormalData, though it can be even twice slower.
 *
 * Parameters:
 *  see adaptiveNonLocalMeans
 */
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
#ifndef FORCE_ANLM_BIG
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
#endif
        adaptiveNonLocalMeansBigData(
            src, dst, ids, filterSigma, imgH, imgW, patchH, patchW,
            patchSigma, regions
        );
#ifndef FORCE_ANLM_BIG
    }
#endif
}

}

#endif
