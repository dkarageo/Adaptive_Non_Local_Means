/**
 * anlm.hpp
 *
 * Version: 0.1
 */

#ifndef __cuda_anlm_hpp__
#define __cuda_anlm_hpp__


namespace cuda
{
    template <class T>
    void
    adaptiveNonLocalMeans(T *src, T *dst, int *ids, T *filterSigma,
                          int imgH, int imgW, int patchH, int patchW,
                          T patchSigma);
}

#endif
