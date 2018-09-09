/**
 * DMat.hpp
 *
 * Version: 0.1
 */

#ifndef __cuda_DMat_hpp__
#define __cuda_DMat_hpp__

#include <cstdio>

namespace cuda
{
    struct DMatPos
    {
        int x;
        int y;

        __device__ DMatPos(int y, int x) : y(y), x(x) {}
    };

    template <class T>
    class DMat
    {
    public:
        T *data;
        size_t height;
        size_t width;
        size_t pitch;

        DMat(size_t width, size_t height);
        DMat(T *hostData, size_t width, size_t height);
        DMat(const DMat &dmat);
        virtual ~DMat();

        void copyFromHost(T *hostData, int width, int height);
        void copyToHost(T *hostDst);

        __device__ inline T& at(int y, int x);
        __device__ inline T& at(const DMatPos &pos);
        __device__ inline T& operator()(int y, int x);

    protected:
        DMat();

    private:
        bool _isOwner;
    };

    template <class T>
    class DMatExpanded : public DMat<T>
    {
    public:
        using DMat<T>::data;
        using DMat<T>::height;
        using DMat<T>::width;
        using DMat<T>::pitch;

        DMatExpanded(size_t width, size_t height,
                     size_t expWidth, size_t expHeight);
        DMatExpanded(T *hostData, size_t width, size_t height,
                     size_t expWidth, size_t expHeight);
        DMatExpanded(const DMatExpanded &dmat);
        ~DMatExpanded();

        void copyFromHost(T *hostData, int width, int height,
                          int vExpand, int hExpand);

    private:
        T *_expData;
        size_t _expWidth;
        size_t _expHeight;
    };

    template <class T>
    __device__ inline T&
    DMat<T>::at(int y, int x)
    {
        return *((T*) ((char*) data + y*pitch) + x);
    }

    template <class T>
    __device__ inline T&
    DMat<T>::at(const DMatPos &pos)
    {
        return *((T*) ((char*) data + pos.y*pitch) + pos.x);
    }

    template <class T>
    __device__ inline T&
    DMat<T>::operator()(int y, int x)
    {
        return *((T*) ((char*) data + y*pitch) + x);
    }
}

#endif
