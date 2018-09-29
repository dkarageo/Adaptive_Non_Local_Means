/**
 * DMat.cu
 *
 * Version: 0.1
 */

#include <iostream>
#include <assert.h>
#include "DMat.hpp"


namespace cuda {


template <class T>
void
swap(DMat<T>& first, DMat<T>& second)
{
    using std::swap;

    swap(first.width, second.width);
    swap(first.height, second.height);
    swap(first.pitch, second.pitch);
    swap(first.data, second.data);
    swap(first._isOwner, second._isOwner);
}

template <class T>
DMat<T>::DMat()
: _isOwner(true),
  height(0),
  width(0),
  pitch(0),
  data(nullptr)
{
    // std::cout << "Calling DMat constructor" << std::endl;
}

template <class T>
DMat<T>::DMat(size_t width, size_t height)
: DMat()
{
    cudaError_t cudaStat;

    this->width = width;
    this->height = height;

    if (width > 0 && height > 0) {
        // std::cout << "Creating matrix: " << width << "x" << height << std::endl;
        cudaStat = cudaMallocPitch(&data, &pitch, sizeof(T) * width, height);
        assert(cudaSuccess == cudaStat);
    } else std::cout << "Creating empty matrix" << std::endl;
}

template <class T>
DMat<T>::DMat(T* hostData, size_t width, size_t height)
: DMat(width, height)
{
    copyFromHost(hostData, width, height);
}

template <class T>
DMat<T>::DMat(const DMat<T> &dmat)
{
    // std::cout << "Copying matrix - isOwner:" << (dmat._isOwner ? "true" : "false")
    //           << " width:" << dmat.width
    //           << " height:" << dmat.height
    //           << " pitch:" << dmat.pitch
    //           << std::endl;

    data = dmat.data;
    width = dmat.width;
    height = dmat.height;
    pitch = dmat.pitch;
    _isOwner = false;
}

template <class T>
DMat<T>::DMat(DMat<T> &&dmat)
{
    swap<T>(*this, dmat);
}

template <class T>
DMat<T>&
DMat<T>::operator=(DMat<T> dmat)
{
    swap<T>(*this, dmat);
    if (data == dmat.data) dmat._isOwner = false;
    return *this;
}

template <class T>
DMat<T>::~DMat()
{
    // std::cout << "Destructing DMat" << std::endl;

    if (_isOwner && data) {
        // std::cout << "is owner" << std::endl;

        cudaError_t cudaStat = cudaFree(data);
        assert(cudaSuccess == cudaStat);
    }
}

template <class T>
void
DMat<T>::copyFromHost(T *hostData, int width, int height)
{
    cudaError_t cudaStat;

    cudaStat = cudaMemcpy2D(data, pitch,
                            hostData, width*sizeof(T),
                            width*sizeof(T), height,
                            cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat);
}

template <class T>
void
DMat<T>::copyToHost(T *hostDst)
{
    cudaError_t cudaStat;

    cudaStat = cudaMemcpy2D(hostDst, width*sizeof(T),
                            data, pitch,
                            width*sizeof(T), height,
                            cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat);
}

template <class T>
DMatExpanded<T>::DMatExpanded(size_t width, size_t height,
                              size_t expWidth, size_t expHeight)
{
    this->width = width;
    this->height = height;
    _expWidth = width + expWidth * 2;
    _expHeight = height + expHeight * 2;

    cudaError_t cudaStat;

    // Allocate a table large enough for the expanded matrix.
    cudaStat = cudaMallocPitch(&_expData, &pitch,
                               sizeof(T) * _expWidth, _expHeight);
    assert(cudaSuccess == cudaStat);

    // Point DMat data to the portion of original table, so DMatExpanded
    // can be a valid DMat too. A displacement of _expHeight rows, followed
    // by _expWidth columns, will do the job. Expansion cells will be seen as
    // pitch padding by DMat.
    data = (T*) ((char*) _expData + expHeight*pitch) + expWidth;
}

template <class T>
DMatExpanded<T>::DMatExpanded(T *hostData, size_t width, size_t height,
                              size_t expWidth, size_t expHeight)
: DMatExpanded(width, height, expWidth, expHeight)
{
    copyFromHost(hostData, width, height, expWidth, expHeight);
}

template <class T>
DMatExpanded<T>::DMatExpanded(const DMatExpanded<T> &dmat)
: DMat<T>(dmat)
{
    _expData = dmat._expData;
    _expWidth = dmat._expWidth;
    _expHeight = dmat._expHeight;
}

template <class T>
DMatExpanded<T>&
DMatExpanded<T>::operator=(const DMatExpanded<T> &dmat)
{
    DMat<T>::operator=(dmat);
    _expData = dmat._expData;
    _expWidth = dmat._expWidth;
    _expHeight = dmat._expHeight;
    return *this;
}

template <class T>
DMatExpanded<T>::~DMatExpanded()
{
    // Restore data attr for parent destructor.
    data = _expData;
}

template <class T>
void
DMatExpanded<T>::copyFromHost(T *hostData, int width, int height,
                              int vExpand, int hExpand)
{
    cudaError_t cudaStat;

    // Copy original table to the centre of expanded matrix.
    DMat<T>::copyFromHost(hostData, width, height);

    // Expand horizontally.
    for (int i = 0; i < hExpand; ++i) {
        cudaStat = cudaMemcpy2D(data-i-1, pitch, data+i, pitch, sizeof(T),
                                height, cudaMemcpyDeviceToDevice);
        assert(cudaSuccess == cudaStat);
        cudaStat = cudaMemcpy2D(data+width+i, pitch, data+width-i-1, pitch, sizeof(T),
                                height, cudaMemcpyDeviceToDevice);
        assert(cudaSuccess == cudaStat);
    }

    // Expand vertically.
    for (int i = 0; i < vExpand; ++i) {
        T *dst = (T*) ((char*) (data - hExpand) - pitch * (i + 1));
        T *src = (T*) ((char*) (data - hExpand) + pitch * i);
        cudaStat = cudaMemcpy2D(dst, pitch, src, pitch,
                                (width+hExpand*2)*sizeof(T),
                                1, cudaMemcpyDeviceToDevice);
        assert(cudaSuccess == cudaStat);
        dst = (T*) ((char*) (data - hExpand) + pitch * (height + i));
        src = (T*) ((char*) (data - hExpand) + pitch * (height - i - 1));
        cudaStat = cudaMemcpy2D(dst, pitch, src, pitch,
                                (width+hExpand*2)*sizeof(T),
                                1, cudaMemcpyDeviceToDevice);
        assert(cudaSuccess == cudaStat);
    }
}


template class DMat<double>;
template class DMat<double*>;
template class DMat<int>;
template class DMat<int*>;
template class DMat<float>;
template class DMat<float*>;
template class DMat<DMatPos>;
template class DMat<DMatPos*>;
template class DMat<DMat<double>>;
template class DMat<DMat<float>>;
template class DMat<DMat<DMatPos>>;

template class DMatExpanded<double>;
template class DMatExpanded<double*>;
template class DMatExpanded<int>;
template class DMatExpanded<int*>;
template class DMatExpanded<float>;
template class DMatExpanded<float*>;
template class DMatExpanded<DMatPos>;
template class DMatExpanded<DMatPos*>;


}
