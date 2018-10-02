/**
 * DMat.hpp
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * Declarations for DMat and DMatExpanded. They provide create-on-host,
 * access-on-device matrices. That means they can be created and filled on host
 * code, while their objects can be provided to device code for accessing
 * their data.
 *
 * Classes defined in DMat.hpp:
 *  -cuda::DMatPos
 *  -cuda::DMat
 *  -cuda::DMatExpanded
 *
 * Version: 0.1
 *
 * License: GNU GPL v3 (see project's license).
 */

#ifndef __cuda_DMat_hpp__
#define __cuda_DMat_hpp__

#include <cstdio>

namespace cuda
{

template <class T>
class DMat;

/**
 * Swap function for DMat.
 */
template <class T>
void
swap(DMat<T>& first, DMat<T>& second);


/**
 * DMatPos is a class that represents a cell position (x, y) in a DMat.
 *
 * It can be created and accessed by device code.
 */
struct DMatPos
{
    int x;
    int y;

    __device__ DMatPos(int y, int x) : y(y), x(x) {}
};


/**
 * DMat provides a matrix that can be created and filled from host code, while
 * its elements can be accessed by device code.
 *
 * DMat is stored in row major order and each row is properly aligned, so that
 * access scheme should be followed for high coalescion.
 *
 * Copies of DMat do not copy the device resources, making passing DMat objects
 * to device code fairly light.
 */
template <class T>
class DMat
{
public:
    // TODO: Provide inline host/device setters/getters without creating a mess.

    T *data;        // Raw pointer to matrix location in device's memory.
    size_t height;  // Height of matrix.
    size_t width;   // Width of matrix .
    size_t pitch;   // Pitch size of each matrix row (in bytes).

    /**
     * Creates a matrix object of 0 size.
     */
    DMat();
    /**
     * Creates a matrix object of size height * width, by allocating
     * required memory on device.
     *
     * Parameters:
     *  -width : Width of created matrix.
     *  -height : Height of created matrix.
     */
    DMat(size_t width, size_t height);
    /**
     * Creates a matrix object of size height * width and fills it with
     * provided data.
     *
     * Parameters:
     *  -hostData : A pointer to a host memory location that contains a
     *          matrix of size height*width in row major order.
     *  -width : Width of created matrix.
     *  -height : Height of created matrix.
     */
    DMat(T* hostData, size_t width, size_t height);
    /**
     * Creates a copy of a DMat that operates on exact the same resources as the
     * original one. All copies are just alt handlers for these resources.
     * Resource management is done through the lifetime methods of the original
     * object. Destroying the original object, makes all copies of it uselles.
     */
    DMat(const DMat& dmat);
    /**
     * Moves the ownership of resources of temporary DMat to the current one.
     * Resources of the current DMat object are freed if it was the original
     * owner of them.
     */
    DMat(DMat&& dmat);
    /**
     * See copy/move constructor for copy/move assignment characteristics.
     */
    DMat<T>& operator=(DMat<T> dmat);
    /**
     * Releases resources of current object if it is the owner of them.
     */
    virtual ~DMat();

    friend void swap<T>(DMat<T>& first, DMat<T>& second);

    /**
     * Fills device matrix from given host matrix.
     *
     * Parameters:
     *  -hostData : A pointer to a host memory location that contains a
     *          matrix of size height*width in row major order.
     *  -width : Width of given host matrix.
     *  -height : Height of given host matrix.
     */
    void copyFromHost(T* hostData, int width, int height);
    /**
     * Copies device matrix to given host location.
     *
     * Parameters:
     *  -hostDst : A pointer to a host array location of size large enough to
     *          hold all elements of matrix.
     */
    void copyToHost(T* hostDst);

    // Device elements accessors.
    __device__ inline T& at(int y, int x);
    __device__ inline T& at(const DMatPos& pos);
    __device__ inline T& operator()(int y, int x);

private:
    bool _isOwner;
};


/**
 * DMatExpanded provides a matrix that can be created and filled on host and
 * accessed on device, with the ability to expand itself by mirroring its
 * rows and columns.
 *
 * A DMatExpanded, can increase its size from MxN size of original DMat to
 * (M+expH*2)x(N+expW*2), where expH and expW are the number of rows and columns
 * to be added to each side vertically and horizontally respectively. That means,
 * that top expH rows are mirrored above the top of the original matrix. Bottom
 * expH rows are mirrored below the bottom of the original matrix. The same
 * happens for expW columns at the right and left of the original matrix.
 *
 * Indices of original data to the new matrix remain exactly the same. That
 * means for example that (0, 0) element of expanded matrix will be the same
 * as (0, 0) element of the original one. Expanded cells are accessed by going
 * beyond the original indices. So an indice of (-1, -1) indicates the element
 * one row below the first row and one column to the left of the element (0, 0).
 */
template <class T>
class DMatExpanded : public DMat<T>
{
public:
    using DMat<T>::data;
    using DMat<T>::height;
    using DMat<T>::width;
    using DMat<T>::pitch;

    /**
     * Creates a device matrix expanded at both sides vertically and
     * horizontally by given expansion sizes.
     *
     * Parameters:
     *  -width : Width of the original matrix.
     *  -height : Height of the original matrix.
     *  -expWidth : Number of collumns to be padded both to the right and to
     *          the left of the original matrix.
     *  -expHeight : Number of rows to be padded both to the top and to the
     *          bottom of the original matrix.
     */
    DMatExpanded(size_t width, size_t height,
                 size_t expWidth, size_t expHeight);
    /**
     * Creates a device matrix expanded at both sides vertically and
     * horizontally by given expansion sizes, while filling it with the given
     * data.
     *
     * Parameters:
     *  -hostData : Pointer to a host array containing a matrix of size
     *          height*width stored in row major order.
     *  -width : Width of the original matrix.
     *  -height : Height of the original matrix.
     *  -expWidth : Number of collumns to be padded both to the right and to
     *          the left of the original matrix.
     *  -expHeight : Number of rows to be padded both to the top and to the
     *          bottom of the original matrix.
     */
    DMatExpanded(T *hostData, size_t width, size_t height,
                 size_t expWidth, size_t expHeight);
    /**
     * Creates a shallow copy of current DMatExpanded object without copying
     * underlying resources.
     */
    DMatExpanded(const DMatExpanded &dmat);
    /**
     * See copy constructor.
     */
    DMatExpanded<T> &operator=(const DMatExpanded<T> &dmat);
    /**
     * Frees DMatExpanded resources, if this object is the owner of them.
     */
    ~DMatExpanded();

    // TODO: Implement a swap idiom and move operations.

    /**
     * Copies to the current device matrix a host matrix, while also expanding
     * it.
     *
     * Parameters:
     *  -hostData : A pointer to a host matrix of size width*height stored in
     *          row major order.
     *  -width : Width of given matrix.
     *  -height : Height of given matrix.
     *  -vExpand : Number of rows to be padded both to the top and to the
     *          bottom of the original matrix.
     *  -hExpand : Number of collumns to be padded both to the right and to
     *          the left of the original matrix.
     */
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
