/**
 * demo.cpp
 *
 * Created by Dimitrios Karageorgiou,
 *  for course "Parallel And Distributed Systems".
 *  Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
 *
 * An application that demonstrates Adaptive Non Local Means (ANLM) implementations
 * provided in current project. It benchmarks both FP32 and FP64 implementations
 * and can test the validity of results by comparing to given precomputed data.
 *
 * Application utilizes custom binary files ended with .karas that are actually
 * a serialized matrix of doubles. They start with two uint32_t fields
 * containing height and width respectively. Then height*width doubles are
 * following. The matrix is stored in row major order.
 *
 * Usage:
 *  ./binfile <noisy_img> <regions> [<filtered_img>]
 * where:
 *  -noisy_img : A .karas file containing the grayscale image to be denoised.
 *          Each cell in the matrix represents luminosity of each pixel in
 *          range [0, 1].
 *  -regions : Number of regions the image will be divided into in order to
 *          apply ANLM.
 *  -filtered_image [optional]: A .karas file containing the precomputed
 *          denoised image. This image is expected to have been computed using
 *          another ANLM implementation with the same mathematical characteristics.
 *          The application will compare the output of current ANLM implementation
 *          output, against these data. It can be seen as a light unit test.
 *
 * Version: 0.1
 */

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include "cuda/anlm.hpp"
#include "cuda/init.hpp"


using namespace std;
using Clock=std::chrono::high_resolution_clock;

template<class T>
vector<int>
calculateEachPixelRegion(const vector<T> &img, int regionsNum);
template<class T>
vector<T>
calculateEachPixelSigma(const vector<T> &img, const vector<int> &ids,
                        int regionsNum);
void
testCudaAnlmFP64(vector<double> noisyImg, int height, int width,
                 int regions, char *testFile);
void
testCudaAnlmFP32(vector<double> noisyImgFP64, int height, int width,
                 int regions, char *testFile);


int main(int argc, char *argv[])
{
    if (argc < 3) {
        cout << "Usage: ./" << argv[0] << " <noisy_img> <regions> [<test_data>]";
        exit(-1);
    }

    char *testFile = nullptr;
    if (argc >= 4) testFile = argv[3];

    // The number of rengions the image will be divided for applying anlm.
    int regions = atoi(argv[2]);
    uint32_t height;
    uint32_t width;

    // Load noisy image from .karas file.
    ifstream dataStream(argv[1], ios_base::binary|ios_base::in);
    if (!dataStream.is_open()) {
        cout << "Failed to open " << argv[1] << endl;
        exit(-1);
    }
    dataStream.read((char*) &height, sizeof(uint32_t));
    dataStream.read((char*) &width, sizeof(uint32_t));
    vector<double> noisyImg(height*width);
    dataStream.read((char*) noisyImg.data(), sizeof(double)*height*width);
    dataStream.close();

    cuda::deviceInit();  // Warm up device.

    cout << "================================================" << endl;
    testCudaAnlmFP64(noisyImg, height, width, regions, testFile);
    cout << "================================================" << endl;
    testCudaAnlmFP32(noisyImg, height, width, regions, testFile);
    cout << "================================================" << endl;

    return 0;
}

void
testCudaAnlmFP64(vector<double> noisyImg, int height, int width,
                 int regions, char *testFile)
{
    // Precompute anlm's arguments that should not be part of the benchmark.
    vector<int> ids = calculateEachPixelRegion<double>(noisyImg, regions);
    vector<double> filterSigma = calculateEachPixelSigma<double>(
            noisyImg, ids, regions);

    vector<double> filteredImg(noisyImg.size());

    // Benchmark anlm implementation.
    auto start = Clock::now();
    cuda::adaptiveNonLocalMeans<double>(noisyImg.data(), filteredImg.data(),
                                        ids.data(), filterSigma.data(),
                                        height, width, 5, 5, (double) 5/3,
                                        regions);
    auto stop = Clock::now();

    cout << "Image size: " << height << "x" << width << endl;
    cout << "CUDA ANLM took "
         << (double) std::chrono::duration_cast<std::chrono::microseconds>(
                stop-start).count()/(double) (1000*1000)
         << " seconds using FP64"
         << endl;

    // Use precomputed anlm data of the same image/args to verify implementation.
    if (testFile) {
        ifstream testStream(testFile, ios_base::binary|ios_base::in);
        if (testStream.is_open()) {
            uint32_t tHeight;
            uint32_t tWidth;
            testStream.read((char*) &tHeight, sizeof(uint32_t));
            testStream.read((char*) &tWidth, sizeof(uint32_t));
            vector<double> testData(tHeight*tWidth);
            testStream.read((char*) testData.data(), sizeof(double)*tHeight*tWidth);
            testStream.close();

            int l = 0;

            bool success = equal(testData.begin(), testData.end(), filteredImg.begin(),
                                 [&] (double a, double b) {
                                     bool s = (a-b) < 1.0e-4 && (a-b) > -1.0e-4;
                                     if (!s) cout << l << " Exp:" << a << " Had:" << b << endl;
                                     l++;
                                     return s;
                                 });

            cout << "CUDA Test: " << (success ? "PASSED" : "FAILED") << endl;
        } else cout << "Failed to open " << testFile << endl;
    }
}

// TODO: Maybe some day find the equal parts of FP64/FP32 and specialize only
// the different ones...
void
testCudaAnlmFP32(vector<double> noisyImgFP64, int height, int width,
                 int regions, char *testFile)
{
    // Convert img from FP64 to FP32.
    vector<float> noisyImg(noisyImgFP64.begin(), noisyImgFP64.end());

    // Precompute anlm's argument that should not be part of the benchmark.
    vector<int> ids = calculateEachPixelRegion<float>(noisyImg, regions);
    vector<float> filterSigma = calculateEachPixelSigma<float>(
            noisyImg, ids, regions);

    vector<float> filteredImg(noisyImg.size());

    // Benchmark anlm implementation.
    auto start = Clock::now();
    cuda::adaptiveNonLocalMeans<float>(noisyImg.data(), filteredImg.data(),
                                       ids.data(), filterSigma.data(),
                                       height, width, 5, 5, (float) 5/3,
                                       regions);
    auto stop = Clock::now();

    cout << "Image size: " << height << "x" << width << endl;
    cout << "CUDA ANLM took "
         << (double) std::chrono::duration_cast<std::chrono::microseconds>(
                stop-start).count()/(double) (1000*1000)
         << " seconds using FP32"
         << endl;

    // Use precomputed anlm data of the same image/args to verify implementation.
    if (testFile) {
        ifstream testStream(testFile, ios_base::binary|ios_base::in);
        if (testStream.is_open()) {
            uint32_t tHeight;
            uint32_t tWidth;
            testStream.read((char*) &tHeight, sizeof(uint32_t));
            testStream.read((char*) &tWidth, sizeof(uint32_t));
            vector<double> testData(tHeight*tWidth);
            testStream.read((char*) testData.data(), sizeof(double)*tHeight*tWidth);
            testStream.close();

            int l = 0;

            bool success = equal(testData.begin(), testData.end(), filteredImg.begin(),
                                 [&] (double a, float b) {
                                     bool s = (a-b) < 1.0e-4 && (a-b) > -1.0e-4;
                                     if (!s) cout << l << " Exp:" << a << " Had:" << b << endl;
                                     l++;
                                     return s;
                                 });

            cout << "CUDA Test: " << (success ? "PASSED" : "FAILED") << endl;
        } else cout << "Failed to open " << testFile << endl;
    }
}

/**
 * Calculates the region each pixel belongs to, in range [0, regionsNum).
 */
template<class T>
vector<int>
calculateEachPixelRegion(const vector<T> &img, int regionsNum)
{
    vector<int> ids(img.size());
    for (int i = 0; i < (int) img.size(); ++i) {
        ids[i] = (int) (round(img[i] * ((T) regionsNum - 1)));
        //if (ids[i] >= regionsNum) ids[i] = regionsNum - 1;  // handle 1.0 case
    }

    return ids;
}

/**
 * Calculates filter sigma of each pixel as std deviation of its region.
 */
template<class T>
vector<T>
calculateEachPixelSigma(const vector<T> &img, const vector<int> &ids,
                        int regionsNum)
{
    vector<T> means(regionsNum, 0);
    vector<int> regionSize(regionsNum, 0);
    // Calculate mean values of each region.
    for (int i = 0; i < (int) img.size(); ++i) {
        means[ids[i]] += img[i];
        ++regionSize[ids[i]];
    }
    for (int i = 0; i < (int) means.size(); ++i) means[i] /= regionSize[i];

    // Calculate std deviation of each region.
    vector<T> stds(regionsNum, 0);
    for (int i = 0; i < (int) img.size(); ++i) {
        T d = img[i] - means[ids[i]];
        stds[ids[i]] += d * d;
    }
    for (int i = 0; i < (int) stds.size(); ++i) stds[i] = sqrt(stds[i]/(regionSize[i]-1));

    // Use as sigma for each pixel the std deviation of its region.
    vector<T> sigmas(img.size());
    for (int i = 0; i < (int) sigmas.size(); ++i) sigmas[i] = stds[ids[i]];

    return sigmas;
}
