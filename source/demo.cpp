/**
 * demo.cpp
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
        cout << "Usage: ./" << argv[0] << " <noisy_img> <regions> [<test_data> <test_ids>]";
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

    // if (dataStream.bad()) cout << "Failed to read (badbit):" << argv[1] << endl;
    // if (dataStream.fail()) cout << "Failed to read (badbit|failbit):" << argv[1] << endl;
    // if (dataStream.eof()) cout << "Failed to read (eofbit):" << argv[1] << endl;
    // cout << "Loaded " << argv[1] << "-Items:" << noisyImg.size()
    //      << "Image size:" << width << "x" << height << endl;

    cuda::deviceInit();

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
    // Precompute anlm's argument that should not be part of the benchmark.
    vector<int> ids = calculateEachPixelRegion<double>(noisyImg, regions);
    vector<double> filterSigma = calculateEachPixelSigma<double>(
            noisyImg, ids, regions);

    vector<double> filteredImg(noisyImg.size());

    // // DEBUG
    //     // for (int i = 0; i < height; ++i) {
    //     //     for (int j = 0; j < width; ++j) {
    //     //         cout << filterSigma[i*width+j] << " ";
    //     //     }
    //     //     cout << endl;
    //     // }
    //
    //     if (argc >= 5) {
    //         char *idsFile = argv[4];
    //
    //         ifstream idsStream(idsFile, ios_base::binary|ios_base::in);
    //         if (idsStream.is_open()) {
    //             uint32_t tHeight;
    //             uint32_t tWidth;
    //             idsStream.read((char*) &tHeight, sizeof(uint32_t));
    //             idsStream.read((char*) &tWidth, sizeof(uint32_t));
    //             vector<double> idsData(tHeight*tWidth);
    //             idsStream.read((char*) idsData.data(), sizeof(double)*tHeight*tWidth);
    //             idsStream.close();
    //
    //             vector<int> testIds(idsData.begin(), idsData.end());
    //
    //             bool success = equal(testIds.begin(), testIds.end(), ids.begin(),
    //                                  [] (int a, int b) {
    //                                      bool s = a == b;
    //                                      if (!s) cout << "Exp:" << a << " Had:" << b << endl;
    //                                      return s;
    //                                  });
    //
    //             cout << "IDs Test: " << (success ? "PASSED" : "FAILED") << endl;
    //         } else cout << "Failed to open " << idsFile << endl;
    //     }
    //
    //     if (argc >= 6) {
    //         char *idsFile = argv[5];
    //
    //         ifstream idsStream(idsFile, ios_base::binary|ios_base::in);
    //         if (idsStream.is_open()) {
    //             uint32_t tHeight;
    //             uint32_t tWidth;
    //             idsStream.read((char*) &tHeight, sizeof(uint32_t));
    //             idsStream.read((char*) &tWidth, sizeof(uint32_t));
    //             vector<double> idsData(tHeight*tWidth);
    //             idsStream.read((char*) idsData.data(), sizeof(double)*tHeight*tWidth);
    //             idsStream.close();
    //
    //             bool success = equal(idsData.begin(), idsData.end(), filterSigma.begin(),
    //                                  [] (double a, double b) {
    //                                      bool s = (a-b) < 0.001 && (a-b) > -0.001;
    //                                      if (!s) cout << "Exp:" << a << " Had:" << b << endl;
    //                                      return s;
    //                                  });
    //
    //             cout << "STDs Test: " << (success ? "PASSED" : "FAILED") << endl;
    //         } else cout << "Failed to open " << idsFile << endl;
    //     }
    // // END DEBUG

    // Benchmark anlm implementation.
    auto start = Clock::now();
    cuda::adaptiveNonLocalMeans<double>(noisyImg.data(), filteredImg.data(),
                                        ids.data(), filterSigma.data(),
                                        height, width, 5, 5, (double) 5/3);
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
                                       height, width, 5, 5, (float) 5/3);
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
