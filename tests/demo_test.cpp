/**
 *
 */

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstdint>


using namespace std;
using Clock=std::chrono::high_resolution_clock;

template<class T>
vector<int>
calculateEachPixelRegion(const vector<T> &img, int regionsNum);
template<class T>
vector<T>
calculateEachPixelSigma(const vector<T> &img, const vector<int> &ids,
                        int regionsNum);


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

    // if (dataStream.bad()) cout << "Failed to read (badbit):" << argv[1] << endl;
    // if (dataStream.fail()) cout << "Failed to read (badbit|failbit):" << argv[1] << endl;
    // if (dataStream.eof()) cout << "Failed to read (eofbit):" << argv[1] << endl;
    // cout << "Loaded " << argv[1] << "-Items:" << noisyImg.size()
    //      << "Image size:" << width << "x" << height << endl;

    // Precompute anlm's argument that should not be part of the benchmark.
    vector<int> ids = calculateEachPixelRegion<double>(noisyImg, regions);
    vector<double> filterSigma = calculateEachPixelSigma<double>(
            noisyImg, ids, regions);

    vector<double> filteredImg(noisyImg.size());

    // Benchmark anlm implementation.
    auto start = Clock::now();
    // cuda::adaptiveNonLocalMeans<double>(noisyImg.data(), filteredImg.data(),
    //                                     ids.data(), filterSigma.data(),
    //                                     height, width, 5, 5, 5/3);
    auto stop = Clock::now();

    cout << "Image size: " << height << "x" << width << endl;
    cout << "CUDA ANLM took "
         << std::chrono::duration_cast<std::chrono::seconds>(stop-start).count()
         << " seconds"
         << endl;

    // Use precomputed anlm data of the same image/args to verify implementation.
    if (testFile) {
        ifstream testStream(testFile, ios_base::binary|ios_base::in);
        uint32_t tHeight;
        uint32_t tWidth;
        testStream.read((char*) &tHeight, sizeof(uint32_t));
        testStream.read((char*) &tWidth, sizeof(uint32_t));
        vector<double> testData(tHeight*tWidth);
        testStream.read((char*) testData.data(), sizeof(double)*tHeight*tWidth);
        testStream.close();

        bool success = equal(testData.begin(), testData.end(), filteredImg.begin(),
                             [] (double a, double b) {
                                 return ((a-b) < 0.01 && (a-b) > -0.01);
                             });

        cout << "CUDA Test: " << (success ? "PASSED" : "FAILED") << endl;
    }

    return 0;
}

/**
 * Calculates the region each pixel belongs to, in range [0, regionsNum).
 */
template<class T>
vector<int>
calculateEachPixelRegion(const vector<T> &img, int regionsNum)
{
    vector<int> ids(img.size());
    for (int i = 0; i < img.size(); ++i) {
        ids[i] = (int) (img[i] * (T) regionsNum);
        if (ids[i] >= regionsNum) ids[i] = regionsNum - 1;  // handle 1.0 case
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
    for (int i = 0; i < img.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= regionsNum) cout << "id:" << ids[i] << endl;
        means[ids[i]] += img[i];
        ++regionSize[ids[i]];
    }
    for (int i = 0; i < means.size(); ++i) means[i] /= regionSize[i];

    // Calculate std deviation of each region.
    vector<T> stds(regionsNum, 0);
    for (int i = 0; i < img.size(); ++i) {
        T d = img[i] - means[ids[i]];
        stds[ids[i]] += d * d;
    }
    for (int i = 0; i < stds.size(); ++i) stds[i] /= regionSize[i];

    // Use as sigma for each pixel the std deviation of its region.
    vector<T> sigmas(img.size());
    for (int i = 0; i < sigmas.size(); ++i) sigmas[i] = stds[ids[i]];

    return sigmas;
}
