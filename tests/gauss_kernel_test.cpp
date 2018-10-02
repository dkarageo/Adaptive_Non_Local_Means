#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>


using namespace std;


template <class T>
vector<T>
calculateGaussianFilter(int m, int n, T sigma)
{
    vector<T> filter(m*n);
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

    T max = filter[m*n/2] / sum;  // greatest kernel will always be on center
    for (int i = 0; i < m*n; ++i) filter[i] /= (sum * max);

    return filter;
}

int main(int argc, char *argv[])
{
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    double sigma = atof(argv[3]);

    cout << m << endl;
    cout << n << endl;
    cout << sigma << endl;

    vector<double> filter = calculateGaussianFilter<double>(m, n, sigma);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << filter[i*m+j] << " ";
        }
        cout << endl;
    }

    return 0;
}
