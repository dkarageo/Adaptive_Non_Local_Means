#include <vector>
#include <cstdlib>
#include <cstdio>
#include "../source/cuda/DMat.hpp"


using namespace std;


__global__ void
printTable(cuda::DMatExpanded<int> table)
{
    if (threadIdx.x != 0 || threadIdx.y != 0 ||
        blockIdx.x != 0 || blockIdx.y != 0)
            return;

    printf("Input:\n");

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%d ", table(i, j));
        }
        printf("\n");
    }

    printf("Expanded:\n");

    for (int i = -2; i < 6; ++i) {
        for (int j = -2; j < 6; ++j) {
            printf("%d ", table(i, j));
        }
        printf("\n");
    }
}

int main()
{
    int nums[] = {1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11, 12,
                  13, 14, 15, 16};

    cuda::DMatExpanded<int> table(nums, (size_t) 4, (size_t) 4, (size_t) 2, (size_t) 2);

    dim3 grid(1, 1);
    dim3 block(1, 1);

    printf("Calling kernel\n");
    printTable<<<grid, block>>>(table);
    printf("Kernel returned\n");

    return 0;
}
