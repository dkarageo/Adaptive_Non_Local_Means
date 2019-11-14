# **Adaptive Non Local Means**

## An efficient CUDA implementation of Adaptive Non Local Means algorithm for image denoising.


Developed by *Dimitrios Karageorgiou*,\
during the course *Parallel And Distributed Systems*,\
*Aristotle University Of Thessaloniki, Greece,*\
*2017-2018.*


**Adaptive Non Local Means** is a variaton of Non Local Means algorithm, allowing for an irregular search window instead of a fixed rectangular area. It also allows the usage of different sigma values for each pixel of the image.

Under current project, an efficient CUDA implementation of Adaptive Non Local Means algorithm is provided. It is intended for use with grayscale images. Consists of two different implementations, one for small and medium sized images and one for large ones, with the first one taking about half the time of the second to complete. Selection of the implementation is done dynamically at runtime. What is considered a small or a large image though, depends on the amount of VRAM the GPU has and the arguments the algorithm is called with. For example a 2048x2048 image in the common case of using a 5x5 patch size (defined the same way as in the original Non Local Means), will need about 512MB VRAM, thus being a small image for most modern GPUs.


### How to compile demo:

`make` : Builds a demo.

`make force_big` : Builds a demo that forces for all images the usage of the implementation intended for the big ones.

Executables are located inside `bin` folder under project's root.

In order to successfully compile the following tools are required:
- Nvidia CUDA Toolkit's compiler `nvcc`.
- A compiler with full C++11 support (like a recent version of g++).
- GNU Make (preferably 3.80 or newer).

In order to test the executable use `make test` (requires previous compilation of the executable using the targets above).


### How to run a series of benchmarks:

`make benchmark` : Runs a series of benchmarks for all datasets contained under *test_datasets* folder.


### How to manually run demo:

While the demo is actually a benchmark and a test runner, its executable can be directly called as following:

`./bin/demo_anlm <noisy_img> <regions> [<filtered_image>]`

where:
- noisy_img : A *.karas* file containing a noisy image.
- regions : Number of regions the image will be divided into in order to apply ANLM.
- filtered_image [optional] : A *.karas* file containing the precalculated filtered image using a 3rd party ANLM imlementation. This option effectively enables testing mode and is intended for comparing the results against different ANLM implementations.

Someone may be wondering what *.karas* files are. They are just custom binary files for storing 2D matrices of *doubles*. They start with two *uint32_t* fields containing height and width of the matrix respectively. Then height*width *doubles* are following. The matrix is stored in row major order. Under `helper_scripts` folder many MATLAB scripts can be found for converting an image file to a *.karas* file and vice-versa, for applying noise to an image, etc.


### How to use included implementation:

- In order to use the provided Adaptive Non Local Means implementation, include `cuda/anlm.hpp` header file.
- In order to change the maximum VRAM allowed to be used before switching to the implementation for big images, set `MAX_DEVICE_DRAM_USAGE` macro to the desired amount in MB. Default amount is 1480MB, because its original target was an old Nvidia GTX 480.


### Licensing:

This project is licensed under GNU GPL v3.0 license. A copy of this license is contained in current project. It applies to all files in this project whether or not it is stated in them.
