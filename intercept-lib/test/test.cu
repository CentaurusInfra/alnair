#include <dlfcn.h>
#include <string.h>
#include <iostream>
#include <cuda.h>


int main() {
    std::cout << "testing cudaMalloc..." << std::endl;
    float *d_x;

    cudaMalloc(&d_x, 1024*8*sizeof(float));

    return 0;
}