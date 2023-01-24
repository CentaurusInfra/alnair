#ifndef __UTIL_FUNC__
#define __UTIL_FUNC__


#include <functional>

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float()> F);

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float(int, int)> F);

template<typename T>
void print_matrix(T* M, int rows, int cols);


template<typename T>
T maxDiff(T* A1, T* A2, int rows, int cols);

template<typename T>
void check_copy(T* dM, T* hM, int d_size, char* label);

extern "C" void perform_matmul();

#endif __UTIL_FUNC__