#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <chrono>

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float()> F) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      M[i * cols + j] = F();
    }
  }
}

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float(int, int)> F) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      M[i * cols + j] = F(i, j);
    }
  }
}

template<typename T>
void print_matrix(T* M, int rows, int cols) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
        std::cout << M[i * cols + j] << " ";
    }
    std::cout << "\n";
  }
}


template<typename T>
T maxDiff(T* A1, T* A2, int rows, int cols){
  T maxDiff = A1[0] - A2[0];
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++){
      T diff = abs(A1[i * cols + j] - A2[i * cols + j]);
      if( diff > maxDiff) {
          maxDiff = diff;
      }
    }
  }

  
  return maxDiff;
}

template<typename T>
void check_copy(T* dM, T* hM, int d_size, char* label) {
  T* cp_host = (T*) malloc(d_size*sizeof(T));

  std::cout << "\n\n" << label << std::endl;

	cudaMemcpy(cp_host, dM, d_size, cudaMemcpyDeviceToHost);
  std::cout << "CPU original" << std::endl;
  print_matrix<T>( hM, 10, 10);
  std::cout << "GPU copy" << std::endl;
  print_matrix<T>( cp_host, 10, 10);

  free(cp_host);
}

#define ROW_TILE_WIDTH  32
#define COL_TILE_WIDTH  32

#define EPSILON         (1e-5)

template<typename T>
__global__
void naive_matrix_multiply(T *A, T *B, T* C, int width, int cRows, int cCols)
{
  __shared__ T shATile[ROW_TILE_WIDTH][COL_TILE_WIDTH];
  __shared__ T shBTile[ROW_TILE_WIDTH][COL_TILE_WIDTH];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  T pValue = 0;

  // iterate for width/COL_TILE_WIDTH number of times
  // to compute the C tile
  for(int p = 0; p < width/COL_TILE_WIDTH; p++) {
    //load values to tiles from A and B
    shATile[threadIdx.y][threadIdx.x] = A[row * width + p * ROW_TILE_WIDTH + threadIdx.x];
    shBTile[threadIdx.y][threadIdx.x] = B[(p * COL_TILE_WIDTH + threadIdx.y) * cCols + col];

    // wait until all threads finish loading values
    __syncthreads();
    // update pValue for this thread
    for(int i = 0; i < COL_TILE_WIDTH;  i++) pValue += shATile[threadIdx.y][i] * shBTile[i][threadIdx.x];
    // wait until all threads finish computing pValue before overwriting
    __syncthreads();

  }
  
  C[row * cCols + col] = pValue;

}

template<typename T>
void naive_matrix_multiply_cpu(T *A, T *B, T* C, int width, int C_rows, int C_cols){
  
  for(int i = 0; i < C_rows; i++)
    for(int j = 0; j < C_cols; j++){
      T value = 0.0f;
      for(int k = 0; k < width; k++){
        value += A[i * width + k] * B[k * C_cols + j];
      }

     
      C[i * C_cols + j] = value;
    }
}

extern "C" void perform_matmul() {
  int A_rows = 1 << 8;
  int A_cols = 1 << 10;
  int B_cols = 1 << 11;

  int B_rows = A_cols;
  int C_rows = A_rows;
  int C_cols = B_cols;
  int A_size = A_rows * A_cols;
  int B_size = B_rows * B_cols;
  int C_size = C_rows * C_cols;
  float *A, *B, *C, *C_host;
  float *A_cpu, *B_cpu, *C_cpu;
  // timing
  cudaEvent_t start_gpu, stop_gpu;
  float gpu_time_ms = 0;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  
  std::cout << "A size: " << A_size << ", B size: " << B_size << ", C Size: " << C_size << std::endl;  
  std::cout << "A: " << A_rows << "x" << A_cols << ", B: " << B_rows << "x" << B_cols <<  ", C: " << C_rows << "x" << C_cols << std::endl;  
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMalloc(&A, A_size*sizeof(float));

  cudaMalloc(&B, B_size*sizeof(float));
  cudaMalloc(&C, C_size*sizeof(float));
  C_host = (float*) malloc(C_size*sizeof(float));


  A_cpu = (float*) malloc(A_size*sizeof(float));
  B_cpu = (float*) malloc(B_size*sizeof(float));
  C_cpu = (float*) malloc(C_size*sizeof(float));

  // initialize A and B matrices
  auto all_ones = []() -> float {
    return 1.0f;
  };

  srand (time(NULL));
  auto rand_numbers = []() -> float {
    return static_cast<float>(rand())/(static_cast<float>(RAND_MAX/1000));
  };

  auto index_based = [](int i, int j) -> float {
    return j;
  };

  initialize_matrix<float>(A_cpu, A_rows, A_cols, rand_numbers);
	cudaMemcpy(A, A_cpu, A_size * sizeof(float), cudaMemcpyHostToDevice);  

  initialize_matrix<float>(B_cpu, B_rows, B_cols, rand_numbers);
	cudaMemcpy(B, B_cpu, B_size * sizeof(float), cudaMemcpyHostToDevice);


  // launch kernel

  dim3 dim_grid(C_cols/COL_TILE_WIDTH, C_rows/ROW_TILE_WIDTH, 1);
  dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

  cudaEventRecord(start_gpu);
  naive_matrix_multiply<float><<<dim_grid, dim_block>>>(A, B, C, A_cols, C_rows, C_cols);
  cudaEventRecord(stop_gpu);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
	cudaMemcpy(C_host, C, C_size * sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop_gpu);
  cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);
  

  // check results on CPU
  auto t1 = std::chrono::system_clock::now();
  naive_matrix_multiply_cpu<float>(A_cpu, B_cpu, C_cpu, A_cols, C_rows, C_cols);
  auto t2 = std::chrono::system_clock::now();

  if(fabsf(maxDiff<float>(C_host, C_cpu, C_rows, C_cols)) <= (float)EPSILON )
     std::cout << "PASS" << std::endl;
  else {
     std::cout << "FAIL" << std::endl;
     std::cout << "GPU result [0:9, 0:9]" << std::endl;
     print_matrix<float>( C_host, 10, 10);
     std::cout << "CPU result [0:9, 0:9]" << std::endl;
     print_matrix<float>( C_cpu, 10, 10);

  }

  auto cpu_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.0f;
  std::cout << "GPU time = " << gpu_time_ms << "ms" << std::endl;
  std::cout << "CPU time = " << cpu_time_ms << "ms" << std::endl;
  std::cout << "Speedup = " << cpu_time_ms/gpu_time_ms << std::endl;
  
  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  free(C_host);

  free(A_cpu);
  free(B_cpu);
  free(C_cpu);  
}