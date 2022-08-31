#include <stdio.h>
#include <cuda.h>


__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
struct Param { int n; float a; float *x; float *y;};

int main(void)
{
	//CUdeviceptr dptr;
	//cuMemAlloc(&dptr, 1024);
	 cudaSetDevice(0);
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(1024*N*sizeof(float));
  y = (float*)malloc(1024*N*sizeof(float));

  cudaMalloc(&d_x, 1024*N*sizeof(float));
  cudaMalloc(&d_y, 1024*N*sizeof(float));
  size_t pitch_a, pitch_b, pitch_c;
  float *a;
  cudaMallocPitch((void**) &a, &pitch_a, sizeof(float) * 10, 10);
  float *b;
  cudaMallocManaged(&b, 1024*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  
  Param param;
  param.n = N;
  param.a = 2.0f;
  param.x = d_x;
  param.y = d_y;
  void *kArgs = {&param};
  dim3 gridDim;
  gridDim.x = (N+255)/256;
  
  dim3 blockDim;
  blockDim.x = 256;
  //cudaLaunchCooperativeKernel((void *)saxpy, gridDim, blockDim, (void**)(N, 2.0f, d_x, d_y), 0, 0);
  
  cudaDeviceSynchronize();
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
  cudaArray* arr;

      //Create Channel Descriptor. float is just for example. Change it to required data type.
      cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();

      //Allocate Memory
      cudaMallocArray(&arr,&channel,5, 1,cudaArrayDefault);
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	cudaExtent extent;

	extent.width = 1; // Note, for cudaArrays the width field is the width in elements, not bytes

	extent.height = 2;

	extent.depth = 3;

	cudaArray *array = 0;

  printf("1000\n");
	cudaMalloc3DArray(&array,&desc,extent,cudaArrayLayered);
  cudaFreeArray(arr);
  cudaFreeArray(array);
  printf("2000\n");

  cudaMipmappedArray_t mip = 0;
  cudaChannelFormatDesc des = cudaCreateChannelDesc<float>();
  cudaMallocMipmappedArray(&mip, &des, extent, 2, 0);
  printf("3000. %lu\n", mip);
  if (mip != 0)
    cudaFreeMipmappedArray(mip);
  printf("4000\n");

  
}
