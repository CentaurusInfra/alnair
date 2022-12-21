#include <storage_gds.cuh>
#include <utils.cuh>

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <cmath>

GDSStorage::~GDSStorage()
{
    if (this->gpu)
        cudaFree(this->device_data);
    this->gpu = false;
    std::cout << "\n GDSStorageDestructor executed";
}

GDSStorage::GDSStorage(const std::vector<int> &_shape) : shape(_shape), flag(HOST_MEM) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  if (flag==GDS_MEM) {
    if (this->gpu) {
        std::cout << "Already registered?" << std::endl;
      	cuFileBufDeregister((char*)this->device_data);
        cudaFree(this->device_data);
    } 
	  cudaMalloc(&(this->device_data), size);
    cuFileBufRegister((char*)this->device_data, size, 0);
    this->gpu = true;
  }
  else 
      std::cout << "Not supported" << std::endl;
}

GDSStorage::GDSStorage(const std::vector<int> &_shape, float value) : shape(_shape) , flag(HOST_MEM){
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  if (flag==GDS_MEM) {
    if (this->gpu) {
        std::cout << "Already registered?" << std::endl;
      	cuFileBufDeregister((char*)this->device_data);
        cudaFree(this->device_data);
    } 
	  cudaMalloc(&(this->device_data), size);
    cuFileBufRegister((char*)this->device_data, size, 0);
    this->gpu = true;
  }
  else 
      std::cout << "Not supported" << std::endl;
}

GDSStorage::GDSStorage(const std::vector<int> &_shape,
                 const std::vector<float> &_data)
    : shape(_shape),  flag(HOST_MEM){
  this->check_size();
}

GDSStorage::GDSStorage(const GDSStorage &other) { *this = other; }

GDSStorage &GDSStorage::operator=(const GDSStorage &other) {
  // if (this != &other) {
  //   this->shape = other.shape;
  //   this->data = other.data;
  // }

  // return *this;
  std::cout << "not supported" << std::endl;
}

GDSStorage::GDSStorage(const std::vector<int> &_shape, int flag) : shape(_shape) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  if (flag==GDS_MEM) {
    if (this->gpu) {
        std::cout << "Already registered?" << std::endl;
      	cuFileBufDeregister((char*)this->device_data);
        cudaFree(this->device_data);
    } 
	  cudaMalloc(&(this->device_data), size);
    cuFileBufRegister((char*)this->device_data, size, 0);
    this->gpu = true;
  }
  else 
    printf("Not implemented\n");
}

GDSStorage::GDSStorage(GDSStorage &&other) 
{ 
  if (flag==GDS_MEM)
    printf("Not implemented\n");
  else 
    *this = std::move(other); 
}

GDSStorage &GDSStorage::operator=(GDSStorage &&other) {
  if (this != &other) {
    this->shape = std::move(other.shape);
    if (flag == GDS_MEM)
        // this->device_data = std::move(other.device_data);
        printf("Not implemented\n");
    else 
        printf("Not implemented\n");
        // this->data = std::move(other.data);
  }
  return *this;
}

void GDSStorage::reshape(const std::vector<int> &_shape) {
  this->shape = _shape;
  this->check_size();
}

void GDSStorage::resize(const std::vector<int> &_shape) {

    if (flag == GDS_MEM) {
        // this->device_data = std::move(other.device_data);
        if (this->gpu) {
            std::cout << "Already registered?" << std::endl;
            cuFileBufDeregister((char*)this->device_data);
            cudaFree(this->device_data);
        } 
        this->shape = _shape;

        int size = 1;
        for (int i = 0; i < _shape.size(); i++) {
            size *= _shape[i];
        }
        cudaMalloc(&(this->device_data), size);
        cuFileBufRegister((char*)this->device_data, size, 0);
        this->gpu = true;           
    }
    else {
        std::cout << "Not supported" << std::endl;
        // this->shape = _shape;

        // int size = 1;
        // for (int i = 0; i < _shape.size(); i++) {
        //     size *= _shape[i];
        // }

        // if (size != this->data.size()) {
        //     this->data.resize(size);
        // }
    }
}

__global__ void storage_xavier(float *a, int size, float scale,
                               curandState *cs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    curand_init(1234, index, 0, &cs[index]);
    a[index] = (curand_uniform(&cs[index]) * 2 - 1) * scale;
  }
}

void GDSStorage::xavier(size_t in_size, size_t out_size) {
  // float *a_ptr = RAW_PTR(this->data);
  float *a_ptr = (float *)this->device_data;
  int size;
  if (flag == HOST_MEM)
        std::cout << "Not supported" << std::endl;
  else
      size = sizeof(this->device_data);

  int grid_size = ceil((float)(size) / BLOCK_SIZE);

  thrust::device_vector<curandState> cs(size);
  curandState *cs_ptr = RAW_PTR(cs);
  float scale = std::sqrt((float)6) / std::sqrt((float)(in_size) + out_size);
  storage_xavier<<<grid_size, BLOCK_SIZE>>>(a_ptr, size, scale, cs_ptr);

  CUDA_POST_KERNEL_CHECK;
}

void GDSStorage::check_size() {
  int size = 1;
  for (int i = 0; i < this->shape.size(); i++) {
    size *= this->shape[i];
  }
  if (flag == HOST_MEM)
        std::cout << "Not supported" << std::endl;
  else
      CHECK_EQ(size, sizeof(this->device_data), "GDSStorage: size error");

}