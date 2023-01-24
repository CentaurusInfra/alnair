#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <sys/stat.h>
// #include <sys/types.h>
// #include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <tuple>
#include <vector>
#include "cufile.h"
#include "gds_test.cuh"

//
// test image reader on device
//
float* read_image(char * minst_data_path, int length) {
    std::unique_ptr<DataSetGDS> gds;
    char * gds_image;
    
    gds.reset(new DataSetGDS(minst_data_path, true));
    gds_image = gds->get_train_data();

    int end = std::min(length, gds->get_train_datasize());
    thrust::host_vector<char> host_img(gds_image, gds_image+end);

    return ((float*)&host_img[0]);
}

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

#define KB(x) ((x)*1024L)
#define TESTFILE "/mnt/test"

__global__ void hello(char *str) {
	printf("Hello World!\n");
	printf("buf: %s\n", str);
}

__global__ void strrev(char *str, int *len) {
	int size = 0;
	while (str[size] != '\0') {
		size++;
	}
	for(int i=0;i<size/2;++i) {
		char t = str[i];
		str[i] = str[size-1-i];
		str[size-1-i] = t;
	}
	/*
	printf("buf: %s\n", str);
	printf("size: %d\n", size);
	*/
	*len = size;
}

__global__ void g_reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
//   return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
//          ((unsigned int)ch3 << 8) + ch4;
}

unsigned int reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
         ((unsigned int)ch3 << 8) + ch4;
}




template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

void test_transform() 
{
    float x[4] = {1.0, 2.0, 3.0, 4.0};

    // transfer to device
    thrust::device_vector<float> d_x(x, x + 4);

	print_vector("d_x", d_x);

    // setup arguments
    square<float>        unary_op;
    thrust::plus<float> binary_op;
    float init = 0;

    // compute norm
	thrust::device_ptr<float> d_ptr = d_x.data();	

    // float norm = std::sqrt( thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op) );
    float norm = std::sqrt( thrust::transform_reduce(d_ptr, d_ptr + sizeof(x), unary_op, init, binary_op) );

    std::cout << "norm is " << norm << std::endl;	
}

int read_image_data(char * file_name, unsigned int batchsize, char* data) {
	int fd;
	int ret;
	char *gpumem_buf, *meta;
	int *sys_len;
	int *gpu_len;
	int parasize=KB(1);

	int bufsize = KB(4);
	// int n_bufsize = n_rows * n_cols * sizeof(float);
	off_t file_offset = 0;
	off_t mem_offset = 0;
	int metasize=16;


	CUfileDescr_t cf_desc; 
	CUfileHandle_t cf_handle;

	cuFileDriverOpen();
	fd = open(file_name, O_RDWR|O_DIRECT);
	cf_desc.handle.fd = fd;
	cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

	cuFileHandleRegister(&cf_handle, &cf_desc);
	unsigned int magic_number = 0;
	unsigned int number_of_images = 0;
	unsigned int n_rows = 0;
	unsigned int n_cols = 0;


	// thrust::device_vector<char> data_tt(bufsize);
	// gpumem_buf = (char*)thrust::raw_pointer_cast(&data[0]);	
	cudaMalloc(&meta, metasize);
	// cuFileBufRegister((char*)meta, metasize, 0);

	ret = cuFileRead(cf_handle, (char*)meta, metasize, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		printf("ret %d\n", ret);
	}

	sys_len = (int*)malloc(parasize);
	cudaMemcpy(sys_len, meta, metasize, cudaMemcpyDeviceToHost);
	magic_number = reverse_int(((int*)sys_len)[0]);
	number_of_images = reverse_int(((int*)sys_len)[1]);
	n_rows = reverse_int(((int*)sys_len)[2]);
	n_cols = reverse_int(((int*)sys_len)[3]);

	std::cout << file_name << std::endl;
	std::cout << "magic number = " << magic_number << std::endl;
	std::cout << "number of images = " << number_of_images << std::endl;
	std::cout << "rows = " << n_rows << std::endl;
	std::cout << "cols = " << n_cols << std::endl;
    int length = std::min(number_of_images, batchsize);
	bufsize = n_rows * n_cols * sizeof(char) * length;


	cudaMalloc(&gpumem_buf, bufsize + 1);
	file_offset = 4 * sizeof(int);
	mem_offset = 0;

	cuFileBufRegister((char*)gpumem_buf, bufsize + 1, 0);

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize + 1, file_offset, mem_offset);

	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		cudaMalloc(&data, length + 1);
		cudaMemcpy(data, gpumem_buf, length, cudaMemcpyDeviceToHost);
		ret = length;
	}
	cuFileBufDeregister((char*)gpumem_buf);
	cudaFree(gpumem_buf);
	close(fd);
	cuFileDriverClose();

    return ret;

}

int read_image_data(char * file_name, unsigned int batchsize, char* data, int* row, int *col) {
	int fd;
	int ret;
	char *gpumem_buf, *meta;
	int *sys_len;
	int *gpu_len;
	int parasize=KB(1);

	int bufsize = KB(4);
	// int n_bufsize = n_rows * n_cols * sizeof(float);
	off_t file_offset = 0;
	off_t mem_offset = 0;
	int metasize=16;


	CUfileDescr_t cf_desc; 
	CUfileHandle_t cf_handle;

	cuFileDriverOpen();
	fd = open(file_name, O_RDWR|O_DIRECT);
	cf_desc.handle.fd = fd;
	cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

	cuFileHandleRegister(&cf_handle, &cf_desc);
	unsigned int magic_number = 0;
	unsigned int number_of_images = 0;
	unsigned int n_rows = 0;
	unsigned int n_cols = 0;


	// thrust::device_vector<char> data_tt(bufsize);
	// gpumem_buf = (char*)thrust::raw_pointer_cast(&data[0]);	
	cudaMalloc(&meta, metasize);
	// cuFileBufRegister((char*)meta, metasize, 0);

	ret = cuFileRead(cf_handle, (char*)meta, metasize, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		printf("ret %d\n", ret);
	}

	sys_len = (int*)malloc(parasize);
	cudaMemcpy(sys_len, meta, metasize, cudaMemcpyDeviceToHost);
	magic_number = reverse_int(((int*)sys_len)[0]);
	number_of_images = reverse_int(((int*)sys_len)[1]);
	n_rows = reverse_int(((int*)sys_len)[2]);
	n_cols = reverse_int(((int*)sys_len)[3]);

	std::cout << file_name << std::endl;
	std::cout << "magic number = " << magic_number << std::endl;
	std::cout << "number of images = " << number_of_images << std::endl;
	std::cout << "rows = " << n_rows << std::endl;
	std::cout << "cols = " << n_cols << std::endl;
    int length = std::min(number_of_images, batchsize);
	bufsize = n_rows * n_cols * sizeof(char) * length;


	cudaMalloc(&gpumem_buf, bufsize + 1);
	file_offset = 4 * sizeof(int);
	mem_offset = 0;

	cuFileBufRegister((char*)gpumem_buf, bufsize + 1, 0);

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize + 1, file_offset, mem_offset);

	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		cudaMalloc(&data, length + 1);
		cudaMemcpy(data, gpumem_buf, length, cudaMemcpyDeviceToHost);
		ret = length;
		*row = n_rows;
		*col = n_cols;
	}
	cuFileBufDeregister((char*)gpumem_buf);
	cudaFree(gpumem_buf);
	close(fd);
	cuFileDriverClose();

    return ret;

}


//
// test cufile read with numpy formatted image data
//
void test_numpy(char * file_name) {
	int fd;
	int ret;
	char *gpumem_buf, *meta;
	int *sys_len;
	int *gpu_len;
	int parasize=KB(1);

	int bufsize = KB(4);
	// int n_bufsize = n_rows * n_cols * sizeof(float);
	off_t file_offset = 0;
	off_t mem_offset = 0;
	int metasize=16;


	CUfileDescr_t cf_desc; 
	CUfileHandle_t cf_handle;

	cuFileDriverOpen();
	fd = open(file_name, O_RDWR|O_DIRECT);
	cf_desc.handle.fd = fd;
	cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

	cuFileHandleRegister(&cf_handle, &cf_desc);
	unsigned int magic_number = 0;
	unsigned int number_of_images = 0;
	unsigned int n_rows = 0;
	unsigned int n_cols = 0;


	// thrust::device_vector<char> data_tt(bufsize);
	// gpumem_buf = (char*)thrust::raw_pointer_cast(&data[0]);	
	cudaMalloc(&meta, metasize);
	// cuFileBufRegister((char*)meta, metasize, 0);

	ret = cuFileRead(cf_handle, (char*)meta, metasize, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		printf("ret %d\n", ret);
	}

	sys_len = (int*)malloc(parasize);
	cudaMemcpy(sys_len, meta, metasize, cudaMemcpyDeviceToHost);
	magic_number = reverse_int(((int*)sys_len)[0]);
	number_of_images = reverse_int(((int*)sys_len)[1]);
	n_rows = reverse_int(((int*)sys_len)[2]);
	n_cols = reverse_int(((int*)sys_len)[3]);

	std::cout << file_name << std::endl;
	std::cout << "magic number = " << magic_number << std::endl;
	std::cout << "number of images = " << number_of_images << std::endl;
	std::cout << "rows = " << n_rows << std::endl;
	std::cout << "cols = " << n_cols << std::endl;
	bufsize = n_rows * n_cols * sizeof(char) * number_of_images;


	cudaMalloc(&gpumem_buf, bufsize);
	file_offset = 4 * sizeof(int);
	mem_offset = 0;

	cuFileBufRegister((char*)gpumem_buf, bufsize, 0);

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d\n", ret); 
	} else {
		printf("ret %d\n", ret);
	}

	cuFileBufDeregister((char*)gpumem_buf);
	cudaFree(gpumem_buf);
	close(fd);
	cuFileDriverClose();
}
