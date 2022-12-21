#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <dataset_gds.cuh>
#include <utils.cuh>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>

DataSetGDS::DataSetGDS(std::string mnist_data_path, bool shuffle)
    : shuffle(shuffle), train_data_index(0), test_data_index(0) {
  // train data
  this->read_images(mnist_data_path + "/train-images-idx3-ubyte",
                    this->train_data);
  // this->read_labels(mnist_data_path + "/train-labels-idx1-ubyte",
  //                   this->train_label);
  // // test data
  // this->read_images(mnist_data_path + "/t10k-images-idx3-ubyte",
  //                   this->test_data);
  // this->read_labels(mnist_data_path + "/t10k-labels-idx1-ubyte",
                    // this->test_label);
}

void DataSetGDS::reset() {
  this->train_data_index = 0;
  this->test_data_index = 0;

  if (shuffle) {
    // keep random seed same
    // unsigned int seed =
    //     std::chrono::system_clock::now().time_since_epoch().count() % 1234;

    // std::shuffle(this->train_data.begin(), this->train_data.end(),
    //              std::default_random_engine(seed));
    // std::shuffle(this->train_label.begin(), this->train_label.end(),
    //              std::default_random_engine(seed));
      std::cout << "Not implemented" << std::endl; 
  }
}

void DataSetGDS::forward(int batch_size, bool is_train) {
  if (is_train) {
    int start = this->train_data_index;
    int data_size = sizeof(this->train_data) / sizeof(char);
    int end = std::min(this->train_data_index + batch_size,
                       data_size);
    this->train_data_index = end;
    int size = end - start;

    // init device memory
    std::vector<int> output_shape{size, 1, this->height, this->width};
    std::vector<int> output_label_shape{size, 10};
    INIT_GDSSTORAGE(this->output, output_shape, GDS_MEM);
    INIT_GDSSTORAGE(this->output_label, output_label_shape, GDS_MEM);
    // thrust::fill(this->output_label->get_data().begin(),
    //              this->output_label->get_data().end(), 0);

    // copy to device memory
    int im_stride = 1 * this->height * this->width;
    int one_hot_stride = 10;

    // thrust::host_vector<
    //     float, thrust::system::cuda::experimental::pinned_allocator<float>>
    //     train_data_buffer;
    // train_data_buffer.reserve(size * im_stride);

    for (int i = start; i < end; i++) {
      // train_data_buffer.insert(train_data_buffer.end(),
      //                          this->train_data[i].begin(),
      //                          this->train_data[i].end());
      this->output_label
          ->get_data()[(i - start) * one_hot_stride + this->train_label[i]] = 1;
    }
    // this->output->get_data() = train_data_buffer;

  } else {
    int start = this->test_data_index;
    int end = std::min(this->test_data_index + batch_size,
                       (int)this->test_data_size);
    this->test_data_index = end;
    int size = end - start;

    // init device memory
    std::vector<int> output_shape{size, 1, this->height, this->width};
    std::vector<int> output_label_shape{size, 10};
    INIT_GDSSTORAGE(this->output, output_shape, GDS_MEM);
    INIT_GDSSTORAGE(this->output_label, output_label_shape, GDS_MEM);
    // thrust::fill(this->output_label->get_data().begin(),
    //              this->output_label->get_data().end(), 0);

    // copy to device memory
    int im_stride = 1 * this->height * this->width;
    int one_hot_stride = 10;

    // thrust::host_vector<
    //     float, thrust::system::cuda::experimental::pinned_allocator<float>>
    //     test_data_buffer;
    // test_data_buffer.reserve(size * im_stride);

    for (int i = start; i < end; i++) {
        // this->output->get_data() = 
    //   test_data_buffer.insert(test_data_buffer.end(),
    //                           this->test_data[i].begin(),
    //                           this->test_data[i].end());
      this->output_label->get_data()[(i - start) * one_hot_stride + this->test_label[i]] = 1;
    }
    // this->output->get_data() = test_data_buffer;
  }
}

bool DataSetGDS::has_next(bool is_train) {
  if (is_train) {
    return this->train_data_index < this->train_data_size;
  } else {
    return this->test_data_index < this->test_data_size;
  }
}

void DataSetGDS::print_im() {
  int size = this->output->get_shape()[0];
  int im_stride = 1 * height * width;

  for (int k = 0; k < size; k++) {
    int max_pos = -1;
    float max_value = -FLT_MAX;
    for (int i = 0; i < 10; i++) {
      //
      //D2Hcopy
      // float val = this->output_label->get_data_float[k * 10 + i];
      // if (val > max_value) {
      //   max_value = val;
      //   max_pos = i;
      // }
    }

    std::cout << max_pos << std::endl;
    const float * data = (float *) this->output->get_data();
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        std::cout << (data[k * im_stride + i * width + j] > 0 ? "* " : "  ");
      }
      std::cout << std::endl;
    }
  }
}

unsigned int DataSetGDS::reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
         ((unsigned int)ch3 << 8) + ch4;
}
void DataSetGDS::read_images(std::string file_name, char * gpumem_buf) {
	int fd;
	int ret;
	char *meta;
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
	// std::cout << file_name << std::endl;

	cuFileDriverOpen();
	fd = open(file_name.c_str(), O_RDWR|O_DIRECT);
	cf_desc.handle.fd = fd;
	cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

	cuFileHandleRegister(&cf_handle, &cf_desc);
	unsigned int magic_number = 0;
	unsigned int number_of_images = 0;
	unsigned int n_rows = 0;
	unsigned int n_cols = 0;


	// thrust::device_vector<char> data_tt(bufsize);
	// gpumem_buf = (char*)thrust::raw_pointer_cast(&data[0]);	

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

	std::cout << "magic number = " << magic_number << std::endl;
	std::cout << "number of images = " << number_of_images << std::endl;
	std::cout << "rows = " << n_rows << std::endl;
	std::cout << "cols = " << n_cols << std::endl;
	bufsize = n_rows * n_cols * sizeof(char) * number_of_images;
  cudaFree(meta);
  free(sys_len);

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

	// cuFileBufDeregister((char*)gpumem_buf);
	// cudaFree(gpumem_buf);
	close(fd);
	cuFileDriverClose();  
}

// void DataSetGDS::read_images(std::string file_name,
//                           char* gpuimg_buf) {
//     int fd;
//     int ret;
    
//     int *sys_len;
//     int *meta;
//     int parasize=KB(1);
//     int bufsize = KB(4);

//     off_t file_offset = 0;
//     off_t mem_offset = 0;
//     int metasize = 4 * sizeof(int);
//     unsigned int magic_number = 0;
//     unsigned int number_of_images = 0;
//     unsigned int n_rows = 0;
//     unsigned int n_cols = 0;

//     sys_len = (int*)malloc(parasize);

//     CUfileDescr_t cf_desc; 
//     CUfileHandle_t cf_handle;

//     cuFileDriverOpen();
//     fd = open(file_name.c_str(), O_RDWR|O_DIRECT);
//     cf_desc.handle.fd = fd;
//     cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
//     cuFileHandleRegister(&cf_handle, &cf_desc);

//     cudaMalloc(&meta, bufsize);
//     // thrust::device_ptr<char> dev_ptr(gpumem_buf);
//     // cuFileBufRegister((char*)meta, bufsize, 0);

//     ret = cuFileRead(cf_handle, (char*)meta, metasize, file_offset, mem_offset);
//     if (ret < 0) {
//       printf("cuFileRead failed : %d\n", ret); 
//     } 
//     cudaMemcpy(sys_len, meta, metasize, cudaMemcpyDeviceToHost);
//     // cuFileBufRegister((char*)meta, bufsize, 0);
//     magic_number = this->reverse_int(((int*)sys_len)[0]);
//     number_of_images = this->reverse_int(((int*)sys_len)[1]);
//     n_rows = this->reverse_int(((int*)sys_len)[2]);
//     n_cols = this->reverse_int(((int*)sys_len)[3]);

//     std::cout << file_name << std::endl;
//     std::cout << "magic number = " << magic_number << std::endl;
//     std::cout << "number of images = " << number_of_images << std::endl;
//     std::cout << "rows = " << n_rows << std::endl;
//     std::cout << "cols = " << n_cols << std::endl;

//     this->height = n_rows;
//     this->width = n_cols;

//     this->train_data_size = n_rows * n_cols * sizeof(char) * number_of_images;
//     int n_bufsize = n_rows * n_cols * sizeof(float);

//     cudaMalloc(&gpuimg_buf, bufsize);
//     file_offset = metasize;
//     mem_offset = 0;

//     cuFileBufRegister((char*)gpuimg_buf, bufsize, 0);

//     ret = cuFileRead(cf_handle, (char*)gpuimg_buf, this->train_data_size, file_offset, mem_offset);
//     cuFileDriverClose();

//     close(fd);

//     std::cout << "data size:" << this->train_data_size << std::endl;    
// }

void DataSetGDS::read_labels(std::string file_name,
                          char * gpulbl_buf) {
    int fd;
    int ret;
    

    int *sys_len;
    int *meta;
    int parasize=KB(1);
    int bufsize = KB(4);

    off_t file_offset = 0;
    off_t mem_offset = 0;
    int metasize=8;
    unsigned int magic_number = 0;
    unsigned int number_of_images = 0;

    sys_len = (int*)malloc(parasize);

    CUfileDescr_t cf_desc; 
    CUfileHandle_t cf_handle;

    cuFileDriverOpen();
  	fd = open(file_name.c_str(), O_RDWR|O_DIRECT);
    cf_desc.handle.fd = fd;
    cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    cuFileHandleRegister(&cf_handle, &cf_desc);

    cudaMalloc(&meta, bufsize);
    // thrust::device_ptr<char> dev_ptr(gpumem_buf);
    // cuFileBufRegister((char*)meta, bufsize, 0);

    ret = cuFileRead(cf_handle, (char*)meta, metasize, file_offset, mem_offset);
    if (ret < 0) {
      std::cout << "cuFileRead failed:" << ret << std::endl; 
    } 
    cudaMemcpy(sys_len, meta, metasize, cudaMemcpyDeviceToHost);
    // magic_number = reverse_int(((int*)sys_len)[0]);
    // number_of_images = reverse_int(((int*)sys_len)[1]);
    magic_number = this->reverse_int(magic_number);
    number_of_images = this->reverse_int(number_of_images);

    std::cout << file_name << std::endl;
    std::cout << "magic number = " << magic_number << std::endl;
    std::cout << "number of images = " << number_of_images << std::endl;

    if (number_of_images > 0 ) {
      cudaMalloc(&gpulbl_buf, bufsize);
      off_t file_offset = metasize;
      off_t mem_offset = 0;
      CUfileDescr_t cf_desc; 
      CUfileHandle_t cf_handle;

      cuFileDriverOpen();

      cf_desc.handle.fd = fd;
      cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
      cuFileHandleRegister(&cf_handle, &cf_desc);
      cuFileBufRegister((char*)gpulbl_buf, bufsize, 0);

      ret = cuFileRead(cf_handle, (char*)gpulbl_buf, number_of_images, file_offset, mem_offset);

      cuFileDriverClose();
    }
    close(fd);
}
