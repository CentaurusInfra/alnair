#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include "cufile.h"
#include <utils.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <tuple>
#include <vector>

#include <dataset_gds.cuh>
// #include <layer_gds.cuh>
// #include <mnist_gds.cuh>
#include "gds_test.cuh"

//   This example computes the norm [1] of a vector.  The norm is 
// computed by squaring all numbers in the vector, summing the 
// squares, and taking the square root of the sum of squares.  In
// Thrust this operation is efficiently implemented with the 
// transform_reduce() algorith.  Specifically, we first transform
// x -> x^2 and the compute a standard plus reduction.  Since there
// is no built-in functor for squaring numbers, we define our own
// square functor.
//
// [1] http://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm


#define BATCH_SIZE 128
#define LEARNING_RATE 0.003
#define L2 0.0001
#define EPOCHS 2
#define BETA 0.99
#define FILE_NAME_LENGTH 256

int main(int argc, char *argv[])
{
	clock_t start, end;
	start = clock();
    int idx = 1;
    float time_epoch_fd, time_epoch_bd;
	clock_t start_e;	

	/* First argument is executable name only */
	printf("\nexe name=%s\n", argv[0]);

	int opt = 0;
	int epochs = EPOCHS;
	int batch_size = BATCH_SIZE;
	char foldername[FILE_NAME_LENGTH];

	while ((opt = getopt(argc, argv, "f:e:b:")) != -1) {
		switch(opt) {
		case 'f':
			strcpy(foldername,optarg);
			break;
		case 'e':
			epochs = std::atoi(optarg);
			break;
		case 'b':
			batch_size = std::atoi(optarg);
			break;
		default:
			std::cout << "option is not supported!" << std::endl; 
		}

	}
	if (sizeof(foldername) < 0) {
		std::cout << "please provide the location of training data" << std::endl; 
		exit(1);
	}

	// std::cout << "data folder: " << foldername <<  std::endl; 
		
	std::unique_ptr<DataSetGDS> dataset;
    // dataset.reset(new DataSetGDS(foldername, false));
	// float * test = read_image(foldername, 256);
	char filename[256];
	strcpy(filename, foldername);
	strcat(filename, "/train-images-idx3-ubyte");
	test_numpy(filename);
	char * imgs;
	int rows, cols;
	int ret = read_image_data(filename, batch_size, imgs, &rows, &cols);
	std::cout << "batchsize:" << batch_size << "ret: " << ret << "rows: " << rows << "cols: "<< cols << std::endl;

	// Minist mnist(filename, LEARNING_RATE, L2, BETA);

	// test(argv[1]);
	// char * mnist_data="/home/steven/dev/DataLoaders_DALI/cuda-neural-network/build/mnist_data/train-images-idx3-ubyte";
	// test_stream(argv[1]);
	// test_transform();

    // while (dataset->has_next(true)) {
   	//   start_e = clock();  
    //   dataset->forward(batch_size, true);
	//   const GDSStorage* labels = dataset->get_label();

    //   time_epoch_fd = (clock() - start_e) * 1000000 / CLOCKS_PER_SEC;

   	//   start_e = clock();  
    //   dataset->backward();
    //   time_epoch_bd = (clock() - start_e) * 1000000 / CLOCKS_PER_SEC;
    // }

}
