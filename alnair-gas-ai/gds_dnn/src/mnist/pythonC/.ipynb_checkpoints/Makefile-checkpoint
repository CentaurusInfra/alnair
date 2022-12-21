CC=g++
NVCC=nvcc
# CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS= -std=c++14 -c -arch=sm_80 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
# LIBDIRS=-L/usr/local/cuda-7.5/lib64

CUDA_INCDIRS=-I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I../../cuda -I../test -I../ 
CUDA_LIBDIRS=-L /usr/local/cuda/targets/x86_64-linux/lib/ -lcufile -L /usr/local/cuda/lib64/ -lcuda -L   -Bstatic -L /usr/local/cuda/lib64/ 
LIBS= -lopenblas -lpthread -lcublas -ldl -lcudart_static -lrt -lcufile -lcuda  -L -Bstatic

.PHONY: all build test clean

all: build libgdsunittests.a gds_func_ops.h
	CC=g++ LDSHARED='$(shell python configure.py)' python setup.py build
	python setup.py install
	# python tests/test.py

install:
	python setup.py install

libgdsunittests.a: gds_add_unit_test.o gds_readimg_unit_test.o stddev_unit_test.o storage_gds.o dataset_gds.o
	rm -f libgdsunittests.a
	ar cru libgdsunittests.a gds_add_unit_test.o temp_add.o 
	ar cru libgdsunittests.a gds_readimg_unit_test.o temp_readimg.o 
	ar cru libgdsunittests.a stddev_unit_test.o  temp_dev.o
	ar cru libgdsunittests.a storage_gds.o  temp_storage.o
	ar cru libgdsunittests.a dataset_gds.o temp_dataset.o
	ranlib libgdsunittests.a

gds_add_unit_test.o:  ../test/gds_add_unit_test.cu
	$(NVCC) $(CUDAFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp_add.o ../test/gds_add_unit_test.cu
	$(NVCC) -dlink --compiler-options '-fPIC' -o gds_add_unit_test.o temp_add.o $(CUDA_LIBDIRS) $(LIBS)

gds_readimg_unit_test.o:  ../test/gds_readimg_unit_test.cu
	$(NVCC) $(CUDAFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp_readimg.o ../test/gds_readimg_unit_test.cu
	$(NVCC) -dlink --compiler-options '-fPIC' -o gds_readimg_unit_test.o temp_readimg.o $(CUDA_LIBDIRS) $(LIBS)

stddev_unit_test.o:  ../test/stddev_unit_test.cu
	$(NVCC) $(CUDAFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp_dev.o ../test/stddev_unit_test.cu
	$(NVCC) -dlink --compiler-options '-fPIC' -o stddev_unit_test.o temp_dev.o $(CUDA_LIBDIRS) $(LIBS)

storage_gds.o:  ../../cuda/storage_gds.cu
	$(NVCC) $(CUDAFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp_storage.o ../../cuda/storage_gds.cu
	$(NVCC) -dlink --compiler-options '-fPIC' -o storage_gds.o temp_storage.o $(CUDA_LIBDIRS) $(LIBS)

dataset_gds.o:  ../dataset_gds.cu
	$(NVCC) $(CUDAFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o temp_dataset.o ../dataset_gds.cu
	$(NVCC) -dlink --compiler-options '-fPIC' -o dataset_gds.o temp_dataset.o $(CUDA_LIBDIRS) $(LIBS)



clean:
	rm -rf *.out *.bin *.exe *.o *.a *.so *.png test build

sanity:
	python test.py
