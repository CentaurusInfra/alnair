NVCC=nvcc
SM := 80

CUDADIR=./cuda
CDIR=./C

BUILDDIR=./build

CFLAGS = -std=c++14 -g -G
GENCODE_FLAGS = -gencode arch=compute_$(SM),code=sm_$(SM)

CUDAINCLUDE = -I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I/usr/local/cuda/include  -I ./cuda
CUDALIB = -lcudart -L/usr/local/cuda/lib -L./ -lgdskernel
#-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

OBJ = $(BUILDDIR)/util_func.o $(BUILDDIR)/main.o

all : $(BUILDDIR)/matmul

$(BUILDDIR)/libgdskernel.a: $(BUILDDIR)/util_func.o
	rm -f $(BUILDDIR)/libgdskernel.a
	ar cru $(BUILDDIR)/libgdskernel.a $(BUILDDIR)/util_func.o $(BUILDDIR)/temp_util_func.o 
	ranlib $(BUILDDIR)/libgdskernel.a

$(BUILDDIR)/matmul: $(BUILDDIR)/main.o $(BUILDDIR)/libgdskernel.a
	${NVCC} --link -o $(BUILDDIR)/matmul $(BUILDDIR)/main.o $(BUILDDIR)/libgdskernel.a 

$(BUILDDIR)/util_func.o: $(CUDADIR)/util_func.cu
	$(NVCC) $(CFLAGS) -rdc=true --compiler-options '-fPIC' $(CUDA_INCDIRS) -c -o $(BUILDDIR)/temp_util_func.o $(CUDADIR)/util_func.cu  $(GENCODE_FLAGS)
	${NVCC} -dlink --compiler-options '-fPIC' -o $(BUILDDIR)/util_func.o $(BUILDDIR)/temp_util_func.o $(CUDA_LIBDIRS) $(LIBS)  $(GENCODE_FLAGS)

$(BUILDDIR)/main.o: $(CDIR)/tiled_matmul.cpp
	g++ $(CUDAINCLUDE) -c $(CDIR)/tiled_matmul.cpp -o $(BUILDDIR)/main.o

clean :
	rm -f  $(BUILDDIR)/*