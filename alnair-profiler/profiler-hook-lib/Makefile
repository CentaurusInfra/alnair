CXX = g++

OBJ_DIR := build/obj
LIB_DIR := build/lib
BIN_DIR := build/bin
_builddirs := $(shell mkdir -p $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR))

CUDA_BIN_DIR := $(shell dirname `which nvcc`)
CUDA_INCLUDE_DIR := $(shell readlink -f $(CUDA_BIN_DIR)/../include)

CXXFLAGS += -fPIC
# CXXFLAGS += -I /usr/local/cuda-11.4/targets/x86_64-linux/include/
OBJ_FILES := $(addprefix $(OBJ_DIR)/, hooks.o cuda_metrics.o)

.PHONY: lib
lib: $(OBJ_DIR)/cuda_interpose.o
	$(CXX) $^ -shared -fPIC $(LDFLAGS)  -o $(LIB_DIR)/libpfinterpose.so -L/usr/local/cuda/lib64 -lcudart -lcuda -lnvidia-ml

$(OBJ_DIR)/%.o: src/%.c
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR)/cuda_interpose.o: src/cuda_interpose.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) -c -o $@ $^


.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/*  $(LIB_DIR)/libpfinterpose.so
