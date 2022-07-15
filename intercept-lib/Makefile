CXX = g++

OBJ_DIR := build/obj
LIB_DIR := build/lib
BIN_DIR := build/bin
_builddirs := $(shell mkdir -p $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR))

CUDA_BIN_DIR := $(shell dirname `which nvcc`)
CUDA_INCLUDE_DIR := $(shell readlink -f $(CUDA_BIN_DIR)/../include)

CXXFLAGS += -fPIC
OBJ_FILES := $(addprefix $(OBJ_DIR)/, register_client.o hooks.o cuda_metrics.o)

.PHONY: lib
lib: $(OBJ_FILES) $(OBJ_DIR)/cuda_interpose.o
	$(CXX) $^ -shared -fPIC $(LDFLAGS) -lcuda -lnvidia-ml -o $(LIB_DIR)/libcuinterpose.so

$(OBJ_DIR)/hooks.o: src/hooks.cc 
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: src/%.c
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

$(OBJ_DIR)/cuda_interpose.o: src/cuda_interpose.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) -c -o $@ $^

$(OBJ_DIR)/cuda_metrics.o: src/cuda_metrics.cc 
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) -c -o $@ $^

.PHONY: clean
clean:
	rm -f $(OBJ_DIR)/* $(BIN_DIR)/client-tool $(LIB_DIR)/libcuinterpose.so
