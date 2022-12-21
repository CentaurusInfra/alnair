#pragma once

#include <layer_gds.cuh>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cufile.h"

#define KB(x) ((x)*1024L)

class DataSetGDS : public GDSLayer {
 public:
  explicit DataSetGDS(std::string minist_data_path, bool shuffle = false);
  void reset();

  void forward(int batch_size, bool is_train);
  bool has_next(bool is_train);

  int get_height() { return this->height; }
  int get_width() { return this->width; }
  GDSStorage* get_label() { return this->output_label.get(); }
  //
  // for unit testing
  //
  char* get_train_data() { return this->train_data; }
  char* get_train_label() { return this->train_label; }
  //

  void print_im();

  int get_train_datasize() { return this->train_data_size;}
  int get_test_datasize() { return this->test_data_size;}

 private:
  unsigned int reverse_int(unsigned int i);  // big endian
  void read_images(std::string file_name,
                   char*);
  void read_labels(std::string file_name, char*);

  char* train_data;
  char* train_label;
  int train_data_index;
  int train_data_size;

  char* test_data;
  char* test_label;
  int test_data_index;
  int test_data_size;

  int height;
  int width;
  bool shuffle;
  std::unique_ptr<GDSStorage> output_label;
};
