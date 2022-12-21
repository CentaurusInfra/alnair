#include <vector>
#include <numeric>
#include <iterator>

void test_operator_add(char * minst_data_path);
double standardDeviation(std::vector<double> v);
float* read_image(char * minst_data_path, int length);
int read_image_data(char * minst_data_path, unsigned int batchsize, char* data, int* rows, int * cols);