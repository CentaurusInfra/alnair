#include <dataset.cuh>
#include <mnist.cuh>
#include <unistd.h>
#include <string>     // std::string, std::stoi

#define BATCH_SIZE 128
#define LEARNING_RATE 0.003
#define L2 0.0001
#define EPOCHS 2
#define BETA 0.99

int main(int argc, char *argv[]) {


/* First argument is executable name only */
printf("\nexe name=%s", argv[0]);

 int opt = 0;
 int epochs = EPOCHS;
 int batch_size = BATCH_SIZE;

 while ((opt = getopt(argc, argv, "e:b:")) != -1) {
    switch(opt) {
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

  // DataSet dataset("./mnist_data", true);
  // dataset.forward(64, true);
  // dataset.print_im();

  auto cudaStatus = cudaSetDevice(0);
  CHECK_EQ(cudaStatus, cudaSuccess,
           "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

  Minist mnist("./mnist_data", LEARNING_RATE, L2, BETA);
  mnist.train(epochs, batch_size);
}
