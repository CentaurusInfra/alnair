# sources="../cuda/storage_gds.cu dataset_gds.cu test/main.cu  mnist_gds.cu ../cuda/blas_gds.cu ../cuda/conv_gds.cu ../cuda/nll_loss_gds.cu ../cuda/softmax_gds.cu"
sources="../cuda/storage_gds.cu dataset_gds.cu test/gds_readimg_unit_test.cu   test/main.cu "
nvcc  -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -g -I /usr/local/cuda/include/  -I /usr/local/cuda/targets/x86_64-linux/lib/ -I ../cuda -I ./ $sources -o gds_test.co -L /usr/local/cuda/targets/x86_64-linux/lib/ -lcufile -L /usr/local/cuda/lib64/ -lcuda -L   -Bstatic -L /usr/local/cuda/lib64/ -lcudart_static -lrt -lpthread -ldl 
# -lcrypto -lssl
# ../../build/mnist_data/train-images-idx3-ubyte