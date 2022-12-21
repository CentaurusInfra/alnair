#!/usr/bin/env python3
import numpy as np
import random
import timeit
import pandas as pd
from matplotlib import pyplot as plt
import unittests as gds

def data_gen(rg):
    lp_time = []
    py_time = []
    np_time = []
    c_time = []

    for l in rg:
        rands = [random.random() for _ in range(0, l)]
        numpy_rands = np.array(rands)
        np_time = np.append(np_time, timeit.timeit(lambda: np.std(numpy_rands), number=1000))
        # print(l, np_time.shape)
        c_time = np.append(c_time, timeit.timeit(lambda: gds.standard_dev(rands), number=1000))
    return np.array([np.transpose(np_time), np.transpose(c_time)])

def test_stddev():
    lens = range(1000, 20000, 1000)
    data = data_gen(rg=lens)

    df = pd.DataFrame(data.transpose(), index=lens, columns=['Numpy', 'C++'])
    plt.figure()
    df.plot()
    plt.legend(loc='best')
    plt.ylabel('Time (Seconds)')
    plt.xlabel('Number of Elements')
    plt.title('1k Runs of Standard Deviation')
    plt.savefig('numpy_vs_c.png')
    plt.show()

def test_system():
    gds.system("ls -l")

def test_add():
    print(gds.add(5, 6, 'testfile'))

def test_readimg():
    batch_size = 256
    mnist_data = "/home/scripts/data/mnist_data/train-images-idx3-ubyte"
    data = gds.gds_read_image_data(mnist_data, batch_size)
    print(data)

test_stddev()
test_system()
test_add()
test_readimg()
