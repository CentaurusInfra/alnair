import os
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

if 'CUDA_PATH' in os.environ:
   CUDA_PATH = os.environ['CUDA_PATH']
else:
   print("Could not find CUDA_PATH in environment variables. Defaulting to /usr/local/cuda!")
   CUDA_PATH = "/usr/local/cuda/"

func_files = [
       'gds_unit_test.cpp', 
       'bind.cpp'
       ]

module1 = Extension('unittests',
                    sources =func_files,
                    libraries=["gdsunittests", "cudart", "cufile"],
              library_dirs = [".", os.path.join(CUDA_PATH, "lib64")],
              include_dirs=[numpy.get_include()])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a unit test package for GDS-framework',
       ext_modules = [module1])

# setup(name = 'myModule', version = '1.0',  \
#    ext_modules = [
#       Extension('myModule', ['myModule.c'], 
#       include_dirs=[np.get_include(), os.path.join(CUDA_PATH, "include")],
#       libraries=["vectoradd", "cudart"],
#       library_dirs = [".", os.path.join(CUDA_PATH, "lib64")]
# )])

