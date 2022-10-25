## Test Instructions
Build the hook.cpp file with the following command in a cuda env

```nvcc -shared -lcuda --compiler-options '-fPIC' hook.cpp -o hook.so```

To test interception before cuda 11.3, choose the first base image in the Dockerfile

```pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime```

Then docker build and run

```docker build -f Dockerfile . -t pyt18```

```docker run -it pyt18 /bin/bash```

Once inside the docker, you can launch a python script to verify the intercept print out
```LD_PRELOAD=./hook.so python ./pyt-test1.py```

Expected outputs are like the following
```
====cuInit hooked====at 1657668149994166572
./pyt-test1.py:156: FutureWarning: The default weight initialization of inception_v3 will be changed in future releases of torchvision. If you wish to keep the old behavior (which leads to long initialization times due to scipy/scipy#11299), please set init_weights=True.
  warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
@@@@==cuMemAlloc hooked====
@@@@==cuMemAlloc hooked====
@@@@==cuMemAlloc hooked====
```
To test interception after cuda 11.3, choose the second base image in the Dockerfile, and follow the same step build docker image and run.
```pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime```


**Notes: the pytorch runtime image does not contain nvcc library, so you cannot build the hook.so within the image. You need to build it on a gpu host machine first.**
