FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
#FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime 

RUN apt-get update
RUN pip install scipy
WORKDIR /intercept-testing
COPY . .

#RUN nvcc -shared -lcuda --compiler-options '-fPIC' hook.cpp -o hook.so
CMD ["sleep", "infinity"]
# LD_PRELOAD=./hook0.so python ./pyt-test1.py
