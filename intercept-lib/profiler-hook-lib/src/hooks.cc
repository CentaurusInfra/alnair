/*
Copyright (c) 2022 Futurewei Technologies.
Author: Steven Wang (@pint1022)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <sstream>
#include <set>
#include <unistd.h>
#include <pthread.h>
#include <nvml.h>
#include <cuda.h>
#include <sys/time.h>
#include "cuda_metrics.h"

#define MAXPROC 1024

//////////////////////////////////////////////////////
//
// cuda metrics
//
/////////////////////////////////////////////////////
extern cuda_metrics_t pf;
extern void log_api_call(const char *pid, const int memUsed, const int kernelCnt, const int tokens);
extern void* profiling_thread_func(void *arg);
pthread_mutex_t mem_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t launch_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t H2D_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t D2H_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t pre_cuinit_ctrl = PTHREAD_ONCE_INIT;
static pthread_once_t post_cuinit_ctrl = PTHREAD_ONCE_INIT;
static volatile bool pre_initialized = false;
static volatile bool post_initialized = false;
static unsigned int proc_id = NO_PID; // GPU proc_id on the pod, assume there is only one GPU assigned, and one process per pod on the GPU

/************************************************/
#include <execinfo.h>
static void print_trace (void)
{
  void *array[10];
  char **strings;
  int size, i;

  size = backtrace (array, 10);
  strings = backtrace_symbols (array, size);
  if (strings != NULL)
  {
    fprintf(stderr, "Obtained %d stack frames.\n", size);
    for (i = 0; i < size; i++)
      fprintf(stderr, "%s\n", strings[i]);
  }

  free (strings);
}
/************************************************/

//example line, k8s v1.21 format
//6:memory:/kubepods/besteffort/podb494d806-bfe7-4c33-8e23-032da1434a90/06b159b3f1cb4c021766a97e5ac82d18284c381223e5539aa510269ee5eed4d3
static std::string get_cgroup() 
{
    std::ifstream fs("/proc/self/cgroup");
    for(std::string line; std::getline(fs, line); ) {
        std::stringstream ss(line);
        std::string item;
        while(std::getline(ss, item, ':')) {
            if(item == "memory") {
                std::getline(ss, item, ':');
                return item;
            }
        }
    }
    fs.close();
    return "";
}

static std::string cgroup = get_cgroup();

static int get_gpu_compute_processes(unsigned int* procCount, nvmlProcessInfo_t* procInfos) 
{
    nvmlReturn_t ret;
    nvmlDevice_t device;
    ret = nvmlDeviceGetHandleByIndex(0, &device);
    if(NVML_SUCCESS != ret) {
        fprintf(stderr, "Failed nvmlDeviceGetHandleByIndex: %s\n", nvmlErrorString(ret));
        return -1;
    }

    ret = nvmlDeviceGetComputeRunningProcesses(device, procCount, procInfos);
    if(NVML_SUCCESS != ret) {
        fprintf(stderr, "Failed nvmlDeviceGetComputeRunningProcesses: %s\n", nvmlErrorString(ret));
        return -1;
    }

    return 0;
}

static void read_pids(std::set<unsigned int>& pids)
{
    std::ifstream fs("/var/lib/alnair/workspace/cgroup.procs");
    for(std::string line; std::getline(fs, line); ) {
        pids.insert(atoi(line.c_str()));
    }
    fs.close();
}

static timespec period = {
    .tv_sec = 0,
    .tv_nsec = 10 * 1000000
};

static unsigned int find_proc()
{
    nvmlReturn_t ret;
    nvmlDevice_t device;
    nvmlProcessUtilizationSample_t sample[1024];
    unsigned int numProc = 1024;
    struct timeval now;
    size_t microsec;

    std::set<unsigned int> pids;
    read_pids(pids);

    ret = nvmlDeviceGetHandleByIndex(0, &device);   //limits: only assume this container mount 1 GPU
    if(NVML_SUCCESS != ret) {
        fprintf(stderr, "Failed nvmlDeviceGetHandleByIndex: %s\n", nvmlErrorString(ret));
        return -1;
    }

    gettimeofday(&now, NULL);
    microsec = (now.tv_sec-1)*1000000 + now.tv_usec;
    ret = nvmlDeviceGetProcessUtilization(device, sample, &numProc, microsec);
    if(NVML_SUCCESS != ret) {
        fprintf(stderr, "Failed nvmlDeviceGetProcessUtilization: %s\n", nvmlErrorString(ret));
        return -1;
    }

    for(int i=0; i < numProc; ++i) {
        unsigned int pid = sample[i].pid;
        if(pids.find(pid) != pids.end()) return pid;
    }

    return NO_PID;
}

static void pre_cuinit(void)
{
    int res = 0;

    // init nvml library
    nvmlReturn_t ret;
    ret = nvmlInit();
    if(NVML_SUCCESS != ret) {
        fprintf(stderr, "Failed nvmlInit: %s\n", nvmlErrorString(ret));
        return;
    }

    pre_initialized = true;
}

static void post_cuinit(void)
{
    CUresult cures = CUDA_SUCCESS;
    CUdevice dev;
    int numSM, numThreadsPerSM, res;
    CUuuid_st uuid;

    // Here we only support compute resource sharing within a single device.
    // If multiple devices are visible, gpuComputeLimit would be 100, 
    // and the previous statement would have already exited. 
    cures = cuDeviceGet(&dev, 0);
    if(CUDA_SUCCESS != cures) {
        fprintf(stderr, "cuDeviceGet failed: %d\n", cures);
    }

    cures = cuDeviceGetAttribute(&numSM, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    if(CUDA_SUCCESS != cures) {
        fprintf(stderr, "# of SM query failed: %d\n", cures);
    }

    cures = cuDeviceGetAttribute(&numThreadsPerSM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    if(CUDA_SUCCESS != cures) {
        fprintf(stderr, "# of threads per SM query failed: %d\n", cures);
    }

    //
    // find proc_id for the GPU process
    //
    proc_id = find_proc();
    if (proc_id < 0) {
        fprintf(stderr,"ERROR: there is no valid process id for GPU process.\n");
    }
    pf.pid = proc_id;
    cures = cuDeviceGetUuid(&uuid, dev);
    
    if(CUDA_SUCCESS != cures) {
        fprintf(stderr, "ERROR: UUid failed, ERRNO %d\n", cures);
    }
    // fprintf(stderr, "UUid %s\n", (char*) &(pf.UUID));

    // initialize for profiler
    pthread_t pf_thread;

    res = pthread_create(&pf_thread, NULL, profiling_thread_func, NULL);
    if(res < 0) {
        fprintf(stderr,"profiler failed to start, errno=%d\n", errno);
    }
    
    post_initialized = true;
}

CUresult cuInit_hook (unsigned int Flags)
{
    CUresult cures = CUDA_SUCCESS;
    int res = 0;
    
    // initialize for GPU memory monitoring
    res = pthread_once(&pre_cuinit_ctrl, pre_cuinit);
    if(res < 0) {
        fprintf(stderr,"pre_cuinit failed, errno=%d\n", errno);
    }


    return cures;
}

CUresult cuInit_posthook (unsigned int Flags)
{
    CUresult cures = CUDA_SUCCESS;
    int res = 0;

    // initialize for GPU compute monitoring
    res = pthread_once(&post_cuinit_ctrl, post_cuinit);
    if(res < 0) {
        fprintf(stderr,"post_cuinit failed, errno=%d\n", errno);
    }
    
    return cures;
}

// CUresult cuInit_posthook (unsigned int Flags)
// {
//     CUresult cures = CUDA_SUCCESS;
//     int res = 0;

//     // initialize for profiler
//     pthread_t pf_thread;

//     res = pthread_create(&pf_thread, NULL, profiling_thread_func, NULL);
//     if(res < 0) {
//         fprintf(stderr,"profiler failed to start, errno=%d\n", errno);
//     }
    
//     return cures;
// }

CUresult cuLaunchKernel_hook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, 
                             unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, 
                             unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, 
                             void** kernelParams, void** extra)
{
    CUresult cures = CUDA_SUCCESS;
    // unsigned int cost = gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ;
    pthread_mutex_lock(&launch_mutex);
    pf.kernelCnt ++;
    pf.Kbegin = std::chrono::steady_clock::now();
    pthread_mutex_unlock(&launch_mutex);


    return cures;
}

CUresult cuLaunchKernel_posthook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, 
                             unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, 
                             unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, 
                             void** kernelParams, void** extra)
{
    CUresult cures = CUDA_SUCCESS;
    // unsigned int cost = gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ;
    pthread_mutex_lock(&launch_mutex);
    pf.kernelRunTime += (std::chrono::steady_clock::now() - pf.Kbegin).count() / 1000 ; //milliseconds
    pthread_mutex_unlock(&launch_mutex);


    return cures;
}

// static CUresult update_mem_usage(size_t bytesize )
// {
//     CUresult cures = CUDA_SUCCESS;

//     pthread_mutex_lock(&mem_mutex);
//     pf.memUsed += bytesize;
//     pthread_mutex_unlock(&mem_mutex);

//     return (cures);
// }

static CUresult update_mem_usage()
{
    // get process ids within the same container
    unsigned long long totalUsage;    
    std::set<unsigned int> pids;

    if (proc_id == NO_PID)
        read_pids(pids);

    // get per process gpu memory usage
    unsigned int numProc = MAXPROC;
    nvmlProcessInfo_t procInfos[MAXPROC];
    int ret = get_gpu_compute_processes(&numProc, procInfos);
    if(ret != 0) return CUDA_SUCCESS;

    totalUsage = 0;
    for(int i=0; i < numProc; ++i) {
        unsigned int pid = procInfos[i].pid;
        if (proc_id == NO_PID) {
            if(pids.find(pid) != pids.end()) {totalUsage += procInfos[i].usedGpuMemory;proc_id = pf.pid = pid; break;};
        } else if(pid == proc_id) {
            totalUsage += procInfos[i].usedGpuMemory;
            break;
        }
    }

//////////////////////////////////
// 
//  profiling memory usage
//
//////////////////////////////////
    pf.memUsed = totalUsage;

    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_hook (CUdeviceptr* dptr, size_t bytesize)
{
    return update_mem_usage();

}


CUresult cuMemFree_hook (CUdeviceptr* dptr)
{
    return update_mem_usage();
}

CUresult cuMemcpyDtoH_prehook(void *dstHost, CUarray srcArray, size_t srcOffset,
                               size_t ByteCount) {
  
  pthread_mutex_lock(&D2H_mutex);
  pf.D2HCnt ++;
  pf.D2Hbegin = std::chrono::steady_clock::now();
  pthread_mutex_unlock(&D2H_mutex);
  
  return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_posthook(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {

  pthread_mutex_lock(&D2H_mutex);
  pf.D2HTime += (pf.D2Hbegin  - std::chrono::steady_clock::now()).count() /1000;
  pthread_mutex_unlock(&D2H_mutex);

  return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_prehook(void *dstHost, CUarray srcArray, size_t srcOffset,
                               size_t ByteCount) {
  pthread_mutex_lock(&H2D_mutex);
  pf.H2DCnt ++;
  pf.H2Dbegin = std::chrono::steady_clock::now();
  pthread_mutex_unlock(&H2D_mutex);
  
  return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_posthook(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  
  pthread_mutex_lock(&H2D_mutex);
  pf.H2DTime += (pf.H2Dbegin  - std::chrono::steady_clock::now()).count() /1000;
  pthread_mutex_unlock(&H2D_mutex);
  return CUDA_SUCCESS;
}
