/*
Copyright (c) 2022 Futurewei Technologies.
Author: Hao Xu (@hxhp)

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
#include <sstream>
#include <set>
#include <unistd.h>
#include <pthread.h>
#include <nvml.h>
#include <cuda.h>
#include <sys/time.h>

#define MAXPROC 1024

extern int register_cgroup(const char *cgroup, const char* alnairID);

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

struct token_bucket {
    pthread_mutex_t mutex;
    unsigned int cur_tokens;
    unsigned int fill_rate;
    unsigned int fill_rate_cap;
    unsigned int max_burst;
    struct timespec period;
};

static timespec period = {
    .tv_sec = 0,
    .tv_nsec = 10 * 1000000
};

static struct token_bucket tb = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cur_tokens = 0,
    .fill_rate = 0,
    .fill_rate_cap = 0,
    .max_burst = 0,
    .period = {
        .tv_sec = 0,
        .tv_nsec = 100 * 1000000
    }
};

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

static unsigned long long get_memory_limit()
{
    std::ifstream fs("/var/lib/alnair/workspace/limits");
    for(std::string line; std::getline(fs, line); ) {
        std::stringstream ss(line);
        std::string item;
        std::getline(ss, item, ':');
        if(item == "vmem") {
            std::getline(ss, item, ':');
            return std::stoull(item);
        }
    }
    fs.close();
    return 0;
}

static int get_compute_limit()
{
    char* var = NULL;
    var = getenv("ALNAIR_VGPU_COMPUTE_PERCENTILE");
    if(!var) {
        return 100;
    } else {
        int ret = atoi(var);
        if(ret <= 0 || ret > 100) return 100;
        return ret;
    }
}

static std::string cgroup = get_cgroup();
static volatile unsigned long long gpuMemLimit = get_memory_limit();
static volatile int gpuComputeLimit = get_compute_limit();
static pthread_once_t pre_cuinit_ctrl = PTHREAD_ONCE_INIT;
static pthread_once_t post_cuinit_ctrl = PTHREAD_ONCE_INIT;
static volatile bool pre_initialized = false;
static volatile bool post_initialized = false;

static std::string parse_containerID(const std::string& cgroup) 
{
    std::size_t begin = cgroup.find("docker-");
    return cgroup.substr(begin+7, cgroup.size()-13-begin);
}

static void read_pids(std::set<unsigned int>& pids)
{
    std::string containerID = parse_containerID(cgroup);
    std::ifstream fs("/var/lib/alnair/workspace/cgroup.procs");
    for(std::string line; std::getline(fs, line); ) {
        pids.insert(atoi(line.c_str()));
    }
    fs.close();
}

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

static int get_current_mem_usage(unsigned long long* totalUsage)
{
    // get process ids within the same container
    std::set<unsigned int> pids;
    read_pids(pids);

    // get per process gpu memory usage
    unsigned int numProc = MAXPROC;
    nvmlProcessInfo_t procInfos[MAXPROC];
    int ret = get_gpu_compute_processes(&numProc, procInfos);
    if(ret != 0) return ret;

    *totalUsage = 0;
    for(int i=0; i < numProc; ++i) {
        unsigned int pid = procInfos[i].pid;
        if(pids.find(pid) != pids.end()) (*totalUsage) += procInfos[i].usedGpuMemory;
    }

    return 0;
}

static CUresult validate_memory(size_t toAllocate)
{
    CUresult cures = CUDA_SUCCESS;
    unsigned long long totalUsed = 0;

    if(!pre_initialized) goto exit;
    
    if(get_current_mem_usage(&totalUsed)) goto exit;

    // TODO handle race condition
    if(totalUsed + toAllocate > gpuMemLimit) cures = CUDA_ERROR_OUT_OF_MEMORY;

exit:
    return cures;
}

static size_t get_size_of(CUarray_format fmt)
{
    size_t bytesize = 1;
    switch(fmt) {
        case CU_AD_FORMAT_UNSIGNED_INT8:
        case CU_AD_FORMAT_SIGNED_INT8:
        case CU_AD_FORMAT_NV12:
            bytesize = 1;
            break;
        case CU_AD_FORMAT_UNSIGNED_INT16:
        case CU_AD_FORMAT_SIGNED_INT16:
        case CU_AD_FORMAT_HALF:
            bytesize = 2;
            break;
        case CU_AD_FORMAT_UNSIGNED_INT32:
        case CU_AD_FORMAT_SIGNED_INT32:
        case CU_AD_FORMAT_FLOAT:
            bytesize = 4;
            break;
    }
    return bytesize;
}

static int get_current_group_usage(unsigned int* groupUsage)
{
    nvmlReturn_t ret;
    nvmlDevice_t device;
    nvmlProcessUtilizationSample_t sample[1024];
    unsigned int numProc = 1024;
    struct timeval now;
    size_t microsec;

    std::set<unsigned int> pids;
    read_pids(pids);

    ret = nvmlDeviceGetHandleByIndex(0, &device);
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

    *groupUsage = 0;
    for(int i=0; i < numProc; ++i) {
        unsigned int pid = sample[i].pid;
        if(pids.find(pid) != pids.end()) (*groupUsage) += sample[i].smUtil;
    }

    return 0;
}

static void adjust_fill_Rate(unsigned int targetUsage, unsigned int curGroupUsage)
{
    
    int diff = targetUsage > curGroupUsage ? targetUsage - curGroupUsage : curGroupUsage - targetUsage;
    unsigned int adjust = 50000 * diff;
    if(diff > targetUsage/2)
        adjust = adjust * diff * 2 / (targetUsage+1);

    if(targetUsage > curGroupUsage)
        tb.fill_rate = tb.fill_rate + adjust > tb.fill_rate_cap ? tb.fill_rate_cap : tb.fill_rate + adjust;
    else
        tb.fill_rate = tb.fill_rate > adjust ? tb.fill_rate - adjust : 0;
}

static void* tb_thread_start(void *arg) 
{
    // If user doesn't want to constrain the compute usage, no need to run this thread.
    // This will always be true if more than one devices are visible.
    if(gpuComputeLimit == 100) return NULL;
    
    while(true) {
        unsigned int curGroupUsage;
        int ret;
        if(!pre_initialized || !post_initialized) goto wait;
        ret = get_current_group_usage(&curGroupUsage);
        if(!ret) adjust_fill_Rate(gpuComputeLimit, curGroupUsage);

	//fprintf(stderr, "cur_tokens: %u, fill_rate: %u\n", tb.cur_tokens, tb.fill_rate);
        pthread_mutex_lock(&tb.mutex);
        tb.cur_tokens = (tb.cur_tokens + tb.fill_rate) > tb.max_burst ? tb.max_burst : tb.cur_tokens + tb.fill_rate;
        pthread_mutex_unlock(&tb.mutex);        

wait:
        nanosleep(&tb.period, NULL);
    }

    return NULL;
}

static void pre_cuinit(void)
{
    int res = 0;
    char* alnairID=NULL;

    // register the cgroup
    alnairID = getenv("ALNAIR_ID");
    if(!alnairID) {
        fprintf(stderr, "ALNAIR_ID env variable is not set!\n");
        return;
    }

    res = register_cgroup(cgroup.c_str(), alnairID);
    if(res != 0) return;
    
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

    // No need to continue if user doesn't want to constrain the compute usage.
    if(gpuComputeLimit == 100) return;

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

    tb.fill_rate_cap = numSM * numThreadsPerSM / (tb.period.tv_nsec / 1000000) * 1500000;
    tb.fill_rate_cap = tb.fill_rate_cap >= 1u<<31 ? (1u<<31)-1 : tb.fill_rate_cap;
    tb.max_burst = tb.fill_rate_cap;
    tb.cur_tokens = tb.fill_rate_cap;
    //fprintf(stderr, "fill_rate_cap: %u, max_burst: %u\n", tb.fill_rate_cap, tb.max_burst);

    // thread to fill the token bucket
    pthread_t tb_thread;
    res = pthread_create(&tb_thread, NULL, tb_thread_start, NULL);
    if(res < 0) {
        fprintf(stderr,"token bucket thread creation failed, errno=%d\n", errno);
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

CUresult cuMemAlloc_hook (CUdeviceptr* dptr, size_t bytesize)
{
    return validate_memory(bytesize);
}

CUresult cuMemAllocManaged_hook(CUdeviceptr *dptr, size_t bytesize, unsigned int flags)
{
    return validate_memory(bytesize);
}

CUresult cuMemAllocPitch_hook(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, 
                              size_t Height, unsigned int ElementSizeBytes)
{
    
    size_t toAllocate = WidthInBytes * Height / 100 * 101;
    return validate_memory(toAllocate);
}

CUresult cuArrayCreate_hook(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
    size_t height = pAllocateArray->Height;
    size_t toAllocate;

    height = (height == 0) ? 1 : height;
    toAllocate = pAllocateArray->NumChannels * pAllocateArray->Width * height;
    toAllocate *= get_size_of(pAllocateArray->Format);
    return validate_memory(toAllocate);  
}

CUresult cuArray3DCreate_hook(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray)
{
    size_t depth = pAllocateArray->Depth;
    size_t height = pAllocateArray->Height;
    size_t toAllocate;

    height = (height == 0) ? 1 : height;
    depth = (depth == 0) ? 1 : depth;
    toAllocate = pAllocateArray->NumChannels * pAllocateArray->Width * height * depth;
    toAllocate *= get_size_of(pAllocateArray->Format);
    return validate_memory(toAllocate);
}

CUresult cuMipmappedArrayCreate_hook(CUmipmappedArray *pHandle, 
                                     const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, 
                                     unsigned int numMipmapLevels)
{
    size_t depth = pMipmappedArrayDesc->Depth;
    size_t height = pMipmappedArrayDesc->Height;
    size_t toAllocate;

    height = (height == 0) ? 1 : height;
    depth = (depth == 0) ? 1 : depth;
    toAllocate = pMipmappedArrayDesc->NumChannels * pMipmappedArrayDesc->Width * height * depth;
    toAllocate *= get_size_of(pMipmappedArrayDesc->Format);
    return validate_memory(toAllocate);
}

CUresult cuLaunchKernel_hook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, 
                             unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, 
                             unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, 
                             void** kernelParams, void** extra)
{
    CUresult cures = CUDA_SUCCESS;
    unsigned int cost = gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY * blockDimZ;
    if(gpuComputeLimit == 100) goto exit;
    if(!pre_initialized || !post_initialized) {
        fprintf(stderr, "pre_cuinit or post_cuinit not finished yet\n");
        goto exit;
    }
    
    while(true) {
        pthread_mutex_lock(&tb.mutex);
        if(tb.cur_tokens >= cost) {
            tb.cur_tokens = tb.cur_tokens - cost;
            pthread_mutex_unlock(&tb.mutex);
            break;
        }
        pthread_mutex_unlock(&tb.mutex);
        nanosleep(&period, NULL);
    }

exit:
    return cures;
}

CUresult cuLaunchCooperativeKernel_hook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, 
                                        unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                                        CUstream hStream, void **kernelParams)
{
    return cuLaunchKernel_hook(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 
                               sharedMemBytes, hStream, kernelParams, NULL);
}

CUresult cuDeviceTotalMem_posthook(size_t* bytes, CUdevice dev)
{
    *bytes = gpuMemLimit;
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo_posthook(size_t* free, size_t* total)
{
    // get process ids within the same container
    std::set<unsigned int> pids;
    read_pids(pids);

    // get per process gpu memory usage
    unsigned int procCount = MAXPROC;
    nvmlProcessInfo_t procInfos[MAXPROC];
    unsigned long long totalUsed = 0;
    int ret = get_gpu_compute_processes(&procCount, procInfos);
    if(ret != 0) {
        return CUDA_SUCCESS;
    }

    for(int i=0; i < procCount; ++i) {
        unsigned int pid = procInfos[i].pid;
        if(pids.find(pid) != pids.end()) totalUsed += procInfos[i].usedGpuMemory;
    }

    *total = gpuMemLimit;
    *free = totalUsed > gpuMemLimit ? 0 : gpuMemLimit - totalUsed;
    return CUDA_SUCCESS;
}
