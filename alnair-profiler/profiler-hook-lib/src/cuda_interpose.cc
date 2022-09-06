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

/*
The CUDA intercept technique used in this file is based on the 
following references:
1) https://osterlund.xyz/posts/2018-03-12-interceptiong-functions-c.html
2) https://stackoverflow.com/questions/37792037/ld-preload-doesnt-affect-dlopen-with-rtld-now
3) The Nvidia example "libcuhook.cpp"

*/

#define _USE_GNU
#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <fstream>
#include <pthread.h>
#include "cuda_metrics.h"
#include <iostream>

#define STRINGIFY(x) STRINGIFY_AUX(x)
#define STRINGIFY_AUX(x) #x


extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }

static void* hooks[SYM_CU_SYMBOLS]; 
// = {
//     NULL, 
//     NULL,
//     NULL,
//     NULL,
//     NULL,
//     NULL,
//     NULL
// };

static void* post_hooks[SYM_CU_SYMBOLS];
//  = {
//     NULL, 
//     NULL,
//     NULL,
//     NULL,
//     NULL,
//     NULL,
//     NULL
// };


static bool enable_timeline[SYM_CU_SYMBOLS][STAT_CNT]; 


static void* real_func[SYM_CU_SYMBOLS];
static void* real_omp_get_num_threads = NULL;
static bool profiling_init = false;

const char PFLOG[] = "./test";
const char metrices_file[] = "metrics.log";
const char profiling_file[] = "timeline.log";
static std::string metfile;
static std::string tlfile;
std::queue<pflog_t> pf_queue;


const std::string funcname[] ={
    "cuInit",
    "cuMemAlloc",
    "cuMemFree",
    "cuLaunchKernel",
    "cuMemcpyH2D", 
    "cuMemcpyD2H",
    "cuGetProcAddress"
    };

cuda_metrics_t pf = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .pid = 0,
    .period = {
        .tv_sec = 0,
        .tv_nsec = 100 * 1000000
    }
};

unsigned int api_stats[SYM_CU_SYMBOLS][STAT_CNT];
//  = {
//     {0, 0},               //cuInit
//     {0, 0},              //cuAlloc
//     {0, 0},              //cuFree
//     {0, 0},              //cuLaunch
//     {0, 0},              //cuH2D
//     {0, 0},               //cuD2H
//     {0, 0}             //cuGetProcAddress
// };

// void log_api_call(const int pid, const int memUsed, const int kernelCnt, const int burst) 
void log_api_call() 
{                                                                                               
    // std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();  
    // auto duration = now.time_since_epoch();                                                     
    std::ofstream fmet (metfile);
    for (int i = 0; i < SYM_CU_SYMBOLS; i++) {
        //print timestamp at nano seconds when a cuda API is called                                                                     
        fmet << "name:" << funcname[i] << "\ncount:" << api_stats[i][0] << "\nburst:"  << api_stats[i][1] << std::endl; 
    }
    fmet.close();                                                                               
}  

void log_api_timing() 
{   
    std::ofstream fmet;                                                                         

    fmet.open(tlfile, std::ios_base::app);                                                                         
    while (!pf_queue.empty()) {
      // process request
      pflog log = pf_queue.front();
      pf_queue.pop();
      fmet <<"name:" << funcname[log.kernelid] << " start:" << log.begin << " burst:"  << log.burst << std::endl;                                     
    }
    fmet.close();                                                                               
}
void* profiling_thread_func(void *arg) 
{
    const char* env_p = std::getenv("PFLOG");


    // std::cout << "env: " << std::getenv("PFLOG") << std::endl;

    if(env_p) {
        metfile = std::string(env_p) + std::string("/")+std::string(metrices_file);
        tlfile = std::string(env_p) + std::string("/")+std::string(profiling_file);
    } else {
        metfile = std::string(PFLOG) + std::string("/")+std::string(metrices_file);
        tlfile = std::string(PFLOG) + std::string("/")+std::string(profiling_file);
        
    }
    std::cout << "pflog dir is: " << metfile << std::endl;

    while(true) {
        unsigned int curGroupUsage;
        int ret;
        // pthread_mutex_lock(&pf.mutex);
        // log_api_call(pf.pid, pf.memUsed, pf.kernelCnt, pf.kernelRunTime);
        log_api_call();
        log_api_timing();
        // pthread_mutex_unlock(&pf.mutex);        
        nanosleep(&pf.period, NULL);
    }

    return NULL;
}

void *libcudaHandle = __libc_dlopen_mode("libcuda.so", RTLD_LAZY);
void *libdlHandle = __libc_dlopen_mode("libdl.so", RTLD_LAZY);



static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static pthread_once_t init_done = PTHREAD_ONCE_INIT;

static void initialize(void)
{
    CUresult cures = CUDA_SUCCESS;
    CUdevice dev;
    int numSM, numThreadsPerSM, res;
    // CUuuid_st uuid;

    // Here we only support compute resource sharing within a single device.
    // If multiple devices are visible, gpuComputeLimit would be 100, 
    // and the previous statement would have already exited. 
    cures = cuDeviceGet(&dev, 0);
    if(CUDA_SUCCESS != cures) {
        fprintf(stderr, "cuDeviceGet failed: %d\n", cures);
    }

    // cures = cuDeviceGetAttribute(&numSM, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    // if(CUDA_SUCCESS != cures) {
    //     fprintf(stderr, "# of SM query failed: %d\n", cures);
    // }

    // cures = cuDeviceGetAttribute(&numThreadsPerSM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    // if(CUDA_SUCCESS != cures) {
    //     fprintf(stderr, "# of threads per SM query failed: %d\n", cures);
    // }

    // cures = cuDeviceGetUuid(&uuid, dev);
    
    // if(CUDA_SUCCESS != cures) {
    //     fprintf(stderr, "ERROR: UUid failed, ERRNO %d\n", cures);
    // }
    // fprintf(stderr, "UUid %s\n", (char*) &(pf.UUID));

    // initialize for profiler
    pthread_t pf_thread;

    res = pthread_create(&pf_thread, NULL, profiling_thread_func, NULL);
    if(res < 0) {
        fprintf(stderr,"profiler failed to start, errno=%d\n", errno);
    }
   
    enable_timeline[SYM_CU_INIT][MET_STAT] = true;                      enable_timeline[SYM_CU_INIT][MET_TIMELINE] = true;
    enable_timeline[SYM_CU_MEM_ALLOC][MET_STAT] = true;                 enable_timeline[SYM_CU_MEM_ALLOC][MET_TIMELINE] = false;
    enable_timeline[SYM_CU_MEM_FREE][MET_STAT] = true;                  enable_timeline[SYM_CU_MEM_FREE][MET_TIMELINE] = false;
    enable_timeline[SYM_CU_LAUNCH_KERNEL][MET_STAT] = true;             enable_timeline[SYM_CU_LAUNCH_KERNEL][MET_TIMELINE] = false;
    enable_timeline[SYM_CU_MEM_H2D][MET_STAT] = true;                   enable_timeline[SYM_CU_MEM_H2D][MET_TIMELINE] = true;
    enable_timeline[SYM_CU_MEM_D2H][MET_STAT] = true;                   enable_timeline[SYM_CU_MEM_D2H][MET_TIMELINE] = true;
    enable_timeline[SYM_CU_HOOK_GET_PROC_ADDRESS][MET_STAT] = true;     enable_timeline[SYM_CU_HOOK_GET_PROC_ADDRESS][MET_TIMELINE] = false;
    
}

typedef void *(*fnDlsym)(void *, const char *);
static void *real_dlsym(void *handle, const char *symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(libdlHandle, "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

#define GENERATE_INTERCEPT_FUNCTION(hooksymbol, funcname, params, ...)                                                                 \
    CUresult funcname params                                                                                                           \
    {                                                                                                                                  \
        CUresult res = CUDA_SUCCESS;                                                                                                   \
        unsigned long  Kbegin, burst;                                                                                                  \
        pthread_once(&init_done, initialize);                                                                                          \
        if (hooksymbol == SYM_CU_MEM_H2D)                                                                                              \
            std::cout << "H2D in intercept" << std::endl;                                                                              \
        if (hooksymbol == SYM_CU_MEM_D2H)                                                                                              \
            std::cout << "D2H in intercept" << std::endl;                                                                              \
        if (hooks[hooksymbol]) {                                                                                                       \
            res = ((CUresult (*)params)hooks[hooksymbol])(__VA_ARGS__);                                                                \
        }                                                                                                                              \
        if(CUDA_SUCCESS != res) return res;                                                                                            \
        if(real_func[hooksymbol] == NULL)                                                                                              \
            real_func[hooksymbol] = real_dlsym(RTLD_NEXT, STRINGIFY(funcname));                                                        \
        if(enable_timeline[hooksymbol][MET_STAT])                                                                                      \
            Kbegin = (std::chrono::system_clock::now().time_since_epoch()).count();                                                    \
        res = ((CUresult (*)params)real_func[hooksymbol])(__VA_ARGS__);                                                                \
        if(enable_timeline[hooksymbol][MET_STAT]) {                                                                                    \
            burst = ((std::chrono::system_clock::now().time_since_epoch()).count() - Kbegin)/1000;                                     \
            api_stats[hooksymbol][0]++;                                                                                                \
            api_stats[hooksymbol][1] += burst;                                                                                         \
            if(enable_timeline[hooksymbol][MET_TIMELINE]) {                                                                            \
                pf_queue.push({hooksymbol, Kbegin, burst});                                                                            \
            }                                                                                                                          \
        }                                                                                                                              \
        if(CUDA_SUCCESS == res && post_hooks[hooksymbol]) {                                                                            \
            res = ((CUresult (*)params)post_hooks[hooksymbol])(__VA_ARGS__);                                                           \
        }                                                                                                                              \
        return res;                                                                                                                    \
    }

GENERATE_INTERCEPT_FUNCTION(SYM_CU_INIT, cuInit, (unsigned int Flags), Flags)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_ALLOC, cuMemAlloc, (CUdeviceptr* dptr, size_t bytesize), dptr, bytesize)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_FREE, cuMemFree, (CUdeviceptr dptr), dptr)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_H2D, cuMemcpyHtoD,( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount ), dstDevice, srcHost, ByteCount)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_D2H, cuMemcpyDtoH, (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount), dstHost, srcDevice, ByteCount)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_LAUNCH_KERNEL, cuLaunchKernel, 
                            (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                             unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                             unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra), 
                            f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 
                            sharedMemBytes, hStream, kernelParams, extra)




void* dlsym(void *handle, const char *symbol) 
{
    if (strncmp(symbol, "cu", 2) != 0) {
        return real_dlsym(handle, symbol);
    }
    
 if (strcmp(symbol, STRINGIFY(cuInit)) == 0) {
        if(real_func[SYM_CU_INIT] == NULL) {
            real_func[SYM_CU_INIT] = real_dlsym(handle, symbol);
        }
        return (void*)(&cuInit);
    }

    if (strcmp(symbol, STRINGIFY(cuMemAlloc)) == 0) {
        if(real_func[SYM_CU_MEM_ALLOC] == NULL) {
            real_func[SYM_CU_MEM_ALLOC] = real_dlsym(handle, symbol);
        }
        return (void*)(&cuMemAlloc);
    }

    if (strcmp(symbol, STRINGIFY(cuMemFree)) == 0) {
        if(real_func[SYM_CU_MEM_FREE] == NULL) {
            real_func[SYM_CU_MEM_FREE] = real_dlsym(handle, symbol);
        }
        return (void*)(&cuMemFree);
    }

    if (strcmp(symbol, STRINGIFY(cuLaunchKernel)) == 0) {
        if(real_func[SYM_CU_LAUNCH_KERNEL] == NULL) {
            real_func[SYM_CU_LAUNCH_KERNEL] = real_dlsym(handle, symbol);
        }
        return (void*)(&cuLaunchKernel);
    }

    if (strcmp(symbol, STRINGIFY(cuMemcpyDtoH)) == 0) {
        if(real_func[SYM_CU_MEM_D2H] == NULL) {
            real_func[SYM_CU_MEM_D2H] = real_dlsym(handle, symbol);
        }        
            std::cout << "D2H in dlsym" << std::endl;                                                                                

        return (void *)(&cuMemcpyDtoH);
    }

    if (strcmp(symbol, STRINGIFY(cuMemcpyHtoD)) == 0) {
        if(real_func[SYM_CU_MEM_H2D] == NULL) {
            real_func[SYM_CU_MEM_H2D] = real_dlsym(handle, symbol);
        }        
            std::cout << "H2D in dlsym" << std::endl;                                                                                
        return (void *)(&cuMemcpyHtoD);
    }

    if (strcmp(symbol, STRINGIFY(cuGetProcAddress)) == 0) {
        return (void *)(&cuGetProcAddress);
    }
    return (real_dlsym(handle, symbol));
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
#ifdef _DEBUG
    printf("Enter %s\n", STRINGIFY(cuGetProcAddress));
    printf("symbol %s, cudaVersion %d, flags %lu\n", symbol, cudaVersion, flags);
#endif
    typedef decltype(&cuGetProcAddress) funcType;
    funcType actualFunc;
    if(!real_func[SYM_CU_HOOK_GET_PROC_ADDRESS])
        actualFunc = (funcType)real_dlsym(libcudaHandle, STRINGIFY_AUX(cuGetProcAddress));
    else
        actualFunc = (funcType)real_func[SYM_CU_HOOK_GET_PROC_ADDRESS];
    CUresult result = actualFunc(symbol, pfn, cudaVersion, flags);

    if(strcmp(symbol, STRINGIFY_AUX(cuGetProcAddress)) == 0) {
        real_func[SYM_CU_HOOK_GET_PROC_ADDRESS] = *pfn;
        *pfn = (void*)(&cuGetProcAddress);

#pragma push_macro("cuMemAlloc")
#undef cuMemAlloc
    } else if (strcmp(symbol, STRINGIFY_AUX(cuMemAlloc)) == 0) {
#pragma pop_macro("cuMemAlloc")
        if(real_func[SYM_CU_MEM_ALLOC] == NULL) {
            real_func[SYM_CU_MEM_ALLOC] = *pfn;
        }
        *pfn = (void *)(&cuMemAlloc);
#pragma push_macro("cuMemFree")
#undef cuMemFree
    } else if (strcmp(symbol, STRINGIFY_AUX(cuMemFree)) == 0) {
#pragma pop_macro("cuMemFree")
        if(real_func[SYM_CU_MEM_FREE] == NULL) {
            real_func[SYM_CU_MEM_FREE] = *pfn;
        }
        *pfn = (void *)(&cuMemFree);
#pragma push_macro("cuLaunchKernel")
#undef cuLaunchKernel
    } else if  (strcmp(symbol, STRINGIFY_AUX(cuLaunchKernel)) == 0) { 
#pragma pop_macro("cuLaunchKernel")
        if(real_func[SYM_CU_LAUNCH_KERNEL] == NULL) {
            real_func[SYM_CU_LAUNCH_KERNEL] = *pfn;
        }
        *pfn = (void *)(&cuLaunchKernel);
#pragma push_macro("cuMemcpyHtoD")
#undef cuMemcpyHtoD
    } else if (strcmp(symbol, STRINGIFY(cuMemcpyHtoD)) == 0) {
#pragma pop_macro("cuMemcpyHtoD")
        if(real_func[SYM_CU_MEM_H2D] == NULL) {
            real_func[SYM_CU_MEM_H2D] = *pfn;
        }        
    std::cout << " getProc H2D" << "cuda version: " << cudaVersion << std::endl;
        api_stats[SYM_CU_MEM_H2D][0]++;                                                                                                

        *pfn = (void *)(&cuMemcpyHtoD);
#pragma push_macro("cuMemcpyDtoH")
#undef cuMemcpyDtoH
    } else if (strcmp(symbol, STRINGIFY(cuMemcpyDtoH)) == 0) {
#pragma pop_macro("cuMemcpyDtoH")
    std::cout << " getProc D2H " << "cuda version: " << cudaVersion << std::endl;
        api_stats[SYM_CU_MEM_D2H][0]++;                                                                                                

        if(real_func[SYM_CU_MEM_D2H] == NULL) {
            real_func[SYM_CU_MEM_D2H] = *pfn;
        }        

        *pfn = (void *)(&cuMemcpyDtoH);
#pragma push_macro("cuInit")

#undef cuInit
    } else if (strcmp(symbol, STRINGIFY(cuInit)) == 0) {
#pragma pop_macro("cuInit")
    std::cout << " getProc cuInit " << "cuda version: " << cudaVersion << std::endl;

        *pfn = (void *)(&cuInit);
    } 
    
    return (result);
}
