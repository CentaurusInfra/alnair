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
#include <map>
#include <cuda_runtime.h>

#define STRINGIFY(x) STRINGIFY_AUX(x)
#define STRINGIFY_AUX(x) #x


extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }

static void* pre_hooks[SYM_CU_SYMBOLS]; 

static void* post_hooks[SYM_CU_SYMBOLS];

//
// the structure to store the logging configuration for each intercepted functions.
//
static bool enable_timeline[SYM_CU_SYMBOLS][STAT_CNT]; 
const char PFLOG[] = "./test";
const char metrices_file[] = "metrics.log";
const char profiling_file[] = "timeline.log";
static std::string metfile;
static std::string tlfile;
//
// this queue is to store timeline data before the data are writen to the log.
//
std::queue<pflog_t> pf_queue;


static void* real_func[SYM_CU_SYMBOLS];
static void* real_omp_get_num_threads = NULL;
static bool profiling_init = false;



const std::string funcname[] ={
    "cuInit",
    "cuMemAlloc",
    "cuMemFree",
    "cuLaunchKernel",
    "cuMemcpyH2D", 
    "cuMemcpyD2H",
    "cuGetProcAddress",
    "cuMemcpyH2D_ASYNC",
    "cudaMemcpy",
    "cudaMemcpyAsync"
    };

cuda_metrics_t pf = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .pid = 0,
    .bytes = 0,
    .mem_used = 0,
    .kind = -1,
    .period = {
        .tv_sec = 0,
        .tv_nsec = 10 * 1000000
    }
};

unsigned int api_stats[SYM_CU_SYMBOLS][STAT_CNT];


// void log_api_call(const int pid, const int memUsed, const int kernelCnt, const int burst) 
void log_api_call() 
{                                                                                               
    // std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();  
    // auto duration = now.time_since_epoch();                                                     
    std::ofstream fmet (metfile);
    fmet << "name:mem_used" << ",count:" << pf.mem_used << std::endl; 

    for (int i = 0; i < SYM_CU_SYMBOLS; i++) {
        //print timestamp at nano seconds when a cuda API is called                                                                     
        fmet << "name:" << funcname[i] << ",count:" << api_stats[i][0] << ",burst:"  << api_stats[i][1] << std::endl; 
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
      fmet <<"name:" << funcname[log.kernelid] << ", start:" << log.begin << ", burst:"  << log.burst << ", bytes:"  << log.bytecount << ", kind:"  << log.kind << std::endl;                                     
    }
    fmet.close();                                                                               
}

pthread_mutex_t mem_mutex = PTHREAD_MUTEX_INITIALIZER;
std::map<CUdeviceptr , size_t> allocation_map;

void update_memory_allocation(CUdeviceptr *ptr) {
  pthread_mutex_lock(&mem_mutex);

  if (ptr) {
    std::map<CUdeviceptr, size_t>::const_iterator pos = allocation_map.find(*ptr);  

    if (pos == allocation_map.end()) {
        fprintf(stderr, "Freeing unknown memory!" );
    } else {
        std::cout << "free mem: " << pos->second << std::endl;
        pf.mem_used -= pos->second ;
        allocation_map.erase(*ptr);
        pf.bytes = pos->second;
    }
  } else {
    allocation_map.clear();
    pf.bytes = pf.mem_used;
    pf.mem_used = 0;
  }
  pthread_mutex_unlock(&mem_mutex);    
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

    while(true) {
        unsigned int curGroupUsage;
        int ret;
        // pthread_mutex_lock(&pf.mutex);
        log_api_call();
        log_api_timing();
        // pthread_mutex_unlock(&pf.mutex);        
        nanosleep(&pf.period, NULL);
    }

    return NULL;
}

void *libcudaHandle = __libc_dlopen_mode("libcuda.so", RTLD_LAZY);
void *libdlHandle = __libc_dlopen_mode("libdl.so", RTLD_LAZY);
void *librtHandle = __libc_dlopen_mode("libcudart.so", RTLD_LAZY);



static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static pthread_once_t init_done = PTHREAD_ONCE_INIT;

CUresult cuMemAlloc_prehook(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemFree_prehook(CUdeviceptr* dptr);

CUresult cuMemcpyDtoH_prehook(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyHtoD_prehook( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount );

static void initialize(void)
{
    CUresult cures = CUDA_SUCCESS;
    CUdevice dev;
    int numSM, numThreadsPerSM, res;

    cures = cuDeviceGet(&dev, 0);
    if(CUDA_SUCCESS != cures) {
    }

    pthread_t pf_thread;

    res = pthread_create(&pf_thread, NULL, profiling_thread_func, NULL);
    if(res < 0) {
        fprintf(stderr,"profiler failed to start, errno=%d\n", errno);
    }

  //
  // the log configuration settings must be set individually for each intercepted functions.
  // this is designed for future improvement and expansion
  // 
    enable_timeline[SYM_CU_INIT][MET_STAT] = true;                      enable_timeline[SYM_CU_INIT][MET_TIMELINE] = true;                  enable_timeline[SYM_CU_INIT][MET_BYTES] = false;
    enable_timeline[SYM_CU_MEM_ALLOC][MET_STAT] = true;                 enable_timeline[SYM_CU_MEM_ALLOC][MET_TIMELINE] = true;            enable_timeline[SYM_CU_MEM_ALLOC][MET_BYTES] = true;
    enable_timeline[SYM_CU_MEM_FREE][MET_STAT] = true;                  enable_timeline[SYM_CU_MEM_FREE][MET_TIMELINE] = true;             enable_timeline[SYM_CU_MEM_FREE][MET_BYTES] = true;
    enable_timeline[SYM_CU_LAUNCH_KERNEL][MET_STAT] = true;             enable_timeline[SYM_CU_LAUNCH_KERNEL][MET_TIMELINE] = false;        enable_timeline[SYM_CU_LAUNCH_KERNEL][MET_BYTES] = false;
    enable_timeline[SYM_CU_MEM_H2D][MET_STAT] = true;                   enable_timeline[SYM_CU_MEM_H2D][MET_TIMELINE] = true;               enable_timeline[SYM_CU_MEM_H2D][MET_BYTES] = true;
    enable_timeline[SYM_CU_MEM_D2H][MET_STAT] = true;                   enable_timeline[SYM_CU_MEM_D2H][MET_TIMELINE] = true;               enable_timeline[SYM_CU_MEM_D2H][MET_BYTES] = true;
    enable_timeline[SYM_CU_HOOK_MEMCPY_HTOD_ASYNC][MET_STAT] = true;    enable_timeline[SYM_CU_HOOK_MEMCPY_HTOD_ASYNC][MET_TIMELINE] = true;enable_timeline[SYM_CU_HOOK_MEMCPY_HTOD_ASYNC][MET_BYTES] = true;
    enable_timeline[SYM_CU_HOOK_GET_PROC_ADDRESS][MET_STAT] = true;     enable_timeline[SYM_CU_HOOK_GET_PROC_ADDRESS][MET_TIMELINE] = false;enable_timeline[SYM_CU_HOOK_GET_PROC_ADDRESS][MET_BYTES] = false;
    enable_timeline[SYM_CU_MEMCPY][MET_STAT] = true;                    enable_timeline[SYM_CU_MEMCPY][MET_TIMELINE] = true;               enable_timeline[SYM_CU_MEMCPY][MET_BYTES] = true;
    enable_timeline[SYM_CU_MEMCPY_ASYNC][MET_STAT] = true;              enable_timeline[SYM_CU_MEMCPY_ASYNC][MET_TIMELINE] = true;         enable_timeline[SYM_CU_MEMCPY_ASYNC][MET_BYTES] = true;

    pre_hooks[SYM_CU_MEM_ALLOC] = (void*) cuMemAlloc_prehook;
    pre_hooks[SYM_CU_MEM_H2D] = (void*) cuMemcpyHtoD_prehook;
    pre_hooks[SYM_CU_MEM_D2H] = (void*) cuMemcpyDtoH_prehook;
    pre_hooks[SYM_CU_MEM_FREE] = (void*) cuMemFree_prehook;
    
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
        pf.bytes = 0;                                                                                                                  \
        if (pre_hooks[hooksymbol]) {                                                                                                   \
            res = ((CUresult (*)params)pre_hooks[hooksymbol])(__VA_ARGS__);                                                            \
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
                pf_queue.push({hooksymbol, Kbegin, burst, pf.bytes, -1});                                                              \
            }                                                                                                                          \
        }                                                                                                                              \
        if(CUDA_SUCCESS == res && post_hooks[hooksymbol]) {                                                                            \
            res = ((CUresult (*)params)post_hooks[hooksymbol])(__VA_ARGS__);                                                           \
        }                                                                                                                              \
        return res;                                                                                                                    \
    }

GENERATE_INTERCEPT_FUNCTION(SYM_CU_INIT, cuInit, (unsigned int Flags), Flags)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_ALLOC, cuMemAlloc, (CUdeviceptr* dptr, size_t ByteCount), dptr, ByteCount)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_FREE, cuMemFree, (CUdeviceptr dptr), dptr)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_H2D, cuMemcpyHtoD,( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount ), dstDevice, srcHost, ByteCount)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_D2H, cuMemcpyDtoH, (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount), dstHost, srcDevice, ByteCount)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_LAUNCH_KERNEL, cuLaunchKernel, 
                            (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                             unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                             unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra), 
                            f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 
                            sharedMemBytes, hStream, kernelParams, extra)



cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
{
    cudaError_t res = cudaSuccess;
    unsigned long  Kbegin, burst;                                                                                                  

    cudaError_t (*lcudaMemcpy) ( void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind  ))real_dlsym(RTLD_NEXT, "cudaMemcpy");
    Kbegin = (std::chrono::system_clock::now().time_since_epoch()).count();                                                        
    res = lcudaMemcpy( dst, src, count, kind );
    api_stats[SYM_CU_MEMCPY][0]++;                                                                                                
    burst = ((std::chrono::system_clock::now().time_since_epoch()).count() - Kbegin)/1000;                                     
    api_stats[SYM_CU_MEMCPY][1] += burst;                                                                                         
    pf_queue.push({SYM_CU_MEMCPY, Kbegin, burst, count, kind});                                                        
    return res;

}

cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str )
{
    cudaError_t res = cudaSuccess;
    unsigned long  Kbegin, burst;                                                                                                  

    cudaError_t (*lcudaMemcpyAsync) ( void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind, cudaStream_t   ))real_dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    Kbegin = (std::chrono::system_clock::now().time_since_epoch()).count();                                                        
    api_stats[SYM_CU_MEMCPY_ASYNC][0]++;                                                                                                
    res = lcudaMemcpyAsync( dst, src, count, kind, str );
    burst = ((std::chrono::system_clock::now().time_since_epoch()).count() - Kbegin)/1000;                                     
    pf_queue.push({SYM_CU_MEMCPY_ASYNC, Kbegin, burst, count, kind});                                                        
    api_stats[SYM_CU_MEMCPY_ASYNC][1] += burst;                                                                                         

    return res;
}



void* dlsym(void *handle, const char *symbol) 
{

    if (strncmp(symbol, "cu", 2) != 0) {
        return real_dlsym(handle, symbol);
    }
//   std::cout << symbol << " in dlsym" << std::endl;                                                                                

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
        api_stats[SYM_CU_MEM_D2H][0]++;                                                                                                

        return (void *)(&cuMemcpyDtoH);
    }

    if (strcmp(symbol, STRINGIFY(cuMemcpyHtoD)) == 0) {
        if(real_func[SYM_CU_MEM_H2D] == NULL) {
            real_func[SYM_CU_MEM_H2D] = real_dlsym(handle, symbol);
        }        
        return (void *)(&cuMemcpyHtoD);
    }

    if (strcmp(symbol, STRINGIFY(cudaMemcpy)) == 0) {
        if(real_func[SYM_CU_MEMCPY] == NULL) {
            real_func[SYM_CU_MEMCPY] = real_dlsym(handle, symbol);
        }        

        return (void *)(&cudaMemcpy);
    }    

    if (strcmp(symbol, STRINGIFY(cudaMemcpyAsync)) == 0) {
        if(real_func[SYM_CU_MEMCPY_ASYNC] == NULL) {
            real_func[SYM_CU_MEMCPY_ASYNC] = real_dlsym(handle, symbol);
        }        

        return (void *)(&cudaMemcpyAsync);
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
#pragma push_macro("cudaMemcpy")
#undef cudaMemcpy
    } else if (strcmp(symbol, STRINGIFY(cudaMemcpy)) == 0) {
#pragma pop_macro("cudaMemcpy")
        if(real_func[SYM_CU_MEMCPY] == NULL) {
            real_func[SYM_CU_MEMCPY] = *pfn;
        }        

        *pfn = (void *)(&cudaMemcpy);
#pragma push_macro("cudaMemcpyAsync")
#undef cudaMemcpyAsync
    } else if (strcmp(symbol, STRINGIFY(cudaMemcpyAsync)) == 0) {
#pragma pop_macro("cudaMemcpyAsync")
        if(real_func[SYM_CU_MEMCPY_ASYNC] == NULL) {
            real_func[SYM_CU_MEMCPY_ASYNC] = *pfn;
        }        

        *pfn = (void *)(&cudaMemcpyAsync);
#pragma push_macro("cuMemcpyHtoD")
#undef cuMemcpyHtoD
    } else if (strcmp(symbol, STRINGIFY(cuMemcpyHtoD)) == 0) {
#pragma pop_macro("cuMemcpyHtoD")
        if(real_func[SYM_CU_MEM_H2D] == NULL) {
            real_func[SYM_CU_MEM_H2D] = *pfn;
        }        
        api_stats[SYM_CU_MEM_H2D][0]++;                                                                                                

        *pfn = (void *)(&cuMemcpyHtoD);
#pragma push_macro("cuMemcpyDtoH")
#undef cuMemcpyDtoH
    } else if (strcmp(symbol, STRINGIFY(cuMemcpyDtoH)) == 0) {
#pragma pop_macro("cuMemcpyDtoH")
        api_stats[SYM_CU_MEM_D2H][0]++;                                                                                                

        if(real_func[SYM_CU_MEM_D2H] == NULL) {
            real_func[SYM_CU_MEM_D2H] = *pfn;
        }        

        *pfn = (void *)(&cuMemcpyDtoH);
#pragma push_macro("cuInit")

#undef cuInit
    } else if (strcmp(symbol, STRINGIFY(cuInit)) == 0) {
#pragma pop_macro("cuInit")

        *pfn = (void *)(&cuInit);
    } 
    
    return (result);
}

CUresult cuMemAlloc_prehook (CUdeviceptr* dptr, size_t bytesize)
{
    
  allocation_map[*dptr] = bytesize;
  pf.bytes = bytesize;
  pf.mem_used += bytesize;
  
  return CUDA_SUCCESS;
}


CUresult cuMemFree_prehook (CUdeviceptr* dptre)
{

    update_memory_allocation(dptre);
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD_prehook(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount) {
  
  pf.bytes = ByteCount;
  
  return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_prehook(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {

  pf.bytes = ByteCount;
  
  return CUDA_SUCCESS;
}
