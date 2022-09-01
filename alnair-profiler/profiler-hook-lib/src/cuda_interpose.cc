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

#define STRINGIFY(x) STRINGIFY_AUX(x)
#define STRINGIFY_AUX(x) #x


extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }

// extern cuda_metrics_t pf;


typedef enum HookSymbolsEnum {
    SYM_CU_INIT,
    SYM_CU_MEM_ALLOC,
    SYM_CU_MEM_FREE,
    SYM_CU_LAUNCH_KERNEL,
    SYM_CU_HOOK_GET_PROC_ADDRESS,
    SYM_CU_MEM_H2D,
    SYM_CU_MEM_D2H,
    SYM_CU_SYMBOLS,
} HookSymbols;

extern CUresult cuInit_hook(unsigned int Flags);
extern CUresult cuInit_posthook(unsigned int Flags);
extern CUresult cuMemAlloc_hook(CUdeviceptr* dptr, size_t bytesize);
extern CUresult cuMemFree_hook(CUdeviceptr *dptr);
extern CUresult cuMemcpyDtoH_posthook(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
extern CUresult cuMemcpyHtoD_posthook( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount );
extern CUresult cuLaunchKernel_hook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, 
                                    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, 
                                    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, 
                                    void** kernelParams, void** extra);
extern CUresult cuLaunchKernel_posthook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, 
                                    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, 
                                    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, 
                                    void** kernelParams, void** extra);

extern  std::queue<pflog> pf_queue;

static void* hooks[SYM_CU_SYMBOLS] = {
    (void*) cuInit_hook, 
    (void*) cuMemAlloc_hook,
    (void*) cuMemFree_hook,
    (void*) cuLaunchKernel_hook,
    NULL,
    NULL,
    NULL
};

static void* post_hooks[SYM_CU_SYMBOLS] = {
    (void*) cuInit_posthook, 
    NULL,
    NULL,
    NULL,
    NULL,
    (void*) cuMemcpyHtoD_posthook,
    (void*) cuMemcpyDtoH_posthook
};

static bool enable_timeline[SYM_CU_SYMBOLS] = {
    true, 
    false,
    false,
    false,
    false,
    true,
    true
};


static void* real_func[SYM_CU_SYMBOLS];
static void* real_omp_get_num_threads = NULL;

void *libcudaHandle = __libc_dlopen_mode("libcuda.so", RTLD_LAZY);
void *libdlHandle = __libc_dlopen_mode("libdl.so", RTLD_LAZY);

typedef void *(*fnDlsym)(void *, const char *);
static void *real_dlsym(void *handle, const char *symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(libdlHandle, "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

void* dlsym(void *handle, const char *symbol) 
{
    // if (strcmp(symbol, "omp_get_num_threads") == 0) {
    //     if(real_omp_get_num_threads == NULL)
    //         real_omp_get_num_threads = (void*)__libc_dlsym(__libc_dlopen_mode("libgomp.so.1", RTLD_LAZY), "omp_get_num_threads");
    //     return real_omp_get_num_threads;
    // }

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
        return (void *)(&cuMemcpyDtoH);
    }

    if (strcmp(symbol, STRINGIFY(cuMemcpyHtoD)) == 0) {
        return (void *)(&cuMemcpyHtoD);
    }

    if (strcmp(symbol, STRINGIFY(cuGetProcAddress)) == 0) {
        return (void *)(&cuGetProcAddress);
    }
    return (real_dlsym(handle, symbol));
}


#define GENERATE_INTERCEPT_FUNCTION(hooksymbol, funcname, params, ...)                                             \
    CUresult funcname params                                                                                       \
    {                                                                                                              \
        CUresult res = CUDA_SUCCESS;                                                                               \
        unsigned long  Kbegin;                                                \
        if (hooks[hooksymbol]) {                                                                                   \
            res = ((CUresult (*)params)hooks[hooksymbol])(__VA_ARGS__);                                            \
        }                                                                                                          \
        if(CUDA_SUCCESS != res) return res;                                                                        \
        if(real_func[hooksymbol] == NULL)                                                                          \
            real_func[hooksymbol] = real_dlsym(RTLD_NEXT, STRINGIFY(funcname));                                    \
        if(enable_timeline[hooksymbol])                                                                            \
            Kbegin = (std::chrono::system_clock::now().time_since_epoch()).count();                                \
        res = ((CUresult (*)params)real_func[hooksymbol])(__VA_ARGS__);                                            \
        if(enable_timeline[hooksymbol])                                                                            \
            pf_queue.push({hooksymbol, Kbegin, ((std::chrono::system_clock::now().time_since_epoch()).count() - Kbegin)/1000});   \
        if(CUDA_SUCCESS == res && post_hooks[hooksymbol]) {                                                        \
            res = ((CUresult (*)params)post_hooks[hooksymbol])(__VA_ARGS__);                                       \
        }                                                                                                          \
        return res;                                                                                                \
    }

GENERATE_INTERCEPT_FUNCTION(SYM_CU_INIT, cuInit, (unsigned int Flags), Flags)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_ALLOC, cuMemAlloc, (CUdeviceptr* dptr, size_t bytesize), dptr, bytesize)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_FREE, cuMemFree, (CUdeviceptr dptr), dptr)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_H2D, cuMemcpyHtoD,( CUdeviceptr dptr, const void* srcHost, size_t ByteCount ), dptr, srcHost, ByteCount)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_MEM_D2H, cuMemcpyDtoH, (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount), dstHost, srcDevice, ByteCount)
GENERATE_INTERCEPT_FUNCTION(SYM_CU_LAUNCH_KERNEL, cuLaunchKernel, 
                            (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                             unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                             unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra), 
                            f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 
                            sharedMemBytes, hStream, kernelParams, extra)

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
#pragma push_macro("cuInit")
#undef cuInit
    } else if (strcmp(symbol, STRINGIFY(cuInit)) == 0) {
#pragma pop_macro("cuInit")
        *pfn = (void *)(&cuInit);
    } 
    
    return (result);
}
