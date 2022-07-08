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

/*
The CUDA intercept technique used in this file is based on the 
following references:
1) https://osterlund.xyz/posts/2018-03-12-interceptiong-functions-c.html
2) https://stackoverflow.com/questions/37792037/ld-preload-doesnt-affect-dlopen-with-rtld-now
3) The Nvidia example "libcuhook.cpp"

A few hard-to-diagnose segmentation faults encounted in the Nvidia example
have been fixed:

Fix 1:
dlsym call can be triggered either by ld-linux.so or manually by the
user program. For a symbol to be looked up successfully by real_dlsym(), 
it looks that the corresponding shared library has to be already loaded
somewhere in the memory.

There are two events in question here: 
1) when a function symbol (e.g. cuInit) is looked up by a dlsym call
2) when the function (e.g. cuInit) is called

When the 1st event happens, our cuInit overwrite function is returned.
However, the cuInit overwrite function may not be immediately called.
Instead, the actual call (2nd event) can happen much later, when 
real_dlsym() can no longer find where the real cuInit symbol is. 
This leads to a NULL value returned by real_dlsym() and thus 
the segmentation fault.

The solution is to save the real addresses when the initial lookups
happen. This is likely to succeed because otherwise a normal run 
(without PRELOAD) would also fail.

In this code, real_func[SYM_CU_SYMBOLS] is used to save the real addresses.

Fix 2:
Sometimes real_dlsym() failes to look up "omp_get_num_threads".
The solution is a workaround. See the code below.
*/

#define _USE_GNU
#include <stdio.h>
#include <dlfcn.h>
#include <cuda.h>
#include <string.h>
#include <chrono>
#include <fstream>

#define STRINGIFY(x) STRINGIFY_AUX(x)
#define STRINGIFY_AUX(x) #x
CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);


extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }

const char metrices_file[] = "/var/lib/alnair/workspace/metrics.log";
void *libcudaHandle = __libc_dlopen_mode("libcuda.so", RTLD_LAZY);
void *libdlHandle = __libc_dlopen_mode("libdl.so", RTLD_LAZY);

typedef enum HookSymbolsEnum {
    SYM_CU_INIT,
    SYM_CU_MEM_ALLOC,
    SYM_CU_MEM_ALLOC_MANAGED,
    SYM_CU_MEM_ALLOC_PITCH,
    SYM_CU_ARRAY_CREATE,
    SYM_CU_ARRAY_3D_CREATE,
    SYM_CU_MIP_ARRAY_CREATE,
    SYM_CU_LAUNCH_KERNEL,
    SYM_CU_LAUNCH_COOP_KERNEL,
    SYM_CU_DEVICE_TOTAL_MEM,
    SYM_CU_MEM_GET_INFO,
    SYM_CU_HOOK_GET_PROC_ADDRESS,
    SYM_CU_SYMBOLS,
} HookSymbols;

extern CUresult cuInit_hook(unsigned int Flags);
extern CUresult cuInit_posthook(unsigned int Flags);
extern CUresult cuMemAlloc_hook(CUdeviceptr* dptr, size_t bytesize);
extern CUresult cuMemAllocManaged_hook(CUdeviceptr *dptr, size_t bytesize, unsigned int flags);
extern CUresult cuMemAllocPitch_hook(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, 
                                     size_t Height, unsigned int ElementSizeBytes);
extern CUresult cuArrayCreate_hook(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray);
extern CUresult cuArray3DCreate_hook(CUarray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pAllocateArray);
extern CUresult cuMipmappedArrayCreate_hook(CUmipmappedArray *pHandle, 
                                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, 
                                       unsigned int numMipmapLevels);
extern CUresult cuLaunchKernel_hook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, 
                                    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, 
                                    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, 
                                    void** kernelParams, void** extra);
extern CUresult cuLaunchCooperativeKernel_hook(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, 
                                               unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                                               unsigned int blockDimZ, unsigned int sharedMemBytes,
                                               CUstream hStream, void **kernelParams);
extern CUresult cuDeviceTotalMem_posthook(size_t* bytes, CUdevice dev);
extern CUresult cuMemGetInfo_posthook(size_t* free, size_t* total);


static void* hooks[SYM_CU_SYMBOLS] = {
    (void*) cuInit_hook, 
    (void*) cuMemAlloc_hook,
    (void*) cuMemAllocManaged_hook,
    (void*) cuMemAllocPitch_hook,
    (void*) cuArrayCreate_hook,
    (void*) cuArray3DCreate_hook,
    (void*) cuMipmappedArrayCreate_hook,
    (void*) cuLaunchKernel_hook,
    (void*) cuLaunchCooperativeKernel_hook,
    NULL,
    NULL
};

static void* post_hooks[SYM_CU_SYMBOLS] = {
    (void*) cuInit_posthook, 
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    (void*) cuDeviceTotalMem_posthook,
    (void*) cuMemGetInfo_posthook
};

static void* real_func[SYM_CU_SYMBOLS];
static void* real_omp_get_num_threads = NULL;

typedef void* (*fnDlsym)(void*, const char*);
static void* real_dlsym(void *handle, const char* symbol)
{
    // static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    // return (*internal_dlsym)(handle, symbol);
//   typedef void *(*fnDlsym)(void *, const char *);
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(libdlHandle, "dlsym");
    return (*internal_dlsym)(handle, symbol);    
}

void* dlsym(void *handle, const char *symbol) 
{
    if (strcmp(symbol, "omp_get_num_threads") == 0) {
        if(real_omp_get_num_threads == NULL)
            real_omp_get_num_threads = (void*)__libc_dlsym(__libc_dlopen_mode("libgomp.so.1", RTLD_LAZY), "omp_get_num_threads");
        return real_omp_get_num_threads;
    }

    if (strncmp(symbol, "cu", 2) != 0) {
        return real_dlsym(handle, symbol);
    }

    if (strcmp(symbol, STRINGIFY(cuInit)) == 0) {
        return (void*)(&cuInit);
    } else if (strcmp(symbol, STRINGIFY(cuMemAlloc)) == 0) {
        return (void*)(&cuMemAlloc);
    } else if (strcmp(symbol, STRINGIFY(cuMemAllocManaged)) == 0) {
        return (void*)(&cuMemAllocManaged);
    } else if (strcmp(symbol, STRINGIFY(cuMemAllocPitch)) == 0) {
        return (void*)(&cuMemAllocPitch);
    } else if (strcmp(symbol, STRINGIFY(cuArrayCreate)) == 0) {
        return (void*)(&cuArrayCreate);
    } else if (strcmp(symbol, STRINGIFY(cuArray3DCreate)) == 0) {
        return (void*)(&cuArray3DCreate);
    } else if (strcmp(symbol, STRINGIFY(cuMipmappedArrayCreate)) == 0) {
        return (void*)(&cuMipmappedArrayCreate);
    } else if (strcmp(symbol, STRINGIFY(cuLaunchKernel)) == 0) {
        return (void*)(&cuLaunchKernel);
    } else if (strcmp(symbol, STRINGIFY(cuLaunchCooperativeKernel)) == 0) {
        return (void*)(&cuLaunchCooperativeKernel);
    } else if (strcmp(symbol, STRINGIFY(cuDeviceTotalMem)) == 0) {
        return (void*)(&cuDeviceTotalMem);
    } else if (strcmp(symbol, STRINGIFY(cuMemGetInfo)) == 0) {
        return (void*)(&cuMemGetInfo);
    }

    return (real_dlsym(handle, symbol));
}


#define SAMPLE_CUDA(symbol)                                                                         \
    {                                                                                               \
        std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();  \
        auto duration = now.time_since_epoch();                                                     \
        std::ofstream fmet;                                                                         \
        fmet.open(metrices_file);                                                                   \
        fmet << duration.count() << ',' << symbol << std::endl;                                     \
        fmet.close();                                                                               \
    }                                                                                               \

#define GENERATE_INTERCEPT_FUNCTION(hookname, hooksymbol, funcname, params, ...)                     \
    CUresult hookname params                                                               \
    {                                                                                      \
        CUresult res = CUDA_SUCCESS;                                                       \
        if (hooks[hooksymbol]) {                                                           \
            res = ((CUresult (*)params)hooks[hooksymbol])(__VA_ARGS__);                    \
        }                                                                                  \
        if(CUDA_SUCCESS != res) return res;                                                \
        if(real_func[hooksymbol] == NULL)                                                  \
            real_func[hooksymbol] = real_dlsym(RTLD_NEXT, STRINGIFY(funcname));            \
        res = ((CUresult (*)params)real_func[hooksymbol])(__VA_ARGS__);                    \
        if(CUDA_SUCCESS == res && post_hooks[hooksymbol]) {                                \
            res = ((CUresult (*)params)post_hooks[hooksymbol])(__VA_ARGS__);               \
        }                                                                                  \
        return res;                                                                        \
    }

GENERATE_INTERCEPT_FUNCTION(hook_cuInit, SYM_CU_INIT, cuInit, (unsigned int Flags), Flags)
GENERATE_INTERCEPT_FUNCTION(hook_cuMemAlloc, SYM_CU_MEM_ALLOC, cuMemAlloc, (CUdeviceptr* dptr, size_t bytesize), dptr, bytesize)
GENERATE_INTERCEPT_FUNCTION(hook_cuMemAllocManaged, SYM_CU_MEM_ALLOC_MANAGED, cuMemAllocManaged, 
                            (CUdeviceptr *dptr, size_t bytesize, unsigned int flags),
                            dptr, bytesize, flags)
GENERATE_INTERCEPT_FUNCTION(hook_cuMemAllocPitch, SYM_CU_MEM_ALLOC_PITCH, cuMemAllocPitch, 
                            (CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, 
                             size_t Height, unsigned int ElementSizeBytes), 
                            dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
GENERATE_INTERCEPT_FUNCTION(hook_cuArrayCreate, SYM_CU_ARRAY_CREATE, cuArrayCreate, 
                            (CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray), 
                            pHandle, pAllocateArray)
GENERATE_INTERCEPT_FUNCTION(hook_cuArray3DCreate, SYM_CU_ARRAY_3D_CREATE, cuArray3DCreate, 
                            (CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray), 
                            pHandle, pAllocateArray)
GENERATE_INTERCEPT_FUNCTION(hook_cuMipmappedArrayCreate, SYM_CU_MIP_ARRAY_CREATE, cuMipmappedArrayCreate, 
                            (CUmipmappedArray *pHandle, 
                             const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels), 
                            pHandle, pMipmappedArrayDesc, numMipmapLevels)                            
GENERATE_INTERCEPT_FUNCTION(hook_cuLaunchKernel, SYM_CU_LAUNCH_KERNEL, cuLaunchKernel, 
                            (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                             unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                             unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra), 
                            f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 
                            sharedMemBytes, hStream, kernelParams, extra)
GENERATE_INTERCEPT_FUNCTION(hook_cuLaunchCooperativeKernel, SYM_CU_LAUNCH_COOP_KERNEL, cuLaunchCooperativeKernel,
                            (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, 
                             unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
                             unsigned int blockDimZ, unsigned int sharedMemBytes, 
                             CUstream hStream, void **kernelParams),
                            f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 
                            sharedMemBytes, hStream, kernelParams)
GENERATE_INTERCEPT_FUNCTION(hook_cuDeviceTotalMem, SYM_CU_DEVICE_TOTAL_MEM, cuDeviceTotalMem, 
                            (size_t* bytes, CUdevice dev), bytes, dev)
GENERATE_INTERCEPT_FUNCTION(hook_cuMemGetInfo, SYM_CU_MEM_GET_INFO, cuMemGetInfo, (size_t* free, size_t* total), free, total)



CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
#ifdef _DEBUG
    //printf("Enter %s\n", SYMBOL_STRING(cuGetProcAddress));
    //printf("symbol %s, cudaVersion %d, flags %lu\n", symbol, cudaVersion, flags);
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
        if(real_func[SYM_CU_MEM_GET_INFO] == NULL) {
            real_func[SYM_CU_MEM_GET_INFO] = *pfn;
        }
        *pfn = (void *)(&hook_cuMemAlloc);
#pragma push_macro("cuMemAllocManaged")
#undef cuMemAllocManaged
    } else if (strcmp(symbol, STRINGIFY_AUX(cuMemAllocManaged)) == 0) {
#pragma pop_macro("cuMemAllocManaged")
        if(real_func[SYM_CU_MEM_ALLOC_MANAGED] == NULL) {
            real_func[SYM_CU_MEM_ALLOC_MANAGED] = *pfn;
        }
        *pfn = (void *)(&hook_cuMemAllocManaged);
#pragma push_macro("cuMemAllocPitch")
#undef cuMemAllocPitch
    } else if (strcmp(symbol, STRINGIFY_AUX(cuMemAllocPitch)) == 0) {
#pragma pop_macro("cuMemAllocPitch")
        if(real_func[SYM_CU_MEM_ALLOC_PITCH] == NULL) {
            real_func[SYM_CU_MEM_ALLOC_PITCH] = *pfn;
        }
        *pfn = (void *)(&hook_cuMemAllocPitch);
#pragma push_macro("cuLaunchKernel")
#undef cuLaunchKernel
    } else if  (strcmp(symbol, STRINGIFY_AUX(cuLaunchKernel)) == 0) { 
#pragma pop_macro("cuLaunchKernel")
        if(real_func[SYM_CU_LAUNCH_KERNEL] == NULL) {
            real_func[SYM_CU_LAUNCH_KERNEL] = *pfn;
        }
        *pfn = (void *)(&hook_cuLaunchKernel);
#pragma push_macro("cuLaunchCooperativeKernel")
#undef cuLaunchCooperativeKernel
    } else if  (strcmp(symbol, STRINGIFY_AUX(cuLaunchCooperativeKernel)) == 0) {
#pragma pop_macro("cuLaunchCooperativeKernel")
        if(real_func[SYM_CU_LAUNCH_COOP_KERNEL] == NULL) {
            real_func[SYM_CU_LAUNCH_COOP_KERNEL] = *pfn;
        }
        *pfn = (void *)(&hook_cuLaunchCooperativeKernel);
#pragma push_macro("cuArrayCreate")
#undef cuArrayCreate
    } else if  (strcmp(symbol, STRINGIFY_AUX(cuArrayCreate)) == 0) {
#pragma pop_macro("cuArrayCreate")
        if(real_func[SYM_CU_ARRAY_CREATE] == NULL) {
            real_func[SYM_CU_ARRAY_CREATE] = *pfn;
        }
        *pfn = (void *)(&hook_cuArrayCreate);
#pragma push_macro("cuArray3DCreate")
#undef cuArray3DCreate
    } else if  (strcmp(symbol, STRINGIFY_AUX(cuArray3DCreate)) == 0) {
#pragma pop_macro("cuArray3DCreate")
        if(real_func[SYM_CU_ARRAY_3D_CREATE] == NULL) {
            real_func[SYM_CU_ARRAY_3D_CREATE] = *pfn;
        }
        *pfn = (void *)(&hook_cuArray3DCreate);
#pragma push_macro("cuMipmappedArrayCreate")
#undef cuMipmappedArrayCreate
    } else if  (strcmp(symbol, STRINGIFY_AUX(cuMipmappedArrayCreate)) == 0) {
#pragma pop_macro("cuMipmappedArrayCreate")
        if(real_func[SYM_CU_MIP_ARRAY_CREATE] == NULL) {
            real_func[SYM_CU_MIP_ARRAY_CREATE] = *pfn;
        }
        *pfn = (void *)(&hook_cuMipmappedArrayCreate);
#pragma push_macro("cuDeviceTotalMem")
#undef cuDeviceTotalMem
    } else if (strcmp(symbol, STRINGIFY(cuDeviceTotalMem)) == 0) {
#pragma pop_macro("cuDeviceTotalMem")
        if(real_func[SYM_CU_DEVICE_TOTAL_MEM] == NULL) {
            real_func[SYM_CU_DEVICE_TOTAL_MEM] = *pfn;
        }
        *pfn = (void *)(&hook_cuDeviceTotalMem);

#pragma push_macro("cuMemGetInfo")
#undef cuMemGetInfo
    } else if (strcmp(symbol, STRINGIFY(cuMemGetInfo)) == 0) {
#pragma pop_macro("cuMemGetInfo")
        if(real_func[SYM_CU_MEM_GET_INFO] == NULL) {
            real_func[SYM_CU_MEM_GET_INFO] = *pfn;
        }
        *pfn = (void *)(&hook_cuMemGetInfo);
#pragma push_macro("cuInit")
#undef cuInit
    } else if (strcmp(symbol, STRINGIFY(cuInit)) == 0) {
#pragma pop_macro("cuInit")
        *pfn = (void *)(&hook_cuInit);
    } 
    
#ifdef _DEBUG
    //printf("Leave %s\n", SYMBOL_STRING(cuGetProcAddress));
#endif
    return (result);
}
