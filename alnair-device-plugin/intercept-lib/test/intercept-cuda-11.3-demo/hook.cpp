//
// nvcc -shared -lcuda --compiler-options '-fPIC' hook.cpp -o hook.so
// 
#include <dlfcn.h>
#include <string.h>
#include <iostream>
#include <cuda.h>

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);

extern "C" {
void *__libc_dlsym(void *map, const char *name);
}
extern "C" {
void *__libc_dlopen_mode(const char *name, int mode);
}


#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)
void *libcudaHandle = __libc_dlopen_mode("libcuda.so", RTLD_LAZY);
void *libdlHandle = __libc_dlopen_mode("libdl.so", RTLD_LAZY);

typedef void *(*fnDlsym)(void *, const char *);

static void *real_dlsym(void *handle, const char *symbol) {
  typedef void *(*fnDlsym)(void *, const char *);
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(libdlHandle, "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

typedef void *(*fn_cuMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
CUresult hook_cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  printf("cuMemAllocc_hook is called\n");
  static void *real_func = (void *)real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(cuMemAlloc)); \

  ((fn_cuMemAlloc)real_func)(dptr, bytesize);
 
  return CUDA_SUCCESS;
}

typedef void *(*fn_cuInit)(unsigned int flag);
CUresult hook_cuInit(unsigned int flag) {
  printf("cuInitc_hook is called\n");
  static void *real_func = (void *)real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(cuInit)); \

  ((fn_cuInit)real_func)(flag);
 
  return CUDA_SUCCESS;
}

/*
 ** interposed functions
 */
void *dlsym(void *handle, const char *symbol) {
  // Early out if not a CUDA driver symbol
  if (strncmp(symbol, "cu", 2) != 0) {
    return (real_dlsym(handle, symbol));
  }
  if (strcmp(symbol, CUDA_SYMBOL_STRING(cuGetProcAddress)) == 0) {
        return (void *)(&cuGetProcAddress);
  }else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0) {
    return (void *)(&cuMemAlloc);
  } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuInit)) == 0) {
    return (void *)(&cuInit);
  } 

  
  // omit cuDeviceTotalMem here so there won't be a deadlock in cudaEventCreate when we are in
  // initialize(). Functions called by client are still being intercepted.
  return (real_dlsym(handle, symbol));
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
#ifdef _DEBUG
    printf("Enter %s\n", CUDA_SYMBOL_STRING(cuGetProcAddress));
    printf("symbol %s, cudaVersion %d, flags %lu\n", symbol, cudaVersion, flags);
#endif
    typedef decltype(&cuGetProcAddress) funcType;
    funcType actualFunc;
    if(strcmp(symbol, "cuGetProcAddress") == 0)
        actualFunc = (funcType)real_dlsym(libcudaHandle, CUDA_SYMBOL_STRING(cuGetProcAddress));

    CUresult result = actualFunc(symbol, pfn, cudaVersion, flags);

    if(strcmp(symbol, CUDA_SYMBOL_STRING(cuGetProcAddress)) == 0) {
        *pfn = (void*)(&cuGetProcAddress);

#pragma push_macro("cuMemAlloc")
#undef cuMemAlloc
    } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0) {
#pragma pop_macro("cuMemAlloc")
        *pfn = (void *)(&hook_cuMemAlloc);
#pragma push_macro("cuInit")
#undef cuInit
    } else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuInit)) == 0) {
#pragma pop_macro("cuInit")
        *pfn = (void *)(&hook_cuInit);
    } 


    return (result);
}