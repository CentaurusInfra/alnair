#include <dlfcn.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include <chrono>
#include <ctime>

#define STRINGIFY(x) STRINGIFY_AUX(x)
#define STRINGIFY_AUX(x) #x

/*
   1) use stringify wrap function name, so cuMemAlloc and cuMemAlloc_v2 can be all intercepted
   2) make sure function input types write, refer to cuda driver api https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html
   3) if function not intercepted succesfully, prob is caused that name of function is not called. e.g., cuMemcpy is not called, instead used cuMemcpyHtoD
*/
extern "C" {void *__libc_dlsym(void *map, const char *name);}
extern "C" {void *__libc_dlopen_mode(const char *name, int maddArgumentode);}

void *libcudaHandle = __libc_dlopen_mode("libcuda.so", RTLD_LAZY);
void *libdlHandle = __libc_dlopen_mode("libdl.so", RTLD_LAZY);

typedef void *(*fnDlsym)(void *, const char *);
static void *real_dlsym(void *handle, const char *symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(libdlHandle, "dlsym");
    return (*internal_dlsym)(handle, symbol);
}


typedef enum HookSymbolsEnum {
    SYM_CU_INIT,
    SYM_CU_MEM_ALLOC,
    SYM_CU_MEM_CPY_HTOD,
    SYM_CU_MEM_CPY,
    SYM_CU_HOOK_GET_PROC_ADDRESS,
    SYM_CU_SYMBOLS,
} HookSymbols;
static void* real_func[SYM_CU_SYMBOLS];

CUresult cuInit(unsigned int flag) {
    std::cout << "====cuInit hooked====at ";
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    std::cout << duration.count() << std::endl;
    if (real_func[SYM_CU_INIT] == NULL) real_func[SYM_CU_INIT] = real_dlsym(RTLD_NEXT, "cuInit");
    return  ((CUresult (*)(unsigned int))real_func[SYM_CU_INIT])(flag);
}

CUresult cuMemcpyHtoD (CUdeviceptr dst, const void* srcHost, size_t ByteCount) {
    std::cout << "@@@@==cuMemcpyHtoD hooked=****===at ";
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    std::cout << duration.count() << std::endl;
    if (real_func[SYM_CU_MEM_CPY_HTOD] == NULL) real_func[SYM_CU_MEM_CPY_HTOD] = real_dlsym(RTLD_NEXT, "cuMemcpyHtoD");
    return  ((CUresult (*)(CUdeviceptr,const void*,size_t))real_func[SYM_CU_MEM_CPY_HTOD])(dst, srcHost, ByteCount);
}
CUresult cuMemcpy (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    std::cout << "@@@@==cuMemcpy hooked=****===" << std::endl;
    if (real_func[SYM_CU_MEM_CPY] == NULL) real_func[SYM_CU_MEM_CPY] = real_dlsym(RTLD_NEXT, "cuMemcpy");
    return  ((CUresult (*)(CUdeviceptr,CUdeviceptr,size_t))real_func[SYM_CU_MEM_CPY])(dst, src, ByteCount);
}

CUresult cuMemAlloc (CUdeviceptr* dptr, size_t bytesize) { 
    std::cout << "@@@@==cuMemAlloc hooked====" << std::endl;
    if (real_func[SYM_CU_MEM_ALLOC] == NULL) real_func[SYM_CU_MEM_ALLOC] = real_dlsym(RTLD_NEXT, "cuMemAlloc");
    return  ((CUresult (*)(CUdeviceptr*, size_t))real_func[SYM_CU_MEM_ALLOC])(dptr, bytesize);
}

void *dlsym(void *handle, const char *symbol)   
{
    if (strcmp(symbol, STRINGIFY(cuMemcpyHtoD)) == 0) {
        if(real_func[SYM_CU_MEM_CPY_HTOD] == NULL) real_func[SYM_CU_MEM_CPY_HTOD] = real_dlsym(handle, symbol); 
        return (void*)(&cuMemcpyHtoD);
    }else if (strcmp(symbol, STRINGIFY(cuInit)) == 0) {
        if(real_func[SYM_CU_INIT] == NULL) real_func[SYM_CU_INIT] = real_dlsym(handle, symbol);
        return (void*)(&cuInit);
    }else if (strcmp(symbol, STRINGIFY(cuMemAlloc)) == 0) {
	    if(real_func[SYM_CU_MEM_ALLOC] == NULL) real_func[SYM_CU_MEM_ALLOC] = real_dlsym(handle, symbol);
        return (void*)(&cuMemAlloc);
    }else if (strcmp(symbol, STRINGIFY(cuGetProcAddress)) == 0) {
        return (void *)(&cuGetProcAddress);
    }
    return (real_dlsym(handle, symbol));
}

CUresult CUDAAPI cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {

    printf("Enter %s\n", STRINGIFY(cuGetProcAddress));
    printf("symbol %s, cudaVersion %d, flags %lu\n", symbol, cudaVersion, flags);

    typedef decltype(&cuGetProcAddress) funcType;
    funcType actualFunc;
    if(!real_func[SYM_CU_HOOK_GET_PROC_ADDRESS])
        actualFunc = (funcType)real_dlsym(libcudaHandle, STRINGIFY(cuGetProcAddress));
    else
        actualFunc = (funcType)real_func[SYM_CU_HOOK_GET_PROC_ADDRESS];
    CUresult result = actualFunc(symbol, pfn, cudaVersion, flags);

    if(strcmp(symbol, STRINGIFY(cuGetProcAddress)) == 0) {
        real_func[SYM_CU_HOOK_GET_PROC_ADDRESS] = *pfn;
        *pfn = (void*)(&cuGetProcAddress);
#pragma push_macro("cuMemAlloc")
#undef cuMemAlloc
    } else if (strcmp(symbol, STRINGIFY(cuMemAlloc)) == 0) {
#pragma pop_macro("cuMemAlloc")
        if(real_func[SYM_CU_MEM_ALLOC] == NULL) {
            real_func[SYM_CU_MEM_ALLOC] = *pfn;
        }
        *pfn = (void *)(&cuMemAlloc);
#pragma push_macro("cuInit")
#undef cuInit
    } else if (strcmp(symbol, STRINGIFY(cuInit)) == 0) {
#pragma pop_macro("cuInit")
        if(real_func[SYM_CU_INIT] == NULL) {
            real_func[SYM_CU_INIT] = *pfn;
        }
        *pfn = (void *)(&cuInit);
    } 

    return (result);
}