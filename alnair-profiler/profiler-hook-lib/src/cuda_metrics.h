/*
Copyright (c) 2022 Futurewei Technologies.
Author: Steven Wang

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

#define NO_PID  -2
#include <chrono>
#include <cuda.h>
#include <queue>


typedef enum HookSymbolsEnum {
    SYM_CU_INIT,
    SYM_CU_MEM_ALLOC,
    SYM_CU_MEM_FREE,
    SYM_CU_LAUNCH_KERNEL,
    SYM_CU_MEM_H2D,
    SYM_CU_MEM_D2H,
    SYM_CU_HOOK_GET_PROC_ADDRESS,
    SYM_CU_HOOK_MEMCPY_HTOD_ASYNC,
    SYM_CU_MEMCPY,
    SYM_CU_MEMCPY_ASYNC,
    SYM_CU_SYMBOLS,
} HookSymbols;

//
// profiling flags: [statics, timeline]
//
typedef enum MetricsEnum {
    MET_STAT,
    MET_TIMELINE,
    MET_BYTES,
    STAT_CNT
} Metrics_ENU;

typedef struct cuda_metrics {
    pthread_mutex_t mutex;
    unsigned int pid; 
    unsigned long bytes;
    unsigned long mem_used;
    int kind;
    struct timespec period;
} cuda_metrics_t;

typedef struct pflog {
    unsigned int kernelid;
    unsigned long  begin;    
    unsigned long  burst;
    unsigned long bytecount;
    int kind;
} pflog_t;
