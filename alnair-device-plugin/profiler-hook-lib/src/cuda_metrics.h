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

typedef struct cuda_metrics {
    pthread_mutex_t mutex;
    unsigned int kernelCnt;
    unsigned long long memUsed;
    unsigned int pid;
    unsigned long kernelRunTime;
    unsigned int H2DCnt;
    unsigned int D2HCnt;
    unsigned long H2DTime;
    unsigned long D2HTime;    
    std::chrono::steady_clock::time_point Kbegin;    
    std::chrono::steady_clock::time_point H2Dbegin;    
    std::chrono::steady_clock::time_point D2Hbegin;    
    unsigned int   UUID;
    struct timespec period;
} cuda_metrics_t;
