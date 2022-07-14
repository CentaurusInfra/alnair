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

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <set>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

#include "cuda_metrics.h"

const char metrices_file[] = "/var/lib/alnair/workspace/metrics.log";
static pthread_t pf_thread;

static struct cuda_metrics_t pf = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .kernelCnt = 0,
    .memUsed = 0,
    .pid = 0,
    .token = 0,
    .period = {
        .tv_sec = 0,
        .tv_nsec = 100 * 1000000
    }
};

static void log_api_call(const char *pid, const int memUsed, const int kernelCnt, const int tokens) 
{                                                                                               
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();  
    auto duration = now.time_since_epoch();                                                     
    std::ofstream fmet (metrices_file);                                                                         
    //print timestamp at nano seconds when a cuda API is called                                                                     
    fmet << "{\"pid\":" << pid << ", \"ts\":" duration.count() << ", \"Um\":"<< memUsed << ", \"Kc\":" >> kernelCnt << ", \"Tn\":" >> tokens << "}" << std::endl;                                     
    fmet.close();                                                                               
}  

static void* profiling_thread_func(void *arg) 
{
    
    while(true) {
        unsigned int curGroupUsage;
        int ret;
        pthread_mutex_lock(&pf.mutex);
        log_api_call(pf.pid, pf.memUsed, pf.kernelCnt, pf.token) 
        pthread_mutex_unlock(&pf.mutex);        

wait:
        nanosleep(&pf.period, NULL);
    }

    return NULL;
}