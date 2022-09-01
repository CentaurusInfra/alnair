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

const char PFLOG[] = "/pflog";
const char metrices_file[] = "metrics.log";
const char profiling_file[] = "pf.log";
static std::string metfile;
static std::string tlfile;
static std::queue<pflog_t> pf_queue;

cuda_metrics_t pf = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .kernelCnt = 0,
    .memUsed = 0,
    .pid = 0,
    .kernelRunTime = 0,
    .H2DCnt = 0,
    .D2HCnt = 0,
    .Kbegin = 0,
    .H2Dbegin = 0,
    .D2Hbegin = 0,
    .UUID = 0,
    .period = {
        .tv_sec = 0,
        .tv_nsec = 100 * 1000000
    }
};

static const char* funcname[] ={
    "cuInit",
    "cuMemAlloc",
    "cuLaunchKernel"
    "cuGetProcAddress" 
    "cuMemcpyH2D", 
    "cuMemcpyH2D"
    };

void log_api_call(const int pid, const int memUsed, const int kernelCnt, const int burst) 
{                                                                                               
    // std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();  
    // auto duration = now.time_since_epoch();                                                     
    std::ofstream fmet (metfile);                                                                         
    //print timestamp at nano seconds when a cuda API is called                                                                     
    fmet << "pid:" << pid << "\nkernel-cnt:" << kernelCnt << "\nmem-used:" << memUsed << "\nburst-tm:"  << burst << std::endl;                                     
    fmet.close();                                                                               
}  

void log_api_timing() 
{                                                                                               
    std::ofstream fmet (tlfile);                                                                         
    while (!pf_queue.empty()) {
      // process request
      pflog log = pf_queue.front();
      pf_queue.pop();
      fmet <<"name:" << funcname[log.kernelid] << "\nstart:" << log.begin << "\nburst:"  << log.burst << std::endl;                                     
    }
    fmet.close();                                                                               
}
void* profiling_thread_func(void *arg) 
{
    std::cout << "====profiling thread runnin ==== ";


    if(const char* env_p = std::getenv("PFLOG")) {
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
        log_api_call(pf.pid, pf.memUsed, pf.kernelCnt, pf.kernelRunTime);
        log_api_timing();
        // pthread_mutex_unlock(&pf.mutex);        
        nanosleep(&pf.period, NULL);
    }

    return NULL;
}