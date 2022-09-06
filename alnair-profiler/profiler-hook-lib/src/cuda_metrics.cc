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

unsigned int api_stats[SYM_CU_SYMBOLS][STAT_CNT] = {
    {0, 0},               //cuInit
    {0, 0},              //cuAlloc
    {0, 0},              //cuFree
    {0, 0},              //cuLaunch
    {0, 0},              //cuH2D
    {0, 0},               //cuD2H
    {0, 0}             //cuGetProcAddress
};

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
    std::cout << "====profiling thread runnin ==== " << std::endl;


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
        // log_api_call(pf.pid, pf.memUsed, pf.kernelCnt, pf.kernelRunTime);
        log_api_call();
        // log_api_timing();
        // pthread_mutex_unlock(&pf.mutex);        
        nanosleep(&pf.period, NULL);
    }

    return NULL;
}