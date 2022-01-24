from prometheus_api_client import PrometheusConnect, MetricsList
from prometheus_api_client.utils import parse_datetime
import pandas as pd
import os
import time
import logging
from kubernetes import config, client
from scipy.stats import norm
import numpy as np
import subprocess
import copy
from pynvml import *
import math
import ast
from pymongo import MongoClient
import pymongo
from datetime import datetime
import pytz
from metadata_store import update_job_metrics_to_db

MEM_UTIL = "DCGM_FI_DEV_MEM_COPY_UTIL"
GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
DOMAIN = "ai.centaurus.io"
def cyclic_pattern_detection(time_series):
    """input pandas series, detect cyclic pattern return True/False
    if True, return frequency, if false, frequency is -1
    """
    # calculate autocorrelation
    auto_corr = [time_series.autocorr(lag=i) for i in range(int(len(time_series)/2))]
    # assume auto_corr value is normal distribution, based on 95% confidence interval, calculate the line for signifence
    critical = norm.ppf(1-0.05/2, loc=np.mean(auto_corr), scale=np.std(auto_corr))
    peak_lag = []
    # select the peak of correlation coefficients
    for i, v in enumerate(auto_corr):
        if v > critical:  # if auto corr value > critical value, consider the correlation is significant
            peak_lag.append(i)
    if len(peak_lag) > 2: # repetitive significant peaks as the rule for cyclic patterns
        lag_diff = pd.Series(peak_lag).diff()  # to calculate period
        period = lag_diff.median()
        return True, period
    else:
        return False, -1

def add_pod_info_to_crd(api, crd_api, pod_name, namespace, ann, node_name):
    """1)get pod's owner by ownereference, 2)patch pod info to owner's annotation for persistent record, 
    sinces pods often got deleted after *job is done
    assume owner is crd, not native k8s object for now, since different api group will be used
    save the max utilization only for now, only patch if current util is greater than the existing annotations
    """
    # 1) get owner reference 
    owner_kind, owner_name, owner_group, owner_version = "","","",""
    try:
        ret = api.list_namespaced_pod(namespace)
        for i in ret.items:
            if i.metadata.name == pod_name:
                ref = i.metadata.owner_references
                # get the first owner and print kind and name
                if ref is not None and len(ref) > 0:
                    owner_kind = ref[0].kind.lower()
                    owner_name = ref[0].name.lower()
                    owner_group, owner_version = ref[0].api_version.lower().split("/")
                    break
    except Exception as e:
        logging.error(e)
        return False
    if owner_name=="" or owner_kind=="" or owner_group=="" or owner_version=="":
        logging.warning("In {} namespace pod {}'s owner reference is not set, add no annotations to owner".format(namespace, pod_name))
        return False
    # 2) get owner's current annonation, update if greater utilization is found
    try:
        res = crd_api.get_namespaced_custom_object(owner_group,owner_version,namespace,
        plural=owner_kind+'s',name=owner_name)
    except Exception as e:
        logging.error("Error: no kind: {} named {}".format(owner_kind, owner_name))
        return False
    # ann example ai.centaurus.io/pod_name:{mem_max:XXX,cpu_max:XXX}
    key = DOMAIN + "/" + pod_name
    need_patch = True
    pod_ann = dict()
    if key in res['metadata']['annotations']: # iteration and compare, 
        need_patch = False
        pod_ann = ast.literal_eval(res['metadata']['annotations'][key]) # convert string to dictionary
        for k, v in pod_ann.items(): 
            domain_k = DOMAIN + "/" + k  # the key in owner's annotation has no domain name
            if k == "node":  # skip the node comparison
                continue
            if float(v) < ann[domain_k]:  # detect greater utilization, update
                pod_ann[k] = ann[domain_k]
                need_patch = True
    else: # simply remove the domain name from new ann
        pod_ann = ann
        ann['node'] = node_name
    # patch the info
    if need_patch:
        crd_ann = dict()
        crd_ann[key] = str(pod_ann).replace(DOMAIN+"/","")
        body = {'metadata':{'annotations':crd_ann}}
        res = crd_api.patch_namespaced_custom_object(owner_group,owner_version,namespace,
        plural=owner_kind+'s',name=owner_name,body=body)
        logging.info("patch crd utilization done {}: {}".format(owner_name,res['metadata']['annotations'][key]))
    return True

def patch_annotation(api, name, ann, namespace="", node_name="", crd_api=None):
    """check if object exists first"""
    ann2 = copy.deepcopy(ann) # copy and then make changes
    # reformat dictionary, in case it is nested, flatten to one level by cast nested dict to string
    for k, v in ann2.items():
        if v is not None:  # if v is None, it means delete the existing annotaion
            ann2[k]=str(v)

    body = {'metadata': {'annotations':ann2}}

    if namespace == "": # patch node
        api.patch_node(name, body)
    else: # patch pod, verify pod existence first
        pod_exist = False
        pods = api.list_namespaced_pod(namespace)
        for i in pods.items:
            if i.metadata.name == name:
                pod_exist = True
                break
        if pod_exist:
            api.patch_namespaced_pod(name, namespace, body)
            # patch pod info to owner custom resources, assume owner is CRD for now, dont handle native object like, job, statefulSet ...
            if crd_api is not None:
                add_pod_info_to_crd(api, crd_api, name, namespace, ann, node_name)
    return True

def collect_gpu_usage_nvml(gpu_idx):
    cur_usage = dict()
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_idx)
    except Exception as e:
        logging.error(e)
        return cur_usage
    cur_usage['mem_used'] =str(math.ceil(nvmlDeviceGetMemoryInfo(handle).used/pow(1024,3))) + 'GB'
    cur_usage['mem_free'] =str(math.ceil(nvmlDeviceGetMemoryInfo(handle).total/pow(1024,3)) - math.ceil(nvmlDeviceGetMemoryInfo(handle).used/pow(1024,3))) + 'GB'
    processes = nvmlDeviceGetComputeRunningProcesses(handle)
    cur_usage['process_cnt'] = len(processes)
    if len(processes) > 0:
        cur_usage['pid-mem'] = [(i.pid, str(math.ceil(i.usedGpuMemory/pow(1024,2)))+'MB') for i in processes]
        #cur_usage['pid-mem-gutil'] =[(i.pid, i.usedGpuMemory) for i in processes]
    return cur_usage


def remove_annotation(api, node_name, pod_name="", ns=""):
    # 1) get pod name and namespace on the node, scan all the keys in annoations, if start with ai.centaurus.io, set to none
    field_selector = 'spec.nodeName='+node_name
    ret = api.list_pod_for_all_namespaces(watch=False, field_selector=field_selector)       
    for i in ret.items:
        if pod_name != "" and (i.metadata.name != pod_name or i.metadata.namespace != ns):
            continue
        if i.metadata.annotations is not None:
            ann_rm = dict()
            for key in i.metadata.annotations:
                if key.startswith("ai.centaurus.io"): # add the removal dict
                    ann_rm[key] = None
            if len(ann_rm) > 0:
                patch_annotation(api, i.metadata.name, ann_rm, i.metadata.namespace) 
                logging.info("Init reset pod {}'s annotation, remove {}".format(i.metadata.name,ann_rm.keys()))

def collect_pod_metrics(api, cur_usage, node_name, gpu_id, pods_ann):
    """
    pods_ann: dict()
    key: dlt-job:default  #podname:namespace
    value (annotations): {"ai.centaurusinfra.io/gpu-memused":[(0,137MB),(1,1125MB)]}, #tuple (gpu_id, memused) 
    """
    # 1) get pod name and namespace on the node
    field_selector = 'spec.nodeName='+node_name
    ret = api.list_pod_for_all_namespaces(watch=False, field_selector=field_selector)
    pod_ns = dict()
    for i in ret.items:
        pod_ns[i.metadata.name] = i.metadata.namespace
    # 2) get pod name by pid
    bashCmd = ["nsenter", "--target", "XXX", "--uts", "hostname"]
    for pid, mem_used in cur_usage['pid-mem']:
        mem_used_float = float(mem_used[:-2])
        bashCmd[2]=str(pid)
        subp = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
        output, error = subp.communicate()
        if error is None:
            pod_name = output.decode("utf-8").rstrip()
            if pod_name in pod_ns:
                # 3) format results to dict, key: "podname:namespace:gpu_id", value:"{DOMAIN + "/gpu-memused":memused}"
                key = pod_name + ":" + pod_ns[pod_name]
                # list format for gpu mem usage
                #e.g. ai.centaurus.io/gpu-memused:[('0','12000MB'),('1','159MB')]
                if key in pods_ann:
                    pods_ann[key][DOMAIN + "/" +node_name + "-gpu-" +str(gpu_id)+"_mem_mb"] = mem_used_float 
                    pods_ann[key][DOMAIN + "/" +node_name + "-gpu-" +str(gpu_id)+"_util"] = cur_usage['max_gpu_util']
                else:
                    value = {DOMAIN + "/" +node_name +"-gpu-" +str(gpu_id)+"_util":cur_usage['max_gpu_util'],
                    DOMAIN + "/" +node_name +"-gpu-" +str(gpu_id)+"_mem_mb":mem_used_float
                    }
                    pods_ann[key] = value
            else:
                logging.error("pod name {} is not in listed all pods,{}".format(pod_name, pod_ns.keys)) # there was a podname key="" incident, not reproduced 
        else:
            logging.error("nsenter failed to acquire pod name,{}".format(error))
    return pods_ann

def get_pod_resource_util(pod_name, ns, promi_connector, duration="30s"):
    """use query to get resource utilization"""
    cpu_usage_value, memory_usage_value, network_usage_value, io_usage_value = 0,0,0,0

    cpu_usage = promi_connector.custom_query(query="sum(rate(container_cpu_usage_seconds_total{container_label_io_kubernetes_pod_name=\"" + pod_name + "\", container_label_io_kubernetes_pod_namespace=\"" + ns + "\"}[" + duration + "]))by(container_label_io_kubernetes_pod_name)")
    if len(cpu_usage) > 0:
        cpu_usage_value = cpu_usage[0]["value"][1]
    
    memory_usage = promi_connector.custom_query(query="sum(rate(container_memory_usage_bytes{container_label_io_kubernetes_pod_name=\"" + pod_name + "\", container_label_io_kubernetes_pod_namespace=\"" + ns + "\"}[" + duration + "]))by(container_label_io_kubernetes_pod_name)")
    if len(memory_usage) > 0:
        memory_usage_value = memory_usage[0]["value"][1]
    
    network_usage = promi_connector.custom_query(query="sum(rate(container_network_transmit_bytes_total{container_label_io_kubernetes_pod_name=\"" + pod_name + "\", container_label_io_kubernetes_pod_namespace=\"" + ns + "\"}[" + duration + "]))by(container_label_io_kubernetes_pod_name)")
    if len(network_usage) > 0:
        network_usage_value = network_usage[0]["value"][1]

    io_usage = promi_connector.custom_query(query="sum(rate(container_fs_write_seconds_total{container_label_io_kubernetes_pod_name=\"" + pod_name + "\", container_label_io_kubernetes_pod_namespace=\"" + ns + "\"}[" + duration + "]))by(container_label_io_kubernetes_pod_name)")
    if len(io_usage) > 0:
        io_usage_value = io_usage[0]["value"][1]

    return cpu_usage_value, memory_usage_value, network_usage_value, io_usage_value 

def profiling(api, url, pod_ip, node_name, ana_window='2m', metrics=MEM_UTIL):
    """if key exists, the value will be replaced,
       add dynamic status
       {ai.centaurus.io/gpu0:{cur_mem_used:4GB, max_gpu_util:60, max_mem_cpy_util:34, cyclic:True, process_cnt:1},
        ai.centaurus.io/gpu1:{cur_mem_used:4GB, max_gpu_util:60, max_mem_cpy_util:34, cyclic:True, process_cnt:2, processes:[{pid:25678, cur_mem_used:3GB},{pid:67234, cur_mem_used:1GB}]}                                 
       }
    """
    node_dict = dict()
    pod_dict = dict()
    promi = PrometheusConnect(url=url, disable_ssl=True)
    # except connection error
    try:
        promi.check_prometheus_connection()
    except Exception as e:
        logging.error(e)
        return node_dict, pod_dict  # if connectioin fails, return empty dict
    instance = pod_ip + ":9400" # tmp fixed
    start_time = parse_datetime(ana_window)
    end_time = parse_datetime("now")
    my_label_config = {"instance": instance}  # select current host metrics
    metric_data = promi.get_metric_range_data(metric_name=metrics,
                                              label_config=my_label_config,
                                              start_time=start_time,
                                              end_time=end_time)
    # reorganize data to label_config and metric_values
    metric_object_list = MetricsList(metric_data)
    for item in metric_object_list: # iterate through all the gpus on the node
        if 'gpu' not in item.label_config: # handle metric config info exception
            continue
        id = item.label_config['gpu']  # predefined key from dcgm (gpu index)
        cur_usage = collect_gpu_usage_nvml(int(id)) # nvml access GPU usage
        # ip = item.label_config['instance']
        key = DOMAIN + "/gpu-" + id
        # analyze mem util curve
        ts = item.metric_values.iloc[:, 1]  # metrics_values are two row df, 1st is timestamp, 2nd is value
        cur_usage['cyclic_pattern'] = False
        if ts.max() > 0:
            cyclic, period = cyclic_pattern_detection(ts)
            if cyclic:
                cur_usage['cyclic_pattern'] = True
                cur_usage['period'] = str(period)       
        cur_usage['max_mem_util'] = ts.max()
        # add gpu id to query condition, query again get the max_gpu_util
        my_label_config['gpu'] = id
        gpu_util_data = promi.get_metric_range_data(metric_name=GPU_UTIL,
                                              label_config=my_label_config,
                                              start_time=start_time,
                                              end_time=end_time)
        gpu_util_list = MetricsList(gpu_util_data)
        if len(gpu_util_list) != 1:
            logging.error("gpu util data read error, expect len {}, not equal to 1".format(len(gpu_util_list)))
        else:
            gpu_util_ts = gpu_util_list[0].metric_values.iloc[:, 1]
            cur_usage['max_gpu_util'] = gpu_util_ts.max()
        # Important: flatten nested dictionary to string, otherwise error "cannot unmarshal string into Go value of type map[string]interface {}""
        #node_dict[key] = str(cur_usage)
        # move the string cast to patch_annotation function
        node_dict[key] = cur_usage
        if "process_cnt" in cur_usage and cur_usage['process_cnt'] > 0:
            collect_pod_metrics(api, cur_usage, node_name, id, pod_dict) # a pod may use multiple GPUs, so the dictionary value is appended
    # add cadvisor metrics to pod
    for k, v in pod_dict.items():
        pod_name, ns = k.split(":")
        cpu, memory, network, io = get_pod_resource_util(pod_name,ns, promi) # the values are str type
        # v[DOMAIN + '/cpu_util'] = str(round(float(cpu)*100,2)) + '%'
        # v[DOMAIN + '/cpu_mem'] = str(round(float(memory)/1e6,2)) + 'MB'
        # v[DOMAIN + '/network'] = str(round(float(network)/1e3,2)) + 'KBps'
        # v[DOMAIN + '/disk_io'] = io
        v[DOMAIN + '/cpu_util'] = round(float(cpu)*100,2) # unit percentage
        v[DOMAIN + '/cpu_mem_mb'] = round(float(memory)/1e6,2) # unit MB
        v[DOMAIN + '/network_mbps'] = round(float(network)/1e6,2) # unit MBps
        v[DOMAIN + '/disk_io'] = round(float(io),2) 
    return node_dict, pod_dict

def load_env_var():
    env_var = dict()
    if 'KUBERNETES_PORT' not in os.environ:
        logging.error("RUNNING cluster is not avaliable")
        return env_var, True
    if "PROMETHEUS_SERVICE_HOST" in os.environ and "PROMETHEUS_SERVICE_PORT" in os.environ:
        # use service name instead of IP, to avoid IP changes during service restart
        url = "http://prometheus:" + os.environ['PROMETHEUS_SERVICE_PORT']
        env_var['url'] = url
    else:
        logging.error("PROMETHEUS_SERVICE_HOST cannot be found in environment variable, "
                      "Please make sure service is launched before profiler deployment")
        return env_var, True
    if "MY_POD_IP" in os.environ:
        env_var['pod_ip'] = os.environ['MY_POD_IP']
    else:
        logging.error("MY_POD_IP cannot be found in environment variables, "
                      "Please check profiler deployment file to include it as env.")
        return env_var, True
    if "MY_HOST_IP" in os.environ:
        env_var['host_ip'] = os.environ['MY_HOST_IP']
    else:
        logging.error("MY_HOST_IP cannot be found in environment variables, "
                      "Please check profiler deployment file to include it as env.")
        return env_var, True
    if "MY_NODE_NAME" in os.environ:
        env_var['node_name'] = os.environ['MY_NODE_NAME']
    else:
        logging.error("MY_HOST_NAME cannot be found in environment variables, "
                      "Please check profiler deployment file to include it as env.")
        return env_var, True
    return env_var, False



def collect_gpu_attributes():
    """if key exists, the value will be replaced
       {ai.centaurus.io/gpu-static:{count:2,
                                    gpus:[{index:0, pcie_bus_id:0000:84:00.0, model:TITANX, mem_size: 12GB, pcie_gen_width:1X16},
                                          {index:1, pcie_bus_id:0000:88:00.0, model:TITANX, mem_size: 12GB, pcie_gen_width:1X16}
                                         ]
                                    } 
        }
    """
    attributes = dict()
    try:
        nvmlInit()
    except Exception as e:
        logging.error(e)
        return attributes, True
    deviceCount = nvmlDeviceGetCount()
    attributes['count']=str(deviceCount)
    # only get gpu0's attributes, assume same GPU card on one server
    try: 
        handle = nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        logging.error(e) 
        return attributes, True
    attributes['model'] = nvmlDeviceGetName(handle).decode("utf-8")
    attributes['mem_size'] = str(math.ceil(nvmlDeviceGetMemoryInfo(handle).total/pow(1024,3))) + 'GB'
    attributes['pcie_gen_width'] = str(nvmlDeviceGetCurrPcieLinkGeneration(handle)) + 'x' + str(nvmlDeviceGetCurrPcieLinkWidth(handle))
    key = DOMAIN + "/gpu-static" 
    annotation = {key:attributes}
    
    return annotation, False

def app_top():
    env_var, err = load_env_var()
    if err:
        logging.error("Not all environment variables are avaliable in the profiler pod")
        exit(1)
    
    # 0) load kubernetes configure & connect to MongoDB
    config.load_incluster_config() 
    core_api = client.CoreV1Api()
    crd_api = client.CustomObjectsApi()
    batch_api = client.BatchV1Api()
    CONNECTION_STRING = "mongodb://mongo:27017/"
    client_connect = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS = 5000)

    # 1) init stage, get gpu static attribute, remove pod annotation from previous run
    gpu_attributes, err = collect_gpu_attributes() 
    while err:
        logging.warning("NVML lib is not installed or No GPUs avaliable, or GPU lost, check back after 30 sec")
        time.sleep(30)
        gpu_attributes, err = collect_gpu_attributes() 
    patch_annotation(core_api, env_var['node_name'], gpu_attributes)
    logging.info("Init add gpu static attributes \n{}".format(gpu_attributes))                 
    # remove pod annotations from ai.centaurus.io,
    remove_annotation(core_api, env_var['node_name'])

    # 3) infinit loop to monitor resource utilization and annonates to node, pod, and crds 
    # keep current annotatations, if no changes, no patch sent               
    node_ann_cur = dict()
    pods_ann_cur = dict() 
    while True:
        # profiling, add gpu dynamic status
        node_ann_new, pods_ann_new = profiling(core_api, env_var['url'], env_var['pod_ip'],env_var['node_name'])
        # update node annotation if changes detected
        if node_ann_new != node_ann_cur:
            patch_annotation(core_api, env_var['node_name'], node_ann_new)
            logging.info("Node change detected, update node's GPU utilization")
            node_ann_cur = node_ann_new
        # update pod annotation
        if pods_ann_new != pods_ann_cur:
            logging.info("Pod change deteacted, update pods GPU utilization")
            for name_ns, values in pods_ann_new.items(): # iterate all the pods needs to be annotated
                pod_name, namespace = name_ns.split(":")
                patch_annotation(core_api, pod_name, values, namespace, env_var['node_name'], crd_api) # patch pod and patch owner crd
            for name_ns, values in pods_ann_cur.items():
                if name_ns not in pods_ann_new: # ended pods or processes
                    pod_name, namespace = name_ns.split(":")
                    logging.info("Remove pod {} annotation for finished process \n".format(pod_name))
                    remove_annotation(core_api,env_var['node_name'],pod_name, namespace)
            pods_ann_cur = pods_ann_new

        ## Store job metrics into MongoDB
        update_job_metrics_to_db(crd_api, batch_api, CONNECTION_STRING, client_connect, env_var['url'])
        
        time.sleep(30)
    
    


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',level=logging.INFO)
    app_top()
