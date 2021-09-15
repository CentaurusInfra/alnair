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

MEM_UTIL = "DCGM_FI_DEV_MEM_COPY_UTIL"
GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
DOMAIN = "ai.centaurus.io"
POD_KEY1 = DOMAIN + "/gpu_mem"
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

def patch_annotation(name, ann, namespace=""):
    if 'KUBERNETES_PORT' in os.environ:
        config.load_incluster_config()
    else:
        logging.error("RUNNING cluster is not avaliable")
        exit(1)
    ann = copy.deepcopy(ann) # copy and then make changes
    # reformat dictionary, in case it is nested, flatten to one level by cast nested dict to string
    for k, v in ann.items():
        if v is not None:  # if v is None, it means delete the existing annotaion
            ann[k]=str(v)

    body = {'metadata': {'annotations':ann}}

    v1 = client.CoreV1Api()
    if namespace == "": # patch node
        v1.patch_node(name, body)
    else: # patch pod
        v1.patch_namespaced_pod(name, namespace, body)
    return True

def collect_cur_usage(gpu_idx):
    cur_usage = dict()
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_idx)
    cur_usage['mem_used'] =str(math.ceil(nvmlDeviceGetMemoryInfo(handle).used/pow(1024,3))) + 'GB'
    cur_usage['mem_free'] =str(math.ceil(nvmlDeviceGetMemoryInfo(handle).total/pow(1024,3)) - math.ceil(nvmlDeviceGetMemoryInfo(handle).used/pow(1024,3))) + 'GB'
    processes = nvmlDeviceGetComputeRunningProcesses(handle)
    cur_usage['process_cnt'] = len(processes)
    if len(processes) > 0:
        cur_usage['pid-mem'] = [(i.pid, str(math.ceil(i.usedGpuMemory/pow(1024,2)))+'MB') for i in processes]
    return cur_usage


def init_remove_annotation(node_name, ann_key):
    if 'KUBERNETES_PORT' in os.environ:
        config.load_incluster_config()
    else:
        logging.error("RUNNING cluster is not avaliable")
        exit(1)
    #body = {'metadata': {'annotations':ann}}
    v1 = client.CoreV1Api()
    # 1) get pod name and namespace on the node
    field_selector = 'spec.nodeName='+node_name
    ret = v1.list_pod_for_all_namespaces(watch=False, field_selector=field_selector)
    for i in ret.items:
        if i.metadata.annotations is not None and POD_KEY1 in i.metadata.annotations:
            patch_annotation(i.metadata.name, {POD_KEY1:None}, i.metadata.namespace) 
            logging.info("Init reset pod {}'s annotation{}".format(i.metadata.name,i.metadata.annotations[POD_KEY1]))

def collect_pod_metrics(pid_mem_array, node_name, gpu_id, pods_ann):
    """
    pods_ann: dict()
    key: dlt-job:default  #podname:namespace
    value (annotations): {"ai.centaurusinfra.io/gpu-memused":[(0,137MB),(1,1125MB)]}, #tuple (gpu_id, memused) 
    """
    if 'KUBERNETES_PORT' in os.environ:
        config.load_incluster_config()
    else:
        logging.error("RUNNING cluster is not avaliable")
        exit(1)
    v1 = client.CoreV1Api()
    # 1) get pod name and namespace on the node
    field_selector = 'spec.nodeName='+node_name
    ret = v1.list_pod_for_all_namespaces(watch=False, field_selector=field_selector)
    pod_ns = dict()
    for i in ret.items:
        pod_ns[i.metadata.name] = i.metadata.namespace
    # 2) get pod name by pid
    bashCmd = ["nsenter", "--target", "XXX", "--uts", "hostname"]
    for pid, mem_used in pid_mem_array:
        bashCmd[2]=str(pid)
        subp = subprocess.Popen(bashCmd, stdout=subprocess.PIPE)
        output, error = subp.communicate()
        if error is None:
            pod_name = output.decode("utf-8").rstrip()
            # 3) format results to dict, key: "podname:namespace:gpu_id", value:"{DOMAIN + "/gpu-memused":memused}"
            key = pod_name + ":" + pod_ns[pod_name]
            if key in pods_ann:
                pods_ann[key][POD_KEY1].append((gpu_id,mem_used)) # append other gpu used by this pod
            else:
                value = {POD_KEY1:[(gpu_id,mem_used)]}
                pods_ann[key] = value
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

def profiling(url, pod_ip, node_name, ana_window='2m', metrics=MEM_UTIL):
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
        return ret_dict  # if connectioin fails, return empty dict
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
        cur_usage = collect_cur_usage(int(id)) # nvml access GPU usage
        if cur_usage['process_cnt'] > 0:
            collect_pod_metrics(cur_usage['pid-mem'], node_name, id, pod_dict) # a pod may use multiple GPUs, so the dictionary value is appended
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
        cur_usage['max_mem_util'] = str(ts.max())
        # Important: flatten nested dictionary to string, otherwise error "cannot unmarshal string into Go value of type map[string]interface {}""
        #node_dict[key] = str(cur_usage)
        # move the string cast to patch_annotation function
        node_dict[key] = cur_usage
    # add cadvisor metrics to pod
    for k, v in pod_dict.items():
        pod_name, ns = k.split(":")
        cpu, memory, network, io = get_pod_resource_util(pod_name,ns, promi)
        v[DOMAIN + '/cpu_util'] = str(round(float(cpu)*100,2)) + '%'
        v[DOMAIN + '/cpu_mem'] = str(round(float(memory)/1e6,2)) + 'MB'
        v[DOMAIN + '/network'] = str(round(float(network)/1e3,2)) + 'KBps'
        v[DOMAIN + '/disk_io'] = io
    return node_dict, pod_dict

def load_config():
    config_dict = dict()
    if "PROMETHEUS_SERVICE_HOST" in os.environ and "PROMETHEUS_SERVICE_PORT" in os.environ:
        # use service name instead of IP, to avoid IP changes during service restart
        url = "http://prometheus:" + os.environ['PROMETHEUS_SERVICE_PORT']
        config_dict['url'] = url
    else:
        logging.error("PROMETHEUS_SERVICE_HOST cannot be found in environment variable, "
                      "Please make sure service is launched before profiler deployment")
        exit(1)
    if "MY_POD_IP" in os.environ:
        config_dict['pod_ip'] = os.environ['MY_POD_IP']
    else:
        logging.error("MY_POD_IP cannot be found in environment variables, "
                      "Please check profiler deployment file to include it as env.")
        exit(1)
    if "MY_HOST_IP" in os.environ:
        config_dict['host_ip'] = os.environ['MY_HOST_IP']
    else:
        logging.error("MY_HOST_IP cannot be found in environment variables, "
                      "Please check profiler deployment file to include it as env.")
        exit(1)
    if "MY_NODE_NAME" in os.environ:
        config_dict['node_name'] = os.environ['MY_NODE_NAME']
    else:
        logging.error("MY_HOST_NAME cannot be found in environment variables, "
                      "Please check profiler deployment file to include it as env.")
        exit(1)
    return config_dict

from pynvml import *
import math

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
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    attributes['count']=str(deviceCount)
    # only get gpu0's attributes, assume same GPU card on one server
    handle = nvmlDeviceGetHandleByIndex(0)
    attributes['model'] = nvmlDeviceGetName(handle).decode("utf-8")
    attributes['mem_size'] = str(math.ceil(nvmlDeviceGetMemoryInfo(handle).total/pow(1024,3))) + 'GB'
    attributes['pcie_gen_width'] = str(nvmlDeviceGetCurrPcieLinkGeneration(handle)) + 'x' + str(nvmlDeviceGetCurrPcieLinkWidth(handle))
    key = DOMAIN + "/gpu-static" 
    annotation = {key:attributes}
    
    return annotation

def app_top():
    node_ann_cur = dict()
    pods_ann_cur = dict() # save this for annotation deletion when process ends
    logging.info("Profiler initialization")
    config_dict = load_config()
    # add gpu static attributes
    gpu_attributes = collect_gpu_attributes()
    patch_annotation(config_dict['node_name'], gpu_attributes)
    logging.info("Init add gpu static attributes \n{}".format(gpu_attributes))
    # Init: remove pod annotatio if there is any, POD_KEY1:None
    init_remove_annotation(config_dict['node_name'], POD_KEY1)
    while True:
        # profiling, add gpu dynamic status
        node_ann_new, pods_ann_new = profiling(config_dict['url'], config_dict['pod_ip'],config_dict['node_name'])
        # update node annotation if changes detected
        if node_ann_new != node_ann_cur:
            patch_annotation(config_dict['node_name'], node_ann_new)
            logging.info("Node change detected, update node's GPU utilization \nnew:{}\nold:{}".format(node_ann_new, node_ann_cur))
            node_ann_cur = node_ann_new
        # update pod annotation
        if pods_ann_new != pods_ann_cur:
            logging.info("Pod change deteacted, update pods GPU utilization \nnew:{}\nold:{}".format(pods_ann_new,pods_ann_cur))
            for name_ns, values in pods_ann_new.items(): # iterate all the pods needs to be annotated
                pod_name, namespace = name_ns.split(":")
                patch_annotation(pod_name, values, namespace) 
            for name_ns, values in pods_ann_cur.items():
                if name_ns not in pods_ann_new: # ended pods or processes
                    pod_name, namespace = name_ns.split(":")
                    logging.info("Remove pod {} annotation for finished process \n".format(pod_name))
                    patch_annotation(pod_name, {POD_KEY1:None}, namespace) # tmp use POD_KEY1
            pods_ann_cur = pods_ann_new
        time.sleep(30)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',level=logging.INFO)
    app_top()
