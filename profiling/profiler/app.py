from prometheus_api_client import PrometheusConnect, MetricsList
from prometheus_api_client.utils import parse_datetime
import pandas as pd
import os
import time
import logging
from kubernetes import config, client
from scipy.stats import norm
import numpy as np

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

def patch_annotation(node_name, ann):
    if 'KUBERNETES_PORT' in os.environ:
        config.load_incluster_config()
    else:
        logging.error("RUNNING cluster is not avaliable")
        exit(1)
    body = {'metadata': {'annotations':ann}}

    v1 = client.CoreV1Api()
    logging.info("Initally add gpu static attributes \n{}".format(body))
    v1.patch_node(node_name, body)
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

def profiling(url, pod_ip, ana_window='2m', metrics=MEM_UTIL):
    """if key exists, the value will be replaced,
       add dynamic status
       {ai.centaurus.io/gpu0:{cur_mem_used:4GB, max_gpu_util:60, max_mem_cpy_util:34, cyclic:True, process_cnt:1},
        ai.centaurus.io/gpu1:{cur_mem_used:4GB, max_gpu_util:60, max_mem_cpy_util:34, cyclic:True, process_cnt:2, processes:[{pid:25678, cur_mem_used:3GB},{pid:67234, cur_mem_used:1GB}]}                                 
       }
    """
    ret_dict = dict()
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
    ret_dict = dict()
    for item in metric_object_list: # iterate through all the gpus on the node
        if 'gpu' not in item.label_config: # handle metric config info exception
            continue
        id = item.label_config['gpu']  # predefined key from dcgm (gpu index)
        # ip = item.label_config['instance']
        key = DOMAIN + "/gpu-" + id
        cur_usage = collect_cur_usage(int(id))
        ts = item.metric_values.iloc[:, 1]  # metrics_values are two row df, 1st is timestamp, 2nd is value
        cur_usage['cyclic_pattern'] = False
        if ts.max() > 0:
            cyclic, period = cyclic_pattern_detection(ts)
            if cyclic:
                cur_usage['cyclic_pattern'] = True
                cur_usage['period'] = str(period)       
        cur_usage['max_mem_util'] = str(ts.max())
        # Important: flatten nested dictionary to string, otherwise error "cannot unmarshal string into Go value of type map[string]interface {}""
        ret_dict[key] = str(cur_usage)
    return ret_dict

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
    annotation = {key:str(attributes)}
    
    return annotation

def app_top():
    current_annotation = dict()
    logging.info("profiler initialization")
    config_dict = load_config()
    # add gpu static attributes
    gpu_attributes = collect_gpu_attributes()
    patch_annotation(config_dict['node_name'], gpu_attributes)
    while True:
        # profiling, add gpu dynamic status
        new_annotation = profiling(url=config_dict['url'], pod_ip=config_dict['pod_ip'])
        # update annotation if changes detected
        if new_annotation != current_annotation:
            patch_annotation(config_dict['node_name'], new_annotation)
            current_annotation = new_annotation
        time.sleep(30)


def delete_annotations():
    """assign key value to None for deletion, not in use in the main flow
    """
    #key = "GPU-0-DCGM_FI_DEV_MEM_COPY_UTIL"
    delete_dict = dict()
    node_name = ""
    for m in ["job-type"]:
        for i in range(8):
            key = DOMAIN +"/GPU-" + str(i) + "-" + m
            logging.debug(key)
            delete_dict[key] = None 
    if 'KUBERNETES_PORT' in os.environ:
        config.load_incluster_config()
    else:
        logging.error("RUNNING cluster is not avaliable")
        exit(1)
    if "MY_NODE_NAME" in os.environ:
        node_name = os.environ['MY_NODE_NAME']
    else:
        logging.error("MY_HOST_NAME cannot be found in environment variables, Please check profiler deployment file to include it as env.")
        exit(1)
    # assign key value to None for deletion
    # body = {'metadata':{'annotations':{key:None}}}
    body = {'metadata':{'annotations': delete_dict}}
    v1 = client.CoreV1Api()
    ret = v1.patch_node(node_name,body)
    logging.info("delete node annotation {}, return {}, return type {}".format(key, ret, type(ret)))
    return True


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',level=logging.INFO)
    #delete_annotations()
    app_top()
