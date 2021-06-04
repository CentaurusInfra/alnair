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
ANN1 = "job-type"
ANN2 = "mem-util-max"

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


def update_annotation(node_name, current_ann, new_ann):
    """
    {annotations:{GPU-0:}}
    """
    if 'KUBERNETES_PORT' in os.environ:
        config.load_incluster_config()
    else:
        logging.error("RUNNING cluster is not avaliable")
        exit(1)
    body = {'metadata': {'annotations': new_ann}}
    v1 = client.CoreV1Api()
    logging.info("Usage change detected, update node annotation to \n{}".format(body))
    v1.patch_node(node_name, body)
    return True


def profiling(url, pod_ip, ana_window='2m', metrics=MEM_UTIL):
    ret_dict = dict()
    promi = PrometheusConnect(url=url, disable_ssl=True)
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
        id = item.label_config['gpu']  # predefined key from dcgm
        # ip = item.label_config['instance']
        key1 = DOMAIN + "/" + "-".join(["GPU", str(id), ANN1])
        key2 = DOMAIN + "/" + "-".join(["GPU", str(id), ANN2])
        ts = item.metric_values.iloc[:, 1]  # metrics_values are two row df, 1st is timestamp, 2nd is value
        if ts.max() ==0:
            job_type = "Empty"
        else:
            cyclic, _ = cyclic_pattern_detection(ts)
            job_type = "DLT" if cyclic else "Others"
        ret_dict[key1] = job_type
        if job_type == "DLT":  # add max utilization
            ret_dict[key2] = str(ts.max())
        else:
            ret_dict[key2] = None  # remove previous annotation if there is any
        logging.debug("{}, job type {}, max usage {}".format(key1, job_type, ts.max()))
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


def app_top():
    current_annotation = dict()
    logging.info("profiler initialization")
    while True:
        # load configuration, logging if config changed
        config_dict = load_config()
        # profiling
        new_annotation = profiling(url=config_dict['url'], pod_ip=config_dict['pod_ip'])
        # update annotation if changes detected
        if new_annotation != current_annotation:
            update_annotation(config_dict['node_name'], current_annotation, new_annotation)
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
