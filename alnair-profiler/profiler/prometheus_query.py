from prometheus_api_client import PrometheusConnect, MetricsList
from prometheus_api_client.utils import parse_datetime
import logging
import datetime
from datetime import timezone

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',level=logging.INFO)
logger = logging.getLogger('promi_query')
logger.setLevel(logging.DEBUG)

def prometheus_connection(url="http://prometheus:9090"):
    promi = PrometheusConnect(url=url, disable_ssl=True)
    # except connection error
    try:
        promi.check_prometheus_connection()
    except Exception as e:
        logging.error(e)
        return None  # if connectioin fails, return empty dict
    return promi

def query_pod_utils(pod_name, ns, duration, url="http://prometheus:9090"):
    promi = prometheus_connection()
    # except connection error
    if promi is None:
        promi = prometheus_connection()  # reconnect once
    if promi is None:
        return {}
    
    duration_str = str(duration.seconds) + "s"
    """use query to get cpu, memory, network, io, resource utilization
        {
        max_cpu_util: {max: xxx, median: xxx, mean: xxx, first_sample_time: xxx, last_sample_time: xxx},
        max_cpu_mem_MB: {max: xxx, median: xxx, mean: xxx, first_sample_time: xxx, last_sample_time: xxx},
        max_network_mbps: {max: xxx, median: xxx, mean: xxx, first_sample_time: xxx, last_sample_time: xxx},
        max_disk_io: {max: xxx, median: xxx, mean: xxx, first_sample_time: xxx, last_sample_time: xxx}
    }
    """
    pod_utils ={}
    max_cpu_util, max_cpu_mem_mb, max_network_mbps, max_disk_io = None, None, None, None
    #sum(rate()) return 1 datapoint, add time period and reolution at the end "sum(rate()) [30m:1s]""
    cpu_util = promi.custom_query(query="sum(rate(container_cpu_usage_seconds_total{container_label_io_kubernetes_pod_name=\"" + pod_name + "\", container_label_io_kubernetes_pod_namespace=\"" + ns + "\"}[5s]))by(container_label_io_kubernetes_pod_name)[" + duration_str + ":1s]")
    if len(cpu_util) > 0:
        cpu_util_ts = [float(i[1]) for i in cpu_util[0]["values"]]
        max_cpu_util = round(max(cpu_util_ts)*100, 2)
        
    cpu_memory_usage = promi.custom_query(query="sum(rate(container_memory_usage_bytes{container_label_io_kubernetes_pod_name=\"" + pod_name + "\", container_label_io_kubernetes_pod_namespace=\"" + ns + "\"}[5s]))by(container_label_io_kubernetes_pod_name)[" + duration_str + ":1s]")
    if len(cpu_memory_usage) > 0:
        cpu_memory_ts = [float(i[1]) for i in cpu_memory_usage[0]["values"]]
        max_cpu_mem_mb = round(max(cpu_memory_ts)/1e6, 2)
    
    network_usage = promi.custom_query(query="sum(rate(container_network_transmit_bytes_total{container_label_io_kubernetes_pod_name=\"" + pod_name + "\", container_label_io_kubernetes_pod_namespace=\"" + ns + "\"}[5s]))by(container_label_io_kubernetes_pod_name)[" + duration_str + ":1s]")
    if len(network_usage) > 0:
        network_usage_ts = [float(i[1]) for i in network_usage[0]["values"]]
        max_network_mbps = round(max(network_usage_ts)/1e6, 2)

    io_usage = promi.custom_query(query="sum(rate(container_fs_write_seconds_total{container_label_io_kubernetes_pod_name=\"" + pod_name + "\", container_label_io_kubernetes_pod_namespace=\"" + ns + "\"}[5s]))by(container_label_io_kubernetes_pod_name)[" + duration_str + ":1s]")
    if len(io_usage) > 0:
        io_usage_ts = [float(i[1]) for i in io_usage[0]["values"]]
        max_disk_io = max(io_usage_ts)
    
    if max_cpu_util is not None:
        pod_utils["max_cpu_util"] = max_cpu_util
    if max_cpu_mem_mb is not None:
        pod_utils["max_cpu_mem_mb"] = max_cpu_mem_mb
    if max_network_mbps is not None:
        pod_utils["max_network_mbps"] = max_network_mbps
    if max_disk_io is not None:
        pod_utils["max_disk_io"] = max_disk_io

    # query the gpu utilization
    gpu_util_dict, gpu_mem_util_dict = query_gpu_utils(promi, pod_name, duration_str)
    if len(gpu_util_dict) > 0:
        pod_utils["gpu_util"] = gpu_util_dict
        pod_utils["gpu_mem_util"] = gpu_mem_util_dict
    
    if len(pod_utils) > 0:
        pod_utils["query_dt"] = datetime.datetime.now(timezone.utc)
    return pod_utils  # if current query (now) is empty, return {}, so previously written results will not be erased/overwrriten


def query_gpu_utils(promi, pod_name, duration_str):
    """return gpu utils in the following format, one pod could use multiple GPUs, or multiple processes on one GPU
    {
        gpu_util:     {gpu_<id>_<pid>: 11,
                       gpu_1_3931783 :0
                       }
        gpu_mem_util: {gpu_0_3931783: 11,
                       gpu_1_3931783 :0
                       }
    }
    Args:
        promi (_type_): _description_
        pod_name (_type_): _description_
        duration_str (_type_): _description_

    Returns:
        dict: _description_
    """
    if promi is None:
        promi = prometheus_connection()  # reconnect once
    if promi is None:
        return {}
    # query the gpu utils
    my_label_config = {"pod_name": pod_name}
    gpu_util_data = promi.get_metric_range_data(metric_name="alnair_gpu_util",
                                                label_config=my_label_config,
                                                start_time=parse_datetime(duration_str),
                                                end_time=parse_datetime("now"))
    gpu_util_dict = gpu_metricslist_parsing(MetricsList(gpu_util_data))
    
    gpu_mem_util_data = promi.get_metric_range_data(metric_name="alnair_gpu_mem_util",
                                                label_config=my_label_config,
                                                start_time=parse_datetime(duration_str),
                                                end_time=parse_datetime("now"))
    gpu_mem_util_dict = gpu_metricslist_parsing(MetricsList(gpu_mem_util_data))
    return gpu_util_dict, gpu_mem_util_dict

def gpu_metricslist_parsing(gpu_metricslist):
    gpu_dict = {}
    if len(gpu_metricslist) > 0:
        for gpu_process in gpu_metricslist:
            gpu_util ={}
            # print(dir(gpu_process))
            # print(gpu_process.label_config)
            # print(gpu_process.metric_values)
            key = "gpu_" + gpu_process.label_config["gpu_idx"] + "_" + gpu_process.label_config["pid"]
            gpu_util_ts = gpu_process.metric_values.iloc[:, 1]
            gpu_dict[key] = max(gpu_util_ts) #util percentage
    return gpu_dict

