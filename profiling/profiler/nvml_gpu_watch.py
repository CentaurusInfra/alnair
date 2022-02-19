from pynvml import *
import subprocess
from prometheus_api_client import PrometheusConnect, MetricsList
from prometheus_api_client.utils import parse_datetime

def get_podname_by_pid(pid):
    bashcmd = ["nsenter", "--target", pid, "--uts", "hostname"]
    subp = subprocess.Popen(bashcmd, stdout=subprocess.PIPE)
    output, error = subp.communicate()
    if error is None:
        podname = output.decode("utf-8").rstrip()
        return podname
    else:
        return ""
        
def get_pod_gpu_metrics(instance, start_time, end_time, url):
    prom = PrometheusConnect(url=url, disable_ssl=True)

    pod_gpu_metrics = {}
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    pod_gpu_metrics = {} # return dictionary with podname as the key
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        use = nvmlDeviceGetUtilizationRates(handle)
        if use.gpu > 0:
            # prometheus query DCGM_FI_DEV_FB_USED and DCGM_FI_DEV_GPU_UTIL, label {gpu=i, instance=instance}
            # get metrics {gpu_util:[],gmem:[]}
            label ={}
            label["gpu"] = str(i)
            label["instance"] = instance
            gpu_util_data = prom.get_metric_range_data("DCGM_FI_DEV_GPU_UTIL", label_config = label, start_time=parse_datetime(start_time), end_time=parse_datetime(end_time))
            gpu_metrics = {}
            if (len(gpu_util_data) > 0):
                gpu_util = gpu_util_data[0]["values"]
                gpu_util_list = []
                gpu_util_percentile = []
                for j in range(len(gpu_util)):
                    gpu_util_list.append(gpu_util[j][1])
                if (len(gpu_util_list) < 9) :
                    gpu_util_percentile = gpu_util_list.copy()
                    gpu_metrics["gpu_util"] = gpu_util_percentile
                else :
                    len_gpu_util_list = len(gpu_util_list)
                    interval_gpu_util_list = len_gpu_util_list / 9
                    k = 0
                    while k < len_gpu_util_list:
                        gpu_util_percentile.append(gpu_util_list[k])
                        k = (int) (k + interval_gpu_util_list)
                    gpu_util_percentile.append(gpu_util_list[-1])
                    gpu_metrics["gpu_util"] = gpu_util_percentile

            gmem_data = prom.get_metric_range_data("DCGM_FI_DEV_FB_USED", label_config = label, start_time=parse_datetime(start_time), end_time=parse_datetime(end_time))
            if (len(gmem_data) > 0):
                gmem = gmem_data[0]["values"]
                gmem_list = []
                gmem_percentile = []
                for j in range(len(gmem)):
                    gmem_list.append(gmem[j][1])
                if (len(gmem_list) < 9) :
                    gmem_percentile = gmem_list.copy()
                    gpu_metrics["gmem"] = gmem_percentile
                else :
                    len_gmem_list = len(gmem_list)
                    interval_gmem_list = len_gmem_list / 9
                    k = 0
                    while k < len_gmem_list:
                        gmem_percentile.append(gmem_list[k])
                        k = (int) (k + interval_gmem_list)
                    gmem_percentile.append(gmem_list[-1])
                    gpu_metrics["gmem"] = gmem_percentile

            processes = nvmlDeviceGetComputeRunningProcesses(handle)
            if len(processes) > 0:
                for m in processes:
                    pod_name = get_podname_by_pid(str(m.pid))
                    if (pod_name == ""):
                        continue
                    pod_gpu_metrics[pod_name] = gpu_metrics

    return pod_gpu_metrics

