from prometheus_api_client.utils import parse_datetime
from pymongo import MongoClient
from datetime import datetime
import pytz


def get_pod_records(pod_name, start_time, end_time, url):
    cpu_info, memory_info, network_info, io_info = [],[],[],[]
    all_metrics = {}

    my_label_config = {"container_label_io_kubernetes_pod_name": pod_name}
    cpu_metric_data = prom.get_metric_range_data(metric_name="container_cpu_usage_seconds_total", label_config=my_label_config, start_time=parse_datetime(start_time), end_time=parse_datetime(end_time))
    memory_metric_data = prom.get_metric_range_data(metric_name="container_memory_usage_bytes", label_config=my_label_config, start_time=parse_datetime(start_time), end_time=parse_datetime(end_time))
    network_metric_data = prom.get_metric_range_data(metric_name="container_network_transmit_bytes_total", label_config=my_label_config, start_time=parse_datetime(start_time), end_time=parse_datetime(end_time))
    io_metric_data = prom.get_metric_range_data(metric_name="container_fs_write_seconds_total", label_config=my_label_config, start_time=parse_datetime(start_time), end_time=parse_datetime(end_time))

    if (len(cpu_metric_data) != 0):
        cpu_sum_list = []
        min_cpu_metric_len = float('inf')
        for i in range(len(cpu_metric_data)):
            min_cpu_metric_len = min(min_cpu_metric_len, len(cpu_metric_data[i]["values"]))
        for i in range(0, min_cpu_metric_len):
            sum_at_timestamp = 0
            for j in range(len(cpu_metric_data)):
                sum_at_timestamp += float(cpu_metric_data[j]["values"][i][1])
            cpu_sum_list.append(sum_at_timestamp)
        cpu_diff = []
        for i in range(1, len(cpu_sum_list)):
            diff = (cpu_sum_list[i] - (cpu_sum_list[i-1])) / (float(cpu_metric_data[0]["values"][i][0]) - float(cpu_metric_data[0]["values"][i-1][0]))
            cpu_diff.append(diff / len(cpu_metric_data))
        if (len(cpu_diff) < 9):
            cpu_info = cpu_diff.copy()
        else :
            for i in range (0, len(cpu_diff), (int) (len(cpu_diff) / 9)):
                cpu_info.append(cpu_diff[i])
            cpu_info.append(cpu_diff[len(cpu_diff) - 1])
        all_metrics["cpu_usage"] = cpu_info

    if (len(memory_metric_data) != 0):
        memory_sum_list = []
        min_memory_metric_len = float('inf')
        for i in range(len(memory_metric_data)):
            min_memory_metric_len = min(min_memory_metric_len, len(memory_metric_data[i]["values"]))
        for i in range(0, min_memory_metric_len):
            sum_at_timestamp = 0
            for j in range(len(memory_metric_data)):
                sum_at_timestamp += float(memory_metric_data[j]["values"][i][1])
            memory_sum_list.append(sum_at_timestamp)
        memory_sum_list.sort()
        if (len(memory_sum_list) < 9) :
            memory_info = memory_sum_list.copy()
        else :
            for i in range (0, len(memory_sum_list), (int) (len(memory_sum_list) / 9)):
                memory_info.append(memory_sum_list[i])
            memory_info.append(memory_sum_list[len(memory_sum_list) - 1])
        all_metrics["memory_usage"] = memory_info

    if (len(network_metric_data) != 0): 
        network_data = network_metric_data[0]["values"]
        if (len(network_data) < 9) :
            network_info = network_data.copy()
        else :
            for j in range (0, len(network_data), (int) (len(network_data) / 9)):
                network_info.append(network_data[j][1])
            network_info.append(network_data[len(network_data) - 1][1])
        all_metrics["network_usage"] = network_info
    
    if (len(io_metric_data) != 0):
        for i in range(len(io_metric_data)):
            io_data = io_metric_data[i]["values"]
            if (len(io_info) < 9) :
                io_info = io_info.copy()
            else :  
                for j in range (0, len(io_data), (int) (len(io_data) / 9)):
                    io_info.append(io_data[j][1])
                io_info.append(io_data[len(io_data) - 1][1])
        all_metrics["io_usage"] = io_info

    return all_metrics


def update_job_metrics_to_db(crd_api, batch_api, url, client_connect) :
    # Force a call to check if current connection is valid
    try:
        info = client_connect.server_info()
    except Exception as e:
        logging.warning(e)
        client_connect = MongoClient(url, serverSelectionTimeoutMS = 5000)

    db = client_connect['alnair']
    col = db['collection1']

    # Get Mpijob metrics
    ret = crd_api.list_cluster_custom_object(group="kubeflow.org", version="v1", plural="mpijobs")
    for item in ret["items"]:
        all_metrics = {}
        job_metric = {}

        job_name = item["metadata"]["name"]

        # Check job's start_time
        start_time = ""
        if (item["metadata"]["creationTimestamp"] is not None):
            start_time = item["metadata"]["creationTimestamp"]
            job_metric["start_time"] = start_time
        
        # Use [job_name + "_" + start_time] as the unique identifier for a job
        job_key = job_name + "_" + start_time

        job_kind = item["kind"]
        job_metric["kind"] = job_kind

        # Check if job_key currently exists in the database collection
        # If True, delete the existing job_key data, and update with new data
        try:
            if (col.count_documents({job_key: {"$exists": True}}) > 0):
                col.delete_one({job_key : {"$exists": True}})
        except Exception as e:
            logging.warning(e)


        # Check status, completion time and duration
        failed = False
        for condition in item["status"]["conditions"]:
            if (condition["type"] == "Failed") :
                job_metric["status"] = "Failed"
                failed = True
        if not failed:
            if ("completionTime" not in item["status"]):
                job_metric["status"] = "Running" 

                duration = datetime.now(tz=pytz.utc) - parse_datetime(start_time)

                days, seconds = duration.days, duration.seconds
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                seconds = seconds % 60 
                job_metric["duration"] = str(days) + " days, " + str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds"
            else:
                job_metric["status"] = "Completed"
                completion_time = item["status"]["completionTime"]
                job_metric["completion_time"] = completion_time

                duration = parse_datetime(completion_time) - parse_datetime(start_time)
                        
                days, seconds = duration.days, duration.seconds
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                seconds = seconds % 60
                job_metric["duration"] = str(days) + " days, " + str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds"
                            
        # Get pod metrics
        pod_metrics = {}
        if (item["metadata"]["annotations"] is not None):
            keys = item["metadata"]["annotations"].keys()
            for key in keys:
                if (key[:15] == "ai.centaurus.io"):
                    pod_name = key[16:]
                    # If job has not completed, then retrieve pod metrics from start_time to current timestamp;
                    # Otherwise, retrieve pod metrics from start_time to complete_time
                    try:
                        if ("completionTime" not in item["status"]):
                            pod_metric = get_pod_records(pod_name, start_time, datetime.now(tz=pytz.utc), url=PROM_SERVER_URL)
                            pod_metrics[pod_name] = pod_metric
                        else:
                            completion_time = item["status"]["completionTime"]
                            pod_metric = get_pod_records(pod_name, start_time, completion_time, url=PROM_SERVER_URL)
                            pod_metrics[pod_name] = pod_metric
                    except Exception as e:
                        print("Failed to get pod metrics")

            job_metric["pod_count"] = len(pod_metrics)
            job_metric["pod_metrics"] = pod_metrics

        # Store [job_key, job_metric] into database collection
        all_metrics[job_key] = job_metric

        try:
            x = col.insert_one(all_metrics)
        except Exception as e:
            logging.warning(e)
        

    # Get Batch job metrics
    ret2 = batch_api.list_job_for_all_namespaces(watch=False)
    for item in ret2.items:
        all_metrics = {}
        job_metric = {}

        job_name = format(item.metadata.name)

        # Check job's start_time
        start_time = ""
        if (item.metadata.creation_timestamp is not None):
            start_time = format(item.metadata.creation_timestamp)
            job_metric["start_time"] = start_time

        # Use [job_name + "_" + start_time] as the unique identifier for a job
        job_key = job_name + "_" + start_time

        job_kind = format(item.kind)
        job_metric["kind"] = job_kind

        # Check if job_key currently exists in the database collection
        # If True, delete the existing job_key data, and update with new data
        try:
            if (col.count_documents({job_key: {"$exists": True}}) > 0):
                col.delete_one({job_key : {"$exists": True}})
        except Exception as e:
            logging.warning(e)

        # Check status, completion time and duration
        failed = False
        for condition in item.status.conditions:
            if (condition.type == "Failed"):
                job_metric["status"] = "Failed"
                failed = True
        if not failed:
            if (item.status.completion_time is None):
                job_metric["status"] = "Running"

                duration = datetime.now(tz=pytz.utc) - parse_datetime(start_time)

                days, seconds = duration.days, duration.seconds
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                seconds = seconds % 60 
                job_metric["duration"] = str(days) + " days, " + str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds"
            else :
                job_metric["status"] = "Completed"
                completion_time = format(item.status.completion_time)
                job_metric["completion_time"] = completion_time

                duration = parse_datetime(completion_time) - parse_datetime(start_time)

                days, seconds = duration.days, duration.seconds
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                seconds = seconds % 60 
                job_metric["duration"] = str(days) + " days, " + str(hours) + " hours, " + str(minutes) + " minutes, " + str(seconds) + " seconds"
            
        # Get pod metrics
        pod_metrics = {}
        if (item.metadata.annotations is not None):
            keys = item.metadata.annotations.keys()
            item_info = {}
            for key in keys:
                if (key[:15] == "ai.centaurus.io"):
                    pod_name = key[16:]
                    # If job has not completed, then retrieve pod metrics from start_time to current timestamp;
                    # Otherwise, retrieve pod metrics from start_time to complete_time
                    try:
                        if (item.status.completion_time is None):
                            pod_metric = get_pod_records(pod_name, start_time, datetime.now(tz=pytz.utc), url=PROM_SERVER_URL)
                            pod_metrics[pod_name] = pod_metric
                        else:
                            completion_time = item.status.completion_time
                            pod_metric = get_pod_records(pod_name, start_time, completion_time, url=PROM_SERVER_URL)
                            pod_metrics[pod_name] = pod_metric
                    except Exception as e:
                        print("Failed to get pod metrics")

            job_metric["pod_count"] = len(pod_metrics)
            job_metric["pod_metrics"] = pod_metrics

        # Store [job_key, job_metric] into database collection
        all_metrics[job_key] = job_metric

        try:
            x = col.insert_one(all_metrics)
        except Exception as e:
            logging.warning(e)


