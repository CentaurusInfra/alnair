from prometheus_api_client import PrometheusConnect, MetricsList,MetricSnapshotDataFrame
from prometheus_api_client.utils import parse_datetime
import pandas as pd
import datetime as dt
import os

def get_all_metrics(start_time='5m', end_time='now',instance='', gpu_id=''):
    """
    all DCGM metrics, on all instances, and all gpus
    save dumped data to csv file
    """
    # save the time first, in case multiple query at different time later 
    start_time=parse_datetime(start_time)
    end_time = parse_datetime(end_time)
    # connect to premtheus server, exit if connection fails
    url = "http://prometheus:9090" # use service name, instead of ip to be more robust
    prom = PrometheusConnect(url=url, disable_ssl=True)
    try:
        prom.check_prometheus_connection()
    except Exception as e:
        logging.error(e)
        exit(1)
    # get all metrics under profiler job, note: some instances/gpus may not have all the metrics due to model variance
    metrics = prom.all_metrics()
    metrics = [a for a in metrics if 'DCGM' in a]
    gpu_util = 'DCGM_FI_DEV_GPU_UTIL'
    label_cfg = {"job":"profiler-pods"}
    # get a screenshot of all the instances (pod ip)
    metric_data = prom.get_current_metric_value(metric_name = gpu_util,label_config = label_cfg)
    metric_df = MetricSnapshotDataFrame(metric_data)
    instances = metric_df.instance.unique()
    ins_gpu = dict()
    for ins in instances:
        # add instance in query
        label_cfg['instance'] = ins
        metric_data = prom.get_current_metric_value(metric_name= gpu_util, label_config=label_cfg)
        metric_df = MetricSnapshotDataFrame(metric_data)
        gpus = metric_df.gpu.unique()
        # put each instance's gpus into dictionary
        ins_gpu[ins] = gpus

    my_label_config = {"job": "profiler-pods", "gpu":gpu_id}  # select gpu0
    #my_label_config = {"instance": instance}  # select all gpu
    # if one particular instance is given, update instances
    if instance !='':
        instances = [instance,]
    for ins in instances:
        if gpu_id != '':
            gpus = [gpu_id,]
        else:
            gpus = ins_gpu[ins]
            print(ins, gpus)
        for gpu in gpus:
            my_label_config={"instance":ins,"gpu":gpu}
            df = pd.DataFrame()
            for metric_name in metrics:
                # select from different metric_name to query
                metric_data = prom.get_metric_range_data(
                                          metric_name=metric_name,
                                          label_config=my_label_config,
                                          start_time=parse_datetime(start_time),
                                          end_time=parse_datetime(end_time))

                # reorganize data to label_config and metric_values
                metric_object_list = MetricsList(metric_data)
                if len(metric_object_list) > 0:
                    if 'datetime' not in df.columns:
                        df['datetime'] = metric_object_list[0].metric_values['ds']
                    df[metric_name] = metric_object_list[0].metric_values['y']

            file_name = "_".join([ins, gpu]) + ".csv"
            df.to_csv(file_name)

if __name__ =="__main__":
    start_time_str = '2021-06-27 03:20:00'
    start_time = dt.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    end_time_str = '2021-06-27 04:10:00'
    end_time = dt.datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
    # if no instance or gpu is provided, default is to dump all
    get_all_metrics(start_time=start_time, end_time=end_time)

    # dump last 5 mins data for one gpu
    #get_all_metrics(start_time='5m', end_time='now', instance='10.244.2.23:9400', gpu_id='0')

