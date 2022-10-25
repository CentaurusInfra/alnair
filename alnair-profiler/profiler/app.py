from prometheus_api_client import PrometheusConnect, MetricsList
from prometheus_api_client.utils import parse_datetime
import os
import logging
from kubernetes import config, client
from pymongo import MongoClient
from pod_event_watch import pod_watch
from kubernetes.client.exceptions import ApiException


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


def app_top():
    env_var, err = load_env_var()
    if err:
        logging.error("Not all environment variables are avaliable in the profiler pod")
        exit(1)
    
    # 0) load kubernetes configure & connect to MongoDB
    config.load_incluster_config() 
    core_api = client.CoreV1Api()
    CONNECTION_STRING = "mongodb://mongo:27017/"
    client_connect = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000, tz_aware=True)
    try:
        client_connect.server_info()
    except Exception as e:
        logging.warning(e)
         
    ##  pod event watch, reconnect every 30 mins
    while True:
        try:
            logging.info("Alnair profiler start watching pods")
            pod_watch(core_api, env_var['node_name'], CONNECTION_STRING, client_connect)
        except ApiException as e:
            logging.error(e)
            if e.status == 410: # Resource too old
                logging.error("pod watch stream exception, resource version too old")    
            else:
                raise
    


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',level=logging.INFO)
    app_top()
