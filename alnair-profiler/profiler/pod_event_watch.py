from kubernetes import watch
import logging
import hashlib
import datetime
from mongo_upsert import metadata2mongo
from prometheus_query import query_pod_utils
from datetime import timezone

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',level=logging.INFO)
logger = logging.getLogger('k8s_events')
logger.setLevel(logging.DEBUG)
DOMAIN = "alnair.centaurusinfra.io"    
# avaliable Pod status phase: Pending/Running/Succeeded/Failed/Unknown
# https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/

MAX_LOOK_BACK_SECONDS = 1800 # to avoid query too long of data, set the look back interval no longer than 30 mins

def pod_watch(v1, node_name, mongo_url, client_connect):
    w = watch.Watch()
    field_selector = 'spec.nodeName='+node_name
    for event in w.stream(v1.list_pod_for_all_namespaces,field_selector=field_selector, timeout_seconds=1800):
        # get the pod with changes
        logger.info("Event type: {}, resource kind: {}, name: {}, phase: {}, resource version: {}".format(event['type'], event['object'].kind, event['object'].metadata.name, 
                                                                  event['object'].status.phase, event['object'].metadata.resource_version))
        # ignore pending pods
        if event['object'].status.phase == "Pending":   
            continue
        # 1. collect metadata
        pod_name = event['object'].metadata.name
        pod_image = event['object'].status.container_statuses[0].image
        namespace = event['object'].metadata.namespace
        pod_creation_ts = event['object'].metadata.creation_timestamp
        pod_status = event['object'].status.phase
        node_name = event['object'].spec.node_name
        owner_ref = event['object'].metadata.owner_references
        if owner_ref is not None and len(owner_ref) > 0:
            pod_owner = owner_ref[0].kind + owner_ref[0].name
        else:
            pod_owner = ""
        #volumes = event['object'].spec.volumes   #leave volumes parsing for now
        key_str = ":".join([pod_name, namespace, pod_creation_ts.strftime('%Y%m%d%H%M%S')]) # kubernetes time resoultion is second, no need to add %f
        container_started_time, container_finished_time, duration = None, None, None
        if pod_status == "Running":
            if event['object'].status.container_statuses[0].state.running is not None:
                container_started_time = event['object'].status.container_statuses[0].state.running.started_at
                duration = datetime.datetime.now(timezone.utc) - container_started_time
        else: # pod_status == "Succeeded":
            if event['object'].status.container_statuses[0].state.terminated is not None:
                container_started_time = event['object'].status.container_statuses[0].state.terminated.started_at
                container_finished_time = event['object'].status.container_statuses[0].state.terminated.finished_at
                duration = container_finished_time - container_started_time

        pod_metadata = {"md5_key": hashlib.md5(key_str.encode('utf-8')).hexdigest(),
                        "pod_name": pod_name, 
                        "image": pod_image,
                        "node_name": node_name, 
                        "namespace": namespace, 
                        "pod_create_time": pod_creation_ts,
                        "pod_status": pod_status,
                        "pod_owner" : pod_owner
                        }
        if container_started_time is not None:
            pod_metadata["container_started_time"] = container_started_time
        
        if container_finished_time is not None:
            pod_metadata["container_finished_time"] = container_finished_time
        
        if duration is not None:
            pod_metadata["duration"] = duration.seconds
            
        # 2. collect dynamic utils data, query prometheus
        pod_utils = {}
        if duration is not None:
            if duration > datetime.timedelta(seconds=3): # for those just start running ignore query 
                if duration > datetime.timedelta(minutes=MAX_LOOK_BACK_SECONDS):
                    duration = datetime.timedelta(minutes=MAX_LOOK_BACK_SECONDS)
                pod_utils = query_pod_utils(pod_name, namespace, duration)
        else:
            pod_utils = query_pod_utils(pod_name, namespace, datetime.timedelta(minutes=MAX_LOOK_BACK_SECONDS))
            
        pod_record = dict(pod_metadata)
        if len(pod_utils) > 0:
            pod_record.update(pod_utils)
        else:
            logging.info("pod {} has just started or no activity in recent {} secs".format(pod_name, MAX_LOOK_BACK_SECONDS))
        # 3. upsert to mongo
        ret = metadata2mongo(mongo_url, client_connect, pod_record)
        if ret:
            logging.info("upsert success")
        else:
            logging.info("upsert fails")
        # 4. patch pod with utils once
        patch = True
        if event['object'].metadata.annotations is not None:
            #logging.info(event['object'].metadata.annotations)
            for key in event['object'].metadata.annotations:
                if key.startswith("alnair.centaurusinfra.io/max") or key.startswith("alnair.centaurusinfra.io/gpu"): 
                    patch = False
                    break
        if len(pod_utils) > 0 and patch:
            # patch pod annotation:
            # reformat dictionary, in case it is nested, flatten to one level by cast nested dict to string
            pod_ann = {}
            for k, v in pod_utils.items():
                if v is not None:  # if v is None, it means delete the existing annotaion
                    pod_ann[DOMAIN + "/" + k]=str(v)
            body = {'metadata': {'annotations': pod_ann}}   
            v1.patch_namespaced_pod(pod_name, namespace, body)
        
        
    logging.info("watch stream ends after timeout")
