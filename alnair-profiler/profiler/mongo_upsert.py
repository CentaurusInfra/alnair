import logging
from pymongo import MongoClient

def metadata2mongo(mongo_url, client_connect, pod_metadata):
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',level=logging.INFO)

    # Force a call to check if current connection is valid
    try:
        info = client_connect.server_info()
    except Exception as e:
        logging.warning(e)
        logging.warning("reconnect to mongo at {}".format(mongo_url))
        client_connect = MongoClient(mongo_url, serverSelectionTimeoutMS = 5000, tz_aware=True)

    db = client_connect['alnair']
    col = db['pod_record']
    try:
        filter = { 'md5_key': pod_metadata["md5_key"] }
        existing_info = col.find_one(filter, {"_id":0}) # second input select/deselect the field
        if existing_info is None or len(existing_info) == 0:
            logging.info("insert a new pod record: {}".format(pod_metadata))
            newvalues = { "$set": pod_metadata }
        else: # compare new value with old value
            diff_dict = {}
            for k, v in pod_metadata.items():
                if k in existing_info:
                    if k.startswith("gpu_"):  # compare the nested value
                        for sub_k, max_gpu in pod_metadata[k].items():
                            if sub_k in existing_info[k]:
                                if max_gpu > existing_info[k][sub_k]:
                                    # add this to diff dict
                                    if k in diff_dict:
                                        diff_dict[k][sub_k] = max_gpu
                                    else:
                                        diff_dict[k] = {sub_k:max_gpu}
                    else:
                        if v > existing_info[k]:
                            diff_dict[k] = v
                else:
                    diff_dict[k] = v
            logging.info("pod record diff: {}".format(diff_dict))
            newvalues = { "$set": diff_dict }
        col.update_one(filter, newvalues,upsert=True)
        return True
    except Exception as e:
        logging.warning(e)
        return False