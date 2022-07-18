import concurrent.futures
import grpc
import boto3
import threading
import json, bson
import configparser
import multiprocessing
import hazelcast as hz
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pytz import timezone
from utils import *


logger = get_logger(name=__name__, level='debug')


class Manager(object):
    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read('/configs/manager/manager.conf')
        self.managerconf = parser['manager']
        self.hzconf = parser['hazelcast']
        mconf = parser['mongodb']
        mongo_client = MongoClient(host=mconf['host'], port=int(mconf['port']), 
                                   username=mconf['username'], password=mconf['password'])
        with open("mongo-schemas/client.json", 'r') as f:
            client_schema = json.load(f)
        with open("mongo-schemas/job.json", 'r') as f:
            job_schema = json.load(f)
                
        collections = mongo_client.Cacher.list_collection_names()
        if 'Client' not in collections:
            self.client_col = mongo_client.Cacher.create_collection(name="Client", validator={"$jsonSchema": client_schema}, validationAction="error")
        else:
            self.client_col = mongo_client.Cacher.Client
            
        if 'Job' not in collections: 
            self.job_col = mongo_client.Cacher.create_collection(name='Job', validator={"$jsonSchema": job_schema}, validationAction="error")
        else:
            self.job_col = mongo_client.Cacher.Job

        self.cacher = hz.HazelcastClient(
            cluster_name="Alnair",
            cluster_members=self.hzconf['cluster'].split(','),
            smart_routing=True,
            client_name='alnair.manager',
            lifecycle_listeners=[
                lambda state: print("Lifecycle event >>>", state),
            ],
            connection_timeout=120)    
        logger.info("start global manager")

    def auth_client(self, cred, s3auth=None, conn_check=False):
        result = self.client_col.find_one(filter={"username": cred.username})
        if result is None:
                rc = pb.RC.NO_USER
        else:
            if cred.password == result['password']:
                if conn_check:
                    rc = pb.RC.CONNECTED if result['status'] else pb.RC.DISCONNECTED
                else:
                    rc = pb.RC.CONNECTED
            else:
                rc = pb.RC.WRONG_PASSWORD
        # check whether to update s3auth information
        if rc == pb.RC.CONNECTED and s3auth is not None and result['s3auth'] != s3auth:
            result = self.client_col.update_one(
                filter={"username": cred.username}, 
                update={"$set": {"s3auth": s3auth}})
            if result.modified_count != 1:
                logger.error("user {} is connected, but failed to update S3 authorization information.".format(cred.username))
                rc = pb.RC.FAILED
        return rc           
    
    def auth_job(self, jobId):
        result = self.cacherdb.Job.find_one(filter={"meta.jobId": jobId})
        return result if result is not None else None
    
    def calculate_chunk_size(self, dataset_info: dict, qos=None):
        """Calculate the proper chunk size based on available memory and dataset size
        # TODO: what is the strategy of deciding the chunk size
        Args:
            dataset_info (dict): meta information of the dataset
            qos (_type_, optional): QoS setting of the dataset. Defaults to None.

        Returns:
            _type_: chunk size in MB. Defaults to: 512MB
        """
        DEFAULT_CHUNK_SIZE = 512*1024*1024
        return DEFAULT_CHUNK_SIZE


class ConnectionService(pb_grpc.ConnectionServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def connect(self, request, context):
        cred, s3auth = request.cred, request.s3auth
        rc = self.manager.auth_client(cred=cred, s3auth=MessageToDict(s3auth))
        if rc == pb.RC.WRONG_PASSWORD:
            resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp="wrong password")
        elif rc == pb.RC.NO_USER:
            if request.createUser:
                try:
                    result = self.manager.client_col.insert_one({
                        "username": cred.username,
                        "password": cred.password,
                        "s3auth": MessageToDict(s3auth),
                        "status": True
                    })
                except Exception as ex:
                    print(ex)
                if result.acknowledged:
                    logger.info("user {} connected".format(cred.username))
                    resp = pb.ConnectResponse(rc=pb.RC.CONNECTED, resp="connection setup")
                else:
                    resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp="connection error")
            else:
                resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp = "not found user {}".format(cred.username))
        elif rc == pb.RC.DISCONNECTED:
            result = self.manager.client_col.update_one(
                filter={
                    "username": cred.username,
                    "password": cred.password,
                },
                update={"$set": {"status": True, "jobs": []}}
            )
            if result['modified_count'] == 0:
                resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp="connection error")
            else:
                resp = pb.ConnectResponse(rc=pb.RC.CONNECTED, resp="connection setup")
                logger.info("user {} connected".format(cred.username))
        else:
            resp = pb.ConnectResponse(rc=pb.RC.CONNECTED, resp="connection setup")
            logger.info("user {} connected".format(cred.username))
        return resp


class RegistrationService(pb_grpc.RegistrationServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager

    def register(self, request, context):
        cred = request.cred
        rc = self.manager.auth_client(cred, conn_check=True)
        jobId = "{}/{}".format(cred.username, request.datasource.name)
        dataset = self.manager.cacher.get_map(request.datasource.name)
        if rc == pb.RC.CONNECTED:
            # get s3 auth
            result = self.manager.client_col.find_one(filter={"$and": [{"username": cred.username, "password": cred.password}]})
            s3auth = result['s3auth']
            s3_session = boto3.Session(
                aws_access_key_id=s3auth['aws_access_key_id'],
                aws_secret_access_key=s3auth['aws_secret_access_key'],
                region_name=s3auth['region_name']
            )
            s3_client = s3_session.client('s3')
            bucket_name = request.datasource.bucket

            # get object keys that are not in MongoDB and Cache Cluster
            saved_keys = {}
            try:
                saved_job = self.manager.job_col.aggregate([
                    {"$match": {"meta.jobId": jobId}},
                    {"$sort": {"meta.createTime": -1}},
                    {"$limit": 1}
                ]).next()
            except:
                saved_job = None
            if saved_job is not None:
                for chunk in saved_job['policy']['chunks']:
                    saved_keys[chunk['name']] = chunk
            
            bucket_objs = {} 
            def list_modified_objects(prefix, page):
                nonlocal bucket_objs
                for info in page['Contents']:
                    info['prefix'] = prefix
                    if saved_job is not None \
                        and info['Key'] in saved_keys \
                        and info['LastModified'].replace(tzinfo=timezone('UTC')).timestamp() == saved_keys[info['Key']]['lastModified'] \
                        and dataset.contains_key(saved_keys[info['Key']]['location'].split(':')[1]):
                        info['Exist'] = True
                    else:
                        info['Exist'] = False
                    bucket_objs[info['Key']] = info
            
            if len(request.datasource.keys) == 0:
                request.datasource.keys = [bucket_name]
            for prefix in request.datasource.keys:
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    for page in pages:
                        futures.append(executor.submit(list_modified_objects, prefix, page))
                concurrent.futures.wait(futures)
            
            chunk_size = self.manager.calculate_chunk_size(bucket_objs)

            # copy data from S3 to CC
            snapshot = {}
            def copy_data(info: dict):
                nonlocal snapshot, dataset
                chunk_keys = []
                if not info['Exist']:
                    if info['Size'] <= chunk_size:
                        value = s3_client.get_object(Bucket=bucket_name, Key=info['Key'])['Body'].read()
                        hash_key = hashing(value)
                        dataset.put(hash_key, bytearray(value))
                        # logger.info("Copy data from s3:{} to cacher:{}".format(info['Key'], hash_key))
                        obj = {'name': info['Key'], 'key': hash_key, 'size': info['Size'], 'lastModified': int(info['LastModified'].timestamp())}
                        chunk_keys.append(obj)
                    else:
                        s3_client.download_file(Bucket=bucket_name, Key=info['Key'], Filename='/tmp/{}'.format(info['Key']))
                        # logger.info("Download large file s3:{}, size: {}B".format(info['Key'], info['Size']))
                        values = {}
                        with open('/tmp/{}'.format(info['Key']), 'rb') as f:
                            value = f.read(chunk_size)
                            while value:
                                hash_key = hashing(value)
                                obj = {'name': info['Key'], 'key': hash_key, 'size': chunk_size, 'lastModified': int(info['LastModified'].timestamp())}
                                chunk_keys.append(obj)
                                values[hash_key] = bytearray(value)
                                # logger.info("Copy data from /tmp/{} to redis:{}".format(info['Key'], hash_key))
                                value = f.read(chunk_size)
                            dataset.put_all(values)
                else:
                    # find hash keys given the bucket key
                    def search_key(bk):
                        if bk in saved_keys:
                            return saved_keys[bk]['location'].split(':')[1]
                        return None
                    chunk_keys.append({
                        'name': info['Key'], 
                        'key': search_key(info['Key']), 
                        'size': info['Size'], 
                        'lastModified': int(info['LastModified'].timestamp())})
                    # logger.info('Key {} exists in Cache Cluster.'.format(info['Key']))
                
                sk = info['prefix']
                if sk not in snapshot:
                    snapshot[sk] = {info['Key']: chunk_keys}
                else:
                    snapshot[sk][info['Key']] = chunk_keys
                return chunk_keys
            
            chunk_keys = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for bk in bucket_objs:
                    info = bucket_objs[bk]
                    futures.append(executor.submit(copy_data, info))
                for future in concurrent.futures.as_completed(futures):
                    chunk_keys.extend(future.result())
            
            # create job snapshot in Cache Clucter
            dataset.put_all(snapshot)
            
            # generate connection authorization for client
            now = datetime.utcnow().timestamp()
            token = MessageToDict(request)
            token['time'] = now
            token = hashing(token)
            
            # save jobinfo to database
            chunks = []
            for ck in chunk_keys:
                chunks.append({
                    "name": ck['name'],
                    "size": ck['size'],
                    "lastModified": ck['lastModified'],
                    "totalAccessTime": 0,
                    "lastAccessTime": bson.timestamp.Timestamp(int(now), inc=1),
                    "location": "cacher:{}".format(ck['key']),
                    "hasBackup": False
                })
            jobInfo = {
                "meta": {
                    "username": cred.username,
                    "jobId": jobId,
                    "datasource": MessageToDict(request.datasource),
                    "resourceInfo": MessageToDict(request.resource),
                    "createTime": bson.timestamp.Timestamp(int(now), inc=1),
                    "token": token,
                    "tokenTimeout": bson.timestamp.Timestamp(int(now+self.manager.managerconf.getint('token_life')), inc=1)
                },
                "QoS": MessageToDict(request.qos),
                "policy": {
                    "createTime": bson.timestamp.Timestamp(int(now), inc=1),
                    "chunkSize": chunk_size,
                    "chunks": chunks
                }
            }
            result = self.manager.job_col.insert_one(jobInfo)
            if result.acknowledged:
                logger.info('user {} register job {}'.format(cred.username, jobId))
                
            resp = pb.RegisterResponse(rc=pb.RC.REGISTERED, regsucc=pb.RegisterSuccess(
                    jinfo=pb.JobInfo(
                        jobId=jobId,
                        token=token,
                        createTime=grpc_ts(now),
                        tokenTimeout=grpc_ts(now+3600),
                        ccauth=pb.CCAuth(cluster=self.manager.hzconf['cluster'].split(','))),
                    policy=pb.Policy(chunkSize=chunk_size, snapshot=list(snapshot.keys()))))
        elif rc == pb.RC.DISCONNECTED:
            resp = pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod, user is not connected."))
        else:
            resp = pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod"))
        
        def evicted_func(event):
            key = event.key,
            value = event.value
            with open('{}/{}'.format(self.managerconf['backup_dir'], key), 'wb') as f:
                f.write(value)
            self.job_col.job_col.update(
                {"policy.chunks": {"$elemMatch": {"key": key}}},
                {"$set": {"policy.chunks.$.hasBackup": True}})
        dataset.add_entry_listener(include_value=True, evicted_func=evicted_func)
        return resp

    def deresgister(self, request, context):
        cred = request.cred
        jobId = request.jinfo.jobId
        rc = self.manager.auth_client(cred)
        if rc in [pb.RC.CONNECTED, pb.RC.DISCONNECTED]:
            result = self.manager.job_col.delete_one(filter={"jobId": jobId})
            if result.acknowledged and result.deleted_count == 1:
                resp = pb.DeregisterResponse("successfully deregister job {}".format(jobId))
                logger.info('user {} deregister job {}'.format(cred.username, jobId))
            else:
                resp = pb.DeregisterResponse(response='failed to deregister job {}'.format(jobId))
        else:
            resp = pb.DeregisterResponse(response="failed to deregister job {}".format(jobId))
        return resp


class HeartbeatService(pb_grpc.HeartbeatServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager

    def call(self, request, context):
        cred = request.cred
        jinfo = request.jinfo
        rc = self.manager.auth_client(cred, conn_check=True)   
        if rc != pb.RC.CONNECTED:
            return
        job = self.manager.auth_job(jinfo.jobId)
        bston_ts = lambda ts: datetime.fromtimestamp(ts)
        if job is not None:
            now = datetime.utcnow().timestamp()
            tokenTimeout = bston_ts(jinfo.tokenTimeout.seconds + jinfo.tokenTimeout.nanos/1e9)
            if tokenTimeout > now:
                token = MessageToDict(jinfo)
                token['time'] = now
                token = hashing(token)
                request.jinfo.token = token
                self.manager.job_col.aggregate([
                    {"$match": {"meta.jobId": jinfo.jobId}},
                    {"$set": {"meta.token": token, "meta.tokenTimeout": bston_ts(now+3600)}}
                ])
            result = self.manager.client_col.aggregate([
                {"$match": {"username": job['username']}},
                {"$set": {"lastHeartbeat":bston_ts(now)}}
            ])
            request.jinfo.tokenTimeout = grpc_ts(now+3600)
            if result.acknowledged and result.modified_count == 1:
                logger.info('heatbeat from user {} for job {}'.format(job['username'], job['jobId']))
                return request           


class CacheMissService(pb_grpc.CacheMissServicer):
    """Note: we assume disk has unlimited space"""
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
    
    def call(self, request, context):
        cred = request.cred
        dataset = self.manager.cacher.get_map(request.jinfo.jobId.split('/')[1])
        rc = self.manager.auth_client(cred, conn_check=True)
        if rc != pb.RC.CONNECTED:
            return
        chunk = self.cacherdb.Job.aggregate([
            {"$project": {
                "_id": 0, 
                "chunk": {
                    "$filter": {
                        "input": "$policy.chunks", 
                        "as": "chunks", 
                        "cond": {"$eq": ["$$chunks.key", request.key]}}
                    }
                }
            },
            {"$unwind": "$chunk"}
        ]).next()['chunk']
        if chunk['location'] == 'disk':
            path = chunk['location'].split(':')[1]
            with open(path, 'rb') as f:
                data = f.read()
            resp = dataset.put_if_absent(chunk['key'], data)
            return pb.CacheMissResponse(response=resp)
        else:
            return pb.CacheMissResponse(response=True)


if __name__ == '__main__':
    manager = Manager()
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
    pb_grpc.add_ConnectionServicer_to_server(ConnectionService(manager), server)
    pb_grpc.add_RegistrationServicer_to_server(RegistrationService(manager), server)
    pb_grpc.add_HeartbeatServicer_to_server(HeartbeatService(manager), server)
    pb_grpc.add_CacheMissServicer_to_server(CacheMissService(manager), server)
    server.add_insecure_port(address="{}:{}".format(manager.managerconf['bind'], manager.managerconf['port']))
    server.start()
    server.wait_for_termination()