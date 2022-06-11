from concurrent import futures
import os
from random import shuffle
import grpc
import boto3
import hashlib
import pickle
import configparser
import utils.databus.databus_pb2 as dbus_pb2
import utils.databus.databus_pb2_grpc as dbus_grpc
from datetime import datetime
from google.protobuf.json_format import MessageToDict
from pymongo.mongo_client import MongoClient
from redis import Redis, RedisCluster
from utils.utils import *


logger = get_logger(name=__name__, level='DEBUG')


class Manager:
    def __init__(self) -> None:
        config = configparser.ConfigParser()
        config.read('cacher.conf')
        self.managerconf = config['manager']
        
        mconf = config['mongo']
        mongo_client = MongoClient(
            host=mconf.host,
            port=mconf.port,
            username=mconf.user,
            password=mconf.password)
        self.cacherdb = mongo_client.Cacher
        
        self.redconf = config['redis']
        if self.redconf.cluster_mode:
            self.redis_client =  RedisCluster(host=self.redconf.host, port=int(self.redconf.port), password=self.redconf.password)
        else:
            self.redis_client = Redis(host=self.redconf.host, port=int(self.redconf.port), password=self.redconf.password)
        
        logger.info("start global manager")
    
    def auth_client(self, username, password, conn_check=False):
        result = self.cacherdb.Client.find_one(filter={"$and": [{"username": username}, {"password": password}]}).pretty()
        if result is not None:
            if conn_check:
                return result if result['status'] else None
            else:
                return result
        else:
            return None
    
    def auth_job(self, jobId):
        result = self.cacherdb.Job.find_one(filter={"meta.jobId": jobId}).pretty()
        return result if result is not None else None
    
    def calculate_chunk_size(self, dataset_info: dict, qos=None):
        DEFAULT_CHUNK_SIZE = 512*1024*1024
        return DEFAULT_CHUNK_SIZE

    def flush_data(self):
        evict_policy = self.redis_client.config_get('maxmemory-policy')
        try:
            pipeline = {
                "allkeys-lru": flush_allkeys_lru,
                "allkeys-lfu": flush_allkeys_lfu
            }[evict_policy]
        except KeyError:
            return

        def flush_allkeys_lru(n=10):
            """Backup the least N recent used keys

            Args:
                backup (int, optional): the number of keys. Defaults to 10.
            """
            return [
                {"$unwind": "$policy.chunks"},
                {"$project": {
                    "_id": 0, 
                    "key": "$policy.chunks.key", 
                    "lastAccessTime": "$policy.chunks.lastAccessTime", 
                    "chunks": {"$regexFind": {"input": "$policy.chunks.location", "regex": "^redis", "options": "m"}}}},
                {"$match": {"chunks": {"$ne": None}}},
                {"$sort": {"lastAccessTime": 1}},
                {"$limit": n},
                {"$project": {"key": 1}},
                {"$group": {"_id": "$_id", "keys": {"$push": "$key"}}},
            ]
            
        def flush_allkeys_lfu(n=10):
            """Backup the least N frequent used keys

            Args:
                n (int, optional): the number of keys. Defaults to 10.
            """
            return [
                {"$unwind": "$policy.chunks"},
                {"$project": {
                    "_id": 0, 
                    "key": "$policy.chunks.key", 
                    "totalAccessTime": "$policy.chunks.totalAccessTime", 
                    "chunks": {"$regexFind": {"input": "$policy.chunks.location", "regex": "^redis", "options": "m"}}}},
                {"$match": {"chunks": {"$ne": None}}},
                {"$sort": {"totalAccessTime": 1}},
                {"$limit": n},
                {"$project": {"key": 1}},
                {"$group": {"_id": "$_id", "keys": {"$push": "$key"}}},
            ]
        
        backup_keys = self.cacherdb.Job.aggregate(pipeline())
        backup_data = self.redis_client.mget(backup_keys)
        for i in range(len(backup_keys)):
            with open('{}/{}'.format(self.managerconf['backup_dir'], backup_keys[i]), 'wb') as f:
                f.write(backup_data[i])
    
    def calculate_free_memory(self):
        max_memory = self.redis_client.config_get('maxmemory')
        assert max_memory > 0
        used_memory = self.redis_client.info['used_memory']
        return max_memory-used_memory
    

class Connection(dbus_grpc.ConnectionServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def connect(self, request, context):
        client = self.manager.auth_client(request.cred.username, request.cred.password)
        if client is not None:
            resp = dbus_pb2.ConnectResponse(
                rc = dbus_pb2.FAILED,
                resp = "please choose a different username"
            )
        else:
            result = self.manager.mongo_client.Cacher.Client.update_one(
                filter={
                    "username": request.cred.username,
                    "password": request.cred.password
                },
                update={"$set": {"status": True, "jobs": []}},
                upsert=request.createUser
            )
            if result['modified_count'] == 0:
                resp = dbus_pb2.ConnectResponse(
                    rc = dbus_pb2.FAILED,
                    resp = "username or password can't be found"
                )
            else:
                resp = dbus_pb2.ConnectResponse(
                    rc = dbus_pb2.SUCCESSFUL,
                    resp = "connection setup"
                )
                logger.info('set up connection with user {}'.format(request.cred.username))
        return resp


class Registration(dbus_grpc.RegistrationServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        self.client_col = manager.cacherdb.Client
        self.job_col = manager.cacherdb.Job

    def register(self, request, context):
        cred = request.cred
        client = self.manager.auth_client(cred.username, cred.password, conn_check=True)
        if client is not None:
            s3auth = request.s3auth
            s3_session = boto3.Session(
                aws_access_key_id=s3auth.aws_access_key_id,
                aws_secret_access_key=s3auth.aws_secret_access_key,
                region_name=s3auth.region_name
            )
            s3 = s3_session.resource('s3')
            s3_client = s3_session.client('s3')
            bucket = s3.Bucket(s3auth.bucket)
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=s3auth.bucket)
            
            dataset_info = {}
            for bucket in page_iterator:
                for file in bucket['Contents']:
                    if file not in s3auth.keys and len(s3auth.keys)>0: continue
                    try:
                        metadata = s3_client.head_object(Bucket=s3auth.bucket, Key=file['Key'])
                        dataset_info.update({file['Key']: metadata})
                    except:
                        print("Failed {}".format(file['Key']))

            chunk_size = self.manager.calculate_chunk_size(dataset_info)
            jobId = "{username}_{datasetinfo}_{now}".format(username=cred.username, datasetinfo=pickle.dumps(dataset_info)[:10], now=str(now).split('.')[0])
            token = hashlib.sha256(pickle.dumps(request)).hexdigest()
            
            chunk_keys = []
            for bkey in dataset_info:
                if dataset_info[bkey]['ContentLength']/1024/1024 <= chunk_size:
                    data = s3_client.get_object(Bucket=s3auth.bucket, Key=bkey)['Body'].read()
                    self.manager.redis_client.set("{}.{}.{}".format(cred.username, jobId, bkey), data)
                else:
                    bucket.download_file(s3auth.bucket, '/tmp/{}'.format(bkey), bkey)
                    with open('/tmp/{}'.format(bkey), 'rb') as f:
                        chunk = f.read(chunk_size)
                        index = 0
                        while chunk:
                            rk = "{}.{}.{}_{}".format(cred.username, jobId, bkey, index)
                            chunk_keys.append(rk)
                            self.manager.redis_client.set(rk)
                        index += 1
                        chunk = f.read(chunk_size)

            self.manager.redis_client.acl_setuser(
                username=jobId, passwords=['+{}'.format(token)], 
                commands=['+get', '+mget'],
                keys=chunk_keys, 
                reset=True, reset_keys=False, reset_passwords=False)
            now = datetime.datetime.utcnow().timestamp()
            
            chunks = []
            for key in chunk_keys:
                chunks.append({
                    "key": key,
                    "totalAccessTime": 0,
                    "lastAccessTime": None,
                    "location": "redis:{}".format(key)
                })
            jobInfo = {
                "meta": {
                    "username": cred.username,
                    "jobId": jobId,
                    "s3auth": MessageToDict(s3auth),
                    "resourceInfo": request.resource,
                    "dataset": request.bucket,
                    "createTime": grpc_ts(now),
                    "token": token,
                    "tokenTimeout": grpc_ts(now+3600)
                },
                "QoS": {
                    "useCache": request.useCache,
                    "flushFreq": request.flushFreq,
                    "durabilityInMem": request.durabilityInMem,
                    "durabilityInDisk": request.durabilityInDisk
                },
                "policy": {
                    "createTime": grpc_ts(now),
                    "chunkSize": chunk_size,
                    "chunks": chunks
                }
            }
            resp = dbus_pb2.RegisterResponse(
                dbus_pb2.RegisterSuccess(
                    jinfo=dbus_pb2.JobInfo(
                        jobId=jobId,
                        token=token,
                        createTime=grpc_ts(now),
                        tokenTimeout=grpc_ts(now+3600),
                        redisauth= dbus_pb2.RedisAuth(host=self.manager.redconf.host, port=self.manager.redconf.port, username=jobId, password=token)
                    ),
                    policy=dbus_pb2.Policy(chunkSize=chunk_size, chunkKeys=chunk_keys)))
            result = self.job_col.insert_one(jobInfo)
            if result.acknowledged:
                logger.info('user {} register job {}'.format(cred.username, jobId))
        else:
            if client is None:
                resp = dbus_pb2.RegisterResponse(dbus_pb2.RegisterError(error="Failed to register the jod, user is not connected."))
            else:
                resp = dbus_pb2.RegisterResponse(dbus_pb2.RegisterError(error="Job already exists, deregister first."))
        return resp

    def deresgister(self, request, context):
        cred = request.cred
        jobId = request.jinfo.jobId
        client = self.manager.auth_client(cred.username, cred.password, conn_check=True)
        if client is not None:
            self.manager.redis_client.acl_deluser(username=cred.username)
            result = self.manager.mongo_client.Cacher.Job.delete_one(filter={"jobId": jobId})
            if result.acknowledged and result.deleted_count == 1:
                resp = dbus_pb2.DeregisterResponse("successfully deregister job {jobId}".format(jobId=jobId))
                logger.info('user {} deregister job {}'.format(cred.username, jobId))
            else:
                resp = dbus_pb2.DeregisterResponse(response='Failed to deregister job {jobId}'.format(jobId=jobId))
        else:
            resp = dbus_pb2.DeregisterResponse(response="client is not connected")
        return resp


class Heartbeat(dbus_grpc.HeartbeatServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager

    def call(self, request, context):
        jinfo = request.jinfo
        job = self.manager.auth_job(jinfo.jobId)
        bston_ts = lambda ts: datetime.fromtimestamp(ts)
        if job is not None:
            now = datetime.utcnow().timestamp()
            tokenTimeout = bston_ts(jinfo.tokenTimeout.seconds + jinfo.tokenTimeout.nanos/1e9)
            if tokenTimeout > now:
                new_token = shuffle(jinfo.token)
                request.jinfo.token = new_token
                self.manager.mongo_client.Cacher.Job.aggregate([
                    {"$match": {"meta.jobId": jinfo.jobId}},
                    {"$set": {"meta.token": new_token, "meta.tokenTimeout": bston_ts(now+3600)}}
                ])
            result = self.manager.mongo_client.Cacher.Client.aggregate([
                {"$match": {"username": job['username']}},
                {"$set": {"lastHeartbeat":bston_ts(now)}}
            ])
            request.jinfo.tokenTimeout = grpc_ts(now+3600)
            if result.acknowledged and result.modified_count == 1:
                logger.info('heatbeat from user {} for job {}'.format(job['username'], job['jobId']))
            return request


class CacheMiss(dbus_grpc.CacheMissServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def call(self, request, context):        
        """Note: we assume disk has unlimited space"""
        chunk = self.cacherdb.Job.find({"policy.chunkKeys": {"$elemMatch": {"key": request.key}}}).pretty()
        if chunk.location.startswith('disk'):
            path = chunk.location.split(':')[1]
            with open(path, 'rb') as f:
                data = f.read()
            resp = self.redis_client.set(name=chunk.key, value=data)
            return dbus_pb2.CacheMissResponse(response=resp)
        else:
            return dbus_pb2.CacheMissResponse(response=True)


class Logger(dbus_grpc.LoggerServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def call(self, request, context):
        """
        TODO:
        1. Analyze training logs
        2. re-load data from S3 without changing the redis keys
        """
        return super().call(request, context)