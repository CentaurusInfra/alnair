from concurrent import futures
import grpc
import boto3
import pickle
import threading
import json
import time
from enum import Enum
import configparser
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from datetime import datetime
from google.protobuf.json_format import MessageToDict
from pymongo.mongo_client import MongoClient
from redis import Redis
from utils import *


KEY_LEN = 20
TOKEN_LEN = 20
logger = get_logger(name=__name__, level='Info')

class ConnectionRC(Enum):
    CONNECTED = 1
    NO_USER = 2
    WRONG_PASSWORD = 3
    DISCONNECTED = 4


class Manager(object):
    def __init__(self) -> None:
        self.redisconfig = configparser.ConfigParser()
        self.redisconfig.read('configs/redis/redis.conf') 
        
        parser = configparser.ConfigParser()
        parser.read('/configs/manager/manager.conf')
        self.managerconf = parser['manager']
        mconf = parser['mongodb']
        mongo_client = MongoClient(host=mconf['host'], port=int(mconf['port']), 
                                   username=mconf['username'], password=mconf['password'])
        with open("mongo-schemas/client.json", 'r') as f:
            client_schema = json.load(f)
        with open("mongo-schemas/job.json", 'r') as f:
            job_schema = json.load(f)
        self.client_col = mongo_client.Cacher.create_collection(name="Client", validator={"$jsonSchema": client_schema}, validationAction="error")
        self.job_col = mongo_client.Cacher.create_collection(name='Job', validator={"$jsonSchema": job_schema}, validationAction="error")
        
        self.redauthconf = parser['redis']
        self.redis_client =  Redis(host=self.redauthconf['host'], port=int(self.redauthconf['port']), 
                                   password=self.redauthconf['password'])
        
        flush_thrd = threading.Thread(target=Manager.flush_data, args=(self,), daemon=True)
        flush_thrd.start()
        
        logger.info("start global manager")

    def auth_client(self, username, password, conn_check=False):
        result = self.client_col.find_one(filter={"username": username})
        if result is None:
                return ConnectionRC.NO_USER
        else:
            if password == result['password']:
                if conn_check:
                    return ConnectionRC.SUCCESS if result['status'] else ConnectionRC.DISCONNECTED
                else:
                    return ConnectionRC.CONNECTED
            else:
                return ConnectionRC.WRONG_PASSWORD
    
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
            _type_: chunk size in MB. Defaults to the maximum Redis value size: 512MB
        """
        DEFAULT_CHUNK_SIZE = 512*1024*1024
        return DEFAULT_CHUNK_SIZE

    def flush_data(self):
        def flush_allkeys_lru():
            """Backup the least N recent used keys

            Args:
                backup (int, optional): the number of keys. Defaults to 10.
            """
            return [
                {"$unwind": "$policy.chunks"},
                {"$project": {
                    "_id": 0, 
                    "key": "$policy.chunks.key", 
                    "hasBackup": "$policy.chunks.hasBackup",
                    "lastAccessTime": "$policy.chunks.lastAccessTime", 
                    "chunk": {"$regexFind": {"input": "$policy.chunks.location", "regex": "^redis", "options": "m"}}}
                 },
                {"$match": {"chunk": {"$ne": None}}},
                {"$sort": {"lastAccessTime": 1}},
                {"$limit": self.managerconf['flush_amount']},
                {"$match": {"hasBackup": {"$eq": False}}},
                {"$project": {"key": 1}},
                {"$group": {"_id": "$_id", "keys": {"$push": "$key"}}}
            ]
            
        def flush_allkeys_lfu():
            """Backup the least N frequent used keys

            Args:
                n (int, optional): the number of keys. Defaults to 10.
            """
            return [
                {"$unwind": "$policy.chunks"},
                {"$project": {
                    "_id": 0, 
                    "key": "$policy.chunks.key", 
                    "hasBackup": "$policy.chunks.hasBackup",
                    "totalAccessTime": "$policy.chunks.totalAccessTime", 
                    "chunks": {"$regexFind": {"input": "$policy.chunks.location", "regex": "^redis", "options": "m"}}}
                },
                {"$match": {"chunks": {"$ne": None}}},
                {"$sort": {"totalAccessTime": 1}},
                {"$limit": self.managerconf['flush_amount']},
                {"$match": {"hasBackup": {"$eq": False}}},
                {"$project": {"key": 1}},
                {"$group": {"_id": "$_id", "keys": {"$push": "$key"}}},
            ]
        
        if self.redisconfig.has_option(section=None, option='maxmemory-policy'):
            evict_policy = self.redisconfig['maxmemory-policy']
        else:
            evict_policy = "allkeys-lru"
        try:
            pipeline = {
                "allkeys-lru": flush_allkeys_lru,
                "allkeys-lfu": flush_allkeys_lfu
            }[evict_policy]()
        except KeyError:
            return
        
        # periodically flush data into disk
        while True:
            backup_keys = self.job_col.aggregate(pipeline).next()['keys']
            backup_data = self.redis_client.mget(backup_keys)
            for i in range(len(backup_keys)):
                with open('{}/{}'.format(self.managerconf['backup_dir'], backup_keys[i]), 'wb') as f:
                    f.write(backup_data[i])
            self.job_col.job_col.update_many(
                {"policy.chunks": {"$elemMatch": {"key": {"$in": backup_keys}}}},
                {"$set": {"policy.chunks.$.hasBackup": True}}
            )
            time.sleep(self.managerconf['flush_frequency'] * 60)


class ConnectionService(pb_grpc.ConnectionServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def connect(self, request, context):
        cred, s3auth = request.cred, request.s3auth
        rc = self.manager.auth_client(cred.username, cred.password)
        if rc == ConnectionRC.WRONG_PASSWORD is not None:
            resp = pb.ConnectResponse(rc=pb.FAILED, resp="wrong password")
        elif rc == ConnectionRC.NO_USER:
            if request.createUser:
                result = self.manager.client_col.insert_one({
                    "username": cred.username,
                    "password": cred.password,
                    "s3auth": {
                        "aws_access_key_id": s3auth.aws_access_key_id,
                        "aws_secret_access_key": s3auth.aws_secret_access_key,
                        "region_name": s3auth.region_name
                    }
                })
                if result.acknowledged:
                    logger.info("user {} connected".format(cred.username))
                    resp = pb.ConnectResponse(rc=pb.SUCCESSFUL, resp="connection setup")
                else:
                    resp = pb.ConnectResponse(rc=pb.FAILED, resp="connection error")
            else:
                resp = pb.ConnectResponse(rc=pb.FAILED, resp = "not found user {}".format(cred.username))
        elif rc == ConnectionRC.DISCONNECTED:
            result = self.manager.client_col.update_one(
                filter={
                    "username": cred.username,
                    "password": cred.password,
                },
                update={"$set": {"status": True, "jobs": []}}
            )
            if result['modified_count'] == 0:
                resp = pb.ConnectResponse(rc=pb.FAILED, resp="connection error")
            else:
                resp = pb.ConnectResponse(rc=pb.SUCCESSFUL, resp="connection setup")
                logger.info("user {} connected".format(cred.username))
        else:
            resp = pb.ConnectResponse(rc=pb.SUCCESSFUL, resp="connection setup")
            logger.info("user {} connected".format(cred.username))
        return resp


class RegistrationService(pb_grpc.RegistrationServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager

    def register(self, request, context):
        cred = request.cred
        rc = self.manager.auth_client(cred.username, cred.password, conn_check=True)
        if rc == ConnectionRC.CONNECTED:
            # get s3 auth
            self.manager.client_col.find_one(filter={})
            result = self.manager.client_col.find_one(filter={"$and": [{"username": cred.username, "password": cred.password}]})
            s3auth = result['s3auth']
            s3_session = boto3.Session(
                aws_access_key_id=s3auth['aws_access_key_id'],
                aws_secret_access_key=s3auth['aws_secret_access_key'],
                region_name=s3auth['region_name']
            )
            s3 = s3_session.resource('s3')
            s3_client = s3_session.client('s3')
            
            bucket_name = request.datasource.bucket
            bucket_keys = request.datasource.keys
            bucket = s3.Bucket(bucket_name)
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket_name)
            
            dataset_info = {}
            for bucket in page_iterator:
                for obj in bucket['Contents']:
                    if obj not in bucket_keys and len(bucket_keys)>0: continue
                    try:
                        metadata = s3_client.head_object(Bucket=bucket_name, Key=obj['Key'])
                        dataset_info.update({obj['Key']: metadata})
                    except:
                        print("Failed {}".format(obj['Key']))

            now = datetime.datetime.utcnow().timestamp()
            chunk_size = self.manager.calculate_chunk_size(dataset_info)
            jobId = "{username}_{datasetinfo}_{now}".format(username=cred.username, datasetinfo=pickle.dumps(dataset_info)[:10], now=str(now).split('.')[0])
            
            token = MessageToDict(request)
            token['time'] = now
            token = hashing(token)[:TOKEN_LEN]
            
            chunk_keys = []
            for bkey in dataset_info:
                if dataset_info[bkey]['ContentLength']/1024/1024 <= chunk_size:
                    value = s3_client.get_object(Bucket=s3auth.bucket, Key=bkey)['Body'].read()
                    key = hashing(value)
                    self.manager.redis_client.set(key, value)
                else:
                    bucket.download_file(s3auth.bucket, '/tmp/{}'.format(bkey), bkey)
                    with open('/tmp/{}'.format(bkey), 'rb') as f:
                        value = f.read(chunk_size)
                        index = 0
                        while value:
                            key = hashing(value)
                            key = "{}_part{}".format(key, index)
                            chunk_keys.append(key)
                            self.manager.redis_client.set(key, value)
                            index += 1
                            value = f.read(chunk_size)

            self.manager.redis_client.acl_setuser(
                username=jobId, passwords=['+{}'.format(token)], 
                commands=['+get', '+mget'],
                keys=chunk_keys, 
                reset=True, reset_keys=False, reset_passwords=False)
            chunks = []
            for key in chunk_keys:
                chunks.append({
                    "key": key,
                    "totalAccessTime": 0,
                    "lastAccessTime": None,
                    "location": "redis:{}".format(key),
                    "hasBackup": False
                })
            jobInfo = {
                "meta": {
                    "username": cred.username,
                    "jobId": jobId,
                    "datasource": MessageToDict(request.datasource),
                    "resourceInfo": MessageToDict(request.resource),
                    "createTime": grpc_ts(now),
                    "token": token,
                    "tokenTimeout": grpc_ts(now+3600)
                },
                "QoS": MessageToDict(request.qos),
                "policy": {
                    "createTime": grpc_ts(now),
                    "chunkSize": chunk_size,
                    "chunks": chunks
                }
            }
            resp = pb.RegisterResponse(
                pb.RegisterSuccess(
                    jinfo=pb.JobInfo(
                        jobId=jobId,
                        token=token,
                        createTime=grpc_ts(now),
                        tokenTimeout=grpc_ts(now+3600),
                        redisauth= pb.RedisAuth(host=self.manager.redauthconf['host'], port=self.manager.redauthconf['port'], username=jobId, password=token)
                    ),
                    policy=pb.Policy(chunkSize=chunk_size, chunkKeys=chunk_keys)))
            result = self.manager.job_col.insert_one(jobInfo)
            if result.acknowledged:
                logger.info('user {} register job {}'.format(cred.username, jobId))
        elif rc == ConnectionRC.DISCONNECTED:
            resp = pb.RegisterResponse(pb.RegisterError(error="failed to register the jod, user is not connected."))
        else:
            resp = pb.RegisterResponse(pb.RegisterError(error="failed to register the jod"))
        return resp

    def deresgister(self, request, context):
        cred = request.cred
        jobId = request.jinfo.jobId
        rc = self.manager.auth_client(cred.username, cred.password)
        if rc in [ConnectionRC.CONNECTED, ConnectionRC.DISCONNECTED]:
            self.manager.redis_client.acl_deluser(username=cred.username)
            result = self.manager.job_col.delete_one(filter={"jobId": jobId})
            # we don't delete dataset from Redis as it might be shared by multiple jobs
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
        rc = self.manager.auth_client(cred.username, cred.password, conn_check=True)   
        if rc != ConnectionRC.CONNECTED:
            return
        job = self.manager.auth_job(jinfo.jobId)
        bston_ts = lambda ts: datetime.fromtimestamp(ts)
        if job is not None:
            now = datetime.utcnow().timestamp()
            tokenTimeout = bston_ts(jinfo.tokenTimeout.seconds + jinfo.tokenTimeout.nanos/1e9)
            if tokenTimeout > now:
                token = MessageToDict(jinfo)
                token['time'] = now
                token = hashing(token)[:TOKEN_LEN]
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
        rc = self.manager.auth_client(cred.username, cred.password, conn_check=True)
        if rc != ConnectionRC.CONNECTED:
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
            resp = self.redis_client.set(name=chunk['key'], value=data)
            return pb.CacheMissResponse(response=resp)
        else:
            return pb.CacheMissResponse(response=True)


if __name__ == '__main__':
    manager = Manager()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb_grpc.add_ConnectionServicer_to_server(ConnectionService(manager), server)
    pb_grpc.add_RegistrationServicer_to_server(RegistrationService(manager), server)
    pb_grpc.add_HeartbeatServicer_to_server(HeartbeatService(manager), server)
    pb_grpc.add_CacheMissServicer_to_server(CacheMissService(manager), server)
    server.add_insecure_port(address="{}:{}".format(manager.managerconf['bind'], manager.managerconf['port']))
    server.start()
    server.wait_for_termination()