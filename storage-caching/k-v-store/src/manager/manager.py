import concurrent.futures
import os
from typing import List
import grpc
import boto3
import threading
import json, bson
import time
import configparser
import random
import multiprocessing
import redis
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from datetime import datetime
from google.protobuf.json_format import ParseDict
from pymongo.mongo_client import MongoClient
from redis import RedisCluster, Redis
from pytz import timezone
from utils import *


logger = get_logger(name=__name__, level='debug')


class Manager(object):
    def __init__(self):
        self.redis_conf = parse_redis_conf('/configs/redis/redis.conf')
        parser = configparser.ConfigParser()
        parser.read('/configs/manager/manager.conf')
        
        try:
            self.managerconf = parser['manager']
            mconf = parser['mongodb']
            self.redis_proxy_conf = parser['redis_proxy']
        except KeyError as err:
            logger.error(err)
        
        mongo_client = MongoClient(host=mconf['host'], port=int(mconf['port']), 
                                   username=mconf['username'], password=mconf['password'])
        with open("mongo-schemas/client.json", 'r') as f:
            client_schema = json.load(f)
        with open("mongo-schemas/job.json", 'r') as f:
            job_schema = json.load(f)
        
        # mongo_client.drop_database("Cacher")
        
        collections = mongo_client.Cacher.list_collection_names()
        if 'Client' not in collections:
            self.client_col = mongo_client.Cacher.create_collection(name="Client", validator={"$jsonSchema": client_schema}, validationAction="error")
        else:
            self.client_col = mongo_client.Cacher.Client
            
        if 'Job' not in collections: 
            self.job_col = mongo_client.Cacher.create_collection(name='Job', validator={"$jsonSchema": job_schema}, validationAction="error")
        else:
            self.job_col = mongo_client.Cacher.Job
    
        if self.managerconf['enable_proxy']:
            host = random.choice(self.redis_proxy_conf['hosts'].split(','))
            self.redis = Redis(host=host, port=self.redis_proxy_conf['port'], 
                               username=os.environ.get('REDIS_PROXY_USERNAME'), password=os.environ.get("REDIS_PROXY_PWD"))
        else:
            self.redis = RedisCluster(host="redis-cluster", port=int(self.redis_conf['port']), password=self.redis_conf['requirepass'])
        
        flush_thrd = threading.Thread(target=Manager.flush_data, args=(self,), daemon=True)
        flush_thrd.start()
        
        logger.info("start global manager")

    def auth_client(self, username, password, conn_check=False):
        result = self.client_col.find_one(filter={"username": username})
        if result is None:
                return pb.RC.NO_USER
        else:
            if password == result['password']:
                if conn_check:
                    return pb.RC.CONNECTED if result['status'] else pb.RC.DISCONNECTED
                else:
                    return pb.RC.CONNECTED
            else:
                return pb.RC.WRONG_PASSWORD
    
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
                {"$limit": int(self.managerconf['flush_amount'])},
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
                {"$limit": int(self.managerconf['flush_amount'])},
                {"$match": {"hasBackup": {"$eq": False}}},
                {"$project": {"key": 1}},
                {"$group": {"_id": "$_id", "keys": {"$push": "$key"}}},
            ]
        
        if 'maxmemory-policy' not in self.redis_conf:
            evict_policy = self.redis_config['maxmemory-policy']
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
            backup_keys = self.job_col.aggregate(pipeline)
            if backup_keys._has_next():
                backup_keys = backup_keys.next()['keys']
                for key in backup_keys:
                    backup_data = self.redis.get(key)
                    with open('{}/{}'.format(self.managerconf['backup_dir'], key), 'wb') as f:
                        f.write(backup_data)
                self.job_col.job_col.update_many(
                    {"policy.chunks": {"$elemMatch": {"key": {"$in": backup_keys}}}},
                    {"$set": {"policy.chunks.$.hasBackup": True}}
                )
            time.sleep(int(self.managerconf['flush_frequency']) * 60)


class ConnectionService(pb_grpc.ConnectionServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def connect(self, request, context):
        cred, s3auth = request.cred, request.s3auth
        rc = self.manager.auth_client(cred.username, cred.password)
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
        rc = self.manager.auth_client(cred.username, cred.password, conn_check=True)
        jobId = "{}/{}".format(cred.username, request.datasource.name)
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

            # get object keys that are not in MongoDB and Redis
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
            def list_modified_objects(client: boto3.client, redis: redis.RedisCluster, bucket_name: str, prefix: str, saved_keys: dict):
                bucket_objs = {}
                paginator = client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
                for page in pages:
                    for info in page['Contents']:
                        if saved_job is not None \
                            and info['Key'] in saved_keys \
                            and info['LastModified'].replace(tzinfo=timezone('UTC')).timestamp() == saved_keys[info['Key']]['lastModified'] \
                            and redis.exists(saved_keys[info['Key']]['location'].split(':')[1]):
                            info['Exist'] = True
                        else:
                            info['Exist'] = False
                        bucket_objs[info['Key']] = info
                return bucket_objs
            bucket_objs = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for bk in request.datasource.keys:
                    futures.append(executor.submit(list_modified_objects, s3_client, self.manager.redis, bucket_name, bk, saved_keys))
                for future in concurrent.futures.as_completed(futures):
                    bucket_objs = {**bucket_objs, **future.result()}

            chunk_size = self.manager.calculate_chunk_size(bucket_objs)

            # copy data from S3 to Redis
            def copy_data(client: boto3.client, redis: redis.RedisCluster, info: dict, bucket_name: str):
                chunk_keys = []
                if not info['Exist']:
                    if info['Size'] <= chunk_size:
                        value = client.get_object(Bucket=bucket_name, Key=info['Key'])['Body'].read()
                        hash_key = hashing(value)
                        redis.set(hash_key, value)
                        logger.info("Copy data from s3:{} to redis:{}".format(info['Key'], hash_key))
                        chunk_keys.append({
                            'name': info['Key'], 
                            'key': hash_key, 
                            'size': info['Size'], 
                            'lastModified': int(info['LastModified'].timestamp())})
                    else:
                        client.download_file(Bucket=bucket_name, Key=info['Key'], Filename='/tmp/{}'.format(info['Key']))
                        logger.info("Download large file s3:{}, size: {}B".format(info['Key'], info['Size']))
                        with open('/tmp/{}'.format(info['Key']), 'rb') as f:
                            value = f.read(chunk_size)
                            while value:
                                hash_key = hashing(value)
                                chunk_keys.append({
                                    'name': '{}'.format(info['Key']), 
                                    'key': hash_key, 
                                    'size': chunk_size, 
                                    'lastModified': int(info['LastModified'].timestamp())})
                                redis.set(hash_key, value)
                                logger.info("Copy data from /tmp/{} to redis:{}".format(info['Key'], hash_key))
                                value = f.read(chunk_size)
                else:
                    # find hash keys given the chunk name            
                    def search_key(name):
                        for ck in saved_job['policy']['chunks']:
                            if name == ck['name']:
                                return ck['location'].split(':')[1]
                        return None
                    chunk_keys.append({
                        'name': info['Key'], 
                        'key': search_key(info['Key']), 
                        'size': info['Size'], 
                        'lastModified': int(info['LastModified'].timestamp())})
                return chunk_keys
            chunk_keys = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for bk in bucket_objs:
                    info = bucket_objs[bk]
                    futures.append(executor.submit(copy_data, s3_client, self.manager.redis, info, bucket_name))
                for future in concurrent.futures.as_completed(futures):
                    chunk_keys.extend(future.result())
            
            # generate connection authorization for client
            now = datetime.utcnow().timestamp()
            token = MessageToDict(request)
            token['time'] = now
            token = hashing(token)     
            if not self.manager.managerconf['enable_proxy']: # ACL command is not supported in proxy mode
                self.manager.redis.acl_setuser(
                    username=jobId, passwords=["+{}".format(token)], 
                    enabled=True,
                    commands=['+get', '+mget', '+info'],
                    keys=list(map(lambda x: x['key'], chunk_keys)),
                    reset=True, reset_keys=False, reset_passwords=False)
                logger.info("Set redis user: {}".format(jobId))
            
            # respond client
            chunks = []
            for ck in chunk_keys:
                chunks.append({
                    "name": ck['name'],
                    "size": ck['size'],
                    "lastModified": ck['lastModified'],
                    "totalAccessTime": 0,
                    "location": "redis:{}".format(ck['key']),
                    "hasBackup": False
                })
            pb_chunks = [ParseDict(ck, pb.ChunkObj(), ignore_unknown_fields=True) for ck in chunk_keys]            
            if self.manager.managerconf['enable_proxy']:
                redisauth= pb.RedisAuth(
                            host=random.choice(self.manager.redis_proxy_conf['hosts'].split(',')), 
                            port=int(self.manager.redis_proxy_conf['port']), 
                            username=os.environ.get("REDIS_PROXY_USERNAME"), 
                            password=os.environ.get("REDIS_PROXY_PWD"))
            else:
                redisauth= pb.RedisAuth(
                            host="redis-cluster", 
                            port=int(self.manager.redis_conf['port']), 
                            username=jobId, 
                            password=token)
            
            # save jobinfo to database
            jobInfo = {
                "meta": {
                    "username": cred.username,
                    "jobId": jobId,
                    "datasource": MessageToDict(request.datasource),
                    "resourceInfo": MessageToDict(request.resource),
                    "createTime": bson.timestamp.Timestamp(int(now), inc=1),
                    "token": token,
                    "tokenTimeout": bson.timestamp.Timestamp(int(now+int(self.manager.managerconf['token_life'])), inc=1)
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
                
            # batch chunks
            batch_size = 100
            for i in range(0, len(pb_chunks), batch_size):
                resp = pb.RegisterResponse(rc=pb.RC.REGISTERED, regsucc=pb.RegisterSuccess(
                        jinfo=pb.JobInfo(
                            jobId=jobId,
                            token=token,
                            createTime=grpc_ts(now),
                            tokenTimeout=grpc_ts(now+3600),
                            redisauth=redisauth),
                        policy=pb.Policy(chunkSize=chunk_size, chunkKeys=pb_chunks[i:i+batch_size])))
                yield resp
        elif rc == pb.RC.DISCONNECTED:
            yield pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod, user is not connected."))
        else:
            yield pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod"))

    def deresgister(self, request, context):
        cred = request.cred
        jobId = request.jinfo.jobId
        rc = self.manager.auth_client(cred.username, cred.password)
        if rc in [pb.RC.CONNECTED, pb.RC.DISCONNECTED]:
            self.manager.redis.acl_deluser(username=cred.username)
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
        rc = self.manager.auth_client(cred.username, cred.password, conn_check=True)
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
            resp = self.redis.set(name=chunk['key'], value=data)
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