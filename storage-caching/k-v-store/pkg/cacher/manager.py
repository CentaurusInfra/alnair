from concurrent import futures
import os
from random import shuffle
import grpc
import boto3
import hashlib
import pickle
import configparser

from requests import request
import utils.databus.databus_pb2 as dbus_pb2
import utils.databus.databus_pb2_grpc as dbus_grpc
import datetime
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict
from pymongo.mongo_client import MongoClient
from pymongo.collection import Collection
from redis import Redis, RedisCluster
from utils.utils import *

logger = get_logger(name=__name__, level='DEBUG')


class Manager:
    def __init__(self) -> None:
        config = configparser.ConfigParser()
        config.read('cacher.conf')
        
        mconf = config['mongo']
        self.mongo_client = MongoClient(
            host=mconf.host,
            port=mconf.port,
            username=mconf.user,
            password=mconf.password,
        )
        
        redconf = config['redis']
        if redconf.cluster_mode:
            self.redis =  RedisCluster(host=redconf.host, port=int(redconf.port), password=redconf.password)
        else:
            self.redis_client = Redis(host=redconf.host, port=int(redconf.port), password=redconf.password)
    
    def auth_client(self, username, password, conn_check=False):
        result = self.mongo_client.Cacher.Client.find_one(filter={"$and": [{"username": username}, {"password": password}]})
        if result is not None:
            if conn_check:
                if request['status']:
                    return result
                else:
                    return None
            else:
                return result
        else:
            return None
    
    def auth_job(self, jobId):
        result = self.mongo_client.Cacher.Job.find_one(filter={"meta.jobId": jobId})
        if result is not None:
            return result
        else:
            return None
            
    def gen_policy(self, request: dbus_pb2.ConnectRequest, dataset_metainfo: dict):
        policy = dbus_pb2.Policy()
        return policy
    
    def copy_data_to_redis(self, bucket, policy: dbus_pb2.Policy):
        pass
    
    def evict_job(self, jobId):
        pass
    

class Connection(dbus_grpc.ConnectionServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def Connect(self, request, context):
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
        return resp



class Registration(dbus_grpc.RegistrationServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        self.client_col = manager.mongo_client['Cacher']['Client']
        self.job_col = manager.mongo_client['Cacher']['Job']

    def Register(self, request, context):
        client = self.manager.auth_client(request.cred.username, request.cred.password, conn_check=True)
        job = self.manager.auth_job(request.meta.jobId)
        if (client is not None) and (job is None):
            s3_session = boto3.Session(
                aws_access_key_id=request.s3conn.aws_access_key_id,
                aws_secret_access_key=request.s3conn.aws_secret_access_key,
                region_name=request.s3conn.region_name
            )
            s3 = s3_session.resource('s3')
            s3_client = s3_session.client()
            bucket = s3.Bucket(request.bucket)
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=request.bucket)
            dataset_metainfo = {}
            for bucket in page_iterator:
                for file in bucket['Contents']:
                    if file not in request.keys:
                        continue
                    try:
                        metadata = s3_client.head_object(Bucket=request.bucket, Key=file['Key'])
                        dataset_metainfo.update({file['Key']: metadata})
                    except:
                        print("Failed {}".format(file['Key']))

            policy = self.manager.gen_policy(request, dataset_metainfo)
            self.manager.copy_data_to_redis(bucket=bucket, policy=policy)
            now = datetime.datetime.utcnow().timestamp()
            req_dict = {
                "meta": {
                    "username": request.cred.username,
                    "jobId": "{username}_{datasetinfo}_{now}".format(
                        username=request.cred.username, 
                        datasetinfo=pickle.dumps(dataset_metainfo)[:10], 
                        now=str(now).split('.')[0]),
                    "s3conn": request.s3conn,
                    "resourceInfo": request.resource,
                    "dataset": request.bucket,
                    "createTime": Timestamp(seconds=int(now), nanos=int(now % 1 * 1e9)),
                    "token": hashlib.sha256(pickle.dumps(request)).hexdigest(),
                    "tokenTimeout": Timestamp(seconds=int(now), nanos=int(now % 1 * 1e9))
                },
                "QoS": {
                    "useCache": request.useCache,
                    "flushFreq": request.flushFreq,
                    "durabilityInMem": request.durabilityInMem,
                    "durabilityInDisk": request.durabilityInDisk
                },
                "policy": {
                    "createTime": Timestamp(seconds=int(now), nanos=int(now % 1 * 1e9)),
                    "chunkSize": policy.chunkSize,
                    "chunkKeys": policy.chunkKeys
                }
            }
            resp = dbus_pb2.RegisterResponse(
                dbus_pb2.RegisterSuccess(
                    jinfo=dbus_pb2.JobInfo(
                        jobId=req_dict['meta']['jobId'],
                        token=req_dict['meta']['token'],
                        createTime=Timestamp(seconds=int(now), nanos=int(now % 1 * 1e9)),
                        tokenTimeout=Timestamp(seconds=int(now)+600, nanos=int(now % 1 * 1e9)),
                    policy=policy)))
            result = self.job_col.insert_one(req_dict)
            assert result.acknowledged
        else:
            if client is None:
                resp = dbus_pb2.RegisterResponse(
                    dbus_pb2.RegisterError(error="Failed to register the jod, user is not connected.")
                )
            else:
                resp = dbus_pb2.RegisterResponse(
                    dbus_pb2.RegisterError(error="Job already exists, deregister first.")
                )
        return resp

    def Deresgister(self, request, context):
        client = self.manager.auth_client(request.cred.username, request.cred.password, conn_check=True)
        if client is not None:
            result = self.manager.mongo_client.Cacher.Job.delete_one(filter={"jobId": request.jinfo.jobId})
            if result.acknowledged and result.deleted_count == 1:
                resp = dbus_pb2.DeregisterResponse("successfully deregister job {jobId}".format(jobId=request.jinfo.jobId))
                if request.deleteDataset:
                    self.manager.evict_job(request.jinfo.jobId)
            else:
                resp = dbus_pb2.DeregisterResponse(response='Failed to deregister job {jobId}'.format(jobId=request.jinfo.jobId))
        else:
            resp = dbus_pb2.DeregisterResponse(response="client is not connected")
        return resp


class Heartbeat(dbus_grpc.HeartbeatServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager

    def HB(self, request, context):
        job = self.manager.auth_job(request.jinfo.jobId)
        if job is not None:
            now = datetime.datetime.utcnow().timestamp()
            tokenTimeout = datetime.fromtimestamp(request.jinfo.tokenTimeout.seconds + request.jinfo.tokenTimeout.nanos/1e9)
            if tokenTimeout > now:
                new_token = shuffle(request.jinfo.token) # TODO: update token
                request.jinfo.token = new_token
            result = self.manager.mongo_client.Cacher.Client.update_one(
                filter={"username": job['username']},
                update={"$set": {"lastHeartbeat": Timestamp(seconds=int(now)+600, nanos=int(now % 1 * 1e9))}})
            if result.acknowledged and result.modified_count == 1:
                logger.info('HB from {0} for {1}'.format(job['username'], job['jobId']))
            return request

class UpdatePolicy(dbus_grpc.UpdatePolicyServicer):
    def Update(self, request, context):
        """
        TODO:
        1. decide whether to preempt dataset and load data from disk based on the eviction strategy
        2. reply new policy
        """
        return super().Update(request, context)


class Logger(dbus_grpc.LoggerServicer):
    def call(self, request, context):
        """
        TODO:
        1. Analyze training logs
        2. re-load data from S3 without changing the redis keys
        """
        return super().call(request, context)